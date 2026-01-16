"""
Step 4: PyTorch Q8/Q4 inference - load from binary file and verify vs FP32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List
from transformers import AutoTokenizer

MODEL_FILE = "./models/smollm2-135m-q8.bin"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
CACHE_DIR = "./models/hf_cache"

QUANT_Q8 = 0
QUANT_Q4 = 1


@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    max_seq_len: int
    rope_theta: float
    rms_norm_eps: float
    quant_type: int
    group_size: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class Q8Tensor:
    """Quantized int8 tensor with scale"""
    def __init__(self, scale: float, data: torch.Tensor):
        self.scale = scale
        self.data = data  # int8 tensor
        self._cache = None

    def dequantize(self) -> torch.Tensor:
        if self._cache is None:
            self._cache = self.data.float() * self.scale
        return self._cache


class Q4Tensor:
    """Quantized int4 tensor with group-wise scales and packed data"""
    def __init__(self, scales: torch.Tensor, data: torch.Tensor, group_size: int, shape: tuple):
        self.scales = scales
        self.data = data  # uint8 packed (2 values per byte)
        self.group_size = group_size
        self.shape = shape
        self.numel = 1
        for s in shape:
            self.numel *= s
        self._cache = None

    def dequantize(self) -> torch.Tensor:
        if self._cache is not None:
            return self._cache

        data = self.data
        low = (data & 0x0F).to(torch.int8)
        high = ((data >> 4) & 0x0F).to(torch.int8)
        low = low - (low >= 8).to(torch.int8) * 16
        high = high - (high >= 8).to(torch.int8) * 16

        vals = torch.empty(data.numel() * 2, dtype=torch.int8)
        vals[0::2] = low
        vals[1::2] = high

        scales = self.scales.repeat_interleave(self.group_size)
        out = vals[:self.numel].float() * scales[:self.numel]
        self._cache = out.view(self.shape)
        return self._cache


class QuantLinear:
    """Linear layer with quantized weights (dequantize on-the-fly)"""
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize and apply linear
        w = self.weight.dequantize()
        return F.linear(x, w)


class RMSNorm:
    """Root Mean Square Layer Normalization"""
    def __init__(self, weight: torch.Tensor, eps: float = 1e-5):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tensors"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TransformerLayer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.input_layernorm: RMSNorm = None
        self.q_proj: QuantLinear = None
        self.k_proj: QuantLinear = None
        self.v_proj: QuantLinear = None
        self.o_proj: QuantLinear = None
        self.post_attention_layernorm: RMSNorm = None
        self.gate_proj: QuantLinear = None
        self.up_proj: QuantLinear = None
        self.down_proj: QuantLinear = None

    def attention(self, x, cos, sin, position_ids, mask, kv_cache):
        batch_size, seq_len, _ = x.shape
        head_dim = self.config.head_dim
        num_heads = self.config.num_heads
        num_kv_heads = self.config.num_kv_heads
        num_kv_groups = num_heads // num_kv_heads

        q = self.q_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin, position_ids)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v)

        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)

        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(attn_output), new_kv_cache

    def mlp(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def __call__(self, x, cos, sin, position_ids, mask, kv_cache):
        residual = x
        x = self.input_layernorm(x)
        x, new_kv_cache = self.attention(x, cos, sin, position_ids, mask, kv_cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_kv_cache


class SmolLM2Q8:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.embed_tokens = None
        self.layers: List[TransformerLayer] = []
        self.norm: RMSNorm = None
        self.rope_cos, self.rope_sin = precompute_rope_freqs(
            config.head_dim, config.max_seq_len, config.rope_theta
        )

    def forward(self, input_ids, kv_cache=None, use_cache=False):
        batch_size, seq_len = input_ids.shape

        # Embedding
        x = F.embedding(input_ids, self.embed_tokens.dequantize())

        # Position IDs
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0].shape[2]
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        else:
            past_len = 0
            position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Causal mask
        total_len = past_len + seq_len
        mask = torch.full((seq_len, total_len), float("-inf"), device=input_ids.device)
        mask = torch.triu(mask, diagonal=past_len + 1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Initialize KV cache
        if use_cache and kv_cache is None:
            kv_cache = [None] * self.config.num_layers

        new_kv_cache = [] if use_cache else None

        # Forward through layers
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_cache = layer(x, self.rope_cos, self.rope_sin, position_ids, mask, layer_cache)
            if use_cache:
                new_kv_cache.append(layer_new_cache)

        # Final norm
        x = self.norm(x)

        # LM head (tied with embeddings)
        logits = F.linear(x, self.embed_tokens.dequantize())

        return logits, new_kv_cache


def read_q8_tensor(f, shape: tuple) -> Q8Tensor:
    """Read a Q8 tensor from file"""
    scale = struct.unpack('f', f.read(4))[0]
    numel = 1
    for s in shape:
        numel *= s
    data = torch.frombuffer(bytearray(f.read(numel)), dtype=torch.int8).view(shape)
    return Q8Tensor(scale, data)


def read_q4_tensor(f, shape: tuple, group_size: int) -> Q4Tensor:
    """Read a Q4 tensor from file"""
    num_groups = struct.unpack('I', f.read(4))[0]
    scales = torch.frombuffer(bytearray(f.read(num_groups * 4)), dtype=torch.float32)
    padded = num_groups * group_size
    packed = torch.frombuffer(bytearray(f.read(padded // 2)), dtype=torch.uint8)
    return Q4Tensor(scales, packed, group_size, shape)


def read_quant_tensor(f, shape: tuple, quant_type: int, group_size: int):
    if quant_type == QUANT_Q8:
        return read_q8_tensor(f, shape)
    if quant_type == QUANT_Q4:
        return read_q4_tensor(f, shape, group_size)
    raise ValueError(f"Unsupported quant_type: {quant_type}")


def read_fp32_tensor(f, shape: tuple) -> torch.Tensor:
    """Read an FP32 tensor from file"""
    numel = 1
    for s in shape:
        numel *= s
    data = torch.frombuffer(bytearray(f.read(numel * 4)), dtype=torch.float32).view(shape)
    return data.clone()


def load_model(filepath: str) -> Tuple[SmolLM2Q8, List[str], List[str]]:
    """Load model from binary file"""
    with open(filepath, 'rb') as f:
        # Header
        magic = f.read(4)
        assert magic == b'SMOL', f"Invalid magic: {magic}"

        version = struct.unpack('I', f.read(4))[0]
        if version == 1:
            quant_type = QUANT_Q8
            group_size = 0
        elif version == 2:
            quant_type = struct.unpack('I', f.read(4))[0]
            group_size = struct.unpack('I', f.read(4))[0]
        else:
            raise AssertionError(f"Unsupported version: {version}")

        hidden_size = struct.unpack('I', f.read(4))[0]
        intermediate_size = struct.unpack('I', f.read(4))[0]
        num_layers = struct.unpack('I', f.read(4))[0]
        num_heads = struct.unpack('I', f.read(4))[0]
        num_kv_heads = struct.unpack('I', f.read(4))[0]
        vocab_size = struct.unpack('I', f.read(4))[0]
        max_seq_len = struct.unpack('I', f.read(4))[0]
        rope_theta = struct.unpack('f', f.read(4))[0]
        rms_norm_eps = struct.unpack('f', f.read(4))[0]

        config = ModelConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            quant_type=quant_type,
            group_size=group_size
        )

        print(f"Loaded config: {config}")
        print(f"Quantization: {'Q4' if quant_type == QUANT_Q4 else 'Q8'}")

        # Tokenizer
        num_vocab = struct.unpack('I', f.read(4))[0]
        num_merges = struct.unpack('I', f.read(4))[0]

        vocab = []
        for _ in range(num_vocab):
            token_len = struct.unpack('I', f.read(4))[0]
            token_bytes = f.read(token_len)
            vocab.append(token_bytes.decode('utf-8'))

        merges = []
        for _ in range(num_merges):
            merge_len = struct.unpack('I', f.read(4))[0]
            merge_bytes = f.read(merge_len)
            merges.append(merge_bytes.decode('utf-8'))

        print(f"Loaded tokenizer: {num_vocab} vocab, {num_merges} merges")

        # Model weights
        model = SmolLM2Q8(config)
        head_dim = config.head_dim

        # Embedding
        model.embed_tokens = read_quant_tensor(f, (vocab_size, hidden_size), quant_type, group_size)

        # Layers
        for i in range(num_layers):
            layer = TransformerLayer(config)
            layer.input_layernorm = RMSNorm(read_fp32_tensor(f, (hidden_size,)), rms_norm_eps)
            layer.q_proj = QuantLinear(read_quant_tensor(f, (num_heads * head_dim, hidden_size), quant_type, group_size))
            layer.k_proj = QuantLinear(read_quant_tensor(f, (num_kv_heads * head_dim, hidden_size), quant_type, group_size))
            layer.v_proj = QuantLinear(read_quant_tensor(f, (num_kv_heads * head_dim, hidden_size), quant_type, group_size))
            layer.o_proj = QuantLinear(read_quant_tensor(f, (hidden_size, num_heads * head_dim), quant_type, group_size))
            layer.post_attention_layernorm = RMSNorm(read_fp32_tensor(f, (hidden_size,)), rms_norm_eps)
            layer.gate_proj = QuantLinear(read_quant_tensor(f, (intermediate_size, hidden_size), quant_type, group_size))
            layer.up_proj = QuantLinear(read_quant_tensor(f, (intermediate_size, hidden_size), quant_type, group_size))
            layer.down_proj = QuantLinear(read_quant_tensor(f, (hidden_size, intermediate_size), quant_type, group_size))
            model.layers.append(layer)

        # Final norm
        model.norm = RMSNorm(read_fp32_tensor(f, (hidden_size,)), rms_norm_eps)

        return model, vocab, merges


def main():
    parser = argparse.ArgumentParser(description="Verify quantized model vs FP32")
    parser.add_argument("--model", type=str, default=MODEL_FILE, help="Path to quantized model file")
    args = parser.parse_args()

    print(f"Loading quantized model from {args.model}...")
    model, vocab, merges = load_model(args.model)
    quant_label = "Q4" if model.config.quant_type == QUANT_Q4 else "Q8"

    print("\nLoading HuggingFace tokenizer for comparison...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    # Test prompts
    test_prompts = [
        "Hello",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    # Load FP32 reference
    print("\nLoading FP32 reference model...")
    from model import SmolLM2, ModelConfig as FP32Config, load_hf_weights
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    hf_model = hf_model.float().cpu()

    fp32_config = FP32Config()
    fp32_model = SmolLM2(fp32_config)
    fp32_model = fp32_model.float().cpu()
    fp32_model.eval()
    load_hf_weights(fp32_model, hf_model)

    print(f"\n=== Verification: {quant_label} vs FP32 ===")

    all_passed = True

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt!r}")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # FP32 forward
        with torch.no_grad():
            fp32_logits, _ = fp32_model(input_ids)

        # Quantized forward
        q_logits, _ = model.forward(input_ids)

        # Compare
        max_diff = (fp32_logits - q_logits).abs().max().item()
        mean_diff = (fp32_logits - q_logits).abs().mean().item()

        print(f"  Max absolute difference: {max_diff:.4f}")
        print(f"  Mean absolute difference: {mean_diff:.4f}")

        # Check top predictions
        fp32_top = torch.topk(fp32_logits[0, -1, :], k=5)
        q_top = torch.topk(q_logits[0, -1, :], k=5)

        print(f"  FP32 top 5: {fp32_top.indices.tolist()}")
        print(f"  {quant_label} top 5: {q_top.indices.tolist()}")

        if fp32_top.indices[0].item() == q_top.indices[0].item():
            print("  ✓ Top-1 prediction matches")
        else:
            print("  ✗ Top-1 prediction differs!")
            all_passed = False

    # Generation test
    print("\n=== Generation Test ===")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Quantized generation
    generated = input_ids.clone()
    kv_cache = None

    for _ in range(20):
        if kv_cache is None:
            logits, kv_cache = model.forward(generated, use_cache=True)
        else:
            logits, kv_cache = model.forward(generated[:, -1:], kv_cache=kv_cache, use_cache=True)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    q_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"{quant_label} generated: {q_text!r}")

    # FP32 generation
    generated = input_ids.clone()
    kv_cache = None

    for _ in range(20):
        with torch.no_grad():
            if kv_cache is None:
                logits, kv_cache = fp32_model(generated, use_cache=True)
            else:
                logits, kv_cache = fp32_model(generated[:, -1:], kv_cache=kv_cache, use_cache=True)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    fp32_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"FP32 generated: {fp32_text!r}")

    if q_text == fp32_text:
        print("✓ Generated text matches!")
    else:
        print("~ Generated text differs (expected with quantization)")

    print("\n" + "=" * 50)
    if all_passed:
        print(f"=== Step 4 Complete: {quant_label} model verified ===")
    else:
        print(f"=== Step 4: Some {quant_label} top-1 predictions differ ===")
    print("=" * 50)


if __name__ == "__main__":
    main()
