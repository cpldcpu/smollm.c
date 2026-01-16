"""
Step 3: Quantize model to Q8 or Q4 and export to custom binary format

Binary format v2:
  Header:
    - magic: "SMOL" (4 bytes)
    - version: uint32 (4 bytes) - version 2 for Q4 support
    - quant_type: uint32 (0=Q8, 1=Q4)
    - group_size: uint32 (for Q4, typically 32 or 64)
    - hidden_size: uint32
    - intermediate_size: uint32
    - num_layers: uint32
    - num_heads: uint32
    - num_kv_heads: uint32
    - vocab_size: uint32
    - max_seq_len: uint32
    - rope_theta: float32
    - rms_norm_eps: float32

  Tokenizer: (same as before)

  Q8 tensor format:
    - scale: float32
    - data: int8[numel]

  Q4 tensor format (group-wise):
    - num_groups: uint32
    - scales: float32[num_groups]
    - data: uint8[numel/2] (packed, 2 values per byte)
"""

import torch
import numpy as np
import struct
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
CACHE_DIR = "./models/hf_cache"

QUANT_Q8 = 0
QUANT_Q4 = 1


def quantize_tensor_q8(tensor: torch.Tensor) -> tuple:
    """Quantize tensor to int8 using per-tensor symmetric quantization"""
    tensor = tensor.float().flatten()
    max_abs = tensor.abs().max().item()
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return scale, quantized


def quantize_tensor_q4(tensor: torch.Tensor, group_size: int = 32) -> tuple:
    """Quantize tensor to int4 using group-wise symmetric quantization
    Returns: (scales, packed_data) where packed_data has 2 values per byte
    """
    tensor = tensor.float().flatten()
    numel = tensor.numel()

    # Pad to multiple of group_size
    if numel % group_size != 0:
        pad_size = group_size - (numel % group_size)
        tensor = torch.cat([tensor, torch.zeros(pad_size)])
        numel = tensor.numel()

    # Reshape to groups
    num_groups = numel // group_size
    tensor = tensor.view(num_groups, group_size)

    # Compute per-group scales
    max_abs = tensor.abs().max(dim=1).values
    scales = max_abs / 7.0  # int4 range is -8 to 7, use 7 for symmetric
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)

    # Quantize each group
    quantized = torch.clamp(torch.round(tensor / scales.unsqueeze(1)), -8, 7).to(torch.int8)
    quantized = quantized.flatten()

    # Pack two int4 values into one byte
    # Lower nibble = even indices, upper nibble = odd indices
    packed = torch.zeros(numel // 2, dtype=torch.uint8)
    for i in range(numel // 2):
        low = quantized[2*i].item() & 0x0F  # Mask to 4 bits (handles negative)
        high = quantized[2*i + 1].item() & 0x0F
        packed[i] = low | (high << 4)

    return scales.float(), packed


def write_q8_tensor(f, tensor: torch.Tensor, name: str):
    """Write a Q8 quantized tensor to file"""
    scale, quantized = quantize_tensor_q8(tensor)
    data = quantized.numpy().tobytes()
    f.write(struct.pack('f', scale))
    f.write(data)
    print(f"  {name}: shape={list(tensor.shape)}, scale={scale:.6f}, size={len(data)+4} bytes (Q8)")


def write_q4_tensor(f, tensor: torch.Tensor, group_size: int, name: str):
    """Write a Q4 quantized tensor to file"""
    original_shape = tensor.shape
    scales, packed = quantize_tensor_q4(tensor, group_size)

    num_groups = scales.numel()
    f.write(struct.pack('I', num_groups))
    f.write(scales.numpy().tobytes())
    f.write(packed.numpy().tobytes())

    total_size = 4 + num_groups * 4 + packed.numel()
    print(f"  {name}: shape={list(original_shape)}, groups={num_groups}, size={total_size} bytes (Q4)")


def write_fp32_tensor(f, tensor: torch.Tensor, name: str):
    """Write an FP32 tensor to file"""
    data = tensor.float().numpy().tobytes()
    f.write(data)
    print(f"  {name}: shape={list(tensor.shape)}, size={len(data)} bytes (FP32)")


def main():
    parser = argparse.ArgumentParser(description='Quantize SmolLM2 model')
    parser.add_argument('--q4', action='store_true', help='Use Q4 quantization (default: Q8)')
    parser.add_argument('--group-size', type=int, default=32, help='Group size for Q4 (default: 32)')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    args = parser.parse_args()

    quant_type = QUANT_Q4 if args.q4 else QUANT_Q8
    group_size = args.group_size

    if args.output:
        output_file = args.output
    else:
        output_file = f"./models/smollm2-135m-{'q4' if args.q4 else 'q8'}.bin"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Quantization: {'Q4' if args.q4 else 'Q8'}")
    if args.q4:
        print(f"Group size: {group_size}")

    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = model.float().cpu()

    config = model.config
    state_dict = model.state_dict()

    print(f"\nModel config:")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  vocab_size: {config.vocab_size}")

    # Get tokenizer data
    vocab = tokenizer.get_vocab()
    vocab_list = sorted(vocab.items(), key=lambda x: x[1])

    # Load merges
    snapshot_dir = None
    for root, dirs, files in os.walk(CACHE_DIR):
        if 'merges.txt' in files:
            snapshot_dir = root
            break

    merges = []
    if snapshot_dir:
        merges_file = os.path.join(snapshot_dir, 'merges.txt')
        with open(merges_file, 'r', encoding='utf-8') as mf:
            for line in mf:
                line = line.strip()
                if line and not line.startswith('#'):
                    merges.append(line)

    print(f"\nTokenizer: {len(vocab_list)} vocab, {len(merges)} merges")
    print(f"\nWriting to {output_file}...")

    # Helper to write quantized tensor based on type
    def write_quant_tensor(f, tensor, name):
        if quant_type == QUANT_Q4:
            write_q4_tensor(f, tensor, group_size, name)
        else:
            write_q8_tensor(f, tensor, name)

    with open(output_file, 'wb') as f:
        # Header (version 2)
        print("\nWriting header...")
        f.write(b'SMOL')
        f.write(struct.pack('I', 2))  # version 2
        f.write(struct.pack('I', quant_type))
        f.write(struct.pack('I', group_size))
        f.write(struct.pack('I', config.hidden_size))
        f.write(struct.pack('I', config.intermediate_size))
        f.write(struct.pack('I', config.num_hidden_layers))
        f.write(struct.pack('I', config.num_attention_heads))
        f.write(struct.pack('I', config.num_key_value_heads))
        f.write(struct.pack('I', config.vocab_size))
        f.write(struct.pack('I', config.max_position_embeddings))
        f.write(struct.pack('f', config.rope_theta))
        f.write(struct.pack('f', config.rms_norm_eps))

        # Tokenizer
        print("\nWriting tokenizer...")
        f.write(struct.pack('I', len(vocab_list)))
        f.write(struct.pack('I', len(merges)))

        for token_str, token_id in vocab_list:
            token_bytes = token_str.encode('utf-8')
            f.write(struct.pack('I', len(token_bytes)))
            f.write(token_bytes)

        for merge in merges:
            merge_bytes = merge.encode('utf-8')
            f.write(struct.pack('I', len(merge_bytes)))
            f.write(merge_bytes)

        # Weights
        print("\nWriting weights...")

        print("Embedding:")
        write_quant_tensor(f, state_dict['model.embed_tokens.weight'], 'embed_tokens')

        for i in range(config.num_hidden_layers):
            print(f"Layer {i}:")
            prefix = f'model.layers.{i}.'

            write_fp32_tensor(f, state_dict[f'{prefix}input_layernorm.weight'], 'input_layernorm')
            write_quant_tensor(f, state_dict[f'{prefix}self_attn.q_proj.weight'], 'q_proj')
            write_quant_tensor(f, state_dict[f'{prefix}self_attn.k_proj.weight'], 'k_proj')
            write_quant_tensor(f, state_dict[f'{prefix}self_attn.v_proj.weight'], 'v_proj')
            write_quant_tensor(f, state_dict[f'{prefix}self_attn.o_proj.weight'], 'o_proj')
            write_fp32_tensor(f, state_dict[f'{prefix}post_attention_layernorm.weight'], 'post_attn_layernorm')
            write_quant_tensor(f, state_dict[f'{prefix}mlp.gate_proj.weight'], 'gate_proj')
            write_quant_tensor(f, state_dict[f'{prefix}mlp.up_proj.weight'], 'up_proj')
            write_quant_tensor(f, state_dict[f'{prefix}mlp.down_proj.weight'], 'down_proj')

        print("Final:")
        write_fp32_tensor(f, state_dict['model.norm.weight'], 'norm')

    file_size = os.path.getsize(output_file)
    print(f"\n=== Quantization Complete ===")
    print(f"Output file: {output_file}")
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
