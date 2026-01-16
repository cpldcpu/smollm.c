"""
Bare PyTorch implementation of SmolLM2-135M (LLaMA architecture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    hidden_size: int = 576
    intermediate_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    vocab_size: int = 49152
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tensors (HuggingFace compatible)"""
    # Compute inverse frequencies: 1 / (theta^(2i/d)) for i in [0, d/2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # Create position indices: [max_seq_len]
    t = torch.arange(max_seq_len)
    # Outer product: [max_seq_len, head_dim/2]
    freqs = torch.outer(t, inv_freq)
    # Duplicate to match full head_dim: [max_seq_len, head_dim]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input (HuggingFace compatible)"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to q and k tensors (HuggingFace compatible)"""
    # q, k: [batch, n_heads, seq_len, head_dim]
    # cos, sin: [max_seq_len, head_dim]
    # position_ids: [batch, seq_len]

    # Gather cos/sin for current positions: [batch, seq_len, head_dim]
    cos = cos[position_ids]  # [batch, seq_len, head_dim]
    sin = sin[position_ids]  # [batch, seq_len, head_dim]

    # Add head dimension: [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA)"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, num_heads * head_dim]
        k = self.k_proj(x)  # [batch, seq_len, num_kv_heads * head_dim]
        v = self.v_proj(x)  # [batch, seq_len, num_kv_heads * head_dim]

        # Reshape: [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, cos, sin, position_ids)

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v)

        # Expand KV heads to match query heads (GQA)
        # [batch, num_kv_heads, kv_seq_len, head_dim] -> [batch, num_heads, kv_seq_len, head_dim]
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]

        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Output projection
        output = self.o_proj(attn_output)

        return output, new_kv_cache


class MLP(nn.Module):
    """SwiGLU MLP"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Single transformer block"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        x, new_kv_cache = self.self_attn(x, cos, sin, position_ids, attention_mask, kv_cache)
        x = residual + x

        # Pre-norm MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_kv_cache


class SmolLM2(nn.Module):
    """SmolLM2 model (LLaMA architecture)"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # LM head (tied with embeddings)
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Precompute RoPE frequencies
        cos, sin = precompute_rope_freqs(config.head_dim, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        x = self.embed_tokens(input_ids)

        # Create position IDs if not provided
        if position_ids is None:
            if kv_cache is not None and kv_cache[0] is not None:
                # During generation with cache
                past_len = kv_cache[0][0].shape[2]
                position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            else:
                position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Create causal mask
        if attention_mask is None:
            if kv_cache is not None and kv_cache[0] is not None:
                past_len = kv_cache[0][0].shape[2]
                total_len = past_len + seq_len
            else:
                past_len = 0
                total_len = seq_len

            # Causal mask: [1, 1, seq_len, total_len]
            mask = torch.full((seq_len, total_len), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=past_len + 1)
            attention_mask = mask.unsqueeze(0).unsqueeze(0)

        # Initialize KV cache if needed
        if use_cache and kv_cache is None:
            kv_cache = [None] * self.config.num_hidden_layers

        new_kv_cache = [] if use_cache else None

        # Forward through layers
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, layer_new_cache = layer(x, self.rope_cos, self.rope_sin, position_ids, attention_mask, layer_cache)
            if use_cache:
                new_kv_cache.append(layer_new_cache)

        # Final norm
        x = self.norm(x)

        # LM head
        if self.config.tie_word_embeddings:
            logits = F.linear(x, self.embed_tokens.weight)
        else:
            logits = self.lm_head(x)

        return logits, new_kv_cache


def load_hf_weights(model: SmolLM2, hf_model) -> None:
    """Load weights from a HuggingFace model"""
    hf_state = hf_model.state_dict()

    # Embedding
    model.embed_tokens.weight.data.copy_(hf_state["model.embed_tokens.weight"])

    # Layers
    for i in range(model.config.num_hidden_layers):
        prefix = f"model.layers.{i}."

        # Attention
        model.layers[i].self_attn.q_proj.weight.data.copy_(hf_state[f"{prefix}self_attn.q_proj.weight"])
        model.layers[i].self_attn.k_proj.weight.data.copy_(hf_state[f"{prefix}self_attn.k_proj.weight"])
        model.layers[i].self_attn.v_proj.weight.data.copy_(hf_state[f"{prefix}self_attn.v_proj.weight"])
        model.layers[i].self_attn.o_proj.weight.data.copy_(hf_state[f"{prefix}self_attn.o_proj.weight"])

        # MLP
        model.layers[i].mlp.gate_proj.weight.data.copy_(hf_state[f"{prefix}mlp.gate_proj.weight"])
        model.layers[i].mlp.up_proj.weight.data.copy_(hf_state[f"{prefix}mlp.up_proj.weight"])
        model.layers[i].mlp.down_proj.weight.data.copy_(hf_state[f"{prefix}mlp.down_proj.weight"])

        # Norms
        model.layers[i].input_layernorm.weight.data.copy_(hf_state[f"{prefix}input_layernorm.weight"])
        model.layers[i].post_attention_layernorm.weight.data.copy_(hf_state[f"{prefix}post_attention_layernorm.weight"])

    # Final norm
    model.norm.weight.data.copy_(hf_state["model.norm.weight"])

    # LM head (tied, so nothing to copy if using tied embeddings)
    if not model.config.tie_word_embeddings and "lm_head.weight" in hf_state:
        model.lm_head.weight.data.copy_(hf_state["lm_head.weight"])
