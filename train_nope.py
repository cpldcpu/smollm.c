"""
Continued pretraining script to generate a NoPE (No Positional Embeddings) checkpoint.

This script:
1. Loads the pretrained SmolLM2-135M model from HuggingFace
2. Modifies the model to remove RoPE (Rotary Positional Embeddings)
3. Performs continued pretraining on text data
4. Saves the NoPE checkpoint for later quantization and inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple
import json

# Import the model definition from model.py
from model import ModelConfig, RMSNorm, MLP


class AttentionNoPE(nn.Module):
    """Multi-head attention WITHOUT positional embeddings (NoPE/DroPE)"""

    def __init__(self, config: ModelConfig, use_qk_norm: bool = False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.use_qk_norm = use_qk_norm

        # Projections (same as before)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
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

        # Apply QKNorm for training stability (DroPE recommendation)
        if self.use_qk_norm:
            q = F.layer_norm(q, (self.head_dim,))
            k = F.layer_norm(k, (self.head_dim,))

        # NO RoPE APPLICATION - this is NoPE/DroPE!

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v)

        # Expand KV heads to match query heads (GQA)
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


class TransformerBlockNoPE(nn.Module):
    """Single transformer block with NoPE attention"""

    def __init__(self, config: ModelConfig, use_qk_norm: bool = False):
        super().__init__()
        self.self_attn = AttentionNoPE(config, use_qk_norm=use_qk_norm)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm attention
        residual = x
        x = self.input_layernorm(x)
        x, new_kv_cache = self.self_attn(x, attention_mask, kv_cache)
        x = residual + x

        # Pre-norm MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, new_kv_cache


class SmolLM2NoPE(nn.Module):
    """SmolLM2 model WITHOUT positional embeddings (NoPE/DroPE)"""

    def __init__(self, config: ModelConfig, use_qk_norm: bool = False):
        super().__init__()
        self.config = config
        self.use_qk_norm = use_qk_norm

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlockNoPE(config, use_qk_norm=use_qk_norm) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # LM head (tied with embeddings)
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        x = self.embed_tokens(input_ids)

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
            x, layer_new_cache = layer(x, attention_mask, layer_cache)
            if use_cache:
                new_kv_cache.append(layer_new_cache)

        # Final norm
        x = self.norm(x)

        # LM head
        if self.config.tie_word_embeddings:
            logits = F.linear(x, self.embed_tokens.weight)
        else:
            logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)

        return logits, loss, new_kv_cache


class TextDataset(Dataset):
    """Simple text dataset for pretraining"""

    def __init__(self, text_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Read and tokenize all text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize entire text
        tokens = tokenizer.encode(text)

        # Split into chunks of max_length
        self.examples = []
        for i in range(0, len(tokens) - max_length, max_length):
            self.examples.append(tokens[i:i + max_length])

        print(f"Created dataset with {len(self.examples)} examples from {text_file}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def convert_rope_to_nope(hf_model, config: ModelConfig, use_qk_norm: bool = False) -> SmolLM2NoPE:
    """Convert a RoPE model to NoPE/DroPE by copying weights"""
    nope_model = SmolLM2NoPE(config, use_qk_norm=use_qk_norm)
    hf_state = hf_model.state_dict()

    # Copy embedding
    nope_model.embed_tokens.weight.data.copy_(hf_state["model.embed_tokens.weight"])

    # Copy all layer weights
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}."

        # Attention weights (Q, K, V, O projections are the same)
        nope_model.layers[i].self_attn.q_proj.weight.data.copy_(hf_state[f"{prefix}self_attn.q_proj.weight"])
        nope_model.layers[i].self_attn.k_proj.weight.data.copy_(hf_state[f"{prefix}self_attn.k_proj.weight"])
        nope_model.layers[i].self_attn.v_proj.weight.data.copy_(hf_state[f"{prefix}self_attn.v_proj.weight"])
        nope_model.layers[i].self_attn.o_proj.weight.data.copy_(hf_state[f"{prefix}self_attn.o_proj.weight"])

        # MLP weights
        nope_model.layers[i].mlp.gate_proj.weight.data.copy_(hf_state[f"{prefix}mlp.gate_proj.weight"])
        nope_model.layers[i].mlp.up_proj.weight.data.copy_(hf_state[f"{prefix}mlp.up_proj.weight"])
        nope_model.layers[i].mlp.down_proj.weight.data.copy_(hf_state[f"{prefix}mlp.down_proj.weight"])

        # Layer norms
        nope_model.layers[i].input_layernorm.weight.data.copy_(hf_state[f"{prefix}input_layernorm.weight"])
        nope_model.layers[i].post_attention_layernorm.weight.data.copy_(hf_state[f"{prefix}post_attention_layernorm.weight"])

    # Copy final norm
    nope_model.norm.weight.data.copy_(hf_state["model.norm.weight"])

    return nope_model


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """Create learning rate scheduler with warmup then cosine decay (DroPE approach)"""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine decay after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def train(args):
    """Main training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load HuggingFace model
    print(f"Loading model from {args.model_name}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map=None
    )

    # Get config
    config = ModelConfig()

    # Convert to NoPE/DroPE model
    print(f"Converting to {'DroPE' if args.use_qk_norm else 'NoPE'} model...")
    model = convert_rope_to_nope(hf_model, config, use_qk_norm=args.use_qk_norm)
    model = model.to(device)
    del hf_model  # Free memory

    if args.use_qk_norm:
        print("Using QKNorm for training stability (DroPE recommendation)")

    # Create dataset
    print("Loading dataset...")
    dataset = TextDataset(args.data_file, tokenizer, max_length=args.seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Calculate total steps for scheduler
    steps_per_epoch = len(dataloader)
    total_steps = args.max_steps if args.max_steps > 0 else (steps_per_epoch * args.num_epochs)

    # Setup learning rate scheduler with warmup
    if args.warmup_steps > 0:
        scheduler = get_warmup_cosine_scheduler(optimizer, args.warmup_steps, total_steps)
        print(f"Using warmup scheduler: {args.warmup_steps} warmup steps, {total_steps} total steps")
    else:
        scheduler = None

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Training budget: {args.max_steps if args.max_steps > 0 else 'full dataset'} steps")
    model.train()

    global_step = 0
    for epoch in range(args.num_epochs):
        total_loss = 0
        num_batches = 0

        for batch_idx, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(device)

            # Forward pass
            logits, loss, _ = model(input_ids, labels=input_ids)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # Update learning rate scheduler
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Logging
            if global_step % args.log_interval == 0:
                avg_loss = total_loss / num_batches
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.learning_rate
                print(f"Epoch {epoch+1}/{args.num_epochs} | Step {global_step} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

            # Early stopping for testing
            if args.max_steps > 0 and global_step >= args.max_steps:
                print(f"Reached max_steps ({args.max_steps}), stopping training.")
                break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        avg_epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed | Average Loss: {avg_epoch_loss:.4f}")

    # Save model
    print(f"\nSaving NoPE model to {args.output_dir}...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), output_dir / "model_nope.pt")

    # Save config
    config_dict = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tie_word_embeddings,
        "use_rope": False  # Mark as NoPE model
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved successfully to {output_dir}")
    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train SmolLM2 with NoPE/DroPE (No/Drop Positional Embeddings)",
        epilog="""
DroPE (Dropping Positional Embeddings) recommendations:
  - Use FineWeb-Edu dataset (same as SmolLM2 pretraining)
  - Training budget: 40B-100B tokens (2-5% of 2T pretraining)
  - Learning rate: 3e-4 with warmup (490 steps)
  - Sequence length: 8192 (same as SmolLM2 pretraining)
  - QKNorm: Recommended for stability with long training

Example (minimal DroPE):
  python train_nope.py --data_file data.txt --seq_length 8192 \\
         --learning_rate 3e-4 --warmup_steps 490 --use_qk_norm \\
         --max_steps 1220703  # 40B tokens / (4 batch * 8192 seq)
        """
    )

    # Model args
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct",
                        help="HuggingFace model to load")

    # Data args
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to text file for training")
    parser.add_argument("--seq_length", type=int, default=8192,
                        help="Sequence length for training (DroPE recommends 8192 for SmolLM2)")

    # Training args
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate (DroPE uses 3e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping (0 to disable)")
    parser.add_argument("--warmup_steps", type=int, default=490,
                        help="Number of warmup steps (DroPE uses 490)")
    parser.add_argument("--use_qk_norm", action="store_true",
                        help="Use QKNorm for training stability (recommended for DroPE)")

    # Logging args
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Maximum number of training steps (0 for no limit). "
                             "For DroPE: 40B tokens = ~1.2M steps at batch=4, seq=8192")

    # Output args
    parser.add_argument("--output_dir", type=str, default="./nope_checkpoint",
                        help="Directory to save the trained model")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
