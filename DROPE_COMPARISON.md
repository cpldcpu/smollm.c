# DroPE Implementation Comparison and Recommendations

This document compares our NoPE implementation with DroPE (Dropping Positional Embeddings) from Sakana AI and provides recommendations for improvement.

## What is DroPE?

**DroPE** (Dropping Positional Embeddings) is a method published by Sakana AI in December 2025 for extending the context window of pretrained LLMs by:

1. Taking a model pretrained with positional embeddings (e.g., RoPE)
2. Dropping the positional embeddings from the architecture
3. Performing short recalibration on the original pretraining data
4. Achieving better length generalization with minimal training cost

**Key Paper**: [Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings](https://arxiv.org/abs/2512.12167) (arXiv:2512.12167)

**Repository**: [SakanaAI/DroPE](https://github.com/SakanaAI/DroPE)

## Our Implementation vs DroPE

### Similarities ✅

1. **Core Approach**: Both remove RoPE after pretraining and perform continued training
2. **Dataset Choice**: Both use FineWeb-Edu (the SmolLM2 pretraining data)
3. **Architecture**: Both modify attention to skip positional embedding application
4. **Binary Format**: Our version 3 format with `use_rope` flag matches the concept

### Critical Differences ⚠️

| Aspect | Our Implementation | DroPE Best Practices | Impact |
|--------|-------------------|---------------------|---------|
| **Training Budget** | 500 steps (~2M tokens) | 30B-100B tokens (2-5% of pretraining) | **CRITICAL** - Insufficient training |
| **Learning Rate** | 5e-6 | 3e-4 with warmup | **HIGH** - Too conservative, slow convergence |
| **Sequence Length** | 512 | 8192 (same as pretraining) | **HIGH** - Won't learn long-range dependencies |
| **LR Warmup** | None | 490 steps | **MEDIUM** - May cause instability |
| **QKNorm** | Not implemented | Added for stability | **MEDIUM** - Large gradients may destabilize |
| **Batch Size** | 4 | Not specified | **LOW** - Probably fine |

## DroPE Results on SmolLM

### SmolLM-360M (600B pretraining tokens)
- **30B recalibration** (5%): Recovers performance, basic length extension
- **60B recalibration** (10%): Near-full recovery, good length extension
- **120B recalibration** (20%): Exceeds baseline, excellent length extension

### SmolLM-1.7B (1T pretraining tokens)
- **20B recalibration** (2%): Full recovery + improved length generalization

### SmolLM2-135M (2T pretraining tokens)
**Recommended**: 40B-100B tokens (2-5% of pretraining budget)

## Recommended Training Configuration for SmolLM2-135M

### Minimal Viable DroPE (40B tokens, ~2% budget)

```python
python train_nope.py \
    --data_file training_data.txt \
    --output_dir smollm2_drope_minimal \
    --batch_size 4 \
    --seq_length 8192 \
    --learning_rate 3e-4 \
    --warmup_steps 490 \
    --max_steps 10000000 \
    --tokens_to_train 40000000000 \
    --weight_decay 0.01 \
    --log_interval 100
```

**Calculation**: 40B tokens / (batch_size=4 × seq_len=8192) = ~1.2M steps

### Standard DroPE (60B tokens, ~3% budget)

```python
python train_nope.py \
    --data_file training_data.txt \
    --output_dir smollm2_drope_standard \
    --batch_size 4 \
    --seq_length 8192 \
    --learning_rate 3e-4 \
    --warmup_steps 490 \
    --max_steps 15000000 \
    --tokens_to_train 60000000000 \
    --weight_decay 0.01 \
    --log_interval 100
```

### Full DroPE (100B tokens, ~5% budget)

```python
python train_nope.py \
    --data_file training_data.txt \
    --output_dir smollm2_drope_full \
    --batch_size 4 \
    --seq_length 8192 \
    --learning_rate 3e-4 \
    --warmup_steps 490 \
    --max_steps 25000000 \
    --tokens_to_train 100000000000 \
    --weight_decay 0.01 \
    --log_interval 100
```

## Implementation Changes Needed

### 1. Add Learning Rate Warmup (CRITICAL)

```python
# In train_nope.py, add warmup scheduler
from torch.optim.lr_scheduler import LambdaLR

def get_warmup_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    return LambdaLR(optimizer, lr_lambda)

# After creating optimizer:
scheduler = get_warmup_scheduler(optimizer, warmup_steps=490)

# In training loop:
scheduler.step()
```

### 2. Increase Sequence Length (CRITICAL)

Change default from 512 to 8192 to match SmolLM2 pretraining context:

```python
parser.add_argument("--seq_length", type=int, default=8192,
                    help="Sequence length (use 8192 for SmolLM2)")
```

### 3. Update Learning Rate (HIGH PRIORITY)

Change from conservative 5e-6 to DroPE's 3e-4:

```python
parser.add_argument("--learning_rate", type=float, default=3e-4,
                    help="Learning rate (DroPE uses 3e-4)")
```

### 4. Add QKNorm (MEDIUM PRIORITY)

Add Query-Key Normalization for training stability:

```python
# In train_nope.py AttentionNoPE class
class AttentionNoPE(nn.Module):
    def __init__(self, config: ModelConfig, use_qk_norm: bool = True):
        super().__init__()
        # ... existing code ...
        self.use_qk_norm = use_qk_norm

    def forward(self, x, ...):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply QKNorm (layer norm on head dimension)
        if self.use_qk_norm:
            q = F.layer_norm(q, (self.head_dim,))
            k = F.layer_norm(k, (self.head_dim,))

        # ... rest of attention ...
```

### 5. Add Token-Based Training Budget (CRITICAL)

Track tokens instead of just steps:

```python
parser.add_argument("--tokens_to_train", type=int, default=40000000000,
                    help="Total tokens to train (default: 40B for minimal DroPE)")

# In training loop:
tokens_seen = global_step * args.batch_size * args.seq_length
if tokens_seen >= args.tokens_to_train:
    break
```

## Dataset Considerations

### Current Setup (Good ✅)
- Using FineWeb-Edu matches SmolLM2 pretraining ✅
- Streaming from HuggingFace ✅

### Recommendations
- **Sample size**: Need much more data for 40B-100B tokens
- **Diversity**: FineWeb-Edu 10BT sample should be sufficient
- **Quality**: FineWeb-Edu already filtered, no additional filtering needed

### Updated Data Preparation

```python
# For 40B token training
python prepare_training_data.py \
    --mode fineweb-edu \
    --num_samples 1000000 \
    --max_tokens 50000000000 \
    --output drope_training_data.txt
```

Note: With streaming, we don't need to download everything upfront.

## Computational Requirements

### GPU Memory for seq_length=8192
- **Batch size 4**: ~40-50GB (A100 80GB recommended)
- **Batch size 2**: ~25-30GB (A6000 48GB, A100 40GB)
- **Batch size 1**: ~15-20GB (RTX 4090 24GB, A10 24GB)

### Training Time Estimates

For 40B tokens on a single A100:
- Seq length 8192, batch 4: ~32K tokens/batch
- Throughput: ~1000 tokens/sec (estimated)
- Total time: 40B / 1000 = 40M seconds ≈ **463 days** single GPU
- **Multi-GPU recommended**: 8×A100 = ~58 days

For 60B tokens: ~87 days on 8×A100

### Cost-Effective Approach
- Use gradient accumulation to simulate larger batches
- Reduce to seq_length=4096 (half of 8192) for 4× speedup
- Start with minimal 40B budget

## Performance Validation

### Expected Results (based on DroPE paper)
- **5B tokens**: 95% performance recovery
- **20B tokens**: Full performance recovery
- **40B+ tokens**: Exceeds baseline on length generalization

### Metrics to Track
1. **Perplexity**: Should match or beat RoPE baseline
2. **Long-context tasks**: Needle in haystack, long QA
3. **Standard benchmarks**: MMLU, ARC, etc.

## Migration Path

### Phase 1: Quick Validation (Recommended First Step)
```bash
# Test with 1B tokens (~500 steps at seq_len=512)
python train_nope.py \
    --data_file training_data.txt \
    --seq_length 512 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --max_steps 500 \
    --log_interval 10
```

### Phase 2: Minimal DroPE (40B tokens)
- Implement warmup scheduler
- Increase seq_length to 8192 (or 4096 if GPU limited)
- Train for 40B tokens
- Validate performance recovery

### Phase 3: Full DroPE (100B tokens)
- Add QKNorm for stability
- Train for 100B tokens
- Validate length generalization

## Summary

**Our implementation has the right idea** but uses **insufficient training scale** based on DroPE research. Key issues:

1. ❌ **500 steps = ~2M tokens** is 0.0001% of SmolLM2's 2T pretraining (need 2-5% = 40B-100B)
2. ❌ **Learning rate too low** (5e-6 vs 3e-4) - will converge very slowly
3. ❌ **Sequence length too short** (512 vs 8192) - won't learn long-range patterns
4. ❌ **No warmup** - may cause training instability
5. ❌ **No QKNorm** - large gradients may destabilize longer training

**To match DroPE quality**, we need to update our default settings to align with their empirically validated approach.

## References

- **DroPE Paper**: [Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings](https://arxiv.org/abs/2512.12167)
- **DroPE Blog**: [pub.sakana.ai/DroPE](https://pub.sakana.ai/DroPE/)
- **Code**: [github.com/SakanaAI/DroPE](https://github.com/SakanaAI/DroPE)
- **SmolLM2 Paper**: [arXiv:2502.02737](https://arxiv.org/abs/2502.02737)
