# Training a Claudling: Implementation Guide

## Overview

This guide explains how to fine-tune SmolLM2-135M (or similar open-weight models) into a helpful, harmless, and honest assistant - a "claudling."

## Architecture

```
Base Model (SmolLM2-135M)
    ↓
Supervised Fine-Tuning (SFT)
    ↓
Direct Preference Optimization (DPO)
    ↓
Claudling (Mini Assistant)
```

## Phase 1: Supervised Fine-Tuning (SFT)

### Objective
Teach the model to follow instructions and hold helpful conversations.

### Dataset Preparation

#### Option A: Public Datasets (Recommended for Starting)

1. **OpenAssistant Conversations** (OASST2)
   ```bash
   # Download from Hugging Face
   from datasets import load_dataset
   dataset = load_dataset("OpenAssistant/oasst2")
   ```
   - 161K messages, 35K conversations
   - High-quality human annotations
   - Multiple languages (filter to English)

2. **Anthropic HH-RLHF**
   ```python
   dataset = load_dataset("Anthropic/hh-rlhf")
   ```
   - Conversations focused on helpfulness and harmlessness
   - Preference pairs available for later DPO phase

3. **UltraChat**
   ```python
   dataset = load_dataset("stingning/ultrachat")
   ```
   - Large-scale synthetic conversations
   - Good for diversity

#### Option B: Custom Dataset Creation

Create conversations in this format:
```json
{
  "conversations": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris. It's been the capital since the 12th century and is known for landmarks like the Eiffel Tower and Louvre Museum."}
  ]
}
```

**Quality guidelines:**
- Clear, helpful responses
- Factually accurate
- Safe and harmless
- Natural conversational tone
- Refuse harmful requests politely

### Training Configuration

**Recommended hyperparameters for SmolLM2-135M:**

```yaml
# Training
learning_rate: 2e-5
warmup_steps: 100
max_steps: 5000
batch_size: 4
gradient_accumulation_steps: 4  # Effective batch size = 16
max_seq_length: 2048

# Optimization
optimizer: adamw
weight_decay: 0.01
lr_scheduler: cosine
gradient_checkpointing: true  # Save memory

# LoRA (Parameter-Efficient Fine-Tuning)
use_lora: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Implementation with TRL (Transformer Reinforcement Learning)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

# LoRA config for efficient training
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)

# Load and format dataset
dataset = load_dataset("OpenAssistant/oasst2")
# ... format dataset into conversations ...

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./sft_output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=500,
    )
)

trainer.train()
```

## Phase 2: Direct Preference Optimization (DPO)

### Objective
Align the model with human preferences by training it to prefer better responses over worse ones.

### Dataset Preparation

**Format:** Each example has a prompt with a "chosen" and "rejected" response.

```json
{
  "prompt": "How do I make a cake?",
  "chosen": "To make a basic cake, you'll need: flour, sugar, eggs, butter, and baking powder. Mix dry ingredients, cream butter and sugar, add eggs, combine with dry ingredients, and bake at 350°F for 25-30 minutes...",
  "rejected": "Just mix stuff and put it in the oven. Easy."
}
```

**Public datasets:**
- Anthropic HH-RLHF (already has preference pairs)
- UltraFeedback
- OpenAssistant with ranking data

### Training Configuration

```python
from trl import DPOTrainer, DPOConfig

# Load SFT model from Phase 1
model = AutoModelForCausalLM.from_pretrained("./sft_output")
ref_model = AutoModelForCausalLM.from_pretrained("./sft_output")  # Reference

# DPO training
dpo_config = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=5e-6,
    beta=0.1,  # KL penalty coefficient
    max_length=2048,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## Phase 3: Export and Integration

### Export to Custom Format

After training, export the model to the SmolC binary format:

```python
# Merge LoRA weights if used
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
finetuned_model = PeftModel.from_pretrained(base_model, "./dpo_output")
merged_model = finetuned_model.merge_and_unload()

# Save to HuggingFace format
merged_model.save_pretrained("./claudling-135m")

# Quantize and export using existing script
# Modify step3_quantize.py to load from ./claudling-135m
```

### Update Inference Code

The existing SmolC inference engine will work with your fine-tuned model once exported!

## Hardware Requirements

**For SmolLM2-135M:**
- **SFT:** 8-16GB GPU (RTX 3090, A10, T4 on Colab)
- **DPO:** Similar to SFT
- **Training time:** 2-6 hours on single GPU

**For larger models (1.7B):**
- **SFT:** 24GB+ GPU (RTX 3090/4090, A100)
- Or use LoRA + gradient checkpointing on 16GB

## Evaluation

### Automated Metrics
1. **Perplexity:** Lower is better
2. **Instruction following:** Use AlpacaEval or MT-Bench
3. **Safety:** Check harmful content generation

### Manual Evaluation
```python
# Test with various prompts
prompts = [
    "What is 2+2?",
    "How do I learn Python?",
    "Write a poem about AI",
    "How do I make a bomb?",  # Should refuse politely
]

for prompt in prompts:
    # Run inference and evaluate response quality
```

## Cost-Effective Alternatives

### Free/Cheap Options
1. **Google Colab:** Free T4 GPU (limited hours)
2. **Kaggle Notebooks:** Free P100 GPU (30h/week)
3. **Lambda Labs:** ~$0.50/hour for A10
4. **RunPod/Vast.ai:** Spot instances for $0.20-0.40/hour

### Parameter-Efficient Training
- **LoRA:** Train only 0.1-1% of parameters
- **QLoRA:** LoRA + 4-bit quantization (even smaller memory)

## Quick Start Script

See `train_claudling.py` for a complete end-to-end training script.

## Datasets Summary

| Dataset | Size | Quality | Use Case |
|---------|------|---------|----------|
| OASST2 | 35K convos | High | SFT (primary) |
| HH-RLHF | 161K examples | High | SFT + DPO |
| UltraChat | 1.4M | Medium | SFT (scale) |
| UltraFeedback | 64K | High | DPO |

## Next Steps

1. Start with SFT on OASST2 (smallest, highest quality)
2. Evaluate on test prompts
3. Apply DPO for preference alignment
4. Export and test with SmolC inference
5. Iterate on data quality and training params

## Advanced: Constitutional AI

For even better alignment:
1. Generate model responses
2. Have model critique its own responses using principles
3. Revise responses based on critiques
4. Train on revised responses

This creates a more robust, self-improving assistant.

## References

- [TRL Documentation](https://huggingface.co/docs/trl)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [Constitutional AI](https://arxiv.org/abs/2212.08073)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
