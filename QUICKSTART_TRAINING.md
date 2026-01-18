# Claudling Training Quick Start

Get a mini-Claude running in 30 minutes (or less)!

## Prerequisites

```bash
# Install training dependencies
pip install -r requirements-training.txt

# Optional: For faster training (requires CUDA)
pip install flash-attn --no-build-isolation
```

## Option 1: Fast Test Run (5-10 minutes)

Perfect for testing the pipeline on limited hardware (even CPU or Colab free tier).

```bash
# Train on a tiny subset (100 examples, 1 epoch)
python train_claudling.py \
    --phase sft \
    --dataset hh-rlhf \
    --max-samples 100 \
    --num-epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 2 \
    --output-dir ./claudling-test

# Evaluate
python evaluate_claudling.py --model ./claudling-test --interactive
```

This won't produce a great model, but validates the entire pipeline works!

## Option 2: Quick Training (2-4 hours, single GPU)

Recommended first real training run.

```bash
# Phase 1: Supervised Fine-Tuning
python train_claudling.py \
    --phase sft \
    --dataset hh-rlhf \
    --max-samples 10000 \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --output-dir ./claudling-sft \
    --gradient-checkpointing \
    --merge-lora

# Phase 2: Preference Optimization (optional but recommended)
python train_claudling.py \
    --phase dpo \
    --sft-model ./claudling-sft_merged \
    --max-samples 5000 \
    --num-epochs 1 \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --learning-rate 5e-6 \
    --output-dir ./claudling-dpo

# Evaluate
python evaluate_claudling.py --model ./claudling-dpo
```

## Option 3: Low-Memory Training (QLoRA)

For GPUs with <16GB VRAM (RTX 3060, etc.)

```bash
python train_claudling.py \
    --phase sft \
    --dataset hh-rlhf \
    --use-qlora \
    --batch-size 1 \
    --gradient-accumulation-steps 16 \
    --gradient-checkpointing \
    --output-dir ./claudling-qlora
```

QLoRA uses 4-bit quantization - your model trains in ~6GB VRAM!

## Option 4: Full Training (6+ hours)

For the best results:

```bash
# Use full dataset
python train_claudling.py \
    --phase sft \
    --dataset hh-rlhf \
    --num-epochs 3 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 \
    --output-dir ./claudling-full \
    --use-tensorboard \
    --merge-lora

# Then DPO
python train_claudling.py \
    --phase dpo \
    --sft-model ./claudling-full_merged \
    --num-epochs 1 \
    --output-dir ./claudling-final
```

## Evaluating Your Claudling

### Interactive Testing
```bash
python evaluate_claudling.py --model ./claudling-sft --interactive
```

### Automated Test Suite
```bash
python evaluate_claudling.py --model ./claudling-sft
```

### Compare Before/After
```bash
python evaluate_claudling.py \
    --model ./claudling-sft \
    --compare-with HuggingFaceTB/SmolLM2-135M-Instruct
```

## Export to SmolC Format

After training, export your model to run with the existing C/Rust inference:

```bash
# 1. Your model is in HuggingFace format at ./claudling-sft_merged
# 2. Modify step3_quantize.py to point to your model:

# In step3_quantize.py, change:
# model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
# to:
# model_name = "./claudling-sft_merged"

python step3_quantize.py

# 3. Your claudling is now at models/smollm2-135m-q8.bin
# 4. Run with existing inference:
./smolc/smolc -m models/smollm2-135m-q8.bin -p "Hello! Who are you?" -n 100
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `--batch-size 1`
2. Increase gradient accumulation: `--gradient-accumulation-steps 16`
3. Enable checkpointing: `--gradient-checkpointing`
4. Use QLoRA: `--use-qlora`
5. Reduce sequence length: `--max-seq-length 1024`

### Training is Slow

1. Use Flash Attention (requires CUDA): `pip install flash-attn`
2. Increase batch size if you have VRAM
3. Use mixed precision (enabled by default with fp16)
4. Consider a larger model - 135M is already tiny!

### Model Gives Bad Responses

1. Train longer (more epochs or samples)
2. Try different dataset (OASST2 vs HH-RLHF)
3. Run DPO phase for better alignment
4. Check if base model is instruction-tuned (use SmolLM2-135M-Instruct, not base)
5. Adjust temperature at inference time

### Loss Not Decreasing

1. Check learning rate (try 1e-5 or 5e-5)
2. Ensure dataset is formatted correctly
3. Verify gradient accumulation isn't too high
4. Check for NaN values in logs

## Hardware Requirements

| Setup | VRAM | Time (10K samples) |
|-------|------|-------------------|
| Full FP16 | 12GB | 2 hours |
| LoRA FP16 | 8GB | 2.5 hours |
| QLoRA 4-bit | 6GB | 4 hours |
| CPU only | 16GB RAM | 24+ hours (not recommended) |

## Free Training Options

1. **Google Colab Free**: T4 GPU (15GB) - ~3-4 hours of training time per day
2. **Kaggle Notebooks**: P100 GPU (16GB) - 30 hours per week
3. **Lambda Labs**: First-time credits
4. **Together.ai**: Free tier with API access

## Next Steps

After training your first claudling:

1. **Improve data quality**: Curate high-quality examples for your use case
2. **Domain specialization**: Fine-tune on coding, creative writing, etc.
3. **Scale up**: Try SmolLM2-360M or SmolLM2-1.7B
4. **Advanced alignment**: Implement Constitutional AI
5. **Distillation**: Use a larger model (Claude/GPT-4) to generate training data
6. **Deployment**: Export to GGUF for llama.cpp, or use existing SmolC

## Example Training Session

```bash
# Full workflow from start to finish

# 1. Install dependencies
pip install -r requirements-training.txt

# 2. Quick test (optional)
python train_claudling.py --phase sft --max-samples 100 --num-epochs 1 --output-dir test
python evaluate_claudling.py --model test --interactive
# Try: "What is 2+2?"

# 3. Real training
python train_claudling.py \
    --phase sft \
    --dataset hh-rlhf \
    --max-samples 10000 \
    --num-epochs 3 \
    --batch-size 4 \
    --gradient-accumulation-steps 4 \
    --output-dir claudling-sft \
    --gradient-checkpointing \
    --merge-lora \
    --use-tensorboard

# 4. Monitor training (in another terminal)
tensorboard --logdir claudling-sft

# 5. Evaluate
python evaluate_claudling.py --model claudling-sft_merged

# 6. DPO (optional)
python train_claudling.py \
    --phase dpo \
    --sft-model claudling-sft_merged \
    --max-samples 5000 \
    --output-dir claudling-final

# 7. Final evaluation
python evaluate_claudling.py --model claudling-final --interactive

# 8. Export to SmolC format
# Edit step3_quantize.py to use ./claudling-final
python step3_quantize.py

# 9. Run with C inference!
cd smolc && make
./smolc -m ../models/smollm2-135m-q8.bin -p "Tell me about yourself" -n 150
```

## Resources

- [TRL Documentation](https://huggingface.co/docs/trl) - Training library
- [PEFT Documentation](https://huggingface.co/docs/peft) - LoRA and QLoRA
- [HH-RLHF Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [OpenAssistant Dataset](https://huggingface.co/datasets/OpenAssistant/oasst2)
- [DPO Paper](https://arxiv.org/abs/2305.18290)

Happy training! 🚀
