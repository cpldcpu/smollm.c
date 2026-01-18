Note (16 Jan 2026): This was an agentic code generation experiment. The entire contents this repository were generated with claude code and Opus 4.5 with minimal intervention. See [auto-generated logs](docs/development_log.md) for details.


# SmolLM2 Inference Engine (C & Rust)

Lightweight inference engines for [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) in C and Rust, with INT8 (Q8) and INT4 (Q4) quantization support.

## Features

- Pure C and Rust implementations with no ML framework dependencies
- Q8 (per-tensor) and Q4 (group-wise) symmetric quantization
- Built-in BPE tokenizer with GPT2 byte encoding
- KV cache for efficient autoregressive generation
- Verified against PyTorch/HuggingFace reference
- **NEW**: Full training pipeline to create your own "Claudling" assistant!

## Project Structure

```
├── smolc/                    # C implementation
│   ├── smolc.c               # Main C inference engine
│   ├── smolc.h               # Header file
│   └── Makefile
├── smolr/                    # Rust implementation
│   ├── src/main.rs           # Main Rust inference engine
│   └── Cargo.toml
├── docs/
│   └── development_log.md    # Detailed development history
├── model.py                  # Bare PyTorch model implementation
├── step1_transformers.py     # HuggingFace reference
├── step2_pytorch.py          # PyTorch verification
├── step3_quantize.py         # Quantization and export
├── step4_pytorch_q8.py       # PyTorch Q8/Q4 inference
├── step6_verify.py           # C vs PyTorch verification
├── step7_verify_rust.py      # Rust vs C verification
├── train_claudling.py        # Training script (SFT + DPO)
├── evaluate_claudling.py     # Model evaluation
├── TRAINING_PLAN.md          # Complete training guide
├── QUICKSTART_TRAINING.md    # Quick start for training
└── models/                   # Model binaries (gitignored)
```

## Quick Start

### Build

```bash
# C
cd smolc && make

# Rust
cd smolr && cargo build --release
```

### Run Inference

```bash
# C
./smolc/smolc -m models/smollm2-135m-q8.bin -p "The capital of France is" -n 30

# Rust
./smolr/target/release/smolr -m models/smollm2-135m-q8.bin -p "The capital of France is" -n 30
```

### Command Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m` | Model path | `../models/smollm2-135m-q8.bin` |
| `-p` | Prompt | `"The capital of France is"` |
| `-n` | Max tokens to generate | 50 |
| `-t` | Temperature (0 = greedy) | 0 |

## Quantization

### Create Q8 Model (129 MB)

```bash
python step3_quantize.py
```

### Create Q4 Model (~65 MB)

```bash
python step3_quantize.py --q4 --group-size 32
```

## Verification

```bash
# Verify C vs PyTorch Q8
python step6_verify.py

# Verify Rust vs C
python step7_verify_rust.py

# Verify quantized vs FP32
python step4_pytorch_q8.py
```

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Architecture | LlamaForCausalLM |
| Layers | 30 |
| Hidden size | 576 |
| Attention heads | 9 (3 KV heads, GQA) |
| Vocabulary | 49,152 |
| Parameters | ~135M |

## Training Your Own "Claudling"

Want to train your own mini-Claude assistant? This repository now includes a complete training pipeline!

### Quick Start

```bash
# Install training dependencies
pip install -r requirements-training.txt

# Train a Claudling (2-4 hours on single GPU)
python train_claudling.py \
    --phase sft \
    --dataset hh-rlhf \
    --max-samples 10000 \
    --output-dir ./claudling-sft \
    --merge-lora

# Evaluate your model
python evaluate_claudling.py --model ./claudling-sft_merged --interactive

# Export to SmolC format and run with C inference!
# (Edit step3_quantize.py to point to your model, then:)
python step3_quantize.py
./smolc/smolc -m models/smollm2-135m-q8.bin -p "Hello! Who are you?" -n 100
```

### Training Methods

1. **Supervised Fine-Tuning (SFT)**: Teach the model conversational patterns
2. **Direct Preference Optimization (DPO)**: Align with human preferences
3. **LoRA/QLoRA**: Efficient training (works on 8GB GPUs)

### Documentation

- **[QUICKSTART_TRAINING.md](QUICKSTART_TRAINING.md)** - Get started in 30 minutes
- **[TRAINING_PLAN.md](TRAINING_PLAN.md)** - Complete training methodology
- [docs/development_log.md](docs/development_log.md) - Inference implementation notes

### What Can You Build?

- Personal assistant tuned to your needs
- Domain-specific helper (coding, writing, tutoring)
- Experiment with alignment techniques
- Learn how modern AI assistants are trained

Training a 135M model takes just 2-4 hours on a single GPU (or free on Colab)!

## Documentation

See [docs/development_log.md](docs/development_log.md) for detailed implementation notes, issues encountered, and solutions.

## License

MIT
