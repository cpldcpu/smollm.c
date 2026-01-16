# SmolLM2 Inference Engine (C & Rust)

Lightweight inference engines for [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) in C and Rust, with INT8 (Q8) and INT4 (Q4) quantization support.

## Features

- Pure C and Rust implementations with no ML framework dependencies
- Q8 (per-tensor) and Q4 (group-wise) symmetric quantization
- Built-in BPE tokenizer with GPT2 byte encoding
- KV cache for efficient autoregressive generation
- Verified against PyTorch/HuggingFace reference

## Project Structure

```
├── smolc/                  # C implementation
│   ├── smolc.c             # Main C inference engine
│   ├── smolc.h             # Header file
│   └── Makefile
├── smolr/                  # Rust implementation
│   ├── src/main.rs         # Main Rust inference engine
│   └── Cargo.toml
├── docs/
│   └── development_log.md  # Detailed development history
├── model.py                # Bare PyTorch model implementation
├── step1_transformers.py   # HuggingFace reference
├── step2_pytorch.py        # PyTorch verification
├── step3_quantize.py       # Quantization and export
├── step4_pytorch_q8.py     # PyTorch Q8/Q4 inference
├── step6_verify.py         # C vs PyTorch verification
├── step7_verify_rust.py    # Rust vs C verification
└── models/                 # Model binaries (gitignored)
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

## Documentation

See [docs/development_log.md](docs/development_log.md) for detailed implementation notes, issues encountered, and solutions.

## License

MIT
