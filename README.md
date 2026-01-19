Note (16 Jan 2026): This was an agentic code generation experiment. The entire contents this repository were generated with claude code and Opus 4.5 with minimal intervention. See [auto-generated logs](docs/development_log.md) for details.


# SmolLM2 Inference Engine (C & Rust)

Lightweight inference engines for [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) in C and Rust, with INT8 (Q8) and INT4 (Q4) quantization support.

## Features

- Pure C and Rust implementations with no ML framework dependencies
- Q8 (per-tensor) and Q4 (group-wise) symmetric quantization
- Built-in BPE tokenizer with GPT2 byte encoding
- KV cache for efficient autoregressive generation
- Verified against PyTorch/HuggingFace reference
- **GGUF support**: Download and convert GGUF Q8_0 models from Hugging Face

## Project Structure

```
├── smolc/                  # C implementation
│   ├── smolc.c             # Main C inference engine (Q8)
│   ├── smolc_full.c        # Full inference engine (Q4/Q8)
│   ├── gguf_to_smol.c      # GGUF→SMOL converter
│   ├── smolc.h             # Header file
│   ├── README_CONVERTER.md # GGUF converter guide
│   └── Makefile
├── smolr/                  # Rust implementation
│   ├── src/main.rs         # Main Rust inference engine
│   └── Cargo.toml
├── docs/
│   └── development_log.md  # Detailed development history
├── download_gguf.py        # Download GGUF models from HuggingFace
├── model.py                # Bare PyTorch model implementation
├── step1_transformers.py   # HuggingFace reference
├── step2_pytorch.py        # PyTorch verification
├── step3_quantize.py       # Quantization and export
├── step4_pytorch_q8.py     # PyTorch Q8/Q4 inference
├── step6_verify.py         # C vs PyTorch verification
├── step7_verify_rust.py    # Rust vs C verification
└── models/                 # Model binaries (gitignored)
```

## Two Workflows

### Option 1: GGUF Ecosystem (Recommended)

Use pre-quantized GGUF models from Hugging Face:

```bash
# 1. Install dependencies
pip install huggingface-hub

# 2. Download GGUF model
python download_gguf.py "SmolLM2"
# Or direct: python download_gguf.py --model bartowski/SmolLM2-135M-Instruct-GGUF

# 3. Convert to SMOL format
./smolc/gguf_to_smol models/smollm2-135m-q8_0.gguf models/smollm2-135m-q8.bin

# 4. Run inference
./smolc/smolc -m models/smollm2-135m-q8.bin -p "Hello, world!" -n 50
```

**Benefits:**
- No PyTorch/transformers needed
- Access to thousands of pre-quantized models
- Smaller download size

See [smolc/README_CONVERTER.md](smolc/README_CONVERTER.md) for details.

### Option 2: Native Workflow

Convert from Hugging Face models yourself:

```bash
# 1. Install dependencies
pip install torch transformers

# 2. Download and quantize
python step1_transformers.py  # Downloads model
python step3_quantize.py      # Creates Q8 binary

# 3. Run inference
./smolc/smolc -m models/smollm2-135m-q8.bin -p "Hello, world!" -n 50
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
