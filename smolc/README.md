# SmolLM2 C Inference Engine

Lightweight C inference engine for SmolLM2-135M with INT8 (Q8) and INT4 (Q4) quantization.

## Features

- Pure ANSI C with no ML framework dependencies
- Q8 (per-tensor) and Q4 (group-wise) symmetric quantization
- Built-in BPE tokenizer with GPT2 byte encoding
- KV cache for efficient autoregressive generation
- Verified against PyTorch Q8 reference

## Building

```bash
make              # Build optimized binaries
make debug        # Build with AddressSanitizer
make clean        # Remove build artifacts
```

## Binaries

| Binary | Description |
|--------|-------------|
| `smolc` | Q8-only inference (smaller, faster) |
| `smolc_full` | Q8 and Q4 support |

## Usage

```bash
./smolc -m ../models/smollm2-135m-q8.bin -p "The capital of France is" -n 30
```

### Command Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `-m` | Model path | `../models/smollm2-135m-q8.bin` |
| `-p` | Prompt | `"The capital of France is"` |
| `-n` | Max tokens to generate | 50 |
| `-t` | Temperature (0 = greedy) | 0 |

## Files

| File | Description |
|------|-------------|
| smolc.c | Q8-only inference engine (~300 lines) |
| smolc_full.c | Full Q8/Q4 inference engine |
| smolc.h | Shared header with tensor types |
| Makefile | Build configuration |

## Verification

```bash
# From project root
python step6_verify.py
```

Compares C output against PyTorch Q8 reference implementation.
