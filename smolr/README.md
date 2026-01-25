# SmolLM2 Rust Inference Engine

Rust implementation of SmolLM2-135M inference with Q8 quantization. Direct translation of the C implementation.

## Features

- Pure Rust with minimal dependencies (only `rand` for sampling)
- Reads existing Q8 binary model files from the C implementation
- Identical output to C implementation
- Same command-line interface as C version

## Building

```bash
cargo build --release
```

## Usage

```bash
./target/release/smolr -m ../models/smollm2-135m-q8.bin -p "The capital of France is" -n 30
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
| src/main.rs | Full inference engine (~450 lines) |
| Cargo.toml | Project configuration |

## Verification

```bash
# From project root
python step7_verify_rust.py
```

Compares Rust output against C reference implementation.

## Dependencies

- `rand` 0.8 — Random number generation for temperature sampling
