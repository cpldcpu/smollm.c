# SmolLM.c GGUF Support

This directory contains GGUF format support for SmolLM2 models with Q8_0 quantization.

## Overview

`smolc_gguf` is a standalone implementation that loads and runs SmolLM2 models in the GGUF format with Q8_0 block quantization. This provides compatibility with the popular GGUF ecosystem used by llama.cpp and other inference engines.

### What is GGUF?

GGUF (GPT-Generated Unified Format) is a binary format for storing machine learning models optimized for inference. It's the successor to GGML/GGMF/GGJT formats and provides:

- Self-contained model files with all metadata included
- Efficient memory-mapped loading
- Standardized quantization formats
- Wide ecosystem compatibility

### What is Q8_0?

Q8_0 is an 8-bit block quantization format:
- **Block size**: 32 weights per block
- **Scale factor**: 1 FP16 (half-precision) scale per block
- **Storage**: 34 bytes per block (2 bytes scale + 32 bytes int8 values)
- **Quality**: Near-lossless, very close to FP16 accuracy
- **Compression**: ~4x smaller than FP32

## Files

- `smolc_gguf.h` - Header with Q8_0 block structures and API
- `smolc_gguf.c` - Complete implementation with GGUF parser and inference engine
- `Makefile` - Build configuration (use `make smolc_gguf`)

## Building

```bash
# Build GGUF loader
make smolc_gguf

# Or build all versions
make all

# Build with debug symbols
make debug
```

## Usage

### Basic Inference

```bash
./smolc_gguf -m path/to/model.gguf -p "Your prompt here" -n 50 -t 0.8
```

### Command-line Options

- `-m <path>` - Path to GGUF model file (default: `../models/smollm2-135m-q8_0.gguf`)
- `-p <text>` - Input prompt (default: "Hello")
- `-n <num>` - Maximum tokens to generate (default: 50)
- `-t <temp>` - Sampling temperature (default: 0.8, use 0 for greedy)
- `-h` - Show help message

### Examples

```bash
# Greedy sampling (deterministic)
./smolc_gguf -m model.gguf -p "The capital of France is" -n 30 -t 0

# Creative sampling
./smolc_gguf -m model.gguf -p "Once upon a time" -n 100 -t 1.0

# Short completion
./smolc_gguf -m model.gguf -p "def fibonacci(n):" -n 50 -t 0.7
```

## Getting GGUF Models

### Option 1: Download from Hugging Face

Many models are available in GGUF format on Hugging Face. Look for models with "GGUF" in the name or files ending in `.gguf`.

Example sources:
- Search for "SmolLM2 GGUF" on Hugging Face
- Look for quantized versions by community members
- Filter by Q8_0 quantization for best quality

### Option 2: Convert from Hugging Face Models

Use `llama.cpp` to convert models:

```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Convert model to GGUF FP16
python convert-hf-to-gguf.py /path/to/smollm2-135m-instruct \
  --outfile smollm2-135m-f16.gguf \
  --outtype f16

# Quantize to Q8_0
./quantize smollm2-135m-f16.gguf smollm2-135m-q8_0.gguf Q8_0
```

### Option 3: Use Existing Tools

- **Ollama**: Export GGUF from Ollama models
- **LM Studio**: Download and export models
- **GPT4All**: Access pre-quantized models

## Implementation Details

### Architecture

The GGUF loader implements:

1. **GGUF Parsing**
   - Header validation (magic: "GGUF", version: 3)
   - Metadata key-value parsing for model config
   - Tensor info array processing
   - Alignment and offset calculation

2. **Q8_0 Dequantization**
   - FP16 to FP32 scale conversion
   - Block-wise int8 to float32 conversion
   - Optimized matrix multiplication with blocks

3. **Transformer Inference**
   - Multi-head attention with GQA (Grouped Query Attention)
   - RoPE (Rotary Position Embeddings)
   - SwiGLU activation in FFN
   - RMSNorm layer normalization
   - KV cache for efficient generation

### Memory Layout

```
GGUF File Structure:
├── Header (magic + version + counts)
├── Metadata KV pairs (model config)
├── Tensor info array (names, dims, types, offsets)
├── Padding (align to 32 bytes)
└── Tensor data (Q8_0 blocks or FP32)

Q8_0 Block (34 bytes):
├── scale: uint16_t (FP16) - 2 bytes
└── qs[32]: int8_t[32]     - 32 bytes
```

### Differences from Custom Format

| Feature | Custom Format (smolc.c) | GGUF (smolc_gguf.c) |
|---------|-------------------------|---------------------|
| Format | Custom "SMOL" binary | Standard GGUF |
| Quantization | Per-tensor Q8 (1 scale) | Per-block Q8_0 (scale per 32 values) |
| Scale precision | FP32 | FP16 |
| Metadata | Fixed header fields | Flexible key-value pairs |
| Compatibility | Custom only | llama.cpp ecosystem |
| Quality | Good | Slightly better (block-wise) |

## Tokenizer Notes

**Current Limitation**: The GGUF implementation includes a placeholder tokenizer. For production use, you should:

1. Parse GGUF tokenizer metadata (tokens, scores, types)
2. Implement proper BPE/SentencePiece tokenizer
3. Handle special tokens (BOS, EOS, PAD)

The current version uses a simple character-level fallback for demonstration.

## Performance

Expected performance on modern CPU:
- **SmolLM2-135M-Q8_0**: ~10-50 tokens/sec (single-threaded)
- **Memory usage**: ~140 MB for model + 50 MB for KV cache

Optimization opportunities:
- SIMD vectorization (AVX2/AVX-512)
- Multi-threading
- Memory-mapped file loading
- Fused operations

## Limitations

1. **Tokenizer**: Placeholder implementation (see notes above)
2. **Q8_0 only**: No support for other GGUF quant types (Q4_K, Q5_K, etc.)
3. **SmolLM2 only**: Architecture-specific (llama-style transformers)
4. **Single-threaded**: No parallel processing yet

## Future Enhancements

- [ ] Full GGUF tokenizer parsing
- [ ] Support for Q4_K_M, Q5_K_S quantization
- [ ] Memory-mapped file loading
- [ ] SIMD optimizations
- [ ] Multi-threading
- [ ] Streaming token output
- [ ] Batch inference
- [ ] Cross-architecture support (ARM, RISC-V)

## References

- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [GGML Quantization](https://github.com/ggml-org/ggml/blob/master/include/ggml/ggml.h)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Hugging Face GGUF Documentation](https://huggingface.co/docs/hub/en/gguf)

## License

Same as parent project.
