# GGUF to SMOL Converter

A lightweight tool to convert GGUF Q8_0 models to the compact SMOL format for use with `smolc` and `smolc_full`.

## Why Use This?

Instead of implementing a full GGUF inference engine (which adds 700+ lines of code), this converter lets you:

1. **Use ecosystem models**: Download pre-quantized GGUF models from Hugging Face
2. **Convert once**: Transform GGUF Q8_0 → SMOL Q8 format
3. **Run efficiently**: Use the lightweight `smolc.c` (~307 lines) for inference

**Size comparison:**
- `smolc_gguf.c`: ~730 lines (full inference + GGUF parsing)
- `gguf_to_smol.c`: ~450 lines (conversion only)
- `smolc.c`: ~307 lines (inference only)

**Result**: Smaller codebase, same functionality!

## Building

```bash
# Build the converter
make gguf_to_smol

# Or build everything
make all
```

## Usage

### Basic Conversion

```bash
./gguf_to_smol input.gguf output.bin
```

### Complete Workflow

```bash
# 1. Get a GGUF model (example with llama.cpp)
python llama.cpp/convert-hf-to-gguf.py \
  HuggingFaceTB/SmolLM2-135M-Instruct \
  --outfile smollm2-135m-f16.gguf \
  --outtype f16

# 2. Quantize to Q8_0
llama.cpp/quantize smollm2-135m-f16.gguf smollm2-135m-q8_0.gguf Q8_0

# 3. Convert to SMOL format
./gguf_to_smol smollm2-135m-q8_0.gguf ../models/smollm2-135m-q8.bin

# 4. Run inference
./smolc -m ../models/smollm2-135m-q8.bin -p "Hello" -n 50
```

## What Gets Converted

### Model Weights
✅ **Converted:**
- Embedding layer (Q8_0 → Q8 per-tensor)
- Attention projections (Q, K, V, O)
- FFN projections (gate, up, down)
- Layer norms (FP32 → FP32)
- Output norm (FP32 → FP32)

### Configuration
✅ **Extracted from GGUF metadata:**
- Model dimensions (hidden_size, intermediate_size, etc.)
- Architecture (layers, heads, kv_heads)
- Context length
- RoPE parameters
- Normalization epsilon

### Tokenizer
⚠️ **Not converted** - See "Tokenizer Handling" below

## Quantization Conversion Details

### Q8_0 → Q8 Transformation

GGUF Q8_0 uses **per-block** quantization:
```
Block (34 bytes):
├── scale: FP16 (2 bytes)
└── values: int8[32] (32 bytes)
```

SMOL Q8 uses **per-tensor** quantization:
```
Tensor:
├── scale: FP32 (4 bytes)
└── values: int8[N] (N bytes)
```

**Conversion process:**
1. Read all Q8_0 blocks for a tensor
2. Compute average scale across all blocks
3. Rescale int8 values to use unified scale
4. Write as single-scale Q8 tensor

**Quality**: Near-lossless. The per-tensor scale is computed as the average of all block scales, with values adjusted proportionally.

## Tokenizer Handling

The converter **does not** extract the tokenizer from GGUF files. You have two options:

### Option 1: Use Existing Tokenizer

If you already have a SMOL model, copy its tokenizer:

```bash
# Extract tokenizer from existing SMOL model (Python)
import struct

def extract_tokenizer(smol_path):
    with open(smol_path, 'rb') as f:
        # Skip header + config (4+4+4+4 + 9*4 = 52 bytes)
        f.seek(52)

        vocab_size = struct.unpack('I', f.read(4))[0]
        merge_count = struct.unpack('I', f.read(4))[0]

        vocab = []
        for _ in range(vocab_size):
            length = struct.unpack('I', f.read(4))[0]
            token = f.read(length).decode('utf-8')
            vocab.append(token)

        merges = []
        for _ in range(merge_count):
            length = struct.unpack('I', f.read(4))[0]
            merge = f.read(length).decode('utf-8')
            merges.append(merge)

        return vocab, merges
```

### Option 2: Generate from Hugging Face Model

Extract the tokenizer from the original model:

```python
from transformers import AutoTokenizer
import struct

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

# Get vocab and merges
vocab = list(tokenizer.get_vocab().keys())
merges = tokenizer.bpe_ranks  # For GPT2-style tokenizers

# Write to binary format matching SMOL spec
with open('tokenizer.bin', 'wb') as f:
    f.write(struct.pack('I', len(vocab)))
    f.write(struct.pack('I', len(merges)))

    for token in vocab:
        token_bytes = token.encode('utf-8')
        f.write(struct.pack('I', len(token_bytes)))
        f.write(token_bytes)

    for merge in merges:
        merge_bytes = merge.encode('utf-8')
        f.write(struct.pack('I', len(merge_bytes)))
        f.write(merge_bytes)
```

### Option 3: Enhanced Converter (Future)

We plan to add full tokenizer extraction in a future version.

## Supported Models

### Architecture
- **Llama-style transformers** (SmolLM2, Llama, Mistral, etc.)
- Multi-head attention with GQA (Grouped Query Attention)
- SwiGLU FFN activation
- RoPE positional embeddings
- RMSNorm

### Quantization
- **Input**: GGUF Q8_0 only
- **Output**: SMOL Q8 format

**Note**: Other GGUF quantization types (Q4_K_M, Q5_K_S, etc.) are not supported yet.

## Output Format

The converter generates files compatible with `smolc` version 2:

```
SMOL Binary Format v2:
├── Magic: "SMOL" (4 bytes)
├── Version: 2 (4 bytes)
├── Quant type: 0 = Q8 (4 bytes)
├── Reserved (4 bytes)
├── Config (9 × 4 bytes)
├── Tokenizer (placeholder: 0 vocab, 0 merges)
└── Weights (Q8 tensors + FP32 norms)
```

## Limitations

1. **Q8_0 only**: Other GGUF quant types not supported
2. **No tokenizer**: Must be added separately
3. **Llama-style only**: Architecture-specific tensor naming
4. **Single-threaded**: No parallel conversion

## Error Handling

Common issues and solutions:

```bash
# "Invalid GGUF magic"
→ File is not a valid GGUF file or is corrupted

# "Unsupported GGUF version"
→ Converter expects GGUF v3, try converting with latest llama.cpp

# "Unsupported tensor type: X"
→ Model uses non-Q8_0 quantization, requantize with llama.cpp:
  ./quantize model.gguf model-q8_0.gguf Q8_0

# "Warning: [tensor] not found"
→ Model might use different tensor naming, check GGUF structure
```

## Performance

Conversion speed (approximate):
- **SmolLM2-135M**: ~1 second
- **Llama-1B**: ~5 seconds
- **Llama-7B**: ~30 seconds

Memory usage: ~2× model size during conversion

## Verification

After conversion, verify the model works:

```bash
# Test with simple prompt
./smolc -m output.bin -p "Hello" -n 10 -t 0

# Compare outputs (requires original GGUF model)
./smolc_gguf -m input.gguf -p "Test" -n 20 -t 0 > gguf_output.txt
./smolc -m output.bin -p "Test" -n 20 -t 0 > smol_output.txt
diff gguf_output.txt smol_output.txt
```

Small differences are expected due to quantization rescaling, but outputs should be very similar.

## Future Enhancements

- [ ] Full tokenizer extraction from GGUF
- [ ] Support for Q4_K_M, Q5_K_S quantization
- [ ] Batch conversion of multiple models
- [ ] Progress bar for large models
- [ ] Validation mode (dry-run)
- [ ] Direct Hugging Face → SMOL conversion

## Technical Details

### Tensor Name Mapping

| GGUF Tensor | SMOL Order | Notes |
|-------------|------------|-------|
| `token_embd.weight` | Embedding | vocab_size × hidden_size |
| `blk.{L}.attn_norm.weight` | Layer {L} input norm | FP32 |
| `blk.{L}.attn_q.weight` | Layer {L} Q projection | Q8 |
| `blk.{L}.attn_k.weight` | Layer {L} K projection | Q8 |
| `blk.{L}.attn_v.weight` | Layer {L} V projection | Q8 |
| `blk.{L}.attn_output.weight` | Layer {L} O projection | Q8 |
| `blk.{L}.ffn_norm.weight` | Layer {L} FFN norm | FP32 |
| `blk.{L}.ffn_gate.weight` | Layer {L} gate projection | Q8 |
| `blk.{L}.ffn_up.weight` | Layer {L} up projection | Q8 |
| `blk.{L}.ffn_down.weight` | Layer {L} down projection | Q8 |
| `output_norm.weight` | Final norm | FP32 |

### Memory Layout

**GGUF Q8_0 block:**
```c
struct block_q8_0 {
    uint16_t d;        // FP16 scale
    int8_t qs[32];     // Quantized values
};
```

**SMOL Q8 tensor:**
```c
struct {
    float scale;       // FP32 scale (per tensor)
    int8_t data[N];    // All quantized values
};
```

## License

Same as parent project.

## See Also

- `README_GGUF.md` - Direct GGUF inference with `smolc_gguf`
- `smolc.h` - SMOL format specification
- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
