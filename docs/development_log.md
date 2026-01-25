# SmolLM2-135M C Inference Engine - Development Log

## Original User Request

> "Claude, we have an empty folder here. I want you to implement a transformer model inference engine in C. Get a smollm2-135m-instruct checkpoint from HF. I want you to work step by step:
> 1. Implement using the transformers lib
> 2. Implement a bare pytorch implementation and verify vs. transformers
> 3. Once that works, quantize and export to Q8
> 4. Update pytorch to Q8 and verify vs. previous implementation
> 5. Implement ANSI C inference engine (in folder smolc)
> 6. The ANSI-C inference engine shall load the Q8 exported model. Verify inference vs. pytorch Q8."

Model source: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct

---

## Clarifying Questions & User Decisions

| Question | User Decision |
|----------|---------------|
| Tokenization in C? | Include full tokenizer in C (end-to-end inference) |
| Quantization scheme? | Per-tensor symmetric quantization |
| Model file format? | Custom binary format |

---

## Implementation Plan

### Step 1: Transformers Reference Implementation
- Download model from HuggingFace
- Generate reference outputs for verification

### Step 2: Bare PyTorch Implementation
- Implement all components manually (RMSNorm, RoPE, GQA, SwiGLU MLP)
- Verify outputs match transformers library exactly

### Step 3: Q8 Quantization & Export
- Implement per-tensor symmetric INT8 quantization
- Export to custom binary format with embedded tokenizer

### Step 4: PyTorch Q8 Inference
- Implement quantized matrix multiplication in PyTorch
- Verify Q8 output matches FP32 within acceptable tolerance

### Step 5: ANSI C Inference Engine
- Implement full inference in portable C
- Include BPE tokenizer with GPT2 byte encoding

### Step 6: Verification
- Compare C inference output to PyTorch Q8 output

---

## Model Architecture (Discovered)

| Parameter | Value |
|-----------|-------|
| Architecture | LlamaForCausalLM |
| Layers | 30 |
| Hidden size | 576 |
| Intermediate size | 1536 |
| Attention heads | 9 |
| KV heads | 3 (Grouped Query Attention) |
| Head dimension | 64 |
| Vocabulary size | 49,152 |
| Max context length | 8,192 |
| RoPE theta | 100,000 |
| Normalization | RMSNorm (eps=1e-5) |
| Activation | SiLU (in SwiGLU MLP) |
| Embeddings | Tied (input = output) |
| Tokenizer | GPT2-style BPE |

---

## Step-by-Step Implementation Details

### Step 1: Transformers Library (✓ Completed)

**File created:** `step1_transformers.py`

**Actions:**
- Downloaded SmolLM2-135M-Instruct from HuggingFace
- Model cached to `models/smollm2-135m-instruct/`
- Generated reference outputs for test prompts

**Issues encountered:**
1. `transformers` module not installed → Fixed with `pip install torch transformers`
2. `torch_dtype` parameter deprecated in newer transformers → Fixed by removing parameter and using `.float().cpu()` conversion

---

### Step 2: Bare PyTorch Implementation (✓ Completed)

**Files created:** `model.py`, `step2_pytorch.py`

**Components implemented:**
- `RMSNorm` - Root Mean Square Layer Normalization
- `RotaryEmbedding` - Rotary Position Embeddings
- `Attention` - Grouped Query Attention with KV cache
- `MLP` - SwiGLU feed-forward network
- `TransformerBlock` - Single transformer layer
- `SmolLM2` - Full model wrapper

**Major Issue: RoPE Implementation Mismatch**

Initial implementation used complex number multiplication:
```python
# WRONG approach
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
```

This produced different results than HuggingFace. Investigation revealed HuggingFace uses a different approach:

```python
# CORRECT approach (HuggingFace compatible)
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq, dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

Key difference: HuggingFace duplicates cos/sin values `[c0,c0,c1,c1,...]` rather than interleaving.

**Verification result:** Logits match exactly with maximum difference < 1e-4

---

### Step 3: Q8 Quantization & Export (✓ Completed)

**File created:** `step3_quantize.py`

**Quantization scheme:**
- Per-tensor symmetric INT8
- Scale = max(|W|) / 127
- Quantized value = round(W / scale)

**Binary format v1:**
```
Header:
  - Magic: "SMOL" (4 bytes)
  - Version: 1 (4 bytes)
  - Config: hidden_size, intermediate_size, num_layers, etc.

Tokenizer:
  - Vocab size, num merges
  - Vocab entries (length + bytes + score)
  - Merge rules (length + bytes)

Weights:
  - For each tensor: scale (float32) + data (int8[])
```

**Output:** `models/smollm2-135m-q8.bin` (129.51 MB)

---

### Step 4: PyTorch Q8 Inference (✓ Completed)

**File created:** `step4_pytorch_q8.py`

**Implementation:**
- Loads custom binary format
- Dequantizes weights on-the-fly during matmul
- Supports KV cache for generation

**Verification:** Q8 output matches FP32 with acceptable quantization noise

---

### Step 5: ANSI C Inference Engine (✓ Completed)

**Files created:**
- `smolc/smolc.h` - Header with structures and function declarations
- `smolc/smolc.c` - Full implementation
- `smolc/Makefile` - Build configuration

**Features:**
- Pure ANSI C, no external dependencies
- Memory-mapped model loading
- Full BPE tokenizer with GPT2 byte encoding
- KV cache for efficient autoregressive generation
- Top-p (nucleus) sampling

**Major Issue: GPT2 Byte Encoding**

Initial output showed garbled characters like `Ġ` instead of spaces.

**Root cause:** GPT2 BPE uses byte-level encoding where:
- Space (0x20) → `Ġ` (U+0120, UTF-8: 0xC4 0xA0)
- Newline (0x0A) → `Ċ` (U+010A, UTF-8: 0xC4 0x8A)

**Fix:** Implemented bidirectional conversion:

```c
// Encoding: convert input text to GPT2 byte representation
static int convert_to_gpt2_bytes(const char *text, char *out, int max_out) {
    int j = 0;
    for (int i = 0; text[i] && j < max_out - 2; i++) {
        if (text[i] == ' ') {
            out[j++] = 0xC4; out[j++] = 0xA0;  // Ġ
        } else if (text[i] == '\n') {
            out[j++] = 0xC4; out[j++] = 0x8A;  // Ċ
        } else {
            out[j++] = text[i];
        }
    }
    return j;
}

// Decoding: convert GPT2 bytes back to readable text
// Ġ → space, Ċ → newline
```

---

### Step 6: Verification (✓ Completed)

**File created:** `step6_verify.py`

**Test prompts:**
1. "The capital of France is"
2. "Hello, my name is"

**Result:** C implementation output matches PyTorch Q8 exactly for both prompts.

---

## Later Enhancement: INT4 Quantization

**User request:** "ok, now add an option for INT4 quantization"

### Changes Made

**step3_quantize.py:**
- Added `--q4` command line flag
- Added `--group-size` parameter (default: 32)
- Implemented group-wise symmetric INT4 quantization
- Two int4 values packed into one byte

```python
def quantize_tensor_q4(tensor, group_size=32):
    # Reshape to groups
    # Compute per-group scale = max_abs / 7
    # Quantize to [-7, 7] range
    # Pack: low nibble + high nibble << 4
```

**Binary format v2:**
- Added `quant_type` field (0=Q8, 1=Q4)
- Added `group_size` field
- Q4 tensors store: num_groups, scales[], packed_data[]

**smolc/smolc.h:**
```c
typedef struct {
    int num_groups;
    int group_size;
    float *scales;      /* [num_groups] */
    uint8_t *data;      /* Packed: 2 int4 values per byte */
    int rows, cols;
} Q4Tensor;

typedef struct {
    int quant_type;     /* QUANT_Q8 or QUANT_Q4 */
    union { Q8Tensor q8; Q4Tensor q4; };
    int rows, cols;
} QuantTensor;
```

**smolc/smolc.c:**
- Added Q4 unpacking with sign extension
- Added Q4 matrix multiplication
- Updated all tensor operations to use QuantTensor union

```c
static inline float unpack_q4(uint8_t byte, int high, float scale) {
    int8_t val;
    if (high) {
        val = (int8_t)(byte >> 4);
        if (val >= 8) val -= 16;  // Sign extend from 4-bit
    } else {
        val = (int8_t)(byte & 0x0F);
        if (val >= 8) val -= 16;
    }
    return (float)val * scale;
}
```

### Q4 Status: Code Complete, Pending Testing

---

## Later Enhancement: Rust Implementation

**User request:** "can you also translate smolc_full.c to rust in a new folder smolr and verify it vs. c and python? It shall use the existing binary model files"

### Implementation

**Files created:**
- `smolr/Cargo.toml` - Rust project configuration
- `smolr/src/main.rs` - Full Rust implementation (~450 lines)

**Features:**
- Pure Rust, only dependency is `rand` for sampling
- Same command-line interface as C version
- Reads existing Q8 binary model files
- Identical output to C implementation

**Key Rust structures:**

```rust
struct Q8Tensor {
    scale: f32,
    data: Vec<i8>,
    rows: usize,
    cols: usize,
}

struct SmolLM2 {
    config: Config,
    weights: ModelWeights,
    tokenizer: Tokenizer,
    kv_caches: Vec<KVCache>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    // Scratch buffers...
}

impl SmolLM2 {
    fn load(path: &str) -> Result<Self, ...>
    fn forward(&mut self, tok: usize, pos: usize) -> &[f32]
    fn tokenize(&self, text: &str) -> Vec<usize>
    fn decode(&self, tok: usize) -> String
    fn generate(&mut self, prompt: &str, max_tokens: usize, temp: f32)
}
```

**GPT2 byte encoding in Rust:**
```rust
fn tokenize(&self, text: &str) -> Vec<usize> {
    // Convert to GPT2 byte encoding
    let mut encoded = String::new();
    for c in text.chars() {
        if c == ' ' {
            encoded.push_str("\u{0120}"); // Ġ
        } else if c == '\n' {
            encoded.push_str("\u{010A}"); // Ċ
        } else {
            encoded.push(c);
        }
    }
    // ... BPE tokenization
}
```

### Verification

**File created:** `step7_verify_rust.py`

**Test results:**
```
--- Prompt: 'The capital of France is' ---
Rust output: The capital of France is Paris, a city known for its historical landmarks...
C output:    The capital of France is Paris, a city known for its historical landmarks...
✓ Rust matches C exactly!

--- Prompt: 'Hello, my name is' ---
Rust output: Hello, my name is Emily, and I'm a professional photographer...
C output:    Hello, my name is Emily, and I'm a professional photographer...
✓ Rust matches C exactly!
```

### Rust Status: ✓ Complete and Verified

---

## Custom Processor: SMOL-32 ISA & Emulator

**User request:** "create a new working folder called processor/ I want you to analyze the minimal c implementation of the inference code and design an ISA that is optimized for this task (assume a 32bit machine)."

### Analysis

Profiled the C inference code's computational breakdown:

| Operation | % of Compute | Notes |
|-----------|-------------|-------|
| Q8 Matmul (W×x) | ~96% | 16 matrix-vector products per layer × 30 layers |
| RMSNorm | ~1.5% | 2 per layer (sum of squares + scale) |
| RoPE | ~0.5% | 12 heads × 64-dim rotations |
| SiLU + element-wise | ~1% | 1536-dim intermediate |
| Softmax + attention | ~1% | Per-head dot products and weighted sum |

Key insight: The dominant operation is fused INT8 dequantize-multiply-accumulate, performing `sum += (float)w_q8 * scale * x_fp32` across weight rows.

### SMOL-32 ISA Design

**Architecture:**
- 32-bit fixed-width instructions
- 32 integer registers (R0=zero), 32 FP registers (F0=0.0)
- 8 vector registers × 16 FP32 lanes (512-bit SIMD)
- Special registers: 64-bit accumulator, Q8 scale, QBASE, FBASE, VL

**Instruction encoding formats:**
- R-type: op[6] | rd[5] | rs1[5] | rs2[5] | func[5] | ext[6]
- I-type: op[6] | rd[5] | rs1[5] | imm[16]
- Branch: op[6] | rd[5] | rs1[5] | cond[3] | offset[13]

**Key custom instructions:**
- `Q8MACINC n` — Fused: ACC += sum(Q8[qbase:qbase+n] × scale × FP32[fbase:fbase+n*4]); advances both pointers
- `QSETSCALE`, `QSETBASE`, `FSETBASE` — Configure Q8 MAC unit
- `ACCZERO`, `ACCREAD` — Accumulator control
- `VREDSQS` — Vector reduction: sum of squares (for RMSNorm)
- `VSILU` — Vector SiLU activation
- `FRSQRT` — Fast reciprocal square root
- `LOOP rs, offset` — Decrement rs; if rs > 0, branch to PC + offset (fuses ADDI+BNEZ, 31% instruction reduction)

**Performance estimate:** ~830 tokens/sec at 500MHz (vs ~3 tok/s on CPU)

### Assembler

**File:** `processor/assembler.py`

Two-pass assembler that translates SMOL-32 assembly to binary:
- Pass 1: Collect labels, compute addresses
- Pass 2: Resolve branch targets, emit 32-bit encodings

Supports full instruction set including vector ops, Q8 MAC, and all branch forms. Includes a disassembler for verification.

**Issues encountered:**
1. **Branch encoding conflict** — Condition code and offset initially overlapped in the 16-bit immediate. Fixed by splitting: imm[15:13]=condition, imm[12:0]=signed word offset.
2. **Shift instructions** — Initially encoded as ALUI (immediate); actually R-type with shift amount in ext field.
3. **Vector load/store** — V-type format had insufficient bits for base register. Switched to I-type format.
4. **Label fixups** — NoneType error when branch targets were forward-references. Fixed with two-pass approach returning fixup tuples.

### Assembly Kernels

| Kernel | File | Bytes | Description |
|--------|------|-------|-------------|
| matmul_q8 | `kernels/matmul_q8.s` | 136 | Q8 matrix-vector multiply using Q8MACINC |
| rmsnorm | `kernels/rmsnorm.s` | 212 | Two-pass RMS normalization with VREDSQS |
| rope | `kernels/rope.s` | 212 | Rotary position embeddings (vector mul/sub/add) |
| attention | `kernels/attention.s` | 532 | Multi-head grouped-query attention with softmax |
| silu_mul | `kernels/silu_mul.s` | 44 | SiLU-gate activation using VSILU |
| residual | `kernels/residual.s` | 40 | Element-wise vector addition (VADD) |
| embed | `kernels/embed.s` | 100 | Int8→float embedding dequantization |
| memcpy | `kernels/memcpy.s` | 32 | Float vector memory copy (for KV cache) |
| forward | `kernels/forward.s` | 1,208 | Full 30-layer forward pass orchestration |

**Total kernel code: 2,516 bytes** (1,308 compute + 1,208 orchestration) for the complete transformer inference engine.

The inner loop of matmul is just 2 instructions: `Q8MACINC 16; LOOP` — demonstrating the ISA's efficiency for this workload.

### C Emulator

**Files:** `processor/emulator.h`, `processor/emulator.c`, `processor/test_emulator.c`

Full instruction interpreter implementing all SMOL-32 instructions:
- Fetch-decode-execute loop
- Flat 256MB address space
- Accurate Q8 MAC with double-precision accumulation
- Vector register file with configurable VL
- Instruction and cycle counting

**Test results:**
```
=== Test: Q8 Matrix-Vector Multiply ===
  Rows=64, Cols=576, Scale=0.0200
  Max difference: 2.19e-05 → PASS

=== Test: RMSNorm ===
  n=576, eps=1.0e-05
  Max difference: 7.15e-07 → PASS

=== Test: Matmul with Real Model Weights ===
  Q_proj: rows=576, cols=576, scale=0.041339
  Max difference: 7.15e-07 → PASS
```

**Issues encountered:**
1. **Q8MACINC only advanced QBASE** — The activation pointer (FBASE) was not advancing between chunks, causing subsequent Q8MACINC calls to read the same 16 floats. Fixed by advancing FBASE by n×4 bytes alongside QBASE.
2. **Opcode defines not shared** — Defines were in emulator.c but needed by test_emulator.c. Moved to emulator.h.

### Full Model Runner (run_model.c)

Loads the complete SmolLM2-135M Q8 model into emulator memory and runs a full forward pass on the SMOL-32 emulator, comparing output against a pure-C reference implementation.

**Memory layout:**
```
0x0000_0000 - 0x0000_9FFF : Kernel code (40KB, 9 kernels at 4KB intervals)
0x000E_0000 - 0x000E_0FFF : Model descriptor (64B header + 30 × 64B layer descriptors)
0x0010_0000 - 0x003F_FFFF : State buffers (3MB: x, xb, xb2, q, k, v, att, hb, hb2, logits)
0x0040_0000 - 0x08FF_FFFF : Model weights (~128MB Q8)
0x0900_0000 - 0x0EFF_FFFF : KV caches
0x0F00_0000 - 0x0F1F_FFFF : RoPE cos/sin tables
0x0FF0_0000 - 0x0FFF_FFFF : Stack (1MB, grows down)
```

**Calling convention:** `cpu_reset_for_call()` sets RA=0 (HALT trap at address 0), SP=top-of-stack, then runs until HALT. The forward kernel is invoked once with R3=token, R4=position, R5=descriptor base.

**Result (assembly forward pass vs pure-C reference):**
```
Assembly forward: 19,010,082 instructions executed
Max logit difference: 4.53e-05
Avg logit difference: 2.43e-05
Mismatches (>0.1): 0 / 49152
Result: PASS - Assembly matches reference!
```

### Assembly Forward Pass (forward.s)

The entire 30-layer transformer forward pass now runs as a single assembly kernel call. The C host only sets up arguments (token, position, descriptor pointer) and reads back the final logits — all intermediate computation, buffering, and looping happens in SMOL-32 assembly.

**Invocation:**
```c
cpu_reset_for_call(cpu, FORWARD_ENTRY);   // 0x9000
cpu->r[3] = token;
cpu->r[4] = pos;
cpu->r[5] = DESC_BASE;                   // 0xE0000
cpu_run(cpu, 500000000000ULL);
// Read logits from BUF_BASE + 0x80000
```

**Model Descriptor (at 0xE0000):**

A data structure in emulator memory that the assembly reads to locate weights and config values:
- 64-byte header: embedding addresses, config scalars (hidden_size, num_layers, etc.), RoPE table base, KV stride
- 30 × 64-byte layer descriptors: pointers to each layer's weight data/scale for all 8 projections + 2 layer norms

**Register allocation (callee-saved across sub-kernel calls):**
```
R16 = desc_base        R17 = position         R18 = layer_desc_ptr
R19 = layer_counter    R20 = cos_addr         R21 = sin_addr
R22 = kv_head_stride   R23 = layer_k_cache    R24 = layer_v_cache
R25 = seq_len (pos+1)  R26 = max_seq_len      R27 = BUF_X (0x100000)
R28 = hidden_size(576) R29 = temp
```

**Key patterns:**
- Sub-kernel calls: `LI R10, addr; JALR RA, R10, 0` (or ADDI workaround for 0x8000)
- Large addresses: `LI R27, 16; SLLI R27, R27, 16` → 0x100000
- KV cache advance: `6 × kv_head_stride_bytes` computed with shifts+adds, no multiplication

**Performance:** 19.0M instructions for a single forward pass (token 1 at position 0). The LOOP instruction optimization reduced this from 27.6M (31% improvement).

### Text Generator (generate.c)

Full text generation program running the SmolLM2 model entirely on the SMOL-32 emulator. Includes BPE tokenization, autoregressive generation loop, and sampling.

**Forward pass:** A single call to the assembly forward kernel (at 0x9000) executes the entire 30-layer pipeline. The host-side `forward()` function just sets up registers and reads back logits — all intermediate data stays in emulator memory throughout.

**Usage:**
```bash
./generate -p "The capital of France is" -n 50 -s 128 -t 0.0
```

**Output:**
```
The capital of France is Paris, a city known for its historical
landmarks, culture, and cultural institutions. Paris is a major...
```

**Issues encountered during development:**
1. **Memory layout overlap** — Weight base at 0x01000000 overlapped with KV base. Fixed by moving weights to 0x00400000 and KV to 0x09000000.
2. **Hand-encoded instruction bugs** — Initial approach of manually encoding kernel instructions led to subtle errors (wrong register fields, wrong opcodes). Switched to using the assembler to produce .bin files.
3. **Matmul input buffer preservation** — Initially reloaded BUF_XB from host between each Q/K/V matmul call. Analysis showed the matmul kernel only reads from FBASE (never writes), so BUF_XB persists across calls. Eliminated unnecessary roundtrips.

### Processor Status: ✓ Full Forward Pass in Assembly, Verified Against C Reference

---

## File Summary

| File | Purpose |
|------|---------|
| `step1_transformers.py` | HuggingFace reference implementation |
| `step2_pytorch.py` | Bare PyTorch verification |
| `step3_quantize.py` | Q8/Q4 quantization and export |
| `step4_pytorch_q8.py` | PyTorch Q8 inference |
| `step6_verify.py` | C vs PyTorch verification |
| `model.py` | Bare PyTorch model components |
| `smolc/smolc.h` | C header file |
| `smolc/smolc.c` | C implementation |
| `smolc/Makefile` | Build configuration |
| `smolr/Cargo.toml` | Rust project configuration |
| `smolr/src/main.rs` | Rust implementation |
| `step7_verify_rust.py` | Rust vs C verification |
| `models/smollm2-135m-q8.bin` | Quantized Q8 model (129.51 MB) |
| `docs/ISA.md` | SMOL-32 instruction set specification |
| `docs/analysis.md` | Computational analysis of inference workload |
| `processor/encoding.h` | C header with opcode/encoding defines |
| `processor/assembler.py` | Assembler + disassembler for SMOL-32 |
| `processor/emulator.h` | CPU emulator header (state + opcodes) |
| `processor/emulator.c` | CPU emulator implementation (256MB, full ISA) |
| `processor/test_emulator.c` | Emulator unit test harness |
| `processor/run_model.c` | Full model runner: emulator vs reference comparison |
| `processor/generate.c` | Text generation using emulator for all compute |
| `processor/Makefile` | Build system for kernels + programs |
| `processor/kernels/matmul_q8.s` | Q8 matrix-vector multiply (Q8MACINC) |
| `processor/kernels/rmsnorm.s` | RMS normalization (VREDSQS + FRSQRT) |
| `processor/kernels/rope.s` | Rotary position embeddings (VMUL/VSUB/VADD) |
| `processor/kernels/attention.s` | Multi-head GQA + softmax (FEXP/FRECIP) |
| `processor/kernels/silu_mul.s` | SiLU-gate activation (VSILU + VMUL) |
| `processor/kernels/residual.s` | Residual addition (VADD) |
| `processor/kernels/embed.s` | Embedding dequant (FCVT.S.W + FMUL) |
| `processor/kernels/memcpy.s` | Float vector copy (LVF/SVF) |
| `processor/kernels/forward.s` | Full 30-layer forward pass in assembly (1216B, 304 instructions) |

---

## Key Technical Insights

1. **RoPE compatibility matters** - Different implementations of rotary embeddings exist; must match the source model's approach exactly.

2. **GPT2 tokenizers use byte-level encoding** - Characters are mapped to printable Unicode characters to avoid issues with raw bytes in vocab files.

3. **Grouped Query Attention** - Using 3 KV heads for 9 query heads reduces memory and compute while maintaining quality.

4. **Per-tensor vs group-wise quantization** - Q8 uses per-tensor scales (simpler), Q4 uses group-wise (better accuracy at lower precision).

5. **Tied embeddings** - Input and output embeddings share the same weights, reducing model size.

---

## Usage Examples

```bash
# Build C inference engine
cd smolc && make

# Run C Q8 inference
./smolc -m ../models/smollm2-135m-q8.bin -p "Hello, how are you?" -n 50

# Build Rust inference engine
cd smolr && cargo build --release

# Run Rust Q8 inference
./target/release/smolr -m ../models/smollm2-135m-q8.bin -p "Hello, how are you?" -n 50

# Create Q4 model (smaller, faster, slightly lower quality)
python step3_quantize.py --q4 --group-size 32

# Run Q4 inference (C)
./smolc -m ../models/smollm2-135m-q4.bin -p "Hello, how are you?" -n 50

# Verify Rust vs C
python step7_verify_rust.py
```

---

*Generated: 2026-01-16, Updated: 2026-01-25*
