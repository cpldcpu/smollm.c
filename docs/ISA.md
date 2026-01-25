# SMOL-32: A 32-bit ISA for Transformer Inference

## Overview

SMOL-32 is a specialized instruction set architecture designed for efficient transformer model inference, specifically optimized for the computational patterns found in LLM inference with INT8 quantized weights.

## Design Rationale

Analysis of the SmolLM2 C inference code reveals these dominant operations:

| Operation | Frequency | Description |
|-----------|-----------|-------------|
| Q8 MAC | ~95% cycles | INT8 weight × FP32 activation accumulation |
| RMSNorm | Per layer | Sum-of-squares, rsqrt, element-wise multiply |
| RoPE | Per layer | Paired rotation (complex multiply pattern) |
| Softmax | Per head | Max reduction, exp, sum, division |
| SiLU | Per layer | x / (1 + exp(-x)) |
| Vector add | Frequent | Residual connections |

The ISA prioritizes:
1. **High-throughput Q8 MAC** - Fused dequantize-multiply-accumulate
2. **Vector processing** - SIMD for parallel element processing
3. **Efficient memory streaming** - Prefetch and burst access for weights
4. **Special function units** - Hardware exp, sqrt, rsqrt

---

## Architectural State

### Scalar Registers (32 × 32-bit)
```
R0-R31    : General purpose integer registers
R0        : Always zero (hardwired)
R1        : Link register (return address)
R2        : Stack pointer
R3-R7     : Argument/return registers
R8-R15    : Caller-saved temporaries
R16-R27   : Callee-saved
R28       : Frame pointer
R29       : Global pointer
R30-R31   : Reserved
```

### Floating-Point Registers (32 × 32-bit)
```
F0-F31    : Single-precision floating-point
F0        : Always 0.0 (hardwired)
```

### Vector Registers (8 × 512-bit = 16 FP32 elements each)
```
V0-V7     : Vector registers (16 × FP32 each)
```

### Special Registers
```
VL        : Vector length (1-16)
ACC       : 64-bit floating-point accumulator
SCALE     : Q8 dequantization scale (FP32)
QBASE     : Q8 weight base address
FBASE     : FP32 activation base address
```

### Status Register (SR)
```
Bit 0     : Zero flag
Bit 1     : Negative flag
Bit 2     : Overflow flag
Bit 3     : Carry flag
Bit 4-7   : Reserved
Bit 8-11  : Current vector length
Bit 12-15 : Rounding mode
```

---

## Instruction Encoding

All instructions are 32 bits. Four primary formats:

### R-Type (Register-Register)
```
31    26 25  21 20  16 15  11 10   6 5      0
[opcode] [ rd ] [ rs1] [ rs2] [func] [  ext ]
   6       5      5      5      5       6
```

### I-Type (Immediate)
```
31    26 25  21 20  16 15                  0
[opcode] [ rd ] [ rs1] [    immediate     ]
   6       5      5            16
```

### V-Type (Vector)
```
31    26 25  23 22  20 19  17 16  14 13   8 7      0
[opcode] [ vd ] [ vs1] [ vs2] [func] [mask] [ ext ]
   6       3      3      3      3      6      8
```

### M-Type (Memory/Matrix)
```
31    26 25  21 20  16 15  13 12          0
[opcode] [ rd ] [ rs1] [mode] [  offset   ]
   6       5      5      3        13
```

---

## Instruction Set

### 1. Load/Store Instructions

| Mnemonic | Encoding | Description |
|----------|----------|-------------|
| `LW rd, imm(rs1)` | I-type | Load word |
| `SW rs2, imm(rs1)` | I-type | Store word |
| `LF fd, imm(rs1)` | I-type | Load float |
| `SF fs2, imm(rs1)` | I-type | Store float |
| `LQ8 rd, imm(rs1)` | I-type | Load Q8 (sign-extend to 32-bit) |
| `LVF vd, rs1, stride` | M-type | Vector load float (strided) |
| `SVF vs, rs1, stride` | M-type | Vector store float (strided) |
| `LVQ8 vd, rs1` | M-type | Vector load Q8 (16 bytes → 16 FP32) |
| `PREFETCH rs1, hint` | M-type | Prefetch cache line |

### 2. Arithmetic Instructions (Scalar Integer)

| Mnemonic | Description |
|----------|-------------|
| `ADD rd, rs1, rs2` | rd = rs1 + rs2 |
| `SUB rd, rs1, rs2` | rd = rs1 - rs2 |
| `MUL rd, rs1, rs2` | rd = rs1 × rs2 (low 32 bits) |
| `ADDI rd, rs1, imm` | rd = rs1 + sign_extend(imm) |
| `SLLI rd, rs1, shamt` | rd = rs1 << shamt |
| `SRLI rd, rs1, shamt` | rd = rs1 >> shamt (logical) |
| `SRAI rd, rs1, shamt` | rd = rs1 >> shamt (arithmetic) |

### 3. Floating-Point Scalar Instructions

| Mnemonic | Description |
|----------|-------------|
| `FADD fd, fs1, fs2` | fd = fs1 + fs2 |
| `FSUB fd, fs1, fs2` | fd = fs1 - fs2 |
| `FMUL fd, fs1, fs2` | fd = fs1 × fs2 |
| `FDIV fd, fs1, fs2` | fd = fs1 / fs2 |
| `FMADD fd, fs1, fs2, fs3` | fd = fs1 × fs2 + fs3 |
| `FMSUB fd, fs1, fs2, fs3` | fd = fs1 × fs2 - fs3 |
| `FSQRT fd, fs1` | fd = √fs1 |
| `FRSQRT fd, fs1` | fd = 1/√fs1 (approximate) |
| `FEXP fd, fs1` | fd = exp(fs1) (approximate) |
| `FRECIP fd, fs1` | fd = 1/fs1 (approximate) |
| `FMIN fd, fs1, fs2` | fd = min(fs1, fs2) |
| `FMAX fd, fs1, fs2` | fd = max(fs1, fs2) |
| `FSILU fd, fs1` | fd = fs1 / (1 + exp(-fs1)) |

### 4. Vector Instructions

| Mnemonic | Description |
|----------|-------------|
| `VSETVL rd, rs1` | Set VL = min(rs1, 16), return actual VL |
| `VADD vd, vs1, vs2` | vd[i] = vs1[i] + vs2[i] |
| `VSUB vd, vs1, vs2` | vd[i] = vs1[i] - vs2[i] |
| `VMUL vd, vs1, vs2` | vd[i] = vs1[i] × vs2[i] |
| `VDIV vd, vs1, vs2` | vd[i] = vs1[i] / vs2[i] |
| `VFMADD vd, vs1, vs2, vs3` | vd[i] = vs1[i] × vs2[i] + vs3[i] |
| `VMULS vd, vs1, fs` | vd[i] = vs1[i] × fs (scalar broadcast) |
| `VADDS vd, vs1, fs` | vd[i] = vs1[i] + fs |
| `VSQRT vd, vs1` | vd[i] = √vs1[i] |
| `VRSQRT vd, vs1` | vd[i] = 1/√vs1[i] |
| `VEXP vd, vs1` | vd[i] = exp(vs1[i]) |
| `VSILU vd, vs1` | vd[i] = SiLU(vs1[i]) |
| `VREDSUM fd, vs1` | fd = Σ vs1[i] (horizontal sum) |
| `VREDMAX fd, vs1` | fd = max(vs1[i]) |
| `VREDMIN fd, vs1` | fd = min(vs1[i]) |
| `VREDSQS fd, vs1` | fd = Σ vs1[i]² (sum of squares) |

### 5. Quantized Matrix Operations (Key Innovation)

These instructions are the heart of efficient Q8 inference:

| Mnemonic | Description |
|----------|-------------|
| `QSETSCALE fs` | SCALE = fs (set dequant scale) |
| `QSETBASE rs` | QBASE = rs (set Q8 weight pointer) |
| `FSETBASE rs` | FBASE = rs (set FP32 activation pointer) |
| `ACCZERO` | ACC = 0.0 (clear accumulator) |
| `ACCREAD fd` | fd = ACC (read accumulator) |
| `Q8MAC n` | ACC += Σ(Q8[QBASE+i] × SCALE × FP32[FBASE+i]) for i=0..n-1 |
| `Q8MACINC n` | Q8MAC + auto-increment QBASE by n, FBASE by n×4 |
| `VQ8MUL vd, vs` | vd[i] = Q8[QBASE+i] × SCALE × vs[i], QBASE += VL |
| `VQ8MAC vs` | ACC += Σ(Q8[QBASE+i] × SCALE × vs[i]), QBASE += VL |

**Q8MAC Pipeline:** Single instruction performs:
1. Load 16 INT8 values from QBASE
2. Convert to FP32 and multiply by SCALE
3. Multiply by 16 FP32 activations from FBASE
4. Accumulate into ACC
5. Advance QBASE by n, FBASE by n×4 (Q8MACINC only)

This fuses what would be ~50 instructions into 1.

### 6. Transformer-Specific Instructions (Optional Fused Ops)

These fused instructions are defined for potential hardware acceleration but are **not used** by the current kernels, which achieve the same results using decomposed vector operations (VMUL, VSUB, VADD, VREDSQS, FRSQRT, VEXP, etc.):

| Mnemonic | Description |
|----------|-------------|
| `ROPE vd1, vd2, vc, vs` | Fused RoPE: vd1 = v1×c - v2×s, vd2 = v2×c + v1×s |
| `VRMS vd, vs1, vs2, feps` | vd[i] = vs1[i] × rsqrt(sum_sq/VL + feps) × vs2[i] |
| `VSOFTMAX vd, vs` | vd = softmax(vs) (full softmax on vector) |
| `VGELU vd, vs` | vd[i] = GELU(vs[i]) |

### 7. Control Flow

| Mnemonic | Description |
|----------|-------------|
| `BEQ rs1, rs2, offset` | Branch if rs1 == rs2 |
| `BNE rs1, rs2, offset` | Branch if rs1 != rs2 |
| `BLT rs1, rs2, offset` | Branch if rs1 < rs2 (signed) |
| `BGE rs1, rs2, offset` | Branch if rs1 >= rs2 (signed) |
| `JAL rd, offset` | rd = PC+4; PC = PC + offset |
| `JALR rd, rs1, offset` | rd = PC+4; PC = rs1 + offset |
| `LOOP rs, offset` | if (--rs > 0) PC += offset |

### 8. Assembler Pseudo-Instructions

These are expanded by the assembler into one or more hardware instructions:

| Mnemonic | Expansion | Description |
|----------|-----------|-------------|
| `LI rd, imm` | `ADDI rd, R0, imm` | Load 16-bit signed immediate |
| `MV rd, rs` | `ADDI rd, rs, 0` | Register move |
| `NOP` | `ADDI R0, R0, 0` | No operation |
| `RET` | `JALR R0, R1, 0` | Return (jump to link register) |
| `BEQZ rs, offset` | `BEQ rs, R0, offset` | Branch if zero |
| `BNEZ rs, offset` | `BNE rs, R0, offset` | Branch if not zero |
| `BGTZ rs, offset` | `BLT R0, rs, offset` | Branch if greater than zero |

**Immediate limitations:** The 16-bit signed immediate field allows values −32768 to +32767. For larger constants:
- `LI rd, hi; SLLI rd, rd, N` — Shift left to construct large addresses (e.g., `LI R27, 16; SLLI R27, R27, 16` → 0x100000)
- `LI rd, 0x7FFF; ADDI rd, rd, 1` — Construct 0x8000 (just beyond signed range)

---

## Memory Map (SmolLM2-135M Configuration)

```
0x0000_0000 - 0x0000_9FFF : Kernel code (40KB, 9 kernels at 4KB-aligned slots)
0x000E_0000 - 0x000E_0FFF : Model descriptor (header + per-layer weight pointers)
0x0010_0000 - 0x003F_FFFF : Activation buffers (3MB: x, xb, xb2, q, k, v, att, hb, hb2, logits)
0x0040_0000 - 0x08FF_FFFF : Model weights (~128MB, Q8 quantized)
0x0900_0000 - 0x0EFF_FFFF : KV caches (30 layers × 3 heads × 8192 positions × 64 dim)
0x0F00_0000 - 0x0F1F_FFFF : RoPE cos/sin precomputed tables
0x0FF0_0000 - 0x0FFF_FFFF : Stack (1MB, grows down from 0x0FFFFFF0)
```

**Kernel address map:**
```
0x0000 : HALT trap (address 0 = stop execution)
0x1000 : matmul_q8   (136B)
0x2000 : rmsnorm     (212B)
0x3000 : rope        (212B)
0x4000 : attention   (532B)
0x5000 : silu_mul    (44B)
0x6000 : residual    (40B)
0x7000 : embed       (100B)
0x8000 : memcpy      (32B)
0x9000 : forward     (1208B) - orchestrates full 30-layer pass
```

---

## Example: Q8 Matrix-Vector Multiply

C code:
```c
void matmul_q8(float *o, Q8Tensor *W, float *x) {
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++)
            sum += W->data[i*cols + j] * W->scale * x[j];
        o[i] = sum;
    }
}
```

SMOL-32 assembly (rows=576, cols=576):
```asm
# Inputs: R3 = output ptr, R4 = W.data, R5 = W.scale, R6 = x ptr
# R7 = rows, R8 = cols

matmul_q8:
    LF      F1, 0(R5)           # F1 = scale
    QSETSCALE F1                 # Set global scale
    FSETBASE R6                  # Set activation base to x

    MV      R9, R7              # R9 = row counter
.row_loop:
    QSETBASE R4                  # Point to current weight row
    FSETBASE R6                  # Reset activation base (Q8MACINC advances it)
    ACCZERO                      # ACC = 0

    MV      R10, R8             # R10 = col counter
    SRLI    R10, R10, 4         # R10 = cols / 16 (vector iterations)

.col_loop:
    Q8MACINC 16                  # ACC += 16 Q8×FP32 products, advance QBASE+FBASE
    ADDI    R10, R10, -1
    BNEZ    R10, .col_loop

    ACCREAD F2                   # F2 = accumulated sum
    SF      F2, 0(R3)           # Store result
    ADDI    R3, R3, 4           # Advance output pointer
    ADD     R4, R4, R8          # Advance to next row

    ADDI    R9, R9, -1
    BNEZ    R9, .row_loop

    RET
```

**Cycle comparison:**
- Scalar: ~576 × 576 × 5 = 1,658,880 cycles
- SMOL-32: ~576 × (576/16) × 2 = 41,472 cycles
- **Speedup: ~40×**

---

## Example: RMSNorm

C code:
```c
void rmsnorm(float *o, float *x, float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}
```

SMOL-32 assembly (n=576):
```asm
# R3 = o, R4 = x, R5 = w, R6 = n, F1 = eps

rmsnorm:
    VSETVL  R10, R6             # VL = min(n, 16)
    MV      R7, R4              # R7 = x ptr
    FMOV    F2, F0              # F2 = 0 (sum of squares)
    MV      R8, R6              # R8 = counter

    # Pass 1: Sum of squares
.sum_loop:
    LVF     V0, R7, 4           # Load 16 floats
    VREDSQS F3, V0              # F3 = sum of squares of V0
    FADD    F2, F2, F3          # Accumulate
    ADDI    R7, R7, 64          # Advance pointer
    ADDI    R8, R8, -16
    BGTZ    R8, .sum_loop

    # Compute scale: 1/sqrt(ss/n + eps)
    FCVT.S.W F3, R6             # F3 = (float)n
    FDIV    F2, F2, F3          # F2 = ss/n
    FADD    F2, F2, F1          # F2 = ss/n + eps
    FRSQRT  F2, F2              # F2 = 1/sqrt(...)

    # Pass 2: Scale and multiply by weights
    MV      R7, R4              # Reset x ptr
    MV      R8, R6              # Reset counter

.scale_loop:
    LVF     V0, R7, 4           # Load x[i:i+16]
    LVF     V1, R5, 4           # Load w[i:i+16]
    VMULS   V0, V0, F2          # V0 = x * scale
    VMUL    V0, V0, V1          # V0 = x * scale * w
    SVF     V0, R3, 4           # Store to output
    ADDI    R7, R7, 64
    ADDI    R5, R5, 64
    ADDI    R3, R3, 64
    ADDI    R8, R8, -16
    BGTZ    R8, .scale_loop

    RET
```

---

## Example: Apply RoPE

C code:
```c
void apply_rope(float *v, int hd, float *c, float *s) {
    int h = hd / 2;
    for (int i = 0; i < h; i++) {
        float v0 = v[i], v1 = v[i + h];
        v[i] = v0 * c[i] - v1 * s[i];
        v[i + h] = v1 * c[i + h] + v0 * s[i + h];
    }
}
```

SMOL-32 assembly (hd=64, so h=32):
```asm
# R3 = v, R4 = hd, R5 = cos, R6 = sin

apply_rope:
    SRLI    R7, R4, 1           # R7 = h = hd/2
    VSETVL  R10, R7             # VL = 16 (or less)
    MV      R8, R7              # Counter
    SLLI    R9, R7, 2           # R9 = h*4 (byte offset to second half)

.rope_loop:
    # Load v[i] and v[i+h]
    LVF     V0, R3, 4           # V0 = v[0:16]
    ADD     R11, R3, R9
    LVF     V1, R11, 4          # V1 = v[h:h+16]

    # Load cos and sin
    LVF     V2, R5, 4           # V2 = cos[0:16]
    LVF     V3, R6, 4           # V3 = sin[0:16]

    # Use fused ROPE instruction
    ROPE    V4, V5, V2, V3      # V4 = V0*V2 - V1*V3, V5 = V1*V2 + V0*V3

    # Store results
    SVF     V4, R3, 4           # v[i] = result
    SVF     V5, R11, 4          # v[i+h] = result

    ADDI    R3, R3, 64
    ADDI    R5, R5, 64
    ADDI    R6, R6, 64
    ADDI    R8, R8, -16
    BGTZ    R8, .rope_loop

    RET
```

---

## Microarchitecture Recommendations

### Pipeline Stages
```
IF → ID → EX1 → EX2 → EX3 → MEM → WB
         │     │     │
         └─────┴─────┴── Vector/MAC pipeline (3 cycles)
```

### Functional Units
1. **Scalar ALU** - Integer arithmetic
2. **FP ALU** - FP add/sub/mul
3. **FP Special** - sqrt, rsqrt, exp, silu (pipelined, 3-5 cycles)
4. **Vector Unit** - 16-wide SIMD FP32
5. **Q8 MAC Unit** - Fused dequant-multiply-accumulate (16 INT8/cycle)
6. **Load/Store Unit** - 64 bytes/cycle bandwidth

### Memory System
- **L1 Data Cache**: 32KB, 4-way, 64B lines
- **Weight Buffer**: 64KB direct-mapped scratchpad for streaming weights
- **Activation Buffer**: 16KB for current layer activations
- **Prefetch Engine**: Automatic weight prefetching based on QBASE

---

## Instruction Encoding Details

### Opcode Map (6 bits)

| Opcode | Range | Category |
|--------|-------|----------|
| 0x00 | 000000 | LOAD |
| 0x01 | 000001 | STORE |
| 0x02 | 000010 | LOAD.FP |
| 0x03 | 000011 | STORE.FP |
| 0x04-0x07 | 0001xx | Integer ALU |
| 0x08-0x0B | 0010xx | FP ALU |
| 0x0C-0x0F | 0011xx | FP Special |
| 0x10-0x17 | 010xxx | Vector Arith |
| 0x18-0x1B | 0110xx | Vector Special |
| 0x1C-0x1F | 0111xx | Q8 Operations |
| 0x20-0x23 | 1000xx | Transformer Fused |
| 0x24-0x2F | 1001xx-1011xx | Reserved |
| 0x30-0x37 | 110xxx | Branch |
| 0x38-0x3F | 111xxx | System/Control |

---

## Performance Estimates

For SmolLM2-135M (576 hidden, 30 layers, 9 heads):

| Operation | Per Token | Cycles (Scalar) | Cycles (SMOL-32) |
|-----------|-----------|-----------------|------------------|
| Embedding lookup | 1 | 576 | 36 |
| Q/K/V projections | 30×3 | 29.8M | 747K |
| Attention (seq=512) | 30×9 | 8.8M | 550K |
| O projection | 30 | 9.9M | 249K |
| MLP (gate+up+down) | 30×3 | 79.6M | 2.0M |
| RMSNorm | 30×3 | 155K | 9.7K |
| **Total** | | **~128M** | **~3.6M** |

**Estimated speedup: ~35×** over scalar implementation.

**Measured (emulator, position 0):** 19.0M instructions for one forward pass. The LOOP instruction optimization reduced this from 27.6M (31% reduction). The remaining overhead vs. the 3.6M compute-only estimate reflects memory operations, addressing, and kernel call setup.

---

## Summary

SMOL-32 achieves efficiency through:
1. **Fused Q8 MAC** - Single instruction for dequantize-multiply-accumulate
2. **Vector processing** - 16-wide SIMD for activation processing
3. **Special functions** - Hardware exp/sqrt/rsqrt/silu
4. **Transformer primitives** - ROPE, VRMS, VSOFTMAX instructions
5. **Memory streaming** - Optimized weight access patterns

The ISA maintains RISC simplicity while adding domain-specific acceleration.
