# SMOL-32 Processor

A custom 32-bit instruction set architecture designed for efficient transformer model inference with INT8 quantized weights. Implemented as both a C emulator and synthesizable Verilog RTL.

## Overview

The SMOL-32 processor runs the entire SmolLM2-135M forward pass in assembly, achieving a 31% instruction reduction through the `LOOP` instruction and fused `Q8MACINC` operations.

**Key stats:**
- 9 assembly kernels totaling 2,516 bytes
- 19.0M instructions per forward pass
- C emulator: 4.53e-05 max logit difference vs reference
- Verilog RTL: Top-5 tokens match exactly (verified via Verilator)

## Building

```bash
make              # Build all kernels and programs
make clean        # Remove build artifacts
```

## Programs

| Program | Description |
|---------|-------------|
| `generate` | Text generation using the SMOL-32 emulator |
| `run_model` | Verification harness (emulator vs C reference) |
| `test_emulator` | Unit tests for individual kernels |

### Usage

```bash
# Generate text
./generate -p "The capital of France is" -n 30 -t 0.0

# Verify emulator matches reference
./run_model

# Run kernel unit tests
./test_emulator
```

## Assembly Kernels

Located in `kernels/`, assembled by `assembler.py`:

| Kernel | Bytes | Description |
|--------|-------|-------------|
| matmul_q8.s | 136 | Q8 matrix-vector multiply (inner loop: `Q8MACINC 16; LOOP`) |
| rmsnorm.s | 212 | RMS normalization using `VREDSQS` + `FRSQRT` |
| rope.s | 212 | Rotary position embeddings |
| attention.s | 532 | Multi-head grouped-query attention with softmax |
| silu_mul.s | 44 | SiLU-gate activation using `VSILU` |
| residual.s | 40 | Element-wise vector addition |
| embed.s | 100 | INT8→FP32 embedding dequantization |
| memcpy.s | 32 | Float vector copy for KV cache |
| forward.s | 1,208 | Full 30-layer forward pass orchestration |

## Architecture

### Memory Map

```
0x0000 - 0x9FFF : Kernel code (9 kernels at 4KB intervals)
0xE0000         : Model descriptor (config + per-layer weight pointers)
0x100000        : Activation buffers (x, xb, q, k, v, att, hb, logits)
0x400000        : Model weights (~128MB Q8)
0x9000000       : KV caches
0xF000000       : RoPE cos/sin tables
0xFF00000       : Stack (1MB, grows down)
```

### Kernel Addresses

```
0x1000 : matmul_q8
0x2000 : rmsnorm
0x3000 : rope
0x4000 : attention
0x5000 : silu_mul
0x6000 : residual
0x7000 : embed
0x8000 : memcpy
0x9000 : forward
```

### Key Instructions

| Instruction | Description |
|-------------|-------------|
| `Q8MACINC n` | Fused: ACC += Σ(Q8×scale×FP32), advances pointers |
| `LOOP rs, off` | rs--; if (rs > 0) branch (fuses ADDI+BNEZ) |
| `VREDSQS fd, vs` | fd = Σ(vs[i]²) — sum of squares for RMSNorm |
| `VSILU vd, vs` | vd[i] = SiLU(vs[i]) — vectorized activation |
| `FRSQRT fd, fs` | fd = 1/√fs — fast reciprocal square root |

## Files

| File | Description |
|------|-------------|
| assembler.py | Two-pass assembler with disassembler |
| emulator.c/h | Full SMOL-32 instruction interpreter |
| generate.c | Text generation program |
| run_model.c | Model verification harness |
| test_emulator.c | Kernel unit tests |
| encoding.h | Opcode definitions (shared) |
| Makefile | Build system |

## Verilog Implementation

The `verilog/` subdirectory contains a synthesizable RTL implementation of the SMOL-32 processor.

```bash
cd verilog
verilator --cc --exe --build --trace -j 0 -Irtl \
    -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-CASEINCOMPLETE -Wno-LATCH -Wno-COMBDLY \
    rtl/smol32_top.v tb/tb_verilator.cpp -o Vsmol32_tb
./obj_dir/Vsmol32_tb --forward
```

**Results:**
- Full 30-layer forward pass: ~910M cycles
- Top-5 predicted tokens match C emulator exactly
- Avg logit difference: 1.18e-4

See [verilog/README.md](verilog/README.md) for details.

## Documentation

- [../docs/ISA.md](../docs/ISA.md) — Full instruction set specification
- [../docs/analysis.md](../docs/analysis.md) — Computational workload analysis
- [../docs/development_log.md](../docs/development_log.md) — Development history
