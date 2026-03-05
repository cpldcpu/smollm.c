# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

SmolLM2-135M inference engine implemented in four phases: PyTorch → C → Rust → custom SMOL-32 ISA with emulator → synthesizable Verilog RTL. Each phase verified against the previous. Built agentically by Claude Opus 4.5.

## Build Commands

```bash
# C inference engine (Q8-only and Q8/Q4 variants)
cd smolc && make

# Rust inference engine
cd smolr && cargo build --release

# Processor: assembles all kernels, builds emulator + test harness + generator
cd processor && make

# Assemble a single kernel
python3 processor/assembler.py processor/kernels/foo.s -o processor/kernels/foo.bin

# Verilog via Verilator (recommended for full tests)
cd processor/verilog && verilator --cc --exe --build --trace -j 0 -Irtl \
    -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-CASEINCOMPLETE -Wno-LATCH -Wno-COMBDLY \
    rtl/smol32_top.v tb/tb_verilator.cpp -o Vsmol32_tb

# Verilog via iverilog (quick unit tests)
cd processor/verilog && iverilog -o sim_out -Irtl rtl/*.v tb/tb_smol32_top.v && vvp sim_out
```

## Verification

```bash
python step6_verify.py                                           # C vs PyTorch Q8
python step7_verify_rust.py                                      # Rust vs C
cd processor && ./run_model                                      # Emulator vs C
cd processor/verilog && ./obj_dir/Vsmol32_tb --forward           # Verilog vs emulator (~15 min)
```

## Quick Test

```bash
cd smolc && ./smolc -m ../models/smollm2-135m-q8.bin -p "The capital of France is" -n 30
cd processor && ./generate -p "The capital of France is" -n 30 -t 0.0
```

## Architecture

- **Model:** SmolLM2-135M-Instruct (LlamaForCausalLM). 30 layers, hidden=576, 9 attn heads / 3 KV heads (GQA), vocab=49152, RoPE theta=100000, SwiGLU MLP, tied embeddings.
- **Quantization:** Q8 per-tensor symmetric (`scale = max|W| / 127`). Q4 group-wise also supported.
- **Model format:** Custom binary, magic `"SMOL"`, version 2. Contains config + BPE tokenizer + quantized weights.
- **GPT2 byte encoding:** Space → `Ġ` (0xC4 0xA0), newline → `Ċ` (0xC4 0x8A). Applies in C, Rust, and emulator tokenizers.
- **SMOL-32 ISA:** 32-bit fixed-width instructions, R/I/V/M-type encoding. 32 int regs (R0=0), 32 FP regs (F0=0.0), 8 vector regs (16×FP32 lanes), Q8 MAC unit with 64-bit accumulator.
- **Key instruction:** `Q8MACINC` — fused dequantize-multiply-accumulate with auto-incrementing pointers. Inner matmul loop is 2 instructions: `Q8MACINC 16; LOOP`.
- **Verilog FPU** uses `real` type for simulation. Not synthesizable as-is; replace with FP IP for synthesis.
- **Memory map:** Kernels at 0x1000–0x9000 (4KB aligned), model descriptor at 0xE0000, activations at 0x100000, weights at 0x400000, KV caches at 0x9000000, stack at 0xFF00000.

## Directory Layout

```
smolc/                  C inference engine (smolc.c, smolc_full.c, smolc.h)
smolr/                  Rust inference engine (src/main.rs)
processor/              Custom ISA: assembler.py, emulator.c/h, kernels/*.s
processor/verilog/rtl/  12 Verilog modules (smol32_top.v is top-level)
processor/verilog/tb/   Testbenches (iverilog + Verilator)
docs/                   ISA.md, analysis.md, development_log.md, verification_report.md
step*.py                Python reference chain (steps 1–7)
models/                 Model binaries (gitignored, ~130MB)
```

## Gotchas

- `models/` is gitignored. Regenerate Q8 model: `python step3_quantize.py` (requires HuggingFace checkpoint in `models/smollm2-135m-instruct/`).
- The processor `Makefile` auto-assembles kernels. Edit `.s` → `make` rebuilds `.bin`.
- Verilog full forward pass (`--forward`) takes ~15 minutes on Verilator.
- RoPE must use HuggingFace's `rotate_half` approach, not complex multiplication. They are mathematically equivalent but numerically different.
- The `LOOP` instruction fuses `ADDI R, R, -1; BNEZ R, label` into one instruction (31% instruction count reduction).
- 9 assembly kernels total, 2516 bytes. `forward.s` orchestrates the full 30-layer pass; the host only sets up token/position/descriptor and reads back logits.
