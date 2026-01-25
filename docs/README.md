# Documentation

This folder contains technical documentation for the SmolLM2-135M inference engine project, which was developed in three phases.

## Development Phases

### Phase 1: C Implementation (`smolc/`)

The initial implementation: PyTorch reference, Q8 quantization, and portable C inference engine.

**Python scripts:**
- `step1_transformers.py` — HuggingFace reference implementation
- `step2_pytorch.py` — Bare PyTorch implementation for verification
- `step3_quantize.py` — Q8/Q4 quantization and binary export
- `step4_pytorch_q8.py` — PyTorch Q8 inference
- `step6_verify.py` — C vs PyTorch verification
- `model.py` — Bare PyTorch model components (RMSNorm, RoPE, GQA, SwiGLU)

**C implementation:**
- `smolc/smolc.c` — Full inference engine with BPE tokenizer
- `smolc/smolc.h` — Header with Q8/Q4 tensor types
- `smolc/Makefile` — Build configuration

**Documentation:**
- [verification_report.md](verification_report.md) — Q4 vs Q8 vs FP32 quantization comparison
- [development_log.md](development_log.md) — Steps 1-6: PyTorch implementation, Q8 export, C implementation

---

### Phase 2: Rust Implementation (`smolr/`)

Direct translation of the C inference engine to Rust.

**Rust implementation:**
- `smolr/src/main.rs` — Full Rust implementation (~450 lines)
- `smolr/Cargo.toml` — Project configuration

**Python verification:**
- `step7_verify_rust.py` — Rust vs C verification

**Documentation:**
- [development_log.md](development_log.md) — "Later Enhancement: Rust Implementation" section

---

### Phase 3: Custom Processor (`processor/`)

A custom 32-bit instruction set architecture and emulator designed for efficient transformer inference.

**Key components:**
- `processor/assembler.py` — Two-pass assembler with disassembler
- `processor/emulator.c/h` — Full SMOL-32 instruction interpreter
- `processor/kernels/*.s` — 9 assembly kernels (2,516 bytes total)
- `processor/generate.c` — Text generation using emulator
- `processor/run_model.c` — Verification harness (emulator vs C reference)

**Documentation:**
- [ISA.md](ISA.md) — SMOL-32 instruction set specification
- [analysis.md](analysis.md) — Computational workload analysis for transformer inference
- [development_log.md](development_log.md) — "SMOL-32 ISA Design" through "Processor Status" sections

---

## Document Index

| Document | Phase | Description |
|----------|-------|-------------|
| [ISA.md](ISA.md) | 3 | SMOL-32 instruction set architecture specification |
| [analysis.md](analysis.md) | 3 | Computational workload analysis for transformer inference |
| [development_log.md](development_log.md) | 1-3 | Complete development history, issues, and solutions |
| [verification_report.md](verification_report.md) | 1 | Quantization verification (Q4 vs Q8 vs FP32) |

---

## Verification Status

All implementations verified against reference:

| Phase | Verification | Status |
|-------|--------------|--------|
| 1 | C vs PyTorch Q8 | **PASS** |
| 2 | Rust vs C | **PASS** |
| 3 | SMOL-32 emulator vs C | **PASS** (max logit diff: 4.53e-05) |

---

## Performance Summary (Phase 3)

| Metric | Value |
|--------|-------|
| Total kernel code | 2,516 bytes |
| Instructions per forward pass | 19.0M |
| Numerical accuracy | 4.53e-05 max logit difference |
