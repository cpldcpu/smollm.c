**📄 [Read the full article: *Towards Self-Replication — Opus Designs Hardware to Run Itself*](https://cpldcpu.github.io/smollm.c/)**

---

**Note (16 Jan 2026):** This was an agentic code generation experiment. The entire contents of this repository were generated with Claude Code and Opus 4.5 with minimal intervention and guidance. See [docs/development_log.md](docs/development_log.md) for details.

**Addendum (25 Jan 2026):** Added phase 3: custom SMOL-32 processor implementation fully designed and implemented by the agent.

**Addendum (1 Feb 2026):** Added phase 4: Verilog implementation of SMOL-32 processor, verified against C emulator with full 30-layer forward pass.

**Addendum (5 March 2026):** Added article co-written with Opus: [ *Towards Self-Replication  Opus Designs Hardware to Run Itself*](https://cpldcpu.github.io/smollm.c/)

---


# SmolLM2-135M Inference Engine

Lightweight inference engines for [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) implemented in four phases: C, Rust, a custom SMOL-32 processor (emulator), and Verilog RTL.

> **Note:** While developed using SmolLM2-135M, the implementation supports the LLaMA2 architecture in general (RMSNorm, RoPE, GQA, SwiGLU MLP). Other LLaMA2-family models can be used by adjusting the quantization export script.

## Development Phases

### Phase 1: C Implementation (`smolc/`)

PyTorch reference, Q8/Q4 quantization, and portable C inference engine.

```bash
cd smolc && make
./smolc -m ../models/smollm2-135m-q8.bin -p "The capital of France is" -n 30
```

See [smolc/README.md](smolc/README.md) for details.

### Phase 2: Rust Implementation (`smolr/`)

Direct translation of the C implementation to Rust.

```bash
cd smolr && cargo build --release
./target/release/smolr -m ../models/smollm2-135m-q8.bin -p "The capital of France is" -n 30
```

See [smolr/README.md](smolr/README.md) for details.

### Phase 3: Custom Processor (`processor/`)

SMOL-32: a custom 32-bit ISA and emulator designed for transformer inference. Runs the entire forward pass in assembly.

```bash
cd processor && make
./generate -p "The capital of France is" -n 30 -t 0.0
```

See [processor/README.md](processor/README.md) for details.

### Phase 4: Verilog Implementation (`processor/verilog/`)

Synthesizable Verilog RTL of the SMOL-32 processor, verified against the C emulator.

```bash
cd processor/verilog
verilator --cc --exe --build --trace -j 0 -Irtl \
    -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-CASEINCOMPLETE -Wno-LATCH -Wno-COMBDLY \
    rtl/smol32_top.v tb/tb_verilator.cpp -o Vsmol32_tb
./obj_dir/Vsmol32_tb --forward
```

See [processor/verilog/README.md](processor/verilog/README.md) for details.

## Project Structure

```
├── smolc/                  # Phase 1: C implementation
│   ├── smolc.c             # Q8-only inference engine
│   ├── smolc_full.c        # Q8/Q4 inference engine
│   └── smolc.h
├── smolr/                  # Phase 2: Rust implementation
│   └── src/main.rs
├── processor/              # Phase 3 & 4: Custom SMOL-32 processor
│   ├── assembler.py        # Two-pass assembler + disassembler
│   ├── emulator.c/h        # Full instruction interpreter
│   ├── generate.c          # Text generation using emulator
│   ├── kernels/            # Assembly kernels (2,516 bytes total)
│   └── verilog/            # Phase 4: Synthesizable RTL
│       ├── rtl/            # Verilog modules
│       └── tb/             # Testbenches (iverilog + Verilator)
├── docs/
│   ├── ISA.md              # SMOL-32 instruction set specification
│   ├── analysis.md         # Computational workload analysis
│   ├── development_log.md  # Complete development history
│   └── verification_report.md
├── step1_transformers.py   # HuggingFace reference
├── step2_pytorch.py        # Bare PyTorch implementation
├── step3_quantize.py       # Q8/Q4 quantization and export
├── step4_pytorch_q8.py     # PyTorch Q8 inference
├── step6_verify.py         # C vs PyTorch verification
├── step7_verify_rust.py    # Rust vs C verification
└── models/                 # Model binaries (gitignored)
```

## Quantization

```bash
# Create Q8 model (129 MB)
python step3_quantize.py

# Create Q4 model (~65 MB)
python step3_quantize.py --q4 --group-size 32
```

## Verification

All implementations verified against reference:

| Phase | Verification | Status |
|-------|--------------|--------|
| 1 | C vs PyTorch Q8 | **PASS** |
| 2 | Rust vs C | **PASS** |
| 3 | SMOL-32 emulator vs C | **PASS** (4.53e-05 max diff) |
| 4 | Verilog vs C emulator | **PASS** (top-5 tokens match exactly) |

```bash
python step6_verify.py                      # C vs PyTorch
python step7_verify_rust.py                 # Rust vs C
cd processor && ./run_model                 # Emulator vs C
cd processor/verilog && ./obj_dir/Vsmol32_tb --forward  # Verilog vs C
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

- [docs/README.md](docs/README.md) — Documentation index organized by phase
- [docs/ISA.md](docs/ISA.md) — SMOL-32 instruction set specification
- [docs/analysis.md](docs/analysis.md) — Computational workload analysis
- [docs/development_log.md](docs/development_log.md) — Complete development history
- [docs/verification_report.md](docs/verification_report.md) — Q4 vs Q8 vs FP32 comparison

## License

MIT
