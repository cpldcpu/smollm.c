# SMOL-32 Verilog Implementation

Synthesizable Verilog RTL implementation of the SMOL-32 processor for transformer inference.

## Status: ✓ Complete

Full 30-layer forward pass verified against C emulator:
- **Top-5 tokens match exactly**
- ~910M cycles for complete inference
- Avg logit difference: 1.18e-4

## Building and Running

### Verilator (recommended for full forward pass)

```bash
# Build
verilator --cc --exe --build --trace -j 0 -Irtl \
    -CFLAGS "-Wno-unused-result" \
    -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-CASEINCOMPLETE -Wno-LATCH -Wno-COMBDLY \
    rtl/smol32_top.v tb/tb_verilator.cpp -o Vsmol32_tb

# Run kernel tests
./obj_dir/Vsmol32_tb

# Run full forward pass (requires model file, ~15 min)
./obj_dir/Vsmol32_tb --forward
```

### Icarus Verilog (for quick tests)

```bash
cd sim
make test      # Run basic test
make wave      # View waveforms in GTKWave
```

## Test Results

```
=== Summary ===
residual kernel:     PASSED
silu_mul kernel:     PASSED
exp/silu accuracy:   PASSED
Full forward pass:   PASSED (top-5 tokens match exactly)

Top-5 (Verilog):          Top-5 (Reference):
  [260] logit=14.3581       [260] logit=14.3582
  [253] logit=13.6248       [253] logit=13.6249
  [216] logit=13.5549       [216] logit=13.5549
  [28] logit=13.5529        [28] logit=13.5530
  [29] logit=13.4923        [29] logit=13.4924
```

## Architecture

Multi-cycle design with 6 states:
1. **FETCH** — Read instruction from memory
2. **DECODE** — Decode opcode, detect HALT (PC=0 trap)
3. **EXECUTE** — ALU/FPU operation, branch decision
4. **MEMORY** — Load/Store memory access
5. **WRITEBACK** — Write result to register file
6. **VEC_MEM/MAC_WAIT** — Multi-cycle vector/Q8 MAC operations

## File Structure

```
processor/verilog/
├── rtl/
│   ├── smol32_defines.vh      # Opcodes, constants, FSM states
│   ├── smol32_regfile.v       # Integer register file (32×32-bit)
│   ├── smol32_regfile_fp.v    # FP register file (32×32-bit)
│   ├── smol32_regfile_vec.v   # Vector register file (8×512-bit)
│   ├── smol32_alu.v           # Integer ALU
│   ├── smol32_fpu.v           # FP ALU + special functions
│   ├── smol32_vecunit.v       # Vector ALU (16-lane SIMD)
│   ├── smol32_vecmem.v        # Vector memory unit (LVF/SVF)
│   ├── smol32_q8mac.v         # Q8 MAC unit (Q8MACINC)
│   ├── smol32_decode.v        # Instruction decoder
│   ├── smol32_control.v       # Multi-cycle control FSM
│   ├── smol32_core.v          # Core integration
│   └── smol32_top.v           # Top-level with memory
├── tb/
│   ├── tb_smol32_top.v        # Iverilog testbench
│   ├── tb_kernel_matmul.v     # Kernel test for iverilog
│   └── tb_verilator.cpp       # Verilator C++ testbench
└── sim/
    └── Makefile               # Build rules for iverilog
```

## Implemented Instructions

### Integer
- Load/Store: LW, SW
- ALU: ADD, SUB, MUL, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU
- Immediate: ADDI, SLLI, SRLI, SRAI
- Control: BEQ, BNE, BLT, BGE, JAL, JALR, LOOP
- System: HALT (PC=0 trap)

### Floating-Point
- Load/Store: LF, SF
- Arithmetic: FADD, FSUB, FMUL, FDIV, FMIN, FMAX
- Unary: FABS, FNEG, FMV
- Conversion: FCVT.W.S, FCVT.S.W
- Special: FSQRT, FRSQRT, FRECIP, FEXP, FLOG, FSIN, FCOS, FSILU, FGELU, FTANH, FSIGMOID

### Vector (16-lane FP32 SIMD)
- Load/Store: LVF, SVF (strided)
- Element-wise: VADD, VSUB, VMUL, VDIV, VMIN, VMAX
- Scalar broadcast: VADD.S, VMUL.S
- Reductions: VREDSUM, VREDMAX, VREDMIN, VREDSQS
- Special: VSQRT, VRSQRT, VEXP, VSILU
- Config: VSETVL

### Q8 MAC
- Config: QSETSCALE, QSETBASE, FSETBASE
- Accumulator: ACCZERO, ACCREAD
- Compute: Q8MAC, Q8MACINC (fused dequant-multiply-accumulate with pointer increment)

## Synthesis Notes

The RTL is written for synthesizability:
- No `initial` blocks in synthesizable modules
- Proper async reset (active low `rst_n`)
- Register-based state machines

**Exception:** The FPU and vector unit use Verilog `real` type for simulation.
For synthesis, replace `smol32_fpu.v` and `smol32_vecunit.v` with:
- Xilinx/Intel Floating-Point IP
- OpenCores FPU
- Custom IEEE 754 pipeline

The `exp_approx` function uses a floating-point decomposition approach:
```verilog
// exp(x) = 2^(x * log2(e)) = 2^n * 2^f
// where n is integer, f is fractional [0,1)
// 2^f approximated with 5-term minimax polynomial
```

## Memory Map

```
0x0000_0000           HALT trap
0x0000_1000           Kernel: matmul_q8
0x0000_2000           Kernel: rmsnorm
0x0000_3000           Kernel: rope
0x0000_4000           Kernel: attention
0x0000_5000           Kernel: silu_mul
0x0000_6000           Kernel: residual
0x0000_7000           Kernel: embed
0x0000_8000           Kernel: memcpy
0x0000_9000           Kernel: forward
0x000E_0000           Model descriptor
0x0010_0000           Activation buffers
0x0040_0000           Model weights (~128MB)
0x0900_0000           KV caches
0x0F00_0000           RoPE tables
0x0FF0_0000           Stack (1MB)
```

## Performance

| Metric | Value |
|--------|-------|
| Cycles per forward pass | ~910M |
| Cycles per instruction | ~4-50 (varies by type) |
| Simple ALU instruction | ~5 cycles |
| Q8MACINC (16 elements) | ~50 cycles |
| LVF/SVF (16 floats) | ~50 cycles |

## Documentation

- [../../docs/ISA.md](../../docs/ISA.md) — Full instruction set specification
- [../../docs/development_log.md](../../docs/development_log.md) — Development history with bug fixes
