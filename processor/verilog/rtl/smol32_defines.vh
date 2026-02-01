/**
 * SMOL-32 Verilog Definitions
 * Converted from encoding.h
 */

`ifndef SMOL32_DEFINES_VH
`define SMOL32_DEFINES_VH

//============================================================================
// Instruction Format Field Positions
//============================================================================

// R-Type: [opcode:6][rd:5][rs1:5][rs2:5][func:5][ext:6]
`define OPCODE_HI   31
`define OPCODE_LO   26
`define RD_HI       25
`define RD_LO       21
`define RS1_HI      20
`define RS1_LO      16
`define RS2_HI      15
`define RS2_LO      11
`define FUNC_HI     10
`define FUNC_LO     6
`define EXT_HI      5
`define EXT_LO      0

// I-Type: [opcode:6][rd:5][rs1:5][imm:16]
`define IMM_HI      15
`define IMM_LO      0

// Branch: condition is in imm[15:13], offset in imm[12:0]
`define BR_COND_HI  15
`define BR_COND_LO  13
`define BR_OFF_HI   12
`define BR_OFF_LO   0

//============================================================================
// Opcodes (6 bits)
//============================================================================

// Load/Store
`define OP_LW       6'h00
`define OP_SW       6'h01
`define OP_LF       6'h02
`define OP_SF       6'h03

// Integer ALU
`define OP_ALU      6'h04
`define OP_ALUI     6'h05

// Shift immediate (uses rs2 field for shift amount in some encodings)
`define OP_SLLI     6'h05  // Same as ALUI, distinguished by func or separate
`define OP_SRLI     6'h06
`define OP_SRAI     6'h07

// FP ALU
`define OP_FPU      6'h08
`define OP_FPUI     6'h09

// FP Special
`define OP_FSPEC    6'h0C

// Vector Arithmetic
`define OP_VARITH   6'h10
`define OP_VSCALAR  6'h11
`define OP_VRED     6'h12

// Vector Special
`define OP_VSPEC    6'h18
`define OP_VLOAD    6'h19
`define OP_VSTORE   6'h1A

// Q8 Operations
`define OP_Q8SET    6'h1C
`define OP_Q8MAC    6'h1D
`define OP_VQ8      6'h1E

// Transformer Fused
`define OP_ROPE     6'h20
`define OP_VRMS     6'h21
`define OP_VSOFTMAX 6'h22
`define OP_TFUSED   6'h23

// Branch
`define OP_BRANCH   6'h30
`define OP_JAL      6'h31
`define OP_JALR     6'h32
`define OP_LOOP     6'h33

// System
`define OP_SYSTEM   6'h38
`define OP_CSR      6'h39
`define OP_FENCE    6'h3A

//============================================================================
// ALU Function Codes (5 bits)
//============================================================================

`define FUNC_ADD    5'h00
`define FUNC_SUB    5'h01
`define FUNC_MUL    5'h02
`define FUNC_DIV    5'h03
`define FUNC_REM    5'h04
`define FUNC_AND    5'h05
`define FUNC_OR     5'h06
`define FUNC_XOR    5'h07
`define FUNC_SLL    5'h08
`define FUNC_SRL    5'h09
`define FUNC_SRA    5'h0A
`define FUNC_SLT    5'h0B
`define FUNC_SLTU   5'h0C

//============================================================================
// FPU Function Codes (5 bits)
//============================================================================

`define FUNC_FADD       5'h00
`define FUNC_FSUB       5'h01
`define FUNC_FMUL       5'h02
`define FUNC_FDIV       5'h03
`define FUNC_FMIN       5'h04
`define FUNC_FMAX       5'h05
`define FUNC_FMADD      5'h06
`define FUNC_FMSUB      5'h07
`define FUNC_FNMADD     5'h08
`define FUNC_FNMSUB     5'h09
`define FUNC_FCVT_W_S   5'h0A
`define FUNC_FCVT_S_W   5'h0B
`define FUNC_FMV        5'h0C
`define FUNC_FABS       5'h0D
`define FUNC_FNEG       5'h0E
`define FUNC_FEQ        5'h10
`define FUNC_FLT        5'h11
`define FUNC_FLE        5'h12

//============================================================================
// FP Special Function Codes (5 bits)
//============================================================================

`define FUNC_FSQRT      5'h00
`define FUNC_FRSQRT     5'h01
`define FUNC_FRECIP     5'h02
`define FUNC_FEXP       5'h03
`define FUNC_FLOG       5'h04
`define FUNC_FSIN       5'h05
`define FUNC_FCOS       5'h06
`define FUNC_FSILU      5'h07
`define FUNC_FGELU      5'h08
`define FUNC_FTANH      5'h09
`define FUNC_FSIGMOID   5'h0A

//============================================================================
// Vector Function Codes (3 bits)
//============================================================================

`define VFUNC_ADD   3'h0
`define VFUNC_SUB   3'h1
`define VFUNC_MUL   3'h2
`define VFUNC_DIV   3'h3
`define VFUNC_FMADD 3'h4
`define VFUNC_MIN   3'h5
`define VFUNC_MAX   3'h6

// Vector reduction
`define VRED_SUM    3'h0
`define VRED_MAX    3'h1
`define VRED_MIN    3'h2
`define VRED_SQS    3'h3

// Vector special
`define VSPEC_SQRT  3'h0
`define VSPEC_RSQRT 3'h1
`define VSPEC_EXP   3'h2
`define VSPEC_SILU  3'h3

//============================================================================
// Q8 Operation Codes (5 bits in func field)
//============================================================================

`define Q8_SETSCALE 5'h00
`define Q8_SETQBASE 5'h01
`define Q8_SETFBASE 5'h02
`define Q8_GETSCALE 5'h03
`define Q8_GETQBASE 5'h04
`define Q8_GETFBASE 5'h05

`define Q8_ACCZERO  5'h00
`define Q8_ACCREAD  5'h01
`define Q8_MAC      5'h02
`define Q8_MACINC   5'h03

//============================================================================
// Branch Condition Codes (3 bits)
//============================================================================

`define BR_EQ   3'h0
`define BR_NE   3'h1
`define BR_LT   3'h2
`define BR_GE   3'h3
`define BR_LTU  3'h4
`define BR_GEU  3'h5
`define BR_GTZ  3'h6
`define BR_LEZ  3'h7

//============================================================================
// Control Unit States
//============================================================================

`define STATE_FETCH     4'h0
`define STATE_DECODE    4'h1
`define STATE_EXECUTE   4'h2
`define STATE_MEMORY    4'h3
`define STATE_WRITEBACK 4'h4
`define STATE_MAC_WAIT  4'h5
`define STATE_VEC_MEM   4'h6
`define STATE_HALT      4'hF

//============================================================================
// ALU Operations (internal encoding)
//============================================================================

`define ALU_ADD     4'h0
`define ALU_SUB     4'h1
`define ALU_MUL     4'h2
`define ALU_AND     4'h3
`define ALU_OR      4'h4
`define ALU_XOR     4'h5
`define ALU_SLL     4'h6
`define ALU_SRL     4'h7
`define ALU_SRA     4'h8
`define ALU_SLT     4'h9
`define ALU_SLTU    4'hA
`define ALU_PASS_A  4'hB
`define ALU_PASS_B  4'hC

//============================================================================
// Memory Size Constants
//============================================================================

`define MEM_SIZE_BYTES  32'h10000000  // 256 MB

`endif // SMOL32_DEFINES_VH
