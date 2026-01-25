/**
 * SMOL-32 Instruction Encoding Definitions
 * 32-bit ISA optimized for transformer inference
 */

#ifndef SMOL32_ENCODING_H
#define SMOL32_ENCODING_H

#include <stdint.h>

/*============================================================================
 * Instruction Format Masks and Shifts
 *============================================================================*/

/* R-Type: [opcode:6][rd:5][rs1:5][rs2:5][func:5][ext:6] */
#define R_OPCODE_MASK   0xFC000000
#define R_OPCODE_SHIFT  26
#define R_RD_MASK       0x03E00000
#define R_RD_SHIFT      21
#define R_RS1_MASK      0x001F0000
#define R_RS1_SHIFT     16
#define R_RS2_MASK      0x0000F800
#define R_RS2_SHIFT     11
#define R_FUNC_MASK     0x000007C0
#define R_FUNC_SHIFT    6
#define R_EXT_MASK      0x0000003F
#define R_EXT_SHIFT     0

/* I-Type: [opcode:6][rd:5][rs1:5][imm:16] */
#define I_OPCODE_MASK   0xFC000000
#define I_OPCODE_SHIFT  26
#define I_RD_MASK       0x03E00000
#define I_RD_SHIFT      21
#define I_RS1_MASK      0x001F0000
#define I_RS1_SHIFT     16
#define I_IMM_MASK      0x0000FFFF
#define I_IMM_SHIFT     0

/* V-Type: [opcode:6][vd:3][vs1:3][vs2:3][func:3][mask:6][ext:8] */
#define V_OPCODE_MASK   0xFC000000
#define V_OPCODE_SHIFT  26
#define V_VD_MASK       0x03800000
#define V_VD_SHIFT      23
#define V_VS1_MASK      0x00700000
#define V_VS1_SHIFT     20
#define V_VS2_MASK      0x000E0000
#define V_VS2_SHIFT     17
#define V_FUNC_MASK     0x0001C000
#define V_FUNC_SHIFT    14
#define V_MASK_MASK     0x00003F00
#define V_MASK_SHIFT    8
#define V_EXT_MASK      0x000000FF
#define V_EXT_SHIFT     0

/* M-Type: [opcode:6][rd:5][rs1:5][mode:3][offset:13] */
#define M_OPCODE_MASK   0xFC000000
#define M_OPCODE_SHIFT  26
#define M_RD_MASK       0x03E00000
#define M_RD_SHIFT      21
#define M_RS1_MASK      0x001F0000
#define M_RS1_SHIFT     16
#define M_MODE_MASK     0x0000E000
#define M_MODE_SHIFT    13
#define M_OFFSET_MASK   0x00001FFF
#define M_OFFSET_SHIFT  0

/*============================================================================
 * Opcodes (6 bits, bits 31:26)
 *============================================================================*/

/* Load/Store */
#define OP_LW           0x00    /* Load word */
#define OP_SW           0x01    /* Store word */
#define OP_LF           0x02    /* Load float */
#define OP_SF           0x03    /* Store float */

/* Integer ALU */
#define OP_ALU          0x04    /* Integer ALU (func selects operation) */
#define OP_ALUI         0x05    /* Integer ALU immediate */

/* FP ALU */
#define OP_FPU          0x08    /* FP ALU (func selects operation) */
#define OP_FPUI         0x09    /* FP immediate operations */

/* FP Special */
#define OP_FSPEC        0x0C    /* FP special functions */

/* Vector Arithmetic */
#define OP_VARITH       0x10    /* Vector arithmetic */
#define OP_VSCALAR      0x11    /* Vector-scalar operations */
#define OP_VRED         0x12    /* Vector reductions */

/* Vector Special */
#define OP_VSPEC        0x18    /* Vector special functions */
#define OP_VLOAD        0x19    /* Vector loads */
#define OP_VSTORE       0x1A    /* Vector stores */

/* Q8 Operations */
#define OP_Q8SET        0x1C    /* Q8 setup (scale, base) */
#define OP_Q8MAC        0x1D    /* Q8 MAC operations */
#define OP_VQ8          0x1E    /* Vector Q8 operations */

/* Transformer Fused */
#define OP_ROPE         0x20    /* Fused RoPE */
#define OP_VRMS         0x21    /* Fused RMSNorm */
#define OP_VSOFTMAX     0x22    /* Vector softmax */
#define OP_TFUSED       0x23    /* Other transformer fused ops */

/* Branch */
#define OP_BRANCH       0x30    /* Conditional branches */
#define OP_JAL          0x31    /* Jump and link */
#define OP_JALR         0x32    /* Jump and link register */
#define OP_LOOP         0x33    /* Loop instruction */

/* System/Control */
#define OP_SYSTEM       0x38    /* System calls */
#define OP_CSR          0x39    /* CSR access */
#define OP_FENCE        0x3A    /* Memory fence */

/*============================================================================
 * ALU Function Codes (5 bits)
 *============================================================================*/

#define FUNC_ADD        0x00
#define FUNC_SUB        0x01
#define FUNC_MUL        0x02
#define FUNC_DIV        0x03
#define FUNC_REM        0x04
#define FUNC_AND        0x05
#define FUNC_OR         0x06
#define FUNC_XOR        0x07
#define FUNC_SLL        0x08
#define FUNC_SRL        0x09
#define FUNC_SRA        0x0A
#define FUNC_SLT        0x0B
#define FUNC_SLTU       0x0C

/*============================================================================
 * FPU Function Codes (5 bits)
 *============================================================================*/

#define FUNC_FADD       0x00
#define FUNC_FSUB       0x01
#define FUNC_FMUL       0x02
#define FUNC_FDIV       0x03
#define FUNC_FMIN       0x04
#define FUNC_FMAX       0x05
#define FUNC_FMADD      0x06
#define FUNC_FMSUB      0x07
#define FUNC_FNMADD     0x08
#define FUNC_FNMSUB     0x09
#define FUNC_FCVT_W_S   0x0A    /* float to int */
#define FUNC_FCVT_S_W   0x0B    /* int to float */
#define FUNC_FMV        0x0C    /* move */
#define FUNC_FABS       0x0D
#define FUNC_FNEG       0x0E
#define FUNC_FEQ        0x10
#define FUNC_FLT        0x11
#define FUNC_FLE        0x12

/*============================================================================
 * FP Special Function Codes (5 bits)
 *============================================================================*/

#define FUNC_FSQRT      0x00
#define FUNC_FRSQRT     0x01    /* reciprocal sqrt (approximate) */
#define FUNC_FRECIP     0x02    /* reciprocal (approximate) */
#define FUNC_FEXP       0x03    /* exp (approximate) */
#define FUNC_FLOG       0x04    /* log (approximate) */
#define FUNC_FSIN       0x05    /* sin (approximate) */
#define FUNC_FCOS       0x06    /* cos (approximate) */
#define FUNC_FSILU      0x07    /* SiLU activation */
#define FUNC_FGELU      0x08    /* GELU activation */
#define FUNC_FTANH      0x09    /* tanh */
#define FUNC_FSIGMOID   0x0A    /* sigmoid */

/*============================================================================
 * Vector Function Codes (3 bits for V-type)
 *============================================================================*/

#define VFUNC_ADD       0x0
#define VFUNC_SUB       0x1
#define VFUNC_MUL       0x2
#define VFUNC_DIV       0x3
#define VFUNC_FMADD     0x4
#define VFUNC_MIN       0x5
#define VFUNC_MAX       0x6

/* Vector reduction functions (for OP_VRED) */
#define VRED_SUM        0x0
#define VRED_MAX        0x1
#define VRED_MIN        0x2
#define VRED_SQS        0x3     /* sum of squares */
#define VRED_PROD       0x4     /* product */

/* Vector special functions (for OP_VSPEC) */
#define VSPEC_SQRT      0x0
#define VSPEC_RSQRT     0x1
#define VSPEC_EXP       0x2
#define VSPEC_SILU      0x3
#define VSPEC_GELU      0x4
#define VSPEC_SOFTMAX   0x5

/*============================================================================
 * Q8 Operation Codes
 *============================================================================*/

/* Q8SET suboperations */
#define Q8_SETSCALE     0x00    /* Set Q8 scale register */
#define Q8_SETQBASE     0x01    /* Set Q8 weight base */
#define Q8_SETFBASE     0x02    /* Set FP32 activation base */
#define Q8_GETSCALE     0x03
#define Q8_GETQBASE     0x04
#define Q8_GETFBASE     0x05

/* Q8MAC suboperations */
#define Q8_ACCZERO      0x00    /* Clear accumulator */
#define Q8_ACCREAD      0x01    /* Read accumulator to FP reg */
#define Q8_MAC          0x02    /* Basic MAC (no pointer inc) */
#define Q8_MACINC       0x03    /* MAC with pointer increment */
#define Q8_MAC16        0x04    /* MAC 16 elements */
#define Q8_MAC16INC     0x05    /* MAC 16 with increment */

/*============================================================================
 * Branch Condition Codes (3 bits in func field)
 *============================================================================*/

#define BR_EQ           0x0     /* Equal */
#define BR_NE           0x1     /* Not equal */
#define BR_LT           0x2     /* Less than (signed) */
#define BR_GE           0x3     /* Greater or equal (signed) */
#define BR_LTU          0x4     /* Less than (unsigned) */
#define BR_GEU          0x5     /* Greater or equal (unsigned) */
#define BR_GTZ          0x6     /* Greater than zero */
#define BR_LEZ          0x7     /* Less or equal zero */

/*============================================================================
 * Memory Access Modes (3 bits)
 *============================================================================*/

#define MEM_BYTE        0x0     /* 8-bit */
#define MEM_HALF        0x1     /* 16-bit */
#define MEM_WORD        0x2     /* 32-bit */
#define MEM_Q8          0x3     /* Q8 (sign-extended) */
#define MEM_BYTEU       0x4     /* 8-bit unsigned */
#define MEM_HALFU       0x5     /* 16-bit unsigned */
#define MEM_BURST       0x6     /* Burst mode */
#define MEM_STREAM      0x7     /* Streaming mode */

/*============================================================================
 * Special Register Indices
 *============================================================================*/

#define SR_VL           0x00    /* Vector length */
#define SR_ACC_LO       0x01    /* Accumulator low 32 bits */
#define SR_ACC_HI       0x02    /* Accumulator high 32 bits */
#define SR_SCALE        0x03    /* Q8 scale */
#define SR_QBASE        0x04    /* Q8 weight base */
#define SR_FBASE        0x05    /* FP32 activation base */
#define SR_STATUS       0x10    /* Status register */
#define SR_CAUSE        0x11    /* Exception cause */
#define SR_EPC          0x12    /* Exception PC */

/*============================================================================
 * Instruction Encoding Helpers
 *============================================================================*/

/* Encode R-type instruction */
static inline uint32_t encode_r(uint8_t op, uint8_t rd, uint8_t rs1,
                                 uint8_t rs2, uint8_t func, uint8_t ext) {
    return ((op & 0x3F) << 26) | ((rd & 0x1F) << 21) | ((rs1 & 0x1F) << 16) |
           ((rs2 & 0x1F) << 11) | ((func & 0x1F) << 6) | (ext & 0x3F);
}

/* Encode I-type instruction */
static inline uint32_t encode_i(uint8_t op, uint8_t rd, uint8_t rs1, int16_t imm) {
    return ((op & 0x3F) << 26) | ((rd & 0x1F) << 21) | ((rs1 & 0x1F) << 16) |
           (imm & 0xFFFF);
}

/* Encode V-type instruction */
static inline uint32_t encode_v(uint8_t op, uint8_t vd, uint8_t vs1, uint8_t vs2,
                                 uint8_t func, uint8_t mask, uint8_t ext) {
    return ((op & 0x3F) << 26) | ((vd & 0x7) << 23) | ((vs1 & 0x7) << 20) |
           ((vs2 & 0x7) << 17) | ((func & 0x7) << 14) | ((mask & 0x3F) << 8) |
           (ext & 0xFF);
}

/* Encode M-type instruction */
static inline uint32_t encode_m(uint8_t op, uint8_t rd, uint8_t rs1,
                                 uint8_t mode, int16_t offset) {
    return ((op & 0x3F) << 26) | ((rd & 0x1F) << 21) | ((rs1 & 0x1F) << 16) |
           ((mode & 0x7) << 13) | (offset & 0x1FFF);
}

/* Decode helpers */
static inline uint8_t decode_opcode(uint32_t insn) {
    return (insn >> 26) & 0x3F;
}

static inline uint8_t decode_rd(uint32_t insn) {
    return (insn >> 21) & 0x1F;
}

static inline uint8_t decode_rs1(uint32_t insn) {
    return (insn >> 16) & 0x1F;
}

static inline uint8_t decode_rs2(uint32_t insn) {
    return (insn >> 11) & 0x1F;
}

static inline uint8_t decode_func(uint32_t insn) {
    return (insn >> 6) & 0x1F;
}

static inline int16_t decode_imm16(uint32_t insn) {
    return (int16_t)(insn & 0xFFFF);
}

/*============================================================================
 * Common Instruction Encodings
 *============================================================================*/

/* NOP = ADD R0, R0, R0 */
#define INSN_NOP        encode_r(OP_ALU, 0, 0, 0, FUNC_ADD, 0)

/* RET = JALR R0, R1, 0 */
#define INSN_RET        encode_i(OP_JALR, 0, 1, 0)

#endif /* SMOL32_ENCODING_H */
