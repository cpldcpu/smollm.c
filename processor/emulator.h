/**
 * SMOL-32 CPU Emulator
 * Cycle-accurate(ish) emulator for testing SMOL-32 programs
 */

#ifndef SMOL32_EMULATOR_H
#define SMOL32_EMULATOR_H

#include <stdint.h>
#include <stddef.h>

#define VEC_LEN 16          /* Max vector elements */
#define MEM_SIZE (256*1024*1024)  /* 256MB address space */

/* Opcodes */
#define OP_LW       0x00
#define OP_SW       0x01
#define OP_LF       0x02
#define OP_SF       0x03
#define OP_ALU      0x04
#define OP_ALUI     0x05
#define OP_FPU      0x08
#define OP_FSPEC    0x0C
#define OP_VARITH   0x10
#define OP_VSCALAR  0x11
#define OP_VRED     0x12
#define OP_VSPEC    0x18
#define OP_VLOAD    0x19
#define OP_VSTORE   0x1A
#define OP_Q8SET    0x1C
#define OP_Q8MAC    0x1D
#define OP_BRANCH   0x30
#define OP_JAL      0x31
#define OP_JALR     0x32
#define OP_LOOP     0x33
#define OP_SYSTEM   0x38

/* ALU functions */
#define F_ADD  0x00
#define F_SUB  0x01
#define F_MUL  0x02
#define F_AND  0x05
#define F_OR   0x06
#define F_XOR  0x07
#define F_SLL  0x08
#define F_SRL  0x09
#define F_SRA  0x0A
#define F_SLT  0x0B

/* FPU functions */
#define F_FADD 0x00
#define F_FSUB 0x01
#define F_FMUL 0x02
#define F_FDIV 0x03
#define F_FMIN 0x04
#define F_FMAX 0x05
#define F_FMADD 0x06
#define F_FMSUB 0x07
#define F_FCVT_W_S 0x0A
#define F_FCVT_S_W 0x0B
#define F_FMOV 0x0C
#define F_FABS 0x0D
#define F_FNEG 0x0E

/* FSPEC functions */
#define F_FSQRT    0x00
#define F_FRSQRT   0x01
#define F_FRECIP   0x02
#define F_FEXP     0x03
#define F_FSILU    0x07

/* Q8 functions */
#define Q8_SETSCALE 0x00
#define Q8_SETQBASE 0x01
#define Q8_SETFBASE 0x02
#define Q8_ACCZERO  0x00
#define Q8_ACCREAD  0x01
#define Q8_MAC      0x02
#define Q8_MACINC   0x03

/* Branch conditions (stored in imm[15:13]) */
#define BR_EQ  0
#define BR_NE  1
#define BR_LT  2
#define BR_GE  3
#define BR_GTZ 6
#define BR_LEZ 7

typedef struct {
    /* Integer registers */
    uint32_t r[32];         /* R0 always 0 */

    /* FP registers */
    float f[32];            /* F0 always 0.0 */

    /* Vector registers: 8 × 16 floats */
    float v[8][VEC_LEN];

    /* Special registers */
    double acc;             /* 64-bit accumulator */
    float scale;            /* Q8 dequant scale */
    uint32_t qbase;         /* Q8 weight base address */
    uint32_t fbase;         /* FP32 activation base address */
    uint32_t vl;            /* Vector length (1-16) */

    /* Program counter */
    uint32_t pc;

    /* Memory (flat byte array) */
    uint8_t *mem;

    /* Stats */
    uint64_t insn_count;
    uint64_t cycle_count;
    int halted;
} SmolCPU;

/* Initialize CPU state */
void cpu_init(SmolCPU *cpu);

/* Free CPU resources */
void cpu_free(SmolCPU *cpu);

/* Load program binary at given address */
int cpu_load_program(SmolCPU *cpu, const char *path, uint32_t addr);

/* Load raw data into memory */
void cpu_load_data(SmolCPU *cpu, uint32_t addr, const void *data, size_t len);

/* Execute one instruction, return 0 on success, -1 on halt/error */
int cpu_step(SmolCPU *cpu);

/* Run until halt or max_steps */
int cpu_run(SmolCPU *cpu, uint64_t max_steps);

/* Memory access helpers */
static inline uint32_t mem_read32(SmolCPU *cpu, uint32_t addr) {
    return *(uint32_t *)(cpu->mem + addr);
}

static inline void mem_write32(SmolCPU *cpu, uint32_t addr, uint32_t val) {
    *(uint32_t *)(cpu->mem + addr) = val;
}

static inline float mem_readf(SmolCPU *cpu, uint32_t addr) {
    return *(float *)(cpu->mem + addr);
}

static inline void mem_writef(SmolCPU *cpu, uint32_t addr, float val) {
    *(float *)(cpu->mem + addr) = val;
}

static inline int8_t mem_read8(SmolCPU *cpu, uint32_t addr) {
    return (int8_t)cpu->mem[addr];
}

#endif /* SMOL32_EMULATOR_H */
