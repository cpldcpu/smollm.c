/**
 * SMOL-32 CPU Emulator Implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "emulator.h"

static inline int32_t sign_extend16(uint16_t val) {
    return (int16_t)val;
}

static inline int32_t sign_extend13(uint16_t val) {
    if (val & 0x1000) return val | 0xFFFFE000;
    return val;
}

static float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

void cpu_init(SmolCPU *cpu) {
    memset(cpu, 0, sizeof(SmolCPU));
    cpu->mem = (uint8_t *)calloc(MEM_SIZE, 1);
    if (!cpu->mem) { fprintf(stderr, "OOM: can't alloc %d MB\n", MEM_SIZE >> 20); exit(1); }
    cpu->vl = VEC_LEN;
    cpu->r[2] = MEM_SIZE - 4096;  /* SP at top of memory minus guard */
}

void cpu_free(SmolCPU *cpu) {
    if (cpu->mem) free(cpu->mem);
    cpu->mem = NULL;
}

int cpu_load_program(SmolCPU *cpu, const char *path, uint32_t addr) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (addr + sz > MEM_SIZE) { fclose(f); return -1; }
    if (fread(cpu->mem + addr, 1, sz, f) != (size_t)sz) { fclose(f); return -1; }
    fclose(f);
    cpu->pc = addr;
    return 0;
}

void cpu_load_data(SmolCPU *cpu, uint32_t addr, const void *data, size_t len) {
    if (addr + len <= MEM_SIZE)
        memcpy(cpu->mem + addr, data, len);
}

int cpu_step(SmolCPU *cpu) {
    if (cpu->halted || cpu->pc >= MEM_SIZE - 4) return -1;

    uint32_t insn = mem_read32(cpu, cpu->pc);
    uint32_t op   = (insn >> 26) & 0x3F;
    uint32_t rd   = (insn >> 21) & 0x1F;
    uint32_t rs1  = (insn >> 16) & 0x1F;
    uint32_t rs2  = (insn >> 11) & 0x1F;
    uint32_t func = (insn >> 6) & 0x1F;
    uint32_t ext  = insn & 0x3F;
    int16_t  imm  = (int16_t)(insn & 0xFFFF);
    uint32_t next_pc = cpu->pc + 4;

    switch (op) {

    /* ===== Load/Store ===== */
    case OP_LW: {
        uint32_t addr = cpu->r[rs1] + imm;
        cpu->r[rd] = mem_read32(cpu, addr);
        break;
    }
    case OP_SW: {
        uint32_t addr = cpu->r[rs1] + imm;
        mem_write32(cpu, addr, cpu->r[rd]);
        break;
    }
    case OP_LF: {
        uint32_t addr = cpu->r[rs1] + imm;
        cpu->f[rd] = mem_readf(cpu, addr);
        break;
    }
    case OP_SF: {
        uint32_t addr = cpu->r[rs1] + imm;
        mem_writef(cpu, addr, cpu->f[rd]);
        break;
    }

    /* ===== Integer ALU ===== */
    case OP_ALU: {
        int32_t a = (int32_t)cpu->r[rs1];
        int32_t b = (int32_t)cpu->r[rs2];
        int32_t res = 0;
        switch (func) {
            case F_ADD: res = a + b; break;
            case F_SUB: res = a - b; break;
            case F_MUL: res = a * b; break;
            case F_AND: res = a & b; break;
            case F_OR:  res = a | b; break;
            case F_XOR: res = a ^ b; break;
            case F_SLL: res = (uint32_t)a << (ext & 0x1F); break;
            case F_SRL: res = (uint32_t)a >> (ext & 0x1F); break;
            case F_SRA: res = a >> (ext & 0x1F); break;
            case F_SLT: res = a < b ? 1 : 0; break;
            default: break;
        }
        if (rd != 0) cpu->r[rd] = (uint32_t)res;
        break;
    }

    /* ===== Integer ALU Immediate ===== */
    case OP_ALUI: {
        int32_t a = (int32_t)cpu->r[rs1];
        int32_t result = a + (int32_t)imm;
        if (rd != 0) cpu->r[rd] = (uint32_t)result;
        break;
    }

    /* ===== FP ALU ===== */
    case OP_FPU: {
        float a = cpu->f[rs1];
        float b = cpu->f[rs2];
        float res = 0;
        switch (func) {
            case F_FADD: res = a + b; break;
            case F_FSUB: res = a - b; break;
            case F_FMUL: res = a * b; break;
            case F_FDIV: res = a / b; break;
            case F_FMIN: res = a < b ? a : b; break;
            case F_FMAX: res = a > b ? a : b; break;
            case F_FMADD: res = a * b + cpu->f[rd]; break;
            case F_FMSUB: res = a * b - cpu->f[rd]; break;
            case F_FCVT_S_W: res = (float)(int32_t)cpu->r[rs1]; break;
            case F_FMOV: res = a; break;
            case F_FABS: res = fabsf(a); break;
            case F_FNEG: res = -a; break;
            default: break;
        }
        if (func == F_FCVT_W_S) {
            if (rd != 0) cpu->r[rd] = (uint32_t)(int32_t)a;
        } else {
            if (rd != 0) cpu->f[rd] = res;
        }
        break;
    }

    /* ===== FP Special ===== */
    case OP_FSPEC: {
        float a = cpu->f[rs1];
        float res = 0;
        switch (func) {
            case F_FSQRT:  res = sqrtf(a); break;
            case F_FRSQRT: res = 1.0f / sqrtf(a); break;
            case F_FRECIP: res = 1.0f / a; break;
            case F_FEXP:   res = expf(a); break;
            case F_FSILU:  res = silu_f(a); break;
            default: res = a; break;
        }
        if (rd != 0) cpu->f[rd] = res;
        break;
    }

    /* ===== Vector Arithmetic ===== */
    case OP_VARITH: {
        uint32_t vd   = (insn >> 23) & 0x7;
        uint32_t vs1  = (insn >> 20) & 0x7;
        uint32_t vs2  = (insn >> 17) & 0x7;
        uint32_t vfn  = (insn >> 14) & 0x7;
        for (uint32_t i = 0; i < cpu->vl; i++) {
            float a = cpu->v[vs1][i], b = cpu->v[vs2][i];
            switch (vfn) {
                case 0: cpu->v[vd][i] = a + b; break;
                case 1: cpu->v[vd][i] = a - b; break;
                case 2: cpu->v[vd][i] = a * b; break;
                case 3: cpu->v[vd][i] = a / b; break;
                default: break;
            }
        }
        break;
    }

    /* ===== Vector-Scalar ===== */
    case OP_VSCALAR: {
        uint32_t vd  = rd & 0x7;
        uint32_t vs1 = rs1 & 0x7;
        float s = cpu->f[rs2];
        switch (func) {
            case 0: for (uint32_t i = 0; i < cpu->vl; i++) cpu->v[vd][i] = cpu->v[vs1][i] + s; break;
            case 1: for (uint32_t i = 0; i < cpu->vl; i++) cpu->v[vd][i] = cpu->v[vs1][i] - s; break;
            case 2: for (uint32_t i = 0; i < cpu->vl; i++) cpu->v[vd][i] = cpu->v[vs1][i] * s; break;
            case 3: for (uint32_t i = 0; i < cpu->vl; i++) cpu->v[vd][i] = cpu->v[vs1][i] / s; break;
            default: break;
        }
        break;
    }

    /* ===== Vector Reductions ===== */
    case OP_VRED: {
        uint32_t vs = rs1 & 0x7;
        float res = 0;
        switch (func) {
            case 0: /* SUM */
                for (uint32_t i = 0; i < cpu->vl; i++) res += cpu->v[vs][i];
                break;
            case 1: /* MAX */
                res = cpu->v[vs][0];
                for (uint32_t i = 1; i < cpu->vl; i++) if (cpu->v[vs][i] > res) res = cpu->v[vs][i];
                break;
            case 2: /* MIN */
                res = cpu->v[vs][0];
                for (uint32_t i = 1; i < cpu->vl; i++) if (cpu->v[vs][i] < res) res = cpu->v[vs][i];
                break;
            case 3: /* SUM OF SQUARES */
                for (uint32_t i = 0; i < cpu->vl; i++) res += cpu->v[vs][i] * cpu->v[vs][i];
                break;
            default: break;
        }
        if (rd != 0) cpu->f[rd] = res;
        break;
    }

    /* ===== Vector Special ===== */
    case OP_VSPEC: {
        uint32_t vd  = (insn >> 23) & 0x7;
        uint32_t vs1 = (insn >> 20) & 0x7;
        uint32_t vfn = (insn >> 14) & 0x7;
        for (uint32_t i = 0; i < cpu->vl; i++) {
            float a = cpu->v[vs1][i];
            switch (vfn) {
                case 0: cpu->v[vd][i] = sqrtf(a); break;
                case 1: cpu->v[vd][i] = 1.0f / sqrtf(a); break;
                case 2: cpu->v[vd][i] = expf(a); break;
                case 3: cpu->v[vd][i] = silu_f(a); break;
                default: cpu->v[vd][i] = a; break;
            }
        }
        break;
    }

    /* ===== Vector Load ===== */
    case OP_VLOAD: {
        uint32_t vd = rd & 0x7;
        uint32_t base = cpu->r[rs1];
        uint32_t stride = (uint16_t)imm;
        for (uint32_t i = 0; i < cpu->vl; i++) {
            cpu->v[vd][i] = mem_readf(cpu, base + i * stride);
        }
        break;
    }

    /* ===== Vector Store ===== */
    case OP_VSTORE: {
        uint32_t vs = rd & 0x7;
        uint32_t base = cpu->r[rs1];
        uint32_t stride = (uint16_t)imm;
        for (uint32_t i = 0; i < cpu->vl; i++) {
            mem_writef(cpu, base + i * stride, cpu->v[vs][i]);
        }
        break;
    }

    /* ===== Q8 Setup ===== */
    case OP_Q8SET: {
        switch (func) {
            case Q8_SETSCALE: cpu->scale = cpu->f[rs1]; break;
            case Q8_SETQBASE: cpu->qbase = cpu->r[rs1]; break;
            case Q8_SETFBASE: cpu->fbase = cpu->r[rs1]; break;
            default: break;
        }
        break;
    }

    /* ===== Q8 MAC ===== */
    case OP_Q8MAC: {
        switch (func) {
            case Q8_ACCZERO:
                cpu->acc = 0.0;
                break;
            case Q8_ACCREAD:
                if (rd != 0) cpu->f[rd] = (float)cpu->acc;
                break;
            case Q8_MAC:
            case Q8_MACINC: {
                uint32_t n = ext ? ext : 16;
                double sum = 0.0;
                for (uint32_t i = 0; i < n; i++) {
                    int8_t w = mem_read8(cpu, cpu->qbase + i);
                    float x = mem_readf(cpu, cpu->fbase + i * 4);
                    sum += (float)w * cpu->scale * x;
                }
                cpu->acc += sum;
                if (func == Q8_MACINC) {
                    cpu->qbase += n;
                    cpu->fbase += n * 4;
                }
                cpu->cycle_count += n / 4;  /* Approximate pipeline cost */
                break;
            }
            default: break;
        }
        break;
    }

    /* ===== Branch ===== */
    case OP_BRANCH: {
        uint32_t cond = ((uint16_t)imm >> 13) & 0x7;
        int32_t offset = sign_extend13((uint16_t)imm & 0x1FFF);
        int32_t a = (int32_t)cpu->r[rd];
        int32_t b = (int32_t)cpu->r[rs1];
        int take = 0;
        switch (cond) {
            case BR_EQ:  take = (a == b); break;
            case BR_NE:  take = (a != b); break;
            case BR_LT:  take = (a < b); break;
            case BR_GE:  take = (a >= b); break;
            case BR_GTZ: take = (a > 0); break;
            case BR_LEZ: take = (a <= 0); break;
            default: break;
        }
        if (take) {
            next_pc = cpu->pc + (offset << 2);
        }
        break;
    }

    /* ===== JAL ===== */
    case OP_JAL: {
        if (rd != 0) cpu->r[rd] = cpu->pc + 4;
        next_pc = cpu->pc + ((int32_t)imm << 2);
        break;
    }

    /* ===== JALR ===== */
    case OP_JALR: {
        uint32_t target = cpu->r[rs1] + imm;
        if (rd != 0) cpu->r[rd] = cpu->pc + 4;
        next_pc = target;
        break;
    }

    /* ===== LOOP ===== */
    case OP_LOOP: {
        /* LOOP rd, offset: rd--; if (rd > 0) PC += offset<<2 */
        if (rd != 0) {
            cpu->r[rd]--;
            if ((int32_t)cpu->r[rd] > 0) {
                next_pc = cpu->pc + (imm << 2);
            }
        }
        break;
    }

    /* ===== System ===== */
    case OP_SYSTEM: {
        if (func == 0x10) {
            /* VSETVL */
            uint32_t requested = cpu->r[rs1];
            cpu->vl = requested < VEC_LEN ? requested : VEC_LEN;
            if (rd != 0) cpu->r[rd] = cpu->vl;
        } else if (func == 0x1F) {
            /* HALT */
            cpu->halted = 1;
            return -1;
        }
        break;
    }

    default:
        fprintf(stderr, "Unknown opcode 0x%02x at PC=0x%08x\n", op, cpu->pc);
        cpu->halted = 1;
        return -1;
    }

    /* Enforce R0=0, F0=0 */
    cpu->r[0] = 0;
    cpu->f[0] = 0.0f;

    cpu->pc = next_pc;
    cpu->insn_count++;
    cpu->cycle_count++;
    return 0;
}

int cpu_run(SmolCPU *cpu, uint64_t max_steps) {
    while (!cpu->halted && cpu->insn_count < max_steps) {
        if (cpu_step(cpu) < 0) break;
    }
    return cpu->halted ? 0 : 1;  /* 0 = halted normally, 1 = hit max */
}
