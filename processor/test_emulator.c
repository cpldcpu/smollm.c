/**
 * SMOL-32 Emulator Test
 * Loads the actual Q8 model and verifies emulator output matches C reference
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "emulator.h"

/* Memory layout for test:
 * 0x0000_0000 - 0x0000_0FFF : Program code (4KB)
 * 0x0001_0000 - 0x0001_FFFF : Scratch buffers (64KB)
 * 0x0010_0000 - 0x0FFF_FFFF : Model weights
 */
#define CODE_BASE    0x00000000
#define SCRATCH_BASE 0x00010000
#define WEIGHT_BASE  0x00100000

/* Scratch buffer offsets */
#define BUF_X        (SCRATCH_BASE + 0x0000)  /* Input activation [576] */
#define BUF_OUT      (SCRATCH_BASE + 0x1000)  /* Output [1536 max] */
#define BUF_OUT2     (SCRATCH_BASE + 0x3000)  /* Second output */

/* Reference C implementations for verification */
static void ref_matmul_q8(float *o, int8_t *w, float scale, float *x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++)
            sum += w[i * cols + j] * scale * x[j];
        o[i] = sum;
    }
}

static void ref_rmsnorm(float *o, float *x, float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}

/* Assemble matmul_q8 kernel directly (pre-assembled from matmul_q8.s) */
static void emit_matmul_program(SmolCPU *cpu, uint32_t addr) {
    /*
     * Program that does:
     *   R3 = output ptr, R4 = weight data, R5 = &scale, R6 = input ptr
     *   R7 = rows, R8 = cols
     *   Runs matmul, then halts
     *
     * We encode the instructions manually using the same encoding as assembler.py
     */
    uint32_t code[] = {
        /* LF F1, 0(R5)     - Load scale */
        0x08250000,
        /* QSETSCALE F1     - Set scale register */
        0x70010000,
        /* FSETBASE R6      - Set activation base */
        0x70060080,
        /* MV R9, R7        - row counter */
        0x11270000,

        /* row_loop: (offset 0x10) */
        /* QSETBASE R4      - Set weight row pointer */
        0x70040040,
        /* FSETBASE R6      - Reset activation base for this row */
        0x70060080,
        /* ACCZERO          - Clear accumulator */
        0x74000000,
        /* MV R10, R8       - col counter */
        0x11480000,
        /* SRLI R10, R10, 4 - cols / 16 */
        0x114A0244,

        /* col_loop: (offset 0x24) */
        /* Q8MACINC 16      - MAC 16 elements, advance QBASE+FBASE */
        0x740000D0,
        /* ADDI R10, R10, -1 */
        0x154AFFFF,
        /* BNEZ R10, -2     - branch back to col_loop */
        0xC1403FFE,

        /* ACCREAD F2       - Read result */
        0x74400040,
        /* SF F2, 0(R3)     - Store to output */
        0x0C430000,
        /* ADDI R3, R3, 4   - Advance output */
        0x14630004,
        /* ADD R4, R4, R8   - Next weight row */
        0x10844000,
        /* ADDI R9, R9, -1  - Decrement row counter */
        0x1529FFFF,
        /* BNEZ R9, -13     - branch back to row_loop */
        0xC1203FF3,

        /* HALT (SYSTEM func=0x1F) */
        0xE00007C0,
    };

    cpu_load_data(cpu, addr, code, sizeof(code));
}

/* Assemble rmsnorm kernel */
static void emit_rmsnorm_program(SmolCPU *cpu, uint32_t addr) {
    /*
     * R3 = output, R4 = input x, R5 = weights w, R6 = n
     * F1 = eps
     * Runs rmsnorm, then halts
     */
    uint32_t code[] = {
        /* MV R7, R4         - Save x pointer */
        0x10E40000,
        /* VSETVL R10, R6    - Set vector length */
        0xE1460400,
        /* FMOV F2, F0       - sum = 0 */
        0x20400300,
        /* MV R8, R6         - counter */
        0x11060000,

        /* sum_sq_loop: (offset 0x10) */
        /* LVF V0, R4, 4    */
        0x64040004,
        /* VREDSQS F3, V0   */
        0x486000C0,
        /* FADD F2, F2, F3  */
        0x20421800,
        /* ADDI R4, R4, 64  */
        0x14840040,
        /* ADDI R8, R8, -16 */
        0x1508FFF0,
        /* BGTZ R8, -5      */
        0xC100DFFB,

        /* FCVT.S.W F3, R6  - F3 = (float)n */
        0x20660000 | (F_FCVT_S_W << 6),
        /* FDIV F2, F2, F3  */
        0x204218C0,
        /* FADD F2, F2, F1  */
        0x20420800,
        /* FRSQRT F2, F2    */
        0x30420040,

        /* MV R4, R7        - Restore x */
        0x10870000,
        /* MV R8, R6        - Reset counter */
        0x11060000,

        /* scale_loop: (offset 0x40) */
        /* LVF V0, R4, 4   */
        0x64040004,
        /* LVF V1, R5, 4   */
        0x64250004,
        /* VMULS V0, V0, F2 */
        0x44001080,
        /* VMUL V0, V0, V1  */
        0x40028000,
        /* SVF V0, R3, 4   */
        0x68030004,
        /* ADDI R4, R4, 64 */
        0x14840040,
        /* ADDI R5, R5, 64 */
        0x14A50040,
        /* ADDI R3, R3, 64 */
        0x14630040,
        /* ADDI R8, R8, -16 */
        0x1508FFF0,
        /* BGTZ R8, -9     */
        0xC100DFF7,

        /* HALT */
        0xE00007C0,
    };

    cpu_load_data(cpu, addr, code, sizeof(code));
}

static float randf(void) {
    return (float)rand() / RAND_MAX * 2.0f - 1.0f;
}

static int test_matmul(SmolCPU *cpu) {
    printf("=== Test: Q8 Matrix-Vector Multiply ===\n");

    int rows = 64, cols = 576;  /* Small test size */
    float scale = 0.02f;

    /* Generate random weights and input */
    int8_t *weights = malloc(rows * cols);
    float *input = malloc(cols * sizeof(float));
    float *ref_output = malloc(rows * sizeof(float));

    srand(42);
    for (int i = 0; i < rows * cols; i++) weights[i] = (int8_t)(rand() % 255 - 127);
    for (int i = 0; i < cols; i++) input[i] = randf();

    /* Compute reference */
    ref_matmul_q8(ref_output, weights, scale, input, rows, cols);

    /* Load into emulator memory */
    uint32_t weight_addr = WEIGHT_BASE;
    uint32_t scale_addr = WEIGHT_BASE + rows * cols;
    uint32_t input_addr = BUF_X;
    uint32_t output_addr = BUF_OUT;

    cpu_load_data(cpu, weight_addr, weights, rows * cols);
    mem_writef(cpu, scale_addr, scale);
    cpu_load_data(cpu, input_addr, input, cols * sizeof(float));

    /* Set up registers */
    cpu->r[3] = output_addr;
    cpu->r[4] = weight_addr;
    cpu->r[5] = scale_addr;
    cpu->r[6] = input_addr;
    cpu->r[7] = rows;
    cpu->r[8] = cols;
    cpu->pc = CODE_BASE;
    cpu->halted = 0;
    cpu->insn_count = 0;
    cpu->cycle_count = 0;

    /* Load and run matmul program */
    emit_matmul_program(cpu, CODE_BASE);
    cpu_run(cpu, 10000000);

    /* Compare results */
    float max_diff = 0;
    int mismatches = 0;
    for (int i = 0; i < rows; i++) {
        float emu_val = mem_readf(cpu, output_addr + i * 4);
        float diff = fabsf(emu_val - ref_output[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-3f) mismatches++;
    }

    printf("  Rows=%d, Cols=%d, Scale=%.4f\n", rows, cols, scale);
    printf("  Max difference: %.6e\n", max_diff);
    printf("  Instructions: %llu, Cycles: %llu\n",
           (unsigned long long)cpu->insn_count, (unsigned long long)cpu->cycle_count);
    printf("  Result: %s\n\n", mismatches == 0 ? "PASS" : "FAIL");

    free(weights); free(input); free(ref_output);
    return mismatches == 0 ? 0 : 1;
}

static int test_rmsnorm(SmolCPU *cpu) {
    printf("=== Test: RMSNorm ===\n");

    int n = 576;
    float eps = 1e-5f;

    float *input = malloc(n * sizeof(float));
    float *weights = malloc(n * sizeof(float));
    float *ref_output = malloc(n * sizeof(float));

    srand(123);
    for (int i = 0; i < n; i++) { input[i] = randf(); weights[i] = randf() * 0.5f + 1.0f; }

    /* Reference */
    ref_rmsnorm(ref_output, input, weights, n, eps);

    /* Load into emulator memory */
    uint32_t input_addr = BUF_X;
    uint32_t weight_addr = WEIGHT_BASE;
    uint32_t output_addr = BUF_OUT;

    cpu_load_data(cpu, input_addr, input, n * sizeof(float));
    cpu_load_data(cpu, weight_addr, weights, n * sizeof(float));

    /* Set up registers */
    cpu->r[3] = output_addr;
    cpu->r[4] = input_addr;
    cpu->r[5] = weight_addr;
    cpu->r[6] = n;
    cpu->f[1] = eps;
    cpu->pc = CODE_BASE;
    cpu->halted = 0;
    cpu->insn_count = 0;
    cpu->cycle_count = 0;

    /* Load and run rmsnorm program */
    emit_rmsnorm_program(cpu, CODE_BASE);
    cpu_run(cpu, 10000000);

    /* Compare */
    float max_diff = 0;
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        float emu_val = mem_readf(cpu, output_addr + i * 4);
        float diff = fabsf(emu_val - ref_output[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4f) mismatches++;
    }

    printf("  n=%d, eps=%.1e\n", n, eps);
    printf("  Max difference: %.6e\n", max_diff);
    printf("  Instructions: %llu, Cycles: %llu\n",
           (unsigned long long)cpu->insn_count, (unsigned long long)cpu->cycle_count);
    printf("  Result: %s\n\n", mismatches == 0 ? "PASS" : "FAIL");

    free(input); free(weights); free(ref_output);
    return mismatches == 0 ? 0 : 1;
}

static int test_matmul_real_model(SmolCPU *cpu) {
    printf("=== Test: Matmul with Real Model Weights ===\n");

    /* Load actual model file */
    const char *model_path = "../models/smollm2-135m-q8.bin";
    FILE *f = fopen(model_path, "rb");
    if (!f) {
        printf("  Skipping (model file not found: %s)\n\n", model_path);
        return 0;
    }

    /* Parse header */
    char magic[4];
    if (fread(magic, 4, 1, f) != 1) { fclose(f); return -1; }
    if (memcmp(magic, "SMOL", 4)) { fclose(f); return -1; }

    uint32_t ver;
    if (fread(&ver, 4, 1, f) != 1) { fclose(f); return -1; }
    if (ver == 2) { uint32_t tmp; (void)fread(&tmp, 4, 1, f); (void)fread(&tmp, 4, 1, f); }

    uint32_t hs, is, nl, nh, nkv, vs, ms;
    float theta, eps;
    (void)fread(&hs, 4, 1, f); (void)fread(&is, 4, 1, f); (void)fread(&nl, 4, 1, f);
    (void)fread(&nh, 4, 1, f); (void)fread(&nkv, 4, 1, f); (void)fread(&vs, 4, 1, f);
    (void)fread(&ms, 4, 1, f); (void)fread(&theta, 4, 1, f); (void)fread(&eps, 4, 1, f);
    (void)is; (void)ms; (void)theta; (void)eps;

    printf("  Model: hidden=%u, layers=%u, heads=%u, kv=%u\n", hs, nl, nh, nkv);

    /* Skip tokenizer */
    uint32_t nv, nm;
    (void)fread(&nv, 4, 1, f); (void)fread(&nm, 4, 1, f);
    for (uint32_t i = 0; i < nv; i++) { uint32_t len; (void)fread(&len, 4, 1, f); fseek(f, len, SEEK_CUR); }
    for (uint32_t i = 0; i < nm; i++) { uint32_t len; (void)fread(&len, 4, 1, f); fseek(f, len, SEEK_CUR); }

    /* Read embedding weight: scale + data[vocab_size * hidden_size] */
    float embed_scale;
    (void)fread(&embed_scale, 4, 1, f);
    size_t embed_size = (size_t)vs * hs;
    fseek(f, embed_size, SEEK_CUR);

    /* Read first layer's input_layernorm */
    float *ln_weight = malloc(hs * sizeof(float));
    (void)fread(ln_weight, sizeof(float), hs, f);

    /* Read first layer's q_proj: scale + data[nh*hd, hs] */
    float q_scale;
    (void)fread(&q_scale, 4, 1, f);
    int q_rows = (int)(nh * (hs / nh)), q_cols = (int)hs;
    int8_t *q_data = malloc(q_rows * q_cols);
    (void)fread(q_data, 1, q_rows * q_cols, f);

    fclose(f);

    /* Test: Apply first layer's q_proj to a random input */
    float *input = malloc(hs * sizeof(float));
    float *ref_output = malloc(q_rows * sizeof(float));
    srand(999);
    for (uint32_t i = 0; i < hs; i++) input[i] = randf() * 0.1f;

    /* Reference */
    ref_matmul_q8(ref_output, q_data, q_scale, input, q_rows, q_cols);

    /* Emulator setup */
    uint32_t weight_addr = WEIGHT_BASE;
    uint32_t scale_addr = WEIGHT_BASE + q_rows * q_cols;
    uint32_t input_addr = BUF_X;
    uint32_t output_addr = BUF_OUT;

    cpu_load_data(cpu, weight_addr, q_data, q_rows * q_cols);
    mem_writef(cpu, scale_addr, q_scale);
    cpu_load_data(cpu, input_addr, input, hs * sizeof(float));

    cpu->r[3] = output_addr;
    cpu->r[4] = weight_addr;
    cpu->r[5] = scale_addr;
    cpu->r[6] = input_addr;
    cpu->r[7] = q_rows;
    cpu->r[8] = q_cols;
    cpu->pc = CODE_BASE;
    cpu->halted = 0;
    cpu->insn_count = 0;
    cpu->cycle_count = 0;

    emit_matmul_program(cpu, CODE_BASE);
    cpu_run(cpu, 100000000);

    /* Compare */
    float max_diff = 0;
    int mismatches = 0;
    for (int i = 0; i < q_rows; i++) {
        float emu_val = mem_readf(cpu, output_addr + i * 4);
        float diff = fabsf(emu_val - ref_output[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-2f) mismatches++;
    }

    printf("  Q_proj: rows=%d, cols=%d, scale=%.6f\n", q_rows, q_cols, q_scale);
    printf("  Max difference: %.6e\n", max_diff);
    printf("  Instructions: %llu\n", (unsigned long long)cpu->insn_count);
    printf("  Result: %s\n\n", mismatches == 0 ? "PASS" : "FAIL");

    free(ln_weight); free(q_data); free(input); free(ref_output);
    return mismatches == 0 ? 0 : 1;
}

int main(void) {
    printf("SMOL-32 Emulator Test Suite\n");
    printf("==========================\n\n");

    SmolCPU cpu;
    cpu_init(&cpu);

    int failures = 0;
    failures += test_matmul(&cpu);
    failures += test_rmsnorm(&cpu);
    failures += test_matmul_real_model(&cpu);

    printf("==========================\n");
    if (failures == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) failed.\n", failures);
    }

    cpu_free(&cpu);
    return failures;
}
