/**
 * SMOL-32 Verilator Testbench
 * High-speed simulation for kernel and forward pass testing
 */

#include <verilated.h>
#include <verilated_vcd_c.h>
#include "Vsmol32_top.h"
#include "Vsmol32_top___024root.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

// Kernel entry points (fixed addresses like the emulator)
#define MATMUL_ENTRY    0x00001000
#define RMSNORM_ENTRY   0x00002000
#define ROPE_ENTRY      0x00003000
#define ATTN_ENTRY      0x00004000
#define SILU_ENTRY      0x00005000
#define RESIDUAL_ENTRY  0x00006000
#define EMBED_ENTRY     0x00007000
#define MEMCPY_ENTRY    0x00008000

// Buffer addresses
#define BUF_BASE        0x00100000
#define BUF_X           (BUF_BASE + 0x00000)
#define BUF_XB          (BUF_BASE + 0x01000)
#define BUF_W           (BUF_BASE + 0x02000)

// Test addresses
#define KERNEL_ENTRY  0x00001000
#define OUTPUT_ADDR   0x00002000
#define WEIGHT_ADDR   0x00002100
#define SCALE_ADDR    0x00002200
#define INPUT_ADDR    0x00002300

// IEEE 754 helpers
union FloatBits {
    float f;
    uint32_t u;
};

static uint32_t float_to_bits(float f) {
    FloatBits fb;
    fb.f = f;
    return fb.u;
}

static float bits_to_float(uint32_t u) {
    FloatBits fb;
    fb.u = u;
    return fb.f;
}

class SmolSimulator {
public:
    Vsmol32_top* top;
    VerilatedVcdC* tfp;
    uint64_t sim_time;
    uint64_t cycle_count;
    bool trace_enabled;

    SmolSimulator(bool enable_trace = false) {
        top = new Vsmol32_top;
        tfp = nullptr;
        sim_time = 0;
        cycle_count = 0;
        trace_enabled = enable_trace;

        if (trace_enabled) {
            Verilated::traceEverOn(true);
            tfp = new VerilatedVcdC;
            top->trace(tfp, 99);
            tfp->open("smol32_verilator.vcd");
        }
    }

    ~SmolSimulator() {
        if (tfp) {
            tfp->close();
            delete tfp;
        }
        delete top;
    }

    void hold_reset() {
        top->rst_n = 0;
        top->clk = 0;
        top->eval();
    }

    void release_reset() {
        top->rst_n = 1;
        top->eval();
    }

    void reset() {
        hold_reset();
        tick(); tick();
        release_reset();
        tick(); tick();
    }

    void tick() {
        top->clk = 0;
        top->eval();
        if (tfp) tfp->dump(sim_time++);

        top->clk = 1;
        top->eval();
        if (tfp) tfp->dump(sim_time++);

        cycle_count++;
    }

    void run_until_halt(uint64_t max_cycles = 1000000) {
        while (!top->halted && cycle_count < max_cycles) {
            tick();
        }
    }

    // Direct memory access using Verilator's rootp access
    void write_mem(uint32_t addr, uint32_t data) {
        top->rootp->smol32_top__DOT__memory[addr / 4] = data;
    }

    uint32_t read_mem(uint32_t addr) {
        return top->rootp->smol32_top__DOT__memory[addr / 4];
    }

    void write_reg(int reg, uint32_t val) {
        // Verilog array is [1:31], Verilator maps to [0:30], so subtract 1
        if (reg > 0 && reg < 32) {
            top->rootp->smol32_top__DOT__core__DOT__regfile__DOT__regs[reg - 1] = val;
        }
    }

    uint32_t read_reg(int reg) {
        // Verilog array is [1:31], Verilator maps to [0:30], so subtract 1
        if (reg == 0) return 0;
        return top->rootp->smol32_top__DOT__core__DOT__regfile__DOT__regs[reg - 1];
    }

    // FP register access (array named 'fregs', same offset adjustment)
    void write_fp_reg(int reg, uint32_t val) {
        if (reg > 0 && reg < 32) {
            top->rootp->smol32_top__DOT__core__DOT__fp_regfile__DOT__fregs[reg - 1] = val;
        }
    }

    uint32_t read_fp_reg(int reg) {
        if (reg == 0) return 0;
        return top->rootp->smol32_top__DOT__core__DOT__fp_regfile__DOT__fregs[reg - 1];
    }

    // Load kernel binary from file
    int load_kernel(const char* path, uint32_t addr) {
        FILE* f = fopen(path, "rb");
        if (!f) {
            printf("Error: Cannot open kernel file: %s\n", path);
            return -1;
        }
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);

        uint8_t* buf = (uint8_t*)malloc(size);
        if (fread(buf, 1, size, f) != (size_t)size) {
            fclose(f);
            free(buf);
            return -1;
        }
        fclose(f);

        // Write to memory (32-bit aligned)
        for (long i = 0; i < size; i += 4) {
            uint32_t word = 0;
            for (int j = 0; j < 4 && (i + j) < size; j++) {
                word |= ((uint32_t)buf[i + j]) << (j * 8);
            }
            write_mem(addr + i, word);
        }

        free(buf);
        printf("  Loaded kernel %s (%ld bytes) at 0x%08X\n", path, size, addr);
        return 0;
    }

    // Reset CPU for kernel call (RA=0 for halt on return)
    void reset_for_call(uint32_t entry) {
        // Clear all integer registers
        for (int i = 1; i < 32; i++) write_reg(i, 0);
        // Clear FP registers
        for (int i = 1; i < 32; i++) write_fp_reg(i, 0);
        // Set RA to 0 (halt trap)
        write_reg(1, 0);
        // Set SP
        write_reg(2, 0x00FF0000);
        // Reset cycle count
        cycle_count = 0;
    }

    // Direct PC write (requires processor to be in a state where PC can be modified)
    void write_pc(uint32_t addr) {
        top->rootp->smol32_top__DOT__core__DOT__pc = addr;
    }

    uint32_t read_pc() {
        return top->rootp->smol32_top__DOT__core__DOT__pc;
    }
};

// Load matmul_q8 kernel instructions
static uint32_t matmul_kernel[] = {
    0x08250000,  // LF F1, 0(R5)
    0x70010000,  // QSETSCALE F1
    0x70060080,  // FSETBASE R6
    0x11270000,  // MV R9, R7
    0x70040040,  // QSETBASE R4
    0x70060080,  // FSETBASE R6
    0x74000000,  // ACCZERO
    0x11480000,  // MV R10, R8
    0x114a0244,  // SRLI R10, R10, 4
    0x740000d0,  // Q8MACINC 16
    0xcd40ffff,  // LOOP R10, -1
    0x74400040,  // ACCREAD F2
    0x0c430000,  // SF F2, 0(R3)
    0x14630004,  // ADDI R3, R3, 4
    0x10844000,  // ADD R4, R4, R8
    0xcd20fff5,  // LOOP R9, -11
    0xc8010000,  // RET
};

int test_simple() {
    // Simple test: ADDI R3, R0, 42; HALT
    printf("=== Simple Verilator Test ===\n");

    SmolSimulator sim(false);

    // Write memory BEFORE holding reset (memory doesn't have reset)
    // HALT trap at 0
    sim.write_mem(0x0000, 0xE0000000);

    // At 0x1000: ADDI R3, R0, 42  ->  op=05, rd=3, rs1=0, imm=42
    // Binary: 000101 00011 00000 0000000000101010 = 0x1460002A
    sim.write_mem(0x1000, 0x1460002A);

    // At 0x1004: JALR R0, R1, 0 (RET - return to R1 which is 0 after reset)
    sim.write_mem(0x1004, 0xC8010000);

    printf("Memory[0x0000] = 0x%08X (expect 0xE0000000)\n", sim.read_mem(0x0000));
    printf("Memory[0x1000] = 0x%08X (expect 0x1460002A)\n", sim.read_mem(0x1000));
    printf("Memory[0x1004] = 0x%08X (expect 0xC8010000)\n", sim.read_mem(0x1004));

    // Reset sequence
    sim.hold_reset();
    sim.tick();
    sim.tick();
    sim.release_reset();
    sim.tick();

    printf("After reset: PC=0x%08X State=%d\n", sim.top->pc, sim.top->state);

    // Run a few cycles with debug
    for (int i = 0; i < 20 && !sim.top->halted; i++) {
        sim.tick();
        printf("Cycle %d: PC=0x%08X State=%d R3=%d halted=%d\n",
               (int)sim.cycle_count, sim.top->pc, sim.top->state, sim.read_reg(3), sim.top->halted);
    }

    printf("\nFinal: Cycles: %lu\n", sim.cycle_count);
    printf("Halted: %d\n", sim.top->halted);
    printf("PC: 0x%08X\n", sim.top->pc);
    printf("R3: %d (expected 42)\n", sim.read_reg(3));

    return (sim.top->halted && sim.read_reg(3) == 42) ? 0 : 1;
}

int test_matmul_kernel() {
    printf("=== Testing matmul_q8 kernel with Verilator ===\n");

    SmolSimulator sim(false);  // No trace for speed

    // Hold in reset while setting up memory
    sim.hold_reset();
    sim.tick();

    // Set up memory while in reset (memory doesn't reset)

    // Place HALT trap at address 0
    sim.write_mem(0x0000, 0xE0000000);

    // Load kernel at 0x1000
    for (int i = 0; i < (int)(sizeof(matmul_kernel)/sizeof(uint32_t)); i++) {
        sim.write_mem(KERNEL_ENTRY + i * 4, matmul_kernel[i]);
    }

    // Initialize output buffer
    sim.write_mem(OUTPUT_ADDR + 0, 0);
    sim.write_mem(OUTPUT_ADDR + 4, 0);

    // Weight data: row 0 = all 1s, row 1 = all 2s (2 rows x 16 cols)
    sim.write_mem(WEIGHT_ADDR + 0x00, 0x01010101);
    sim.write_mem(WEIGHT_ADDR + 0x04, 0x01010101);
    sim.write_mem(WEIGHT_ADDR + 0x08, 0x01010101);
    sim.write_mem(WEIGHT_ADDR + 0x0C, 0x01010101);
    sim.write_mem(WEIGHT_ADDR + 0x10, 0x02020202);
    sim.write_mem(WEIGHT_ADDR + 0x14, 0x02020202);
    sim.write_mem(WEIGHT_ADDR + 0x18, 0x02020202);
    sim.write_mem(WEIGHT_ADDR + 0x1C, 0x02020202);

    // Scale = 1.0
    sim.write_mem(SCALE_ADDR, float_to_bits(1.0f));

    // Input = all 1.0s
    for (int i = 0; i < 16; i++) {
        sim.write_mem(INPUT_ADDR + i * 4, float_to_bits(1.0f));
    }

    // Release reset first, then set registers
    // (registers are cleared during reset, so set them after)
    sim.release_reset();
    sim.top->eval();  // Let reset propagate

    // Now set up registers after reset is released
    sim.write_reg(1, 0x00000000);  // RA = 0 (halt on return)
    sim.write_reg(3, OUTPUT_ADDR); // output ptr
    sim.write_reg(4, WEIGHT_ADDR); // weight ptr
    sim.write_reg(5, SCALE_ADDR);  // scale ptr
    sim.write_reg(6, INPUT_ADDR);  // input ptr
    sim.write_reg(7, 2);           // rows = 2
    sim.write_reg(8, 16);          // cols = 16
    sim.top->eval();  // Let register writes take effect

    // Verify registers are set
    printf("After setup: R7=%d R8=%d R3=0x%X\n",
           sim.read_reg(7), sim.read_reg(8), sim.read_reg(3));

    // Run until halt
    sim.run_until_halt(100000);

    printf("Cycles: %lu\n", sim.cycle_count);
    printf("Halted: %d\n", sim.top->halted);
    printf("Final PC: 0x%08X\n", sim.top->pc);
    printf("State: %d\n", sim.top->state);
    printf("R3 (out): 0x%08X\n", sim.read_reg(3));
    printf("R7 (rows): %d\n", sim.read_reg(7));
    printf("R8 (cols): %d\n", sim.read_reg(8));
    printf("R9 (row_cnt): %d\n", sim.read_reg(9));
    printf("R10 (col_cnt): %d\n", sim.read_reg(10));

    // Check results
    uint32_t out0 = sim.read_mem(OUTPUT_ADDR + 0);
    uint32_t out1 = sim.read_mem(OUTPUT_ADDR + 4);

    float f0 = bits_to_float(out0);
    float f1 = bits_to_float(out1);

    printf("output[0] = 0x%08X (%.1f, expected 16.0)\n", out0, f0);
    printf("output[1] = 0x%08X (%.1f, expected 32.0)\n", out1, f1);

    int passed = (out0 == 0x41800000 && out1 == 0x42000000);
    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed ? 0 : 1;
}

// Reference RMSNorm implementation for verification
static void ref_rmsnorm(float *o, float *x, float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}

int test_rmsnorm_kernel() {
    printf("\n=== Testing rmsnorm kernel with Verilator ===\n");

    SmolSimulator sim(false);

    // Set up memory before reset
    sim.hold_reset();
    sim.tick();

    // Place HALT trap at address 0
    sim.write_mem(0x0000, 0xE0000000);

    // Load rmsnorm kernel at 0x1000 (where PC starts after reset)
    if (sim.load_kernel("../kernels/rmsnorm.bin", 0x1000) != 0) {
        printf("SKIPPED - kernel binary not found\n");
        return 0;  // Skip, don't fail
    }

    // Test with n=32 floats (2 vector loads worth)
    const int n = 32;
    const float eps = 1e-5f;

    // Set up input data (x = 1.0, 2.0, 3.0, ...)
    float x_ref[n], w_ref[n], o_ref[n];
    for (int i = 0; i < n; i++) {
        x_ref[i] = (float)(i + 1);
        w_ref[i] = 1.0f;  // Weights = 1 for simple test
        sim.write_mem(BUF_X + i * 4, float_to_bits(x_ref[i]));
        sim.write_mem(BUF_W + i * 4, float_to_bits(w_ref[i]));
    }

    // Clear output buffer
    for (int i = 0; i < n; i++) {
        sim.write_mem(BUF_XB + i * 4, 0);
    }

    // Compute reference result
    ref_rmsnorm(o_ref, x_ref, w_ref, n, eps);

    // Release reset and set up registers (like matmul test)
    sim.release_reset();
    sim.top->eval();

    // Set up registers after reset (don't use reset_for_call)
    sim.write_reg(1, 0);            // RA = 0 (halt on return)
    sim.write_reg(3, BUF_XB);       // output ptr
    sim.write_reg(4, BUF_X);        // input ptr
    sim.write_reg(5, BUF_W);        // weight ptr
    sim.write_reg(6, n);            // dimension
    sim.write_fp_reg(1, float_to_bits(eps));  // F1 = eps

    sim.top->eval();

    printf("Running rmsnorm kernel (n=%d, eps=%.1e)...\n", n, eps);

    // Run with detailed debug
    for (int i = 0; i < 1000 && !sim.top->halted; i++) {
        sim.tick();
        uint32_t state = sim.top->state;
        uint32_t pc = sim.top->pc;
        uint32_t ir = sim.top->rootp->smol32_top__DOT__core__DOT__ir;
        uint32_t opcode = (ir >> 26) & 0x3F;

        // Print at WRITEBACK for key FP operations
        if (state == 4) {
            // All FP operations (FPU=0x08, FSPEC=0x0C)
            if (opcode == 0x08 || opcode == 0x0C) {
                uint32_t f1 = sim.read_fp_reg(1);
                uint32_t f2 = sim.read_fp_reg(2);
                uint32_t f3 = sim.read_fp_reg(3);
                uint32_t func = (ir >> 6) & 0x1F;
                uint32_t rd = (ir >> 21) & 0x1F;
                printf("  FP: PC=0x%X op=0x%02X func=%d rd=%d F1=%.6f F2=%.6f F3=%.6f\n",
                       pc, opcode, func, rd, bits_to_float(f1), bits_to_float(f2), bits_to_float(f3));
            }
            // VMULS (VSCALAR opcode 0x11)
            if (opcode == 0x11) {
                uint32_t f2 = sim.read_fp_reg(2);
                uint32_t rs2 = (ir >> 11) & 0x1F;
                uint32_t fs2 = sim.read_fp_reg(rs2);
                uint32_t v0_lane0 = sim.top->rootp->smol32_top__DOT__core__DOT__vec_regfile__DOT__vregs[0][0];
                printf("  VMULS: PC=0x%X F2=%.6f rs2=%d F[rs2]=%.6f V0[0]=%.6f\n",
                       pc, bits_to_float(f2), rs2, bits_to_float(fs2), bits_to_float(v0_lane0));
            }
            // VMUL (VARITH opcode 0x10)
            if (opcode == 0x10) {
                uint32_t v0_lane0 = sim.top->rootp->smol32_top__DOT__core__DOT__vec_regfile__DOT__vregs[0][0];
                uint32_t v1_lane0 = sim.top->rootp->smol32_top__DOT__core__DOT__vec_regfile__DOT__vregs[1][0];
                printf("  VMUL: PC=0x%X V0[0]=%.6f V1[0]=%.6f\n",
                       pc, bits_to_float(v0_lane0), bits_to_float(v1_lane0));
            }
            // VREDSQS (VRED opcode 0x12)
            if (opcode == 0x12) {
                uint32_t f2 = sim.read_fp_reg(2);
                uint32_t f3 = sim.read_fp_reg(3);
                printf("  VREDSQS: PC=0x%X F2=%.6f F3=%.6f\n", pc, bits_to_float(f2), bits_to_float(f3));
            }
        }
    }

    sim.run_until_halt(10000);

    printf("Cycles: %lu\n", sim.cycle_count);
    printf("Halted: %d\n", sim.top->halted);
    printf("Final PC: 0x%08X\n", sim.top->pc);

    if (!sim.top->halted) {
        printf("FAILED - did not halt\n");
        return 1;
    }

    // Check results
    int errors = 0;
    float max_error = 0;
    for (int i = 0; i < n; i++) {
        uint32_t out_bits = sim.read_mem(BUF_XB + i * 4);
        float out_f = bits_to_float(out_bits);
        float error = fabsf(out_f - o_ref[i]);
        if (error > max_error) max_error = error;
        if (error > 0.001f) {
            if (errors < 5) {
                printf("  Error at [%d]: got %.6f, expected %.6f (diff=%.6f)\n",
                       i, out_f, o_ref[i], error);
            }
            errors++;
        }
    }

    printf("Max error: %.6f, Errors: %d/%d\n", max_error, errors, n);
    printf("%s\n", errors == 0 ? "PASSED" : "FAILED");

    return errors == 0 ? 0 : 1;
}

int test_residual_kernel() {
    printf("\n=== Testing residual kernel with Verilator ===\n");

    SmolSimulator sim(false);

    sim.hold_reset();
    sim.tick();

    // Place HALT trap at address 0
    sim.write_mem(0x0000, 0xE0000000);

    // Load residual kernel at 0x1000 (where PC starts)
    if (sim.load_kernel("../kernels/residual.bin", 0x1000) != 0) {
        printf("SKIPPED - kernel binary not found\n");
        return 0;
    }

    // Test with n=16 floats
    const int n = 16;

    // Set up input data
    for (int i = 0; i < n; i++) {
        sim.write_mem(BUF_X + i * 4, float_to_bits((float)(i + 1)));       // x = 1,2,3...
        sim.write_mem(BUF_XB + i * 4, float_to_bits((float)(i * 10)));     // xb = 0,10,20...
    }

    // Release reset and set up registers
    sim.release_reset();
    sim.top->eval();

    // Set up registers after reset
    sim.write_reg(1, 0);        // RA = 0 (halt on return)
    sim.write_reg(3, BUF_X);    // x
    sim.write_reg(4, BUF_XB);   // xb
    sim.write_reg(5, n);        // n

    sim.top->eval();

    printf("Running residual kernel (n=%d)...\n", n);

    sim.run_until_halt(1000);

    printf("Cycles: %lu, Halted: %d\n", sim.cycle_count, sim.top->halted);

    if (!sim.top->halted) {
        printf("FAILED - did not halt\n");
        return 1;
    }

    // Check results: x[i] should be (i+1) + (i*10)
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = (float)(i + 1) + (float)(i * 10);
        float got = bits_to_float(sim.read_mem(BUF_X + i * 4));
        if (fabsf(got - expected) > 0.001f) {
            if (errors < 5) printf("  Error at [%d]: got %.1f, expected %.1f\n", i, got, expected);
            errors++;
        }
    }

    printf("%s\n", errors == 0 ? "PASSED" : "FAILED");
    return errors == 0 ? 0 : 1;
}

// Test exp/silu numerical accuracy across a range of inputs
int test_exp_silu_accuracy() {
    printf("\n=== Testing exp/silu numerical accuracy ===\n");

    // Test values including edge cases and the problematic value 11.43
    float test_values[] = {
        0.0f, 0.5f, 1.0f, 2.0f, 3.0f, 5.0f,
        -0.5f, -1.0f, -2.0f, -3.0f, -5.0f,
        10.0f, 11.0f, 11.43f, 12.0f, 15.0f, 20.0f,  // Large positive (silu input)
        -10.0f, -11.43f, -15.0f, -20.0f,            // Large negative (exp(-x) for silu)
        0.001f, -0.001f                              // Small values
    };
    int n_tests = sizeof(test_values) / sizeof(test_values[0]);

    printf("Testing %d values for exp() and silu()...\n", n_tests);

    int exp_errors = 0, silu_errors = 0;
    float max_exp_err = 0, max_silu_err = 0;
    float worst_exp_input = 0, worst_silu_input = 0;

    for (int i = 0; i < n_tests; i++) {
        float x = test_values[i];

        // C reference
        float c_exp = expf(x);
        float c_silu = x / (1.0f + expf(-x));

        // Helper lambda: exp_approx using 2^(x * log2(e)) decomposition
        auto exp_approx_test = [](double x) -> double {
            double t = x * 1.4426950408889634;  // log2(e)
            int n = (t >= 0) ? (int)t : ((int)t - ((t == (int)t) ? 0 : 1));
            double f = t - n;
            double pow2_f = 1.0 + f * (0.6931471805599453 +
                           f * (0.2402265069591007 +
                           f * (0.0555041086648216 +
                           f * (0.0096181291076285 +
                           f * 0.0013333558146428))));
            if (n > 127) return 3.4e38;
            if (n < -126) return 0.0;
            return pow2_f * pow(2.0, n);
        };

        // Test exp(x)
        double v_exp = exp_approx_test(x);

        // Test silu(x) = x / (1 + exp(-x))
        double v_exp_neg = exp_approx_test(-x);
        double v_silu = x / (1.0 + v_exp_neg);

        // Compute relative errors
        float exp_rel_err = (c_exp != 0) ? fabsf((float)v_exp - c_exp) / fabsf(c_exp) : fabsf((float)v_exp);
        float silu_rel_err = (c_silu != 0) ? fabsf((float)v_silu - c_silu) / fabsf(c_silu) : fabsf((float)v_silu);

        // Track max errors
        if (exp_rel_err > max_exp_err) { max_exp_err = exp_rel_err; worst_exp_input = x; }
        if (silu_rel_err > max_silu_err) { max_silu_err = silu_rel_err; worst_silu_input = x; }

        // Error threshold: 1e-4 relative error (reasonable for single-precision)
        if (exp_rel_err > 1e-4f) exp_errors++;
        if (silu_rel_err > 1e-4f) silu_errors++;

        // Print detailed info for problematic values
        if (exp_rel_err > 1e-4f || silu_rel_err > 1e-4f || fabsf(x) > 10.0f) {
            printf("  x=%.4f: exp(C)=%.6e, exp(V)=%.6e, rel_err=%.2e | silu(C)=%.6f, silu(V)=%.6f, rel_err=%.2e%s%s\n",
                   x, c_exp, (float)v_exp, exp_rel_err,
                   c_silu, (float)v_silu, silu_rel_err,
                   (exp_rel_err > 1e-4f) ? " [EXP ERR]" : "",
                   (silu_rel_err > 1e-4f) ? " [SILU ERR]" : "");
        }
    }

    printf("Results: exp errors=%d/%d (max_rel=%.2e at x=%.2f), silu errors=%d/%d (max_rel=%.2e at x=%.2f)\n",
           exp_errors, n_tests, max_exp_err, worst_exp_input,
           silu_errors, n_tests, max_silu_err, worst_silu_input);

    if (exp_errors == 0 && silu_errors == 0) {
        printf("PASSED\n");
        return 0;
    } else {
        printf("FAILED\n");
        return 1;
    }
}

int test_silu_mul_kernel() {
    printf("\n=== Testing silu_mul kernel with Verilator ===\n");

    SmolSimulator sim(false);

    sim.hold_reset();
    sim.tick();

    sim.write_mem(0x0000, 0xE0000000);

    if (sim.load_kernel("../kernels/silu_mul.bin", 0x1000) != 0) {
        printf("SKIPPED - kernel binary not found\n");
        return 0;
    }

    const int n = 16;

    // gate = 1.0, up = 2.0 for all elements
    for (int i = 0; i < n; i++) {
        sim.write_mem(BUF_X + i * 4, float_to_bits(1.0f));    // gate
        sim.write_mem(BUF_XB + i * 4, float_to_bits(2.0f));   // up
    }

    sim.release_reset();
    sim.top->eval();

    // Set up registers after reset
    sim.write_reg(1, 0);        // RA = 0 (halt on return)
    sim.write_reg(3, BUF_X);    // gate
    sim.write_reg(4, BUF_XB);   // up
    sim.write_reg(5, n);        // n

    sim.top->eval();

    printf("Running silu_mul kernel (n=%d)...\n", n);

    // Debug first 200 cycles to see VSILU/VMUL/SVF execution in detail
    for (int i = 0; i < 200 && !sim.top->halted; i++) {
        sim.tick();
        uint32_t state = sim.top->state;
        uint32_t pc = sim.top->pc;
        uint32_t ir = sim.top->rootp->smol32_top__DOT__core__DOT__ir;
        uint32_t opcode = (ir >> 26) & 0x3F;

        // Get vecmem state
        uint32_t vecmem_state = sim.top->rootp->smol32_top__DOT__core__DOT__vecmem__DOT__state;
        uint32_t vecmem_lane = sim.top->rootp->smol32_top__DOT__core__DOT__vecmem__DOT__lane_count;
        uint32_t vecmem_wdata = sim.top->rootp->smol32_top__DOT__core__DOT__vec_mem_wdata;

        // Get V0 contents
        uint32_t v0_lane0 = sim.top->rootp->smol32_top__DOT__core__DOT__vec_regfile__DOT__vregs[0][0];
        uint32_t v0_lane1 = sim.top->rootp->smol32_top__DOT__core__DOT__vec_regfile__DOT__vregs[0][1];

        // Print for LVF (0x19), VSPEC/VSILU (0x18), VMUL (0x10), SVF (0x1A)
        bool is_vec_instr = (opcode == 0x19 || opcode == 0x18 || opcode == 0x10 || opcode == 0x1A);

        // Print on state 2 (EXECUTE) or state 4 (WRITEBACK) for vector instructions
        // Also print when vecmem is active (states 1-4)
        if (is_vec_instr && (state == 2 || state == 4 || state == 6)) {
            uint32_t vec_we = sim.top->rootp->smol32_top__DOT__core__DOT__vec_regfile_we;
            printf("  C%d: St=%d PC=0x%X op=0x%02X vecWE=%d vmSt=%d vmLane=%d vmWdata=0x%08X(%.4f) V0[0]=%.4f V0[1]=%.4f\n",
                   i, state, pc, opcode, vec_we, vecmem_state, vecmem_lane,
                   vecmem_wdata, bits_to_float(vecmem_wdata),
                   bits_to_float(v0_lane0), bits_to_float(v0_lane1));
        }

        // Also print when vecmem is writing (vec_mem_we=1)
        uint32_t vec_mem_we = sim.top->rootp->smol32_top__DOT__core__DOT__vec_mem_we;
        if (vec_mem_we && opcode == 0x1A) {
            uint32_t vec_mem_addr = sim.top->rootp->smol32_top__DOT__core__DOT__vec_mem_addr;
            printf("    --> WRITE: addr=0x%X data=0x%08X(%.4f) lane=%d\n",
                   vec_mem_addr, vecmem_wdata, bits_to_float(vecmem_wdata), vecmem_lane);
        }
    }

    sim.run_until_halt(1000);

    printf("Cycles: %lu, Halted: %d\n", sim.cycle_count, sim.top->halted);

    if (!sim.top->halted) {
        printf("FAILED - did not halt\n");
        return 1;
    }

    // silu(1.0) = 1.0 / (1 + exp(-1.0)) ≈ 0.7311
    // result = 0.7311 * 2.0 ≈ 1.4621
    float expected = 1.0f / (1.0f + expf(-1.0f)) * 2.0f;
    float got = bits_to_float(sim.read_mem(BUF_X));

    // Debug: print actual stored values
    printf("First 4 outputs: ");
    for (int i = 0; i < 4; i++) {
        uint32_t val = sim.read_mem(BUF_X + i * 4);
        printf("0x%08X(%.4f) ", val, bits_to_float(val));
    }
    printf("\n");

    printf("Result[0] = %.4f (expected %.4f)\n", got, expected);

    int passed = fabsf(got - expected) < 0.01f;
    printf("%s\n", passed ? "PASSED" : "FAILED");

    return passed ? 0 : 1;
}

// ============================================================================
// Full Forward Pass Test
// ============================================================================

// Memory layout (same as emulator)
#define CODE_BASE       0x00000000
#define FORWARD_ENTRY   0x00009000
#define DESC_BASE       0x000E0000

#define WEIGHT_BASE     0x00400000
#define KV_BASE         0x09000000
#define ROPE_BASE       0x0F000000

// Model config
struct ModelConfig {
    int hidden_size, intermediate_size, num_layers;
    int num_heads, num_kv_heads, vocab_size, max_seq_len;
    float rope_theta, rms_norm_eps;
    int head_dim;
};

// Layer weight addresses
struct LayerAddrs {
    uint32_t input_ln;
    uint32_t q_data, q_scale_addr;
    uint32_t k_data, k_scale_addr;
    uint32_t v_data, v_scale_addr;
    uint32_t o_data, o_scale_addr;
    uint32_t post_ln;
    uint32_t gate_data, gate_scale_addr;
    uint32_t up_data, up_scale_addr;
    uint32_t down_data, down_scale_addr;
};

// Reference implementations for forward pass comparison (reuse existing ref_rmsnorm)
static void fwd_matmul_q8(float *o, int8_t *data, float scale, float *x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        int8_t *row = data + (int64_t)i * cols;
        for (int j = 0; j < cols; j++) sum += row[j] * scale * x[j];
        o[i] = sum;
    }
}

static float fwd_silu(float x) { return x / (1.0f + expf(-x)); }

static void fwd_softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void fwd_apply_rope(float *v, int hd, float *c, float *s) {
    int h = hd / 2;
    for (int i = 0; i < h; i++) {
        float v0 = v[i], v1 = v[i + h];
        v[i] = v0 * c[i] - v1 * s[i];
        v[i + h] = v1 * c[i + h] + v0 * s[i + h];
    }
}

int test_forward_pass() {
    printf("\n=== Testing Full Forward Pass with Verilator ===\n");
    printf("WARNING: This test requires ~256MB RAM and may take several minutes\n");
    fflush(stdout);

    SmolSimulator sim(false);

    // Hold reset while setting up memory
    sim.hold_reset();
    sim.tick();

    // Place HALT trap at address 0
    sim.write_mem(0x0000, 0xE0000000);

    // Load all kernel binaries
    printf("Loading kernels...\n");
    if (sim.load_kernel("../kernels/matmul_q8.bin", 0x1000) != 0 ||
        sim.load_kernel("../kernels/rmsnorm.bin", 0x2000) != 0 ||
        sim.load_kernel("../kernels/rope.bin", 0x3000) != 0 ||
        sim.load_kernel("../kernels/attention.bin", 0x4000) != 0 ||
        sim.load_kernel("../kernels/silu_mul.bin", 0x5000) != 0 ||
        sim.load_kernel("../kernels/residual.bin", 0x6000) != 0 ||
        sim.load_kernel("../kernels/embed.bin", 0x7000) != 0 ||
        sim.load_kernel("../kernels/memcpy.bin", 0x8000) != 0 ||
        sim.load_kernel("../kernels/forward.bin", 0x9000) != 0) {
        printf("SKIPPED - kernel binaries not found (run 'make kernels' in processor/)\n");
        return 0;
    }

    // Load model binary
    printf("Loading model...\n");
    const char* model_path = "../../models/smollm2-135m-q8.bin";
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        // Try alternate path
        model_path = "../../../models/smollm2-135m-q8.bin";
        f = fopen(model_path, "rb");
    }
    if (!f) {
        printf("SKIPPED - model file not found\n");
        return 0;
    }

    // Read header
    char magic[5] = {0};
    fread(magic, 4, 1, f);
    if (strcmp(magic, "SMOL") != 0) {
        printf("FAILED - invalid model magic\n");
        fclose(f);
        return 1;
    }

    uint32_t version;
    fread(&version, 4, 1, f);
    if (version == 2) {
        uint32_t skip[2];
        fread(skip, 4, 2, f); // quant_type, group_size
    }

    ModelConfig cfg;
    fread(&cfg.hidden_size, 4, 1, f);
    fread(&cfg.intermediate_size, 4, 1, f);
    fread(&cfg.num_layers, 4, 1, f);
    fread(&cfg.num_heads, 4, 1, f);
    fread(&cfg.num_kv_heads, 4, 1, f);
    fread(&cfg.vocab_size, 4, 1, f);
    fread(&cfg.max_seq_len, 4, 1, f);
    fread(&cfg.rope_theta, 4, 1, f);
    fread(&cfg.rms_norm_eps, 4, 1, f);
    cfg.head_dim = cfg.hidden_size / cfg.num_heads;

    printf("Config: hidden=%d, layers=%d, heads=%d, vocab=%d\n",
           cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.vocab_size);

    // Skip tokenizer
    uint32_t nv, nm;
    fread(&nv, 4, 1, f);
    fread(&nm, 4, 1, f);
    for (uint32_t i = 0; i < nv; i++) {
        uint32_t len;
        fread(&len, 4, 1, f);
        fseek(f, len, SEEK_CUR);
    }
    for (uint32_t i = 0; i < nm; i++) {
        uint32_t len;
        fread(&len, 4, 1, f);
        fseek(f, len, SEEK_CUR);
    }

    int hs = cfg.hidden_size, hd = cfg.head_dim;
    int nh = cfg.num_heads, nkv = cfg.num_kv_heads;

    // Load embedding
    float embed_scale;
    fread(&embed_scale, 4, 1, f);
    size_t embed_sz = (size_t)cfg.vocab_size * hs;
    int8_t* embed_data = (int8_t*)malloc(embed_sz);
    fread(embed_data, 1, embed_sz, f);

    uint32_t wptr = WEIGHT_BASE;
    uint32_t embed_data_addr = wptr;

    // Load weights to Verilator memory
    printf("Loading weights to simulator memory...\n");
    for (size_t i = 0; i < embed_sz; i += 4) {
        uint32_t word = 0;
        for (int j = 0; j < 4 && (i + j) < embed_sz; j++) {
            word |= ((uint32_t)(uint8_t)embed_data[i + j]) << (j * 8);
        }
        sim.write_mem(wptr + i, word);
    }
    wptr += embed_sz;
    wptr = (wptr + 3) & ~3u;

    uint32_t embed_scale_addr = wptr;
    sim.write_mem(wptr, float_to_bits(embed_scale));
    wptr += 4;

    // Allocate layer addresses
    LayerAddrs* layer_addrs = (LayerAddrs*)calloc(cfg.num_layers, sizeof(LayerAddrs));

    // Host-side data for reference computation
    struct LayerData {
        float* input_ln;
        int8_t* q_data; float q_scale;
        int8_t* k_data; float k_scale;
        int8_t* v_data; float v_scale;
        int8_t* o_data; float o_scale;
        float* post_ln;
        int8_t* gate_data; float gate_scale;
        int8_t* up_data; float up_scale;
        int8_t* down_data; float down_scale;
    };
    LayerData* layers = (LayerData*)calloc(cfg.num_layers, sizeof(LayerData));

    printf("Loading %d layers...\n", cfg.num_layers);
    for (int l = 0; l < cfg.num_layers; l++) {
        LayerAddrs* la = &layer_addrs[l];

        // input_layernorm
        layers[l].input_ln = (float*)malloc(hs * sizeof(float));
        fread(layers[l].input_ln, sizeof(float), hs, f);
        la->input_ln = wptr;
        for (int i = 0; i < hs; i++) {
            sim.write_mem(wptr + i * 4, float_to_bits(layers[l].input_ln[i]));
        }
        wptr += hs * sizeof(float);

        // q_proj
        int q_rows = nh * hd, q_cols = hs;
        fread(&layers[l].q_scale, 4, 1, f);
        layers[l].q_data = (int8_t*)malloc(q_rows * q_cols);
        fread(layers[l].q_data, 1, q_rows * q_cols, f);
        la->q_scale_addr = wptr;
        sim.write_mem(wptr, float_to_bits(layers[l].q_scale));
        wptr += 4;
        la->q_data = wptr;
        for (int i = 0; i < q_rows * q_cols; i += 4) {
            uint32_t word = 0;
            for (int j = 0; j < 4 && (i + j) < q_rows * q_cols; j++) {
                word |= ((uint32_t)(uint8_t)layers[l].q_data[i + j]) << (j * 8);
            }
            sim.write_mem(wptr + i, word);
        }
        wptr += q_rows * q_cols;
        wptr = (wptr + 3) & ~3u;

        // k_proj
        int k_rows = nkv * hd, k_cols = hs;
        fread(&layers[l].k_scale, 4, 1, f);
        layers[l].k_data = (int8_t*)malloc(k_rows * k_cols);
        fread(layers[l].k_data, 1, k_rows * k_cols, f);
        la->k_scale_addr = wptr;
        sim.write_mem(wptr, float_to_bits(layers[l].k_scale));
        wptr += 4;
        la->k_data = wptr;
        for (int i = 0; i < k_rows * k_cols; i += 4) {
            uint32_t word = 0;
            for (int j = 0; j < 4 && (i + j) < k_rows * k_cols; j++) {
                word |= ((uint32_t)(uint8_t)layers[l].k_data[i + j]) << (j * 8);
            }
            sim.write_mem(wptr + i, word);
        }
        wptr += k_rows * k_cols;
        wptr = (wptr + 3) & ~3u;

        // v_proj
        fread(&layers[l].v_scale, 4, 1, f);
        layers[l].v_data = (int8_t*)malloc(k_rows * k_cols);
        fread(layers[l].v_data, 1, k_rows * k_cols, f);
        la->v_scale_addr = wptr;
        sim.write_mem(wptr, float_to_bits(layers[l].v_scale));
        wptr += 4;
        la->v_data = wptr;
        for (int i = 0; i < k_rows * k_cols; i += 4) {
            uint32_t word = 0;
            for (int j = 0; j < 4 && (i + j) < k_rows * k_cols; j++) {
                word |= ((uint32_t)(uint8_t)layers[l].v_data[i + j]) << (j * 8);
            }
            sim.write_mem(wptr + i, word);
        }
        wptr += k_rows * k_cols;
        wptr = (wptr + 3) & ~3u;

        // o_proj
        int o_rows = hs, o_cols = nh * hd;
        fread(&layers[l].o_scale, 4, 1, f);
        layers[l].o_data = (int8_t*)malloc(o_rows * o_cols);
        fread(layers[l].o_data, 1, o_rows * o_cols, f);
        la->o_scale_addr = wptr;
        sim.write_mem(wptr, float_to_bits(layers[l].o_scale));
        wptr += 4;
        la->o_data = wptr;
        for (int i = 0; i < o_rows * o_cols; i += 4) {
            uint32_t word = 0;
            for (int j = 0; j < 4 && (i + j) < o_rows * o_cols; j++) {
                word |= ((uint32_t)(uint8_t)layers[l].o_data[i + j]) << (j * 8);
            }
            sim.write_mem(wptr + i, word);
        }
        wptr += o_rows * o_cols;
        wptr = (wptr + 3) & ~3u;

        // post_attention_layernorm
        layers[l].post_ln = (float*)malloc(hs * sizeof(float));
        fread(layers[l].post_ln, sizeof(float), hs, f);
        la->post_ln = wptr;
        for (int i = 0; i < hs; i++) {
            sim.write_mem(wptr + i * 4, float_to_bits(layers[l].post_ln[i]));
        }
        wptr += hs * sizeof(float);

        // gate_proj
        int g_rows = cfg.intermediate_size, g_cols = hs;
        fread(&layers[l].gate_scale, 4, 1, f);
        layers[l].gate_data = (int8_t*)malloc(g_rows * g_cols);
        fread(layers[l].gate_data, 1, g_rows * g_cols, f);
        la->gate_scale_addr = wptr;
        sim.write_mem(wptr, float_to_bits(layers[l].gate_scale));
        wptr += 4;
        la->gate_data = wptr;
        for (int i = 0; i < g_rows * g_cols; i += 4) {
            uint32_t word = 0;
            for (int j = 0; j < 4 && (i + j) < g_rows * g_cols; j++) {
                word |= ((uint32_t)(uint8_t)layers[l].gate_data[i + j]) << (j * 8);
            }
            sim.write_mem(wptr + i, word);
        }
        wptr += g_rows * g_cols;
        wptr = (wptr + 3) & ~3u;

        // up_proj
        fread(&layers[l].up_scale, 4, 1, f);
        layers[l].up_data = (int8_t*)malloc(g_rows * g_cols);
        fread(layers[l].up_data, 1, g_rows * g_cols, f);
        la->up_scale_addr = wptr;
        sim.write_mem(wptr, float_to_bits(layers[l].up_scale));
        wptr += 4;
        la->up_data = wptr;
        for (int i = 0; i < g_rows * g_cols; i += 4) {
            uint32_t word = 0;
            for (int j = 0; j < 4 && (i + j) < g_rows * g_cols; j++) {
                word |= ((uint32_t)(uint8_t)layers[l].up_data[i + j]) << (j * 8);
            }
            sim.write_mem(wptr + i, word);
        }
        wptr += g_rows * g_cols;
        wptr = (wptr + 3) & ~3u;

        // down_proj
        int d_rows = hs, d_cols = cfg.intermediate_size;
        fread(&layers[l].down_scale, 4, 1, f);
        layers[l].down_data = (int8_t*)malloc(d_rows * d_cols);
        fread(layers[l].down_data, 1, d_rows * d_cols, f);
        la->down_scale_addr = wptr;
        sim.write_mem(wptr, float_to_bits(layers[l].down_scale));
        wptr += 4;
        la->down_data = wptr;
        for (int i = 0; i < d_rows * d_cols; i += 4) {
            uint32_t word = 0;
            for (int j = 0; j < 4 && (i + j) < d_rows * d_cols; j++) {
                word |= ((uint32_t)(uint8_t)layers[l].down_data[i + j]) << (j * 8);
            }
            sim.write_mem(wptr + i, word);
        }
        wptr += d_rows * d_cols;
        wptr = (wptr + 3) & ~3u;

        if (l == 0) printf("  Layer 0 loaded, wptr=0x%08X\n", wptr);
        if (l % 10 == 9) printf("  Layer %d loaded...\n", l + 1);
    }

    // final_norm
    float* final_norm = (float*)malloc(hs * sizeof(float));
    fread(final_norm, sizeof(float), hs, f);
    uint32_t final_norm_addr = wptr;
    for (int i = 0; i < hs; i++) {
        sim.write_mem(wptr + i * 4, float_to_bits(final_norm[i]));
    }
    wptr += hs * sizeof(float);

    fclose(f);
    printf("Total weight memory: %u bytes (%.1f MB)\n", wptr - WEIGHT_BASE, (wptr - WEIGHT_BASE) / 1048576.0f);

    // Precompute RoPE tables
    int max_seq = 64; // Cap for testing
    cfg.max_seq_len = max_seq;
    float* rope_cos = (float*)malloc(max_seq * hd * sizeof(float));
    float* rope_sin = (float*)malloc(max_seq * hd * sizeof(float));
    for (int p = 0; p < max_seq; p++) {
        for (int i = 0; i < hd / 2; i++) {
            float freq = 1.0f / powf(cfg.rope_theta, (float)(2 * i) / hd);
            float c = cosf(p * freq), s = sinf(p * freq);
            rope_cos[p * hd + i] = rope_cos[p * hd + hd/2 + i] = c;
            rope_sin[p * hd + i] = rope_sin[p * hd + hd/2 + i] = s;
        }
    }

    uint32_t rope_cos_addr = ROPE_BASE;
    uint32_t rope_sin_addr = ROPE_BASE + max_seq * hd * sizeof(float);
    for (int i = 0; i < max_seq * hd; i++) {
        sim.write_mem(rope_cos_addr + i * 4, float_to_bits(rope_cos[i]));
        sim.write_mem(rope_sin_addr + i * 4, float_to_bits(rope_sin[i]));
    }

    // Populate descriptor
    printf("Setting up model descriptor...\n");
    uint32_t d = DESC_BASE;
    sim.write_mem(d + 0x00, embed_data_addr);
    sim.write_mem(d + 0x04, embed_scale_addr);
    sim.write_mem(d + 0x08, final_norm_addr);
    sim.write_mem(d + 0x0C, KV_BASE);
    sim.write_mem(d + 0x10, cfg.hidden_size);
    sim.write_mem(d + 0x14, cfg.head_dim);
    sim.write_mem(d + 0x18, cfg.num_heads);
    sim.write_mem(d + 0x1C, cfg.num_kv_heads);
    sim.write_mem(d + 0x20, cfg.intermediate_size);
    sim.write_mem(d + 0x24, cfg.num_layers);
    sim.write_mem(d + 0x28, cfg.max_seq_len);
    sim.write_mem(d + 0x2C, float_to_bits(cfg.rms_norm_eps));
    sim.write_mem(d + 0x30, cfg.vocab_size);
    sim.write_mem(d + 0x34, ROPE_BASE);
    sim.write_mem(d + 0x38, cfg.max_seq_len * cfg.head_dim * 4);  // kv_head_stride
    sim.write_mem(d + 0x3C, 0);

    for (int l = 0; l < cfg.num_layers; l++) {
        uint32_t ld = d + 0x40 + l * 64;
        LayerAddrs* la = &layer_addrs[l];
        sim.write_mem(ld + 0x00, la->input_ln);
        sim.write_mem(ld + 0x04, la->q_data);
        sim.write_mem(ld + 0x08, la->q_scale_addr);
        sim.write_mem(ld + 0x0C, la->k_data);
        sim.write_mem(ld + 0x10, la->k_scale_addr);
        sim.write_mem(ld + 0x14, la->v_data);
        sim.write_mem(ld + 0x18, la->v_scale_addr);
        sim.write_mem(ld + 0x1C, la->o_data);
        sim.write_mem(ld + 0x20, la->o_scale_addr);
        sim.write_mem(ld + 0x24, la->post_ln);
        sim.write_mem(ld + 0x28, la->gate_data);
        sim.write_mem(ld + 0x2C, la->gate_scale_addr);
        sim.write_mem(ld + 0x30, la->up_data);
        sim.write_mem(ld + 0x34, la->up_scale_addr);
        sim.write_mem(ld + 0x38, la->down_data);
        sim.write_mem(ld + 0x3C, la->down_scale_addr);
    }

    // Run reference forward pass
    printf("\nComputing reference forward pass...\n");
    int token = 1, pos = 0;
    int ng = nh / nkv;

    static float x[576], xb[576], xb2[576], q[576], k[192], v[192];
    static float hb[1536], hb2[1536];
    static float att[9 * 64];
    static float* k_caches[30];
    static float* v_caches[30];

    for (int l = 0; l < cfg.num_layers; l++) {
        k_caches[l] = (float*)calloc(nkv * max_seq * hd, sizeof(float));
        v_caches[l] = (float*)calloc(nkv * max_seq * hd, sizeof(float));
    }

    // Embedding
    int8_t* emb = embed_data + token * hs;
    for (int i = 0; i < hs; i++) x[i] = emb[i] * embed_scale;

    // Debug: print first few reference embedding values
    printf("\nReference embedding (first 8 values):\n");
    printf("  Input INT8: ");
    for (int i = 0; i < 8; i++) printf("%d ", emb[i]);
    printf("\n  Scale: %.6f\n", embed_scale);
    printf("  Output: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", x[i]);
    printf("\n");

    // Save original embedding for checkpoint comparison (x[] gets modified during layers)
    float x_embed_saved[8];
    for (int i = 0; i < 8; i++) x_embed_saved[i] = x[i];

    // Save x after each layer for checkpoint comparison
    float x_per_layer[30][8];

    float* rc = rope_cos + pos * hd;
    float* rs = rope_sin + pos * hd;

    for (int l = 0; l < cfg.num_layers; l++) {
        ref_rmsnorm(xb, x, layers[l].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q, layers[l].q_data, layers[l].q_scale, xb, nh * hd, hs);
        fwd_matmul_q8(k, layers[l].k_data, layers[l].k_scale, xb, nkv * hd, hs);
        fwd_matmul_q8(v, layers[l].v_data, layers[l].v_scale, xb, nkv * hd, hs);

        for (int h = 0; h < nh; h++) fwd_apply_rope(q + h * hd, hd, rc, rs);
        for (int h = 0; h < nkv; h++) fwd_apply_rope(k + h * hd, hd, rc, rs);

        for (int h = 0; h < nkv; h++) {
            memcpy(k_caches[l] + h * max_seq * hd + pos * hd, k + h * hd, hd * sizeof(float));
            memcpy(v_caches[l] + h * max_seq * hd + pos * hd, v + h * hd, hd * sizeof(float));
        }
        int slen = pos + 1;

        memset(xb2, 0, hs * sizeof(float));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float* qh = q + h * hd;
            float* kc = k_caches[l] + kvh * max_seq * hd;
            float* vc = v_caches[l] + kvh * max_seq * hd;
            float sc = 1.0f / sqrtf((float)hd);

            for (int t = 0; t < slen; t++) {
                float score = 0;
                float* kt = kc + t * hd;
                for (int dd = 0; dd < hd; dd++) score += qh[dd] * kt[dd];
                att[h * max_seq + t] = score * sc;
            }
            fwd_softmax(att + h * max_seq, slen);

            float* oh = xb2 + h * hd;
            for (int t = 0; t < slen; t++) {
                float a = att[h * max_seq + t];
                float* vt = vc + t * hd;
                for (int dd = 0; dd < hd; dd++) oh[dd] += a * vt[dd];
            }
        }

        fwd_matmul_q8(xb, layers[l].o_data, layers[l].o_scale, xb2, hs, nh * hd);
        for (int i = 0; i < hs; i++) x[i] += xb[i];

        ref_rmsnorm(xb, x, layers[l].post_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(hb, layers[l].gate_data, layers[l].gate_scale, xb, cfg.intermediate_size, hs);
        fwd_matmul_q8(hb2, layers[l].up_data, layers[l].up_scale, xb, cfg.intermediate_size, hs);
        for (int i = 0; i < cfg.intermediate_size; i++) hb[i] = fwd_silu(hb[i]) * hb2[i];
        fwd_matmul_q8(xb, layers[l].down_data, layers[l].down_scale, hb, hs, cfg.intermediate_size);
        for (int i = 0; i < hs; i++) x[i] += xb[i];

        // Save x for layer checkpoint comparison
        for (int i = 0; i < 8; i++) x_per_layer[l][i] = x[i];

        if (l % 10 == 9) printf("  Reference layer %d done\n", l + 1);
    }

    ref_rmsnorm(x, x, final_norm, hs, cfg.rms_norm_eps);

    float* ref_logits = (float*)malloc(cfg.vocab_size * sizeof(float));
    for (int i = 0; i < cfg.vocab_size; i++) {
        float sum = 0;
        int8_t* row = embed_data + (int64_t)i * hs;
        for (int j = 0; j < hs; j++) sum += x[j] * row[j] * embed_scale;
        ref_logits[i] = sum;
    }

    // Run Verilog forward pass
    printf("\nRunning Verilog forward pass (token=%d, pos=%d)...\n", token, pos);
    printf("This may take a while (~19M instructions)...\n");

    sim.release_reset();
    sim.top->eval();

    // Set PC to forward kernel entry point (0x9000)
    sim.write_pc(FORWARD_ENTRY);

    // Set up registers for forward kernel call
    sim.write_reg(1, 0);          // RA = 0 (halt on return)
    sim.write_reg(2, 0x00FF0000); // SP = stack pointer (valid memory region)
    sim.write_reg(3, token);      // token
    sim.write_reg(4, pos);        // position
    sim.write_reg(5, DESC_BASE);  // descriptor base
    sim.top->eval();

    printf("Starting at PC=0x%08X\n", sim.read_pc());
    fflush(stdout);

    // Debug: Trace first 500 cycles to see RA changes
    printf("\n=== Debug trace (first 500 cycles) ===\n");
    uint32_t prev_ra = 0;
    for (int i = 0; i < 500 && !sim.top->halted; i++) {
        uint32_t cur_pc = sim.top->pc;
        uint32_t cur_ra = sim.read_reg(1);
        if (cur_ra != prev_ra || i < 20) {
            printf("  C%d: PC=0x%08X, RA=0x%08X, state=%d\n", i, cur_pc, cur_ra, sim.top->state);
            fflush(stdout);
            prev_ra = cur_ra;
        }
        sim.tick();
    }
    printf("=== End debug trace ===\n\n");
    fflush(stdout);

    // Checkpoint 1: Run until embed returns (RA goes back to ~0x9084 -> PC in forward.s)
    // Run until PC is in rmsnorm range (0x2XXX) which indicates embedding is done
    printf("Running until embedding completes...\n");
    uint64_t embed_cycles = 0;
    while (!sim.top->halted && sim.cycle_count < 100000000) {
        sim.tick();
        embed_cycles++;
        // Check if we're now in rmsnorm kernel (PC around 0x2XXX)
        uint32_t pc = sim.top->pc;
        if (pc >= 0x2000 && pc < 0x3000) {
            printf("Reached rmsnorm kernel at cycle %lu\n", sim.cycle_count);
            break;
        }
    }

    // Read BUF_X after embedding
    printf("\nVerilog embedding result (BUF_X at 0x100000, first 8 values):\n  ");
    float verilog_x[8];
    for (int i = 0; i < 8; i++) {
        verilog_x[i] = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
        printf("%.4f ", verilog_x[i]);
    }
    printf("\n");

    // Compare with reference (saved before layer processing)
    printf("Reference embedding (first 8 values, saved):\n  ");
    for (int i = 0; i < 8; i++) printf("%.4f ", x_embed_saved[i]);
    printf("\n");

    // Check for match
    float max_embed_diff = 0;
    for (int i = 0; i < 8; i++) {
        float diff = fabsf(verilog_x[i] - x_embed_saved[i]);
        if (diff > max_embed_diff) max_embed_diff = diff;
    }
    printf("Max embedding diff (first 8): %.6e\n", max_embed_diff);
    if (max_embed_diff > 0.01f) {
        printf("*** EMBEDDING MISMATCH - debugging needed ***\n");
    } else {
        printf("Embedding OK! Proceeding with layer processing...\n");
    }
    fflush(stdout);

    // Checkpoint 2: Run until Q projection completes
    // Q projection is the first matmul in layer 1, output goes to BUF_Q (0x103000)
    // We run until we exit the matmul kernel back to forward.s
    printf("\nRunning until Q projection completes...\n");
    uint32_t prev_pc = 0;
    uint32_t matmul_entries = 0;
    uint32_t matmul_exits = 0;
    bool in_matmul = false;

    while (!sim.top->halted && sim.cycle_count < 100000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;

        // Track matmul kernel entries/exits
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            matmul_entries++;
            printf("  Entered matmul #%d at cycle %lu\n", matmul_entries, sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            matmul_exits++;
            printf("  Exited matmul #%d at cycle %lu, PC=0x%08X\n", matmul_exits, sim.cycle_count, pc);
            if (matmul_exits >= 1) {
                printf("Q projection complete.\n");
                break;
            }
        }
        in_matmul = now_in_matmul;
        prev_pc = pc;

        if (sim.cycle_count % 10000000 == 0) {
            printf("  ...cycle %lu M, PC=0x%08X\n", sim.cycle_count / 1000000, pc);
        }
    }

    // Read BUF_Q after Q projection (0x103000)
    uint32_t buf_q = BUF_BASE + 0x3000;
    printf("\nVerilog BUF_Q (Q projection output, first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_q + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Also read BUF_XB after rmsnorm (0x101000) which is the input to Q projection
    uint32_t buf_xb = BUF_BASE + 0x1000;
    printf("Verilog BUF_XB (rmsnorm output, first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_xb + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference Q projection
    {
        float x_tmp[576], xb_tmp[576], q_tmp[576];
        int8_t* emb_tmp = embed_data + token * hs;
        for (int i = 0; i < hs; i++) x_tmp[i] = emb_tmp[i] * embed_scale;

        ref_rmsnorm(xb_tmp, x_tmp, layers[0].input_ln, hs, cfg.rms_norm_eps);

        printf("Reference BUF_XB (rmsnorm output, first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", xb_tmp[i]);
        printf("\n");

        fwd_matmul_q8(q_tmp, layers[0].q_data, layers[0].q_scale, xb_tmp, nh * hd, hs);

        printf("Reference BUF_Q (Q projection output, first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", q_tmp[i]);
        printf("\n");

        // Compare
        float max_xb_diff = 0, max_q_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vxb = bits_to_float(sim.read_mem(buf_xb + i * 4));
            float vq = bits_to_float(sim.read_mem(buf_q + i * 4));
            float xb_diff = fabsf(vxb - xb_tmp[i]);
            float q_diff = fabsf(vq - q_tmp[i]);
            if (xb_diff > max_xb_diff) max_xb_diff = xb_diff;
            if (q_diff > max_q_diff) max_q_diff = q_diff;
        }
        printf("Max diff - BUF_XB: %.6e, BUF_Q: %.6e\n", max_xb_diff, max_q_diff);
        if (max_xb_diff > 0.01f) printf("*** RMSNORM OUTPUT MISMATCH ***\n");
        if (max_q_diff > 0.01f) printf("*** Q PROJECTION MISMATCH ***\n");
    }
    fflush(stdout);

    // Checkpoint 3: Run through K, V projections and RoPE (until attention kernel)
    printf("\nRunning through K, V projections and RoPE...\n");
    int more_matmuls = 0;
    in_matmul = false;
    bool in_rope = false;
    bool rope_done = false;
    while (!sim.top->halted && sim.cycle_count < 50000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;

        // Track matmul #2 (K proj) and #3 (V proj)
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            more_matmuls++;
            printf("  Entered matmul #%d at cycle %lu\n", matmul_entries + more_matmuls, sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited matmul #%d at cycle %lu\n", matmul_entries + more_matmuls, sim.cycle_count);
        }
        in_matmul = now_in_matmul;

        // Track RoPE kernel (0x3000)
        bool now_in_rope = (pc >= 0x3000 && pc < 0x4000);
        if (now_in_rope && !in_rope) {
            printf("  Entered RoPE at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_rope && in_rope && !rope_done) {
            printf("  Exited RoPE at cycle %lu\n", sim.cycle_count);
            rope_done = true;
        }
        in_rope = now_in_rope;

        // Stop when we reach attention (0x4000)
        if (pc >= 0x4000 && pc < 0x5000) {
            printf("Reached attention kernel at cycle %lu\n", sim.cycle_count);
            break;
        }
    }

    // Check K projection output (BUF_K at 0x104000)
    uint32_t buf_k = BUF_BASE + 0x4000;
    printf("\nVerilog BUF_K (K projection output, first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_k + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Check V projection output (BUF_V at 0x105000)
    uint32_t buf_v = BUF_BASE + 0x5000;
    printf("Verilog BUF_V (V projection output, first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_v + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Check Q after RoPE
    printf("Verilog BUF_Q after RoPE (first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_q + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference K, V, and Q after RoPE
    {
        float x_tmp[576], xb_tmp[576], q_tmp[576], k_tmp[192], v_tmp[192];
        int8_t* emb_tmp = embed_data + token * hs;
        for (int i = 0; i < hs; i++) x_tmp[i] = emb_tmp[i] * embed_scale;

        ref_rmsnorm(xb_tmp, x_tmp, layers[0].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q_tmp, layers[0].q_data, layers[0].q_scale, xb_tmp, nh * hd, hs);
        fwd_matmul_q8(k_tmp, layers[0].k_data, layers[0].k_scale, xb_tmp, nkv * hd, hs);
        fwd_matmul_q8(v_tmp, layers[0].v_data, layers[0].v_scale, xb_tmp, nkv * hd, hs);

        printf("Reference BUF_K (first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", k_tmp[i]);
        printf("\n");

        printf("Reference BUF_V (first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", v_tmp[i]);
        printf("\n");

        // Apply RoPE
        float* rc_tmp = rope_cos;  // pos=0
        float* rs_tmp = rope_sin;
        for (int h = 0; h < nh; h++) fwd_apply_rope(q_tmp + h * hd, hd, rc_tmp, rs_tmp);
        for (int h = 0; h < nkv; h++) fwd_apply_rope(k_tmp + h * hd, hd, rc_tmp, rs_tmp);

        printf("Reference BUF_Q after RoPE (first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", q_tmp[i]);
        printf("\n");

        // Compare
        float max_k_diff = 0, max_v_diff = 0, max_q_rope_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vk = bits_to_float(sim.read_mem(buf_k + i * 4));
            float vv = bits_to_float(sim.read_mem(buf_v + i * 4));
            float vq = bits_to_float(sim.read_mem(buf_q + i * 4));
            if (fabsf(vk - k_tmp[i]) > max_k_diff) max_k_diff = fabsf(vk - k_tmp[i]);
            if (fabsf(vv - v_tmp[i]) > max_v_diff) max_v_diff = fabsf(vv - v_tmp[i]);
            if (fabsf(vq - q_tmp[i]) > max_q_rope_diff) max_q_rope_diff = fabsf(vq - q_tmp[i]);
        }
        printf("Max diff - K: %.6e, V: %.6e, Q(rope): %.6e\n", max_k_diff, max_v_diff, max_q_rope_diff);
        if (max_k_diff > 0.01f) printf("*** K PROJECTION MISMATCH ***\n");
        if (max_v_diff > 0.01f) printf("*** V PROJECTION MISMATCH ***\n");
        if (max_q_rope_diff > 0.01f) printf("*** Q ROPE MISMATCH ***\n");
    }
    fflush(stdout);

    // Checkpoint 4: Run through attention and check output
    printf("\nRunning attention kernel...\n");
    bool in_attn = true;  // We just entered attention
    while (!sim.top->halted && sim.cycle_count < 100000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;

        bool now_in_attn = (pc >= 0x4000 && pc < 0x5000);
        if (!now_in_attn && in_attn) {
            printf("  Exited attention at cycle %lu, PC=0x%08X\n", sim.cycle_count, pc);
            break;
        }
        in_attn = now_in_attn;
    }

    // Check attention output (BUF_XB2 at 0x102000)
    uint32_t buf_xb2 = BUF_BASE + 0x2000;
    printf("\nVerilog BUF_XB2 (attention output, first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_xb2 + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference attention output
    // For pos=0 with slen=1, softmax gives 1.0 weight on single position
    // So output = V values
    {
        float x_tmp[576], xb_tmp[576], q_tmp[576], k_tmp[192], v_tmp[192];
        int8_t* emb_tmp = embed_data + token * hs;
        for (int i = 0; i < hs; i++) x_tmp[i] = emb_tmp[i] * embed_scale;

        ref_rmsnorm(xb_tmp, x_tmp, layers[0].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q_tmp, layers[0].q_data, layers[0].q_scale, xb_tmp, nh * hd, hs);
        fwd_matmul_q8(k_tmp, layers[0].k_data, layers[0].k_scale, xb_tmp, nkv * hd, hs);
        fwd_matmul_q8(v_tmp, layers[0].v_data, layers[0].v_scale, xb_tmp, nkv * hd, hs);

        // Apply RoPE
        float* rc_tmp = rope_cos;
        float* rs_tmp = rope_sin;
        for (int h = 0; h < nh; h++) fwd_apply_rope(q_tmp + h * hd, hd, rc_tmp, rs_tmp);
        for (int h = 0; h < nkv; h++) fwd_apply_rope(k_tmp + h * hd, hd, rc_tmp, rs_tmp);

        // For slen=1, attention output = V scaled by attention weights (which are all 1.0 after softmax)
        float xb2_ref[576];
        memset(xb2_ref, 0, sizeof(xb2_ref));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;  // KV head for this query head
            float* vh = v_tmp + kvh * hd;
            float* oh = xb2_ref + h * hd;
            for (int d = 0; d < hd; d++) oh[d] = vh[d];  // att weight = 1.0
        }

        printf("Reference BUF_XB2 (attention output, first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", xb2_ref[i]);
        printf("\n");

        float max_attn_diff = 0;
        for (int i = 0; i < 8; i++) {
            float va = bits_to_float(sim.read_mem(buf_xb2 + i * 4));
            float diff = fabsf(va - xb2_ref[i]);
            if (diff > max_attn_diff) max_attn_diff = diff;
        }
        printf("Max diff - Attention output: %.6e\n", max_attn_diff);
        if (max_attn_diff > 0.01f) printf("*** ATTENTION OUTPUT MISMATCH ***\n");
    }
    fflush(stdout);

    // Checkpoint 5: Run through O projection only and check output
    printf("\nRunning O projection (matmul #4)...\n");
    in_matmul = false;
    while (!sim.top->halted && sim.cycle_count < 100000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            printf("  Entered O projection at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited O projection at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_matmul = now_in_matmul;
    }

    // Check BUF_XB after O projection (before residual)
    printf("\nVerilog BUF_XB (O projection output, first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_xb + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference O projection
    {
        float x_ref[576], xb_ref[576], xb2_ref[576], q_ref[576], k_ref[192], v_ref[192];
        int8_t* emb_tmp = embed_data + token * hs;
        for (int i = 0; i < hs; i++) x_ref[i] = emb_tmp[i] * embed_scale;

        ref_rmsnorm(xb_ref, x_ref, layers[0].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q_ref, layers[0].q_data, layers[0].q_scale, xb_ref, nh * hd, hs);
        fwd_matmul_q8(k_ref, layers[0].k_data, layers[0].k_scale, xb_ref, nkv * hd, hs);
        fwd_matmul_q8(v_ref, layers[0].v_data, layers[0].v_scale, xb_ref, nkv * hd, hs);

        // Attention output (slen=1, so output = V)
        memset(xb2_ref, 0, sizeof(xb2_ref));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float* vh = v_ref + kvh * hd;
            float* oh = xb2_ref + h * hd;
            for (int d = 0; d < hd; d++) oh[d] = vh[d];
        }

        // O projection
        fwd_matmul_q8(xb_ref, layers[0].o_data, layers[0].o_scale, xb2_ref, hs, nh * hd);

        printf("Reference BUF_XB (O projection output, first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", xb_ref[i]);
        printf("\n");

        float max_o_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vo = bits_to_float(sim.read_mem(buf_xb + i * 4));
            float diff = fabsf(vo - xb_ref[i]);
            if (diff > max_o_diff) max_o_diff = diff;
        }
        printf("Max diff - O projection: %.6e\n", max_o_diff);
        if (max_o_diff > 0.01f) printf("*** O PROJECTION MISMATCH ***\n");
    }

    // Checkpoint 6: Run through first residual and check BUF_X
    printf("\nRunning first residual...\n");
    bool in_residual = false;
    while (!sim.top->halted && sim.cycle_count < 100000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_residual = (pc >= 0x6000 && pc < 0x7000);
        if (now_in_residual && !in_residual) {
            printf("  Entered residual at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_residual && in_residual) {
            printf("  Exited residual at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_residual = now_in_residual;
    }

    // Check BUF_X after first residual
    printf("\nVerilog BUF_X after first residual (first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference x after first residual
    {
        float x_ref[576], xb_ref[576], xb2_ref[576], q_ref[576], k_ref[192], v_ref[192];
        int8_t* emb_tmp = embed_data + token * hs;
        for (int i = 0; i < hs; i++) x_ref[i] = emb_tmp[i] * embed_scale;

        ref_rmsnorm(xb_ref, x_ref, layers[0].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q_ref, layers[0].q_data, layers[0].q_scale, xb_ref, nh * hd, hs);
        fwd_matmul_q8(k_ref, layers[0].k_data, layers[0].k_scale, xb_ref, nkv * hd, hs);
        fwd_matmul_q8(v_ref, layers[0].v_data, layers[0].v_scale, xb_ref, nkv * hd, hs);

        // Attention output
        memset(xb2_ref, 0, sizeof(xb2_ref));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float* vh = v_ref + kvh * hd;
            float* oh = xb2_ref + h * hd;
            for (int d = 0; d < hd; d++) oh[d] = vh[d];
        }

        // O projection
        fwd_matmul_q8(xb_ref, layers[0].o_data, layers[0].o_scale, xb2_ref, hs, nh * hd);

        // First residual: x = x + xb
        for (int i = 0; i < hs; i++) x_ref[i] += xb_ref[i];

        printf("Reference BUF_X after first residual (first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", x_ref[i]);
        printf("\n");

        float max_res_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vx = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
            float diff = fabsf(vx - x_ref[i]);
            if (diff > max_res_diff) max_res_diff = diff;
        }
        printf("Max diff - After first residual: %.6e\n", max_res_diff);
        if (max_res_diff > 0.01f) printf("*** FIRST RESIDUAL MISMATCH ***\n");
    }

    // Checkpoint 7: Run through post-attention rmsnorm and check BUF_XB
    printf("\nRunning post-attention rmsnorm...\n");
    bool in_rmsnorm2 = false;
    while (!sim.top->halted && sim.cycle_count < 100000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_rmsnorm = (pc >= 0x2000 && pc < 0x3000);
        if (now_in_rmsnorm && !in_rmsnorm2) {
            printf("  Entered rmsnorm at cycle %lu\n", sim.cycle_count);
            in_rmsnorm2 = true;
        }
        if (!now_in_rmsnorm && in_rmsnorm2) {
            printf("  Exited rmsnorm at cycle %lu\n", sim.cycle_count);
            break;
        }
    }

    // Check BUF_XB after post-attention rmsnorm
    printf("\nVerilog BUF_XB after post-attn rmsnorm (first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_xb + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference for post-attention rmsnorm
    {
        float x_ref[576], xb_ref[576], xb2_ref[576], q_ref[576], k_ref[192], v_ref[192];
        int8_t* emb_tmp = embed_data + token * hs;
        for (int i = 0; i < hs; i++) x_ref[i] = emb_tmp[i] * embed_scale;

        ref_rmsnorm(xb_ref, x_ref, layers[0].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q_ref, layers[0].q_data, layers[0].q_scale, xb_ref, nh * hd, hs);
        fwd_matmul_q8(k_ref, layers[0].k_data, layers[0].k_scale, xb_ref, nkv * hd, hs);
        fwd_matmul_q8(v_ref, layers[0].v_data, layers[0].v_scale, xb_ref, nkv * hd, hs);

        // Attention output
        memset(xb2_ref, 0, sizeof(xb2_ref));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float* vh = v_ref + kvh * hd;
            float* oh = xb2_ref + h * hd;
            for (int d = 0; d < hd; d++) oh[d] = vh[d];
        }

        // O projection and first residual
        fwd_matmul_q8(xb_ref, layers[0].o_data, layers[0].o_scale, xb2_ref, hs, nh * hd);
        for (int i = 0; i < hs; i++) x_ref[i] += xb_ref[i];

        // Post-attention rmsnorm
        ref_rmsnorm(xb_ref, x_ref, layers[0].post_ln, hs, cfg.rms_norm_eps);

        printf("Reference BUF_XB after post-attn rmsnorm (first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", xb_ref[i]);
        printf("\n");

        float max_postnorm_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vxb = bits_to_float(sim.read_mem(buf_xb + i * 4));
            float diff = fabsf(vxb - xb_ref[i]);
            if (diff > max_postnorm_diff) max_postnorm_diff = diff;
        }
        printf("Max diff - Post-attn rmsnorm: %.6e\n", max_postnorm_diff);
        if (max_postnorm_diff > 0.01f) printf("*** POST-ATTN RMSNORM MISMATCH ***\n");
    }
    fflush(stdout);

    // Checkpoint 8: Run through gate projection (matmul #5)
    printf("\nRunning gate projection (matmul #5)...\n");
    in_matmul = false;
    while (!sim.top->halted && sim.cycle_count < 200000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            printf("  Entered gate matmul at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited gate matmul at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_matmul = now_in_matmul;
    }

    // Gate output is in BUF_HB at 0x150000 (NOT BUF_BASE + 0x3000 which is BUF_Q!)
    uint32_t buf_hb = 0x150000;
    printf("\nVerilog BUF_HB (gate output, first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_hb + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference gate projection
    {
        float x_ref[576], xb_ref[576], xb2_ref[576], q_ref[576], k_ref[192], v_ref[192];
        float hb_ref[1536];
        int8_t* emb_tmp = embed_data + token * hs;
        for (int i = 0; i < hs; i++) x_ref[i] = emb_tmp[i] * embed_scale;

        ref_rmsnorm(xb_ref, x_ref, layers[0].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q_ref, layers[0].q_data, layers[0].q_scale, xb_ref, nh * hd, hs);
        fwd_matmul_q8(k_ref, layers[0].k_data, layers[0].k_scale, xb_ref, nkv * hd, hs);
        fwd_matmul_q8(v_ref, layers[0].v_data, layers[0].v_scale, xb_ref, nkv * hd, hs);

        // Attention output
        memset(xb2_ref, 0, sizeof(xb2_ref));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float* vh = v_ref + kvh * hd;
            float* oh = xb2_ref + h * hd;
            for (int d = 0; d < hd; d++) oh[d] = vh[d];
        }

        // O projection and first residual
        fwd_matmul_q8(xb_ref, layers[0].o_data, layers[0].o_scale, xb2_ref, hs, nh * hd);
        for (int i = 0; i < hs; i++) x_ref[i] += xb_ref[i];

        // Post-attention rmsnorm
        ref_rmsnorm(xb_ref, x_ref, layers[0].post_ln, hs, cfg.rms_norm_eps);

        // Gate projection
        fwd_matmul_q8(hb_ref, layers[0].gate_data, layers[0].gate_scale, xb_ref, cfg.intermediate_size, hs);

        printf("Reference BUF_HB (gate output, first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", hb_ref[i]);
        printf("\n");

        float max_gate_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vhb = bits_to_float(sim.read_mem(buf_hb + i * 4));
            float diff = fabsf(vhb - hb_ref[i]);
            if (diff > max_gate_diff) max_gate_diff = diff;
        }
        printf("Max diff - Gate projection: %.6e\n", max_gate_diff);
        if (max_gate_diff > 0.01f) printf("*** GATE PROJECTION MISMATCH ***\n");
    }
    fflush(stdout);

    // Checkpoint 9: Run through up projection (matmul #6)
    printf("\nRunning up projection (matmul #6)...\n");
    in_matmul = false;
    while (!sim.top->halted && sim.cycle_count < 300000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            printf("  Entered up matmul at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited up matmul at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_matmul = now_in_matmul;
    }

    // Up output is in BUF_HB2 at 0x152000 (NOT BUF_BASE + 0x5000 which is BUF_V!)
    uint32_t buf_hb2 = 0x152000;
    printf("\nVerilog BUF_HB2 (up output, first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_hb2 + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference up projection
    {
        float x_ref[576], xb_ref[576], xb2_ref[576], q_ref[576], k_ref[192], v_ref[192];
        float hb_ref[1536], hb2_ref[1536];
        int8_t* emb_tmp = embed_data + token * hs;
        for (int i = 0; i < hs; i++) x_ref[i] = emb_tmp[i] * embed_scale;

        ref_rmsnorm(xb_ref, x_ref, layers[0].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q_ref, layers[0].q_data, layers[0].q_scale, xb_ref, nh * hd, hs);
        fwd_matmul_q8(k_ref, layers[0].k_data, layers[0].k_scale, xb_ref, nkv * hd, hs);
        fwd_matmul_q8(v_ref, layers[0].v_data, layers[0].v_scale, xb_ref, nkv * hd, hs);

        // Attention output
        memset(xb2_ref, 0, sizeof(xb2_ref));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float* vh = v_ref + kvh * hd;
            float* oh = xb2_ref + h * hd;
            for (int d = 0; d < hd; d++) oh[d] = vh[d];
        }

        // O projection and first residual
        fwd_matmul_q8(xb_ref, layers[0].o_data, layers[0].o_scale, xb2_ref, hs, nh * hd);
        for (int i = 0; i < hs; i++) x_ref[i] += xb_ref[i];

        // Post-attention rmsnorm
        ref_rmsnorm(xb_ref, x_ref, layers[0].post_ln, hs, cfg.rms_norm_eps);

        // Gate and up projections
        fwd_matmul_q8(hb_ref, layers[0].gate_data, layers[0].gate_scale, xb_ref, cfg.intermediate_size, hs);
        fwd_matmul_q8(hb2_ref, layers[0].up_data, layers[0].up_scale, xb_ref, cfg.intermediate_size, hs);

        printf("Reference BUF_HB2 (up output, first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", hb2_ref[i]);
        printf("\n");

        float max_up_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vhb2 = bits_to_float(sim.read_mem(buf_hb2 + i * 4));
            float diff = fabsf(vhb2 - hb2_ref[i]);
            if (diff > max_up_diff) max_up_diff = diff;
        }
        printf("Max diff - Up projection: %.6e\n", max_up_diff);
        if (max_up_diff > 0.01f) printf("*** UP PROJECTION MISMATCH ***\n");
    }
    fflush(stdout);

    // Checkpoint 10: Run through silu_mul
    printf("\nRunning silu_mul...\n");
    bool in_silu = false;
    while (!sim.top->halted && sim.cycle_count < 400000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_silu = (pc >= 0x5000 && pc < 0x6000);
        if (now_in_silu && !in_silu) {
            printf("  Entered silu_mul at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_silu && in_silu) {
            printf("  Exited silu_mul at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_silu = now_in_silu;
    }

    // silu_mul output overwrites BUF_HB
    printf("\nVerilog BUF_HB (after silu_mul, first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_hb + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference silu_mul
    {
        float x_ref[576], xb_ref[576], xb2_ref[576], q_ref[576], k_ref[192], v_ref[192];
        float hb_ref[1536], hb2_ref[1536];
        int8_t* emb_tmp = embed_data + token * hs;
        for (int i = 0; i < hs; i++) x_ref[i] = emb_tmp[i] * embed_scale;

        ref_rmsnorm(xb_ref, x_ref, layers[0].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q_ref, layers[0].q_data, layers[0].q_scale, xb_ref, nh * hd, hs);
        fwd_matmul_q8(k_ref, layers[0].k_data, layers[0].k_scale, xb_ref, nkv * hd, hs);
        fwd_matmul_q8(v_ref, layers[0].v_data, layers[0].v_scale, xb_ref, nkv * hd, hs);

        // Attention output
        memset(xb2_ref, 0, sizeof(xb2_ref));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float* vh = v_ref + kvh * hd;
            float* oh = xb2_ref + h * hd;
            for (int d = 0; d < hd; d++) oh[d] = vh[d];
        }

        // O projection and first residual
        fwd_matmul_q8(xb_ref, layers[0].o_data, layers[0].o_scale, xb2_ref, hs, nh * hd);
        for (int i = 0; i < hs; i++) x_ref[i] += xb_ref[i];

        // Post-attention rmsnorm
        ref_rmsnorm(xb_ref, x_ref, layers[0].post_ln, hs, cfg.rms_norm_eps);

        // Gate and up projections
        fwd_matmul_q8(hb_ref, layers[0].gate_data, layers[0].gate_scale, xb_ref, cfg.intermediate_size, hs);
        fwd_matmul_q8(hb2_ref, layers[0].up_data, layers[0].up_scale, xb_ref, cfg.intermediate_size, hs);

        // silu_mul: hb = silu(hb) * hb2
        for (int i = 0; i < cfg.intermediate_size; i++) {
            hb_ref[i] = fwd_silu(hb_ref[i]) * hb2_ref[i];
        }

        printf("Reference BUF_HB (after silu_mul, first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", hb_ref[i]);
        printf("\n");

        float max_silu_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vhb = bits_to_float(sim.read_mem(buf_hb + i * 4));
            float diff = fabsf(vhb - hb_ref[i]);
            if (diff > max_silu_diff) max_silu_diff = diff;
        }
        printf("Max diff - silu_mul: %.6e\n", max_silu_diff);
        if (max_silu_diff > 0.01f) printf("*** SILU_MUL MISMATCH ***\n");
    }
    fflush(stdout);

    // Checkpoint 11: Run through down projection (matmul #7)
    printf("\nRunning down projection (matmul #7)...\n");
    in_matmul = false;
    while (!sim.top->halted && sim.cycle_count < 500000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            printf("  Entered down matmul at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited down matmul at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_matmul = now_in_matmul;
    }

    // Down output is in BUF_XB at 0x101000
    printf("\nVerilog BUF_XB (down projection output, first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_xb + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference down projection
    {
        float x_ref[576], xb_ref[576], xb2_ref[576], q_ref[576], k_ref[192], v_ref[192];
        float hb_ref[1536], hb2_ref[1536];
        int8_t* emb_tmp = embed_data + token * hs;
        for (int i = 0; i < hs; i++) x_ref[i] = emb_tmp[i] * embed_scale;

        ref_rmsnorm(xb_ref, x_ref, layers[0].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q_ref, layers[0].q_data, layers[0].q_scale, xb_ref, nh * hd, hs);
        fwd_matmul_q8(k_ref, layers[0].k_data, layers[0].k_scale, xb_ref, nkv * hd, hs);
        fwd_matmul_q8(v_ref, layers[0].v_data, layers[0].v_scale, xb_ref, nkv * hd, hs);

        // Attention output
        memset(xb2_ref, 0, sizeof(xb2_ref));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float* vh = v_ref + kvh * hd;
            float* oh = xb2_ref + h * hd;
            for (int d = 0; d < hd; d++) oh[d] = vh[d];
        }

        // O projection and first residual
        fwd_matmul_q8(xb_ref, layers[0].o_data, layers[0].o_scale, xb2_ref, hs, nh * hd);
        for (int i = 0; i < hs; i++) x_ref[i] += xb_ref[i];

        // Post-attention rmsnorm
        ref_rmsnorm(xb_ref, x_ref, layers[0].post_ln, hs, cfg.rms_norm_eps);

        // Gate and up projections
        fwd_matmul_q8(hb_ref, layers[0].gate_data, layers[0].gate_scale, xb_ref, cfg.intermediate_size, hs);
        fwd_matmul_q8(hb2_ref, layers[0].up_data, layers[0].up_scale, xb_ref, cfg.intermediate_size, hs);

        // silu_mul
        for (int i = 0; i < cfg.intermediate_size; i++) {
            hb_ref[i] = fwd_silu(hb_ref[i]) * hb2_ref[i];
        }

        // Down projection
        fwd_matmul_q8(xb_ref, layers[0].down_data, layers[0].down_scale, hb_ref, hs, cfg.intermediate_size);

        printf("Reference BUF_XB (down projection output, first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", xb_ref[i]);
        printf("\n");

        float max_down_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vxb = bits_to_float(sim.read_mem(buf_xb + i * 4));
            float diff = fabsf(vxb - xb_ref[i]);
            if (diff > max_down_diff) max_down_diff = diff;
        }
        printf("Max diff - Down projection: %.6e\n", max_down_diff);
        if (max_down_diff > 0.01f) printf("*** DOWN PROJECTION MISMATCH ***\n");
    }
    fflush(stdout);

    // Checkpoint 12: Run through second residual
    printf("\nRunning second residual...\n");
    in_residual = false;
    while (!sim.top->halted && sim.cycle_count < 600000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_residual = (pc >= 0x6000 && pc < 0x7000);
        if (now_in_residual && !in_residual) {
            printf("  Entered residual at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_residual && in_residual) {
            printf("  Exited residual at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_residual = now_in_residual;
    }

    // Check BUF_X after second residual (this is layer 0 complete output)
    printf("\nVerilog BUF_X after second residual (layer 0 complete, first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference layer 0 complete output
    {
        float x_ref[576], xb_ref[576], xb2_ref[576], q_ref[576], k_ref[192], v_ref[192];
        float hb_ref[1536], hb2_ref[1536];
        int8_t* emb_tmp = embed_data + token * hs;
        for (int i = 0; i < hs; i++) x_ref[i] = emb_tmp[i] * embed_scale;

        ref_rmsnorm(xb_ref, x_ref, layers[0].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q_ref, layers[0].q_data, layers[0].q_scale, xb_ref, nh * hd, hs);
        fwd_matmul_q8(k_ref, layers[0].k_data, layers[0].k_scale, xb_ref, nkv * hd, hs);
        fwd_matmul_q8(v_ref, layers[0].v_data, layers[0].v_scale, xb_ref, nkv * hd, hs);

        // Attention output
        memset(xb2_ref, 0, sizeof(xb2_ref));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float* vh = v_ref + kvh * hd;
            float* oh = xb2_ref + h * hd;
            for (int d = 0; d < hd; d++) oh[d] = vh[d];
        }

        // O projection and first residual
        fwd_matmul_q8(xb_ref, layers[0].o_data, layers[0].o_scale, xb2_ref, hs, nh * hd);
        for (int i = 0; i < hs; i++) x_ref[i] += xb_ref[i];

        // Post-attention rmsnorm
        ref_rmsnorm(xb_ref, x_ref, layers[0].post_ln, hs, cfg.rms_norm_eps);

        // MLP
        fwd_matmul_q8(hb_ref, layers[0].gate_data, layers[0].gate_scale, xb_ref, cfg.intermediate_size, hs);
        fwd_matmul_q8(hb2_ref, layers[0].up_data, layers[0].up_scale, xb_ref, cfg.intermediate_size, hs);
        for (int i = 0; i < cfg.intermediate_size; i++) {
            hb_ref[i] = fwd_silu(hb_ref[i]) * hb2_ref[i];
        }
        fwd_matmul_q8(xb_ref, layers[0].down_data, layers[0].down_scale, hb_ref, hs, cfg.intermediate_size);

        // Second residual
        for (int i = 0; i < hs; i++) x_ref[i] += xb_ref[i];

        printf("Reference BUF_X after second residual (layer 0 complete, first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", x_ref[i]);
        printf("\n");

        float max_layer0_diff = 0;
        int max_diff_idx = 0;
        int mismatch_count = 0;
        // Check ALL 576 elements, not just first 8!
        for (int i = 0; i < hs; i++) {
            float vx = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
            float diff = fabsf(vx - x_ref[i]);
            if (diff > max_layer0_diff) {
                max_layer0_diff = diff;
                max_diff_idx = i;
            }
            if (diff > 0.01f) mismatch_count++;
        }
        printf("Max diff - Layer 0 complete (ALL %d elements): %.6e at index %d\n",
               hs, max_layer0_diff, max_diff_idx);
        printf("Mismatches (>0.01) in layer 0: %d / %d\n", mismatch_count, hs);
        if (max_layer0_diff > 0.01f) {
            printf("*** LAYER 0 COMPLETE MISMATCH ***\n");
            printf("  Verilog x[%d] = %.6f\n", max_diff_idx,
                   bits_to_float(sim.read_mem(BUF_BASE + max_diff_idx * 4)));
            printf("  Ref     x[%d] = %.6f\n", max_diff_idx, x_ref[max_diff_idx]);
        }
    }

    // Save ALL 576 elements of Verilog's layer 0 output for layer 1 reference computation
    float saved_layer0_out[576];
    for (int i = 0; i < hs; i++) {
        saved_layer0_out[i] = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
    }
    printf("Saved all %d elements of Verilog layer 0 output for layer 1 reference\n", hs);

    printf("\n=== Layer 0 debug complete. Checking layer 1 setup... ===\n");

    // Verify layer 1 descriptor is correct
    uint32_t layer1_desc = DESC_BASE + 0x40 + 0x40;  // Layer 1 descriptor
    printf("Layer 1 descriptor at 0x%08X:\n", layer1_desc);
    printf("  input_ln addr:    0x%08X (expected: 0x%08X)\n",
           sim.read_mem(layer1_desc + 0x00), layer_addrs[1].input_ln);
    printf("  q_data addr:      0x%08X (expected: 0x%08X)\n",
           sim.read_mem(layer1_desc + 0x04), layer_addrs[1].q_data);
    printf("  q_scale addr:     0x%08X (expected: 0x%08X)\n",
           sim.read_mem(layer1_desc + 0x08), layer_addrs[1].q_scale_addr);

    // Check if layer 1 weights differ from layer 0
    printf("\nLayer 0 q_data addr: 0x%08X, Layer 1 q_data addr: 0x%08X\n",
           layer_addrs[0].q_data, layer_addrs[1].q_data);

    // Run until layer 1 rmsnorm starts (should be right after layer 0 finishes)
    printf("\nRunning layer 1 input rmsnorm...\n");
    bool in_rmsnorm_l1 = false;
    while (!sim.top->halted && sim.cycle_count < 100000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_rmsnorm = (pc >= 0x2000 && pc < 0x3000);
        if (now_in_rmsnorm && !in_rmsnorm_l1) {
            printf("  Entered layer 1 rmsnorm at cycle %lu\n", sim.cycle_count);
            in_rmsnorm_l1 = true;
        }
        if (!now_in_rmsnorm && in_rmsnorm_l1) {
            printf("  Exited layer 1 rmsnorm at cycle %lu\n", sim.cycle_count);
            break;
        }
    }

    // Check BUF_XB after layer 1 rmsnorm
    printf("\nVerilog BUF_XB after layer 1 rmsnorm (first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_xb + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference layer 1 rmsnorm output using actual Verilog BUF_X as input
    {
        float x_l1[576], xb_l1[576];
        // Read BUF_X from Verilog memory (layer 0 output)
        for (int i = 0; i < hs; i++) {
            x_l1[i] = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
        }
        // Apply layer 1 rmsnorm weights to get expected BUF_XB
        ref_rmsnorm(xb_l1, x_l1, layers[1].input_ln, hs, cfg.rms_norm_eps);

        printf("Reference BUF_XB after layer 1 rmsnorm (first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", xb_l1[i]);
        printf("\n");

        float max_l1_rmsnorm_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vxb = bits_to_float(sim.read_mem(buf_xb + i * 4));
            float diff = fabsf(vxb - xb_l1[i]);
            if (diff > max_l1_rmsnorm_diff) max_l1_rmsnorm_diff = diff;
        }
        printf("Max diff - Layer 1 rmsnorm: %.6e\n", max_l1_rmsnorm_diff);
        if (max_l1_rmsnorm_diff > 0.01f) printf("*** LAYER 1 RMSNORM MISMATCH ***\n");
    }
    fflush(stdout);

    // Run through layer 1 Q projection
    printf("\nRunning layer 1 Q projection...\n");
    in_matmul = false;
    while (!sim.top->halted && sim.cycle_count < 100000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            printf("  Entered layer 1 Q matmul at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited layer 1 Q matmul at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_matmul = now_in_matmul;
    }

    // Check layer 1 Q projection output (buf_q already defined at BUF_BASE + 0x3000)
    printf("\nVerilog BUF_Q after layer 1 Q projection (first 8 values):\n  ");
    for (int i = 0; i < 8; i++) {
        float val = bits_to_float(sim.read_mem(buf_q + i * 4));
        printf("%.4f ", val);
    }
    printf("\n");

    // Compute reference layer 1 Q projection
    {
        float x_l1[576], xb_l1[576], q_l1[576];
        for (int i = 0; i < hs; i++) {
            x_l1[i] = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
        }
        ref_rmsnorm(xb_l1, x_l1, layers[1].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(q_l1, layers[1].q_data, layers[1].q_scale, xb_l1, nh * hd, hs);

        printf("Reference BUF_Q after layer 1 Q projection (first 8 values):\n  ");
        for (int i = 0; i < 8; i++) printf("%.4f ", q_l1[i]);
        printf("\n");

        float max_l1_q_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vq = bits_to_float(sim.read_mem(buf_q + i * 4));
            float diff = fabsf(vq - q_l1[i]);
            if (diff > max_l1_q_diff) max_l1_q_diff = diff;
        }
        printf("Max diff - Layer 1 Q projection: %.6e\n", max_l1_q_diff);
        if (max_l1_q_diff > 0.01f) printf("*** LAYER 1 Q PROJECTION MISMATCH ***\n");
    }
    fflush(stdout);

    // Run through layer 1 K and V projections
    printf("\nRunning layer 1 K projection...\n");
    in_matmul = false;
    while (!sim.top->halted && sim.cycle_count < 100000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            printf("  Entered layer 1 K matmul at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited layer 1 K matmul at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_matmul = now_in_matmul;
    }

    printf("Running layer 1 V projection...\n");
    in_matmul = false;
    while (!sim.top->halted && sim.cycle_count < 100000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            printf("  Entered layer 1 V matmul at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited layer 1 V matmul at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_matmul = now_in_matmul;
    }

    // Check layer 1 K and V outputs (buf_k, buf_v already defined)
    printf("\nVerilog layer 1 K (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(buf_k + i * 4)));
    printf("\nVerilog layer 1 V (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(buf_v + i * 4)));
    printf("\n");

    // Compute reference K, V for layer 1
    {
        float x_l1[576], xb_l1[576], k_l1[192], v_l1[192];
        for (int i = 0; i < hs; i++) x_l1[i] = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
        ref_rmsnorm(xb_l1, x_l1, layers[1].input_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(k_l1, layers[1].k_data, layers[1].k_scale, xb_l1, nkv * hd, hs);
        fwd_matmul_q8(v_l1, layers[1].v_data, layers[1].v_scale, xb_l1, nkv * hd, hs);

        printf("Reference layer 1 K (first 8): ");
        for (int i = 0; i < 8; i++) printf("%.4f ", k_l1[i]);
        printf("\nReference layer 1 V (first 8): ");
        for (int i = 0; i < 8; i++) printf("%.4f ", v_l1[i]);
        printf("\n");

        float max_k_diff = 0, max_v_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vk = bits_to_float(sim.read_mem(buf_k + i * 4));
            float vv = bits_to_float(sim.read_mem(buf_v + i * 4));
            float dk = fabsf(vk - k_l1[i]), dv = fabsf(vv - v_l1[i]);
            if (dk > max_k_diff) max_k_diff = dk;
            if (dv > max_v_diff) max_v_diff = dv;
        }
        printf("Max diff - K: %.6e, V: %.6e\n", max_k_diff, max_v_diff);
    }
    fflush(stdout);

    // Run through layer 1 RoPE, attention, and O projection
    printf("\nRunning layer 1 RoPE, attention, O projection...\n");
    in_matmul = false;  // Reset for O matmul detection
    while (!sim.top->halted && sim.cycle_count < 200000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            printf("  Entered layer 1 O matmul at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited layer 1 O matmul at cycle %lu\n", sim.cycle_count);
            break;  // Only one O projection matmul
        }
        in_matmul = now_in_matmul;
    }

    // Check layer 1 O projection output (in BUF_XB)
    printf("\nVerilog layer 1 O projection (BUF_XB first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(buf_xb + i * 4)));
    printf("\n");

    // Run through layer 1 first residual
    printf("Running layer 1 first residual...\n");
    in_residual = false;
    while (!sim.top->halted && sim.cycle_count < 200000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_residual = (pc >= 0x6000 && pc < 0x7000);
        if (now_in_residual && !in_residual) {
            printf("  Entered layer 1 first residual at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_residual && in_residual) {
            printf("  Exited layer 1 first residual at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_residual = now_in_residual;
    }

    // Check BUF_X after first residual
    printf("\nVerilog BUF_X after layer 1 first residual (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(BUF_BASE + i * 4)));
    printf("\n");

    // Compute reference for layer 1 through first residual
    {
        float x_l1[576], xb_l1[576], xb2_l1[576], q_l1[576], k_l1[192], v_l1[192];

        // Read BUF_X (layer 0 output)
        for (int i = 0; i < hs; i++) x_l1[i] = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
        // WAIT - this reads the current BUF_X which is after layer 1 first residual!
        // We need to use the saved layer 0 output. But we don't have full 576 elements saved.
        // Let me use the reference x values from x_per_layer[0].
    }

    // Now run layer 1 post-attention rmsnorm
    printf("\nRunning layer 1 post-attention rmsnorm...\n");
    bool in_rmsnorm_l1_post = false;
    while (!sim.top->halted && sim.cycle_count < 200000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_rmsnorm = (pc >= 0x2000 && pc < 0x3000);
        if (now_in_rmsnorm && !in_rmsnorm_l1_post) {
            printf("  Entered at cycle %lu\n", sim.cycle_count);
            in_rmsnorm_l1_post = true;
        }
        if (!now_in_rmsnorm && in_rmsnorm_l1_post) {
            printf("  Exited at cycle %lu\n", sim.cycle_count);
            break;
        }
    }

    // Read BUF_XB (post-attn rmsnorm output) and BUF_X (first residual output)
    printf("\nVerilog after layer 1 post-attn rmsnorm:\n");
    printf("  BUF_X (first residual out, first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(BUF_BASE + i * 4)));
    printf("\n  BUF_XB (post-attn rmsnorm, first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(buf_xb + i * 4)));
    printf("\n");

    // Compute reference layer 1 post-attn rmsnorm
    {
        // Read current BUF_X (layer 1 first residual output) for reference computation
        float x_after_res1[576], xb_post[576];
        for (int i = 0; i < hs; i++) {
            x_after_res1[i] = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
        }
        // Apply layer 1 post-attn rmsnorm
        ref_rmsnorm(xb_post, x_after_res1, layers[1].post_ln, hs, cfg.rms_norm_eps);

        printf("  Reference BUF_XB (post-attn rmsnorm, first 8): ");
        for (int i = 0; i < 8; i++) printf("%.4f ", xb_post[i]);
        printf("\n");

        float max_post_diff = 0;
        for (int i = 0; i < 8; i++) {
            float vxb = bits_to_float(sim.read_mem(buf_xb + i * 4));
            float diff = fabsf(vxb - xb_post[i]);
            if (diff > max_post_diff) max_post_diff = diff;
        }
        printf("  Max diff: %.6e\n", max_post_diff);
        if (max_post_diff > 0.01f) printf("  *** LAYER 1 POST-ATTN RMSNORM MISMATCH ***\n");
    }
    fflush(stdout);

    // Run layer 1 gate projection
    printf("\nRunning layer 1 gate projection...\n");
    in_matmul = false;
    while (!sim.top->halted && sim.cycle_count < 200000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            printf("  Entered at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_matmul = now_in_matmul;
    }

    // Check gate output
    printf("Verilog BUF_HB (gate, first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(buf_hb + i * 4)));
    printf("\n");

    // Compute reference gate
    {
        float x_tmp[576], xb_tmp[576], gate_tmp[1536];
        for (int i = 0; i < hs; i++) x_tmp[i] = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
        ref_rmsnorm(xb_tmp, x_tmp, layers[1].post_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(gate_tmp, layers[1].gate_data, layers[1].gate_scale, xb_tmp, cfg.intermediate_size, hs);

        printf("Reference BUF_HB (gate, first 8): ");
        for (int i = 0; i < 8; i++) printf("%.4f ", gate_tmp[i]);
        printf("\n");

        float max_gate_diff = 0;
        int max_gate_idx = 0;
        int gate_mismatch = 0;
        // Check ALL 1536 elements
        for (int i = 0; i < cfg.intermediate_size; i++) {
            float vg = bits_to_float(sim.read_mem(buf_hb + i * 4));
            float diff = fabsf(vg - gate_tmp[i]);
            if (diff > max_gate_diff) {
                max_gate_diff = diff;
                max_gate_idx = i;
            }
            if (diff > 0.01f) gate_mismatch++;
        }
        printf("Max diff - gate (ALL %d): %.6e at idx %d\n", cfg.intermediate_size, max_gate_diff, max_gate_idx);
        if (gate_mismatch > 0) {
            printf("*** GATE MISMATCH: %d elements differ ***\n", gate_mismatch);
        }
        if (max_gate_diff > 0.01f) printf("*** LAYER 1 GATE MISMATCH ***\n");
    }

    // Run layer 1 up projection
    printf("\nRunning layer 1 up projection...\n");
    in_matmul = false;
    while (!sim.top->halted && sim.cycle_count < 200000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            printf("  Entered at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_matmul = now_in_matmul;
    }

    // Check up output
    printf("Verilog BUF_HB2 (up, first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(buf_hb2 + i * 4)));
    printf("\n");

    // Compute reference up
    {
        float x_tmp[576], xb_tmp[576], up_tmp[1536];
        for (int i = 0; i < hs; i++) x_tmp[i] = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
        ref_rmsnorm(xb_tmp, x_tmp, layers[1].post_ln, hs, cfg.rms_norm_eps);
        fwd_matmul_q8(up_tmp, layers[1].up_data, layers[1].up_scale, xb_tmp, cfg.intermediate_size, hs);

        printf("Reference BUF_HB2 (up, first 8): ");
        for (int i = 0; i < 8; i++) printf("%.4f ", up_tmp[i]);
        printf("\n");

        float max_up_diff = 0;
        int max_up_idx = 0;
        int up_mismatch = 0;
        // Check ALL 1536 elements
        for (int i = 0; i < cfg.intermediate_size; i++) {
            float vu = bits_to_float(sim.read_mem(buf_hb2 + i * 4));
            float diff = fabsf(vu - up_tmp[i]);
            if (diff > max_up_diff) {
                max_up_diff = diff;
                max_up_idx = i;
            }
            if (diff > 0.01f) up_mismatch++;
        }
        printf("Max diff - up (ALL %d): %.6e at idx %d\n", cfg.intermediate_size, max_up_diff, max_up_idx);
        if (up_mismatch > 0) {
            printf("*** UP MISMATCH: %d elements differ ***\n", up_mismatch);
        }
        if (max_up_diff > 0.01f) printf("*** LAYER 1 UP MISMATCH ***\n");
    }

    // Capture gate[83] and up[83] BEFORE silu_mul runs
    printf("\n=== Pre-silu_mul debug ===\n");
    float pre_gate_83 = bits_to_float(sim.read_mem(buf_hb + 83 * 4));
    float pre_up_83 = bits_to_float(sim.read_mem(buf_hb2 + 83 * 4));
    printf("  BEFORE silu_mul: gate[83]=%.6f, up[83]=%.6f\n", pre_gate_83, pre_up_83);
    printf("  Expected silu_mul[83] = silu(%.6f) * %.6f = %.6f * %.6f = %.6f\n",
           pre_gate_83, pre_up_83, fwd_silu(pre_gate_83), pre_up_83,
           fwd_silu(pre_gate_83) * pre_up_83);

    // Find max gate value and check all large gates
    float max_gate = 0;
    int max_gate_idx = 0;
    int large_gate_count = 0;
    for (int i = 0; i < cfg.intermediate_size; i++) {
        float g = bits_to_float(sim.read_mem(buf_hb + i * 4));
        if (fabsf(g) > max_gate) {
            max_gate = fabsf(g);
            max_gate_idx = i;
        }
        if (fabsf(g) > 10.0f) large_gate_count++;
    }
    printf("  Max |gate| = %.6f at index %d, large gates (>10): %d\n",
           max_gate, max_gate_idx, large_gate_count);

    // Run layer 1 silu_mul
    printf("\nRunning layer 1 silu_mul...\n");
    bool in_silu_l1 = false;
    while (!sim.top->halted && sim.cycle_count < 200000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_silu = (pc >= 0x5000 && pc < 0x6000);
        if (now_in_silu && !in_silu_l1) {
            printf("  Entered at cycle %lu\n", sim.cycle_count);
            in_silu_l1 = true;
        }
        if (!now_in_silu && in_silu_l1) {
            printf("  Exited at cycle %lu\n", sim.cycle_count);
            break;
        }
    }

    // silu_mul output is in BUF_HB - compare ALL 1536 elements
    printf("Verilog BUF_HB (after silu_mul, first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(buf_hb + i * 4)));
    printf("\n");

    // Compute reference silu_mul for layer 1 using current Verilog state
    {
        float l1_xb[576], l1_hb[1536], l1_hb2[1536];
        // Read BUF_XB (post-attn rmsnorm output) from Verilog
        for (int i = 0; i < hs; i++) l1_xb[i] = bits_to_float(sim.read_mem(buf_xb + i * 4));

        // Compute gate and up using layer 1 weights
        fwd_matmul_q8(l1_hb, layers[1].gate_data, layers[1].gate_scale, l1_xb, cfg.intermediate_size, hs);
        fwd_matmul_q8(l1_hb2, layers[1].up_data, layers[1].up_scale, l1_xb, cfg.intermediate_size, hs);

        // Compute silu_mul
        for (int i = 0; i < cfg.intermediate_size; i++) {
            l1_hb[i] = fwd_silu(l1_hb[i]) * l1_hb2[i];
        }

        // Compare ALL 1536 elements with Verilog's BUF_HB
        float max_silu_diff = 0;
        int max_silu_idx = 0;
        int silu_mismatch = 0;
        for (int i = 0; i < cfg.intermediate_size; i++) {
            float ver = bits_to_float(sim.read_mem(buf_hb + i * 4));
            float diff = fabsf(ver - l1_hb[i]);
            if (diff > max_silu_diff) {
                max_silu_diff = diff;
                max_silu_idx = i;
            }
            if (diff > 0.01f) silu_mismatch++;
        }
        printf("Max diff - Layer 1 silu_mul (ALL %d elements): %.6e at index %d\n",
               cfg.intermediate_size, max_silu_diff, max_silu_idx);
        if (silu_mismatch > 0) {
            printf("*** SILU_MUL MISMATCH: %d elements differ by >0.01 ***\n", silu_mismatch);
            printf("  Verilog silu_mul[%d] = %.6f\n", max_silu_idx,
                   bits_to_float(sim.read_mem(buf_hb + max_silu_idx * 4)));
            printf("  Ref     silu_mul[%d] = %.6f\n", max_silu_idx, l1_hb[max_silu_idx]);

            // Debug: Read gate[idx] and up[idx] BEFORE silu_mul modified them
            // We need to recompute gate and up to check the inputs
            float gate_val = 0, up_val = 0;
            // Recompute gate and up for this specific index
            {
                float l1_xb_dbg[576];
                for (int i = 0; i < hs; i++) l1_xb_dbg[i] = bits_to_float(sim.read_mem(buf_xb + i * 4));
                // gate[idx] = sum(gate_W[idx,j] * xb[j])
                int8_t* gate_row = layers[1].gate_data + (int64_t)max_silu_idx * hs;
                float gate_acc = 0;
                for (int j = 0; j < hs; j++) gate_acc += gate_row[j] * layers[1].gate_scale * l1_xb_dbg[j];
                gate_val = gate_acc;

                int8_t* up_row = layers[1].up_data + (int64_t)max_silu_idx * hs;
                float up_acc = 0;
                for (int j = 0; j < hs; j++) up_acc += up_row[j] * layers[1].up_scale * l1_xb_dbg[j];
                up_val = up_acc;
            }
            printf("  Debug: computed gate[%d] = %.6f, up[%d] = %.6f\n",
                   max_silu_idx, gate_val, max_silu_idx, up_val);
            printf("  Debug: silu(gate) = %.6f\n", fwd_silu(gate_val));
            printf("  Debug: expected silu_mul = %.6f\n", fwd_silu(gate_val) * up_val);

            // Also read what Verilog has in BUF_HB and BUF_HB2 BEFORE silu_mul
            // Note: BUF_HB has been overwritten by silu_mul! We can't read original gate anymore
            // But BUF_HB2 (up) should still be intact
            float ver_up = bits_to_float(sim.read_mem(buf_hb2 + max_silu_idx * 4));
            printf("  Verilog BUF_HB2[%d] (up) = %.6f\n", max_silu_idx, ver_up);
        }
    }

    // Run layer 1 down projection
    printf("\nRunning layer 1 down projection...\n");
    in_matmul = false;
    while (!sim.top->halted && sim.cycle_count < 200000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_matmul = (pc >= 0x1000 && pc < 0x2000);
        if (now_in_matmul && !in_matmul) {
            printf("  Entered at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_matmul && in_matmul) {
            printf("  Exited at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_matmul = now_in_matmul;
    }

    // Check down output in BUF_XB
    printf("Verilog BUF_XB (down, first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(buf_xb + i * 4)));
    printf("\n");

    // Run layer 1 second residual
    printf("\nRunning layer 1 second residual...\n");
    in_residual = false;
    while (!sim.top->halted && sim.cycle_count < 200000000) {
        sim.tick();
        uint32_t pc = sim.top->pc;
        bool now_in_residual = (pc >= 0x6000 && pc < 0x7000);
        if (now_in_residual && !in_residual) {
            printf("  Entered at cycle %lu\n", sim.cycle_count);
        }
        if (!now_in_residual && in_residual) {
            printf("  Exited at cycle %lu\n", sim.cycle_count);
            break;
        }
        in_residual = now_in_residual;
    }

    // Check final layer 1 output
    printf("\nVerilog BUF_X after layer 1 second residual (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(BUF_BASE + i * 4)));
    printf("\n");

    // === Compute COMPLETE reference for layer 1 using saved_layer0_out (all 576 elements) ===
    printf("\n=== Computing complete layer 1 reference using saved layer 0 output ===\n");

    float ref_x[576], ref_xb[576], ref_xb2[576], ref_q[576], ref_k[192], ref_v[192];
    float ref_hb[1536], ref_hb2[1536];

    // Copy saved layer 0 output
    for (int i = 0; i < hs; i++) ref_x[i] = saved_layer0_out[i];

    printf("Layer 0 output (saved, first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", ref_x[i]);
    printf("\n");

    // Layer 1 rmsnorm
    ref_rmsnorm(ref_xb, ref_x, layers[1].input_ln, hs, cfg.rms_norm_eps);
    printf("Ref layer 1 rmsnorm (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", ref_xb[i]);
    printf("\n");

    // Layer 1 Q, K, V projections
    fwd_matmul_q8(ref_q, layers[1].q_data, layers[1].q_scale, ref_xb, nh * hd, hs);
    fwd_matmul_q8(ref_k, layers[1].k_data, layers[1].k_scale, ref_xb, nkv * hd, hs);
    fwd_matmul_q8(ref_v, layers[1].v_data, layers[1].v_scale, ref_xb, nkv * hd, hs);

    // RoPE (position 0)
    for (int h = 0; h < nh; h++) fwd_apply_rope(ref_q + h * hd, hd, rope_cos, rope_sin);
    for (int h = 0; h < nkv; h++) fwd_apply_rope(ref_k + h * hd, hd, rope_cos, rope_sin);

    // Attention (at position 0, attention output = V)
    memset(ref_xb2, 0, sizeof(ref_xb2));
    for (int h = 0; h < nh; h++) {
        int kvh = h / ng;
        float* vh = ref_v + kvh * hd;
        float* oh = ref_xb2 + h * hd;
        for (int d = 0; d < hd; d++) oh[d] = vh[d];
    }

    // O projection
    fwd_matmul_q8(ref_xb, layers[1].o_data, layers[1].o_scale, ref_xb2, hs, nh * hd);
    printf("Ref layer 1 O proj (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", ref_xb[i]);
    printf("\n");

    // First residual
    for (int i = 0; i < hs; i++) ref_x[i] += ref_xb[i];
    printf("Ref layer 1 first res (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", ref_x[i]);
    printf("\n");

    // Post-attention rmsnorm
    ref_rmsnorm(ref_xb, ref_x, layers[1].post_ln, hs, cfg.rms_norm_eps);
    printf("Ref layer 1 post-attn rmsnorm (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", ref_xb[i]);
    printf("\n");

    // Gate and Up projections
    fwd_matmul_q8(ref_hb, layers[1].gate_data, layers[1].gate_scale, ref_xb, cfg.intermediate_size, hs);
    fwd_matmul_q8(ref_hb2, layers[1].up_data, layers[1].up_scale, ref_xb, cfg.intermediate_size, hs);
    printf("Ref layer 1 gate (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", ref_hb[i]);
    printf("\n");
    printf("Ref layer 1 up (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", ref_hb2[i]);
    printf("\n");

    // SILU_MUL
    for (int i = 0; i < cfg.intermediate_size; i++) {
        ref_hb[i] = fwd_silu(ref_hb[i]) * ref_hb2[i];
    }
    printf("Ref layer 1 silu_mul (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", ref_hb[i]);
    printf("\n");

    // Down projection
    fwd_matmul_q8(ref_xb, layers[1].down_data, layers[1].down_scale, ref_hb, hs, cfg.intermediate_size);
    printf("Ref layer 1 down (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", ref_xb[i]);
    printf("\n");

    // Second residual (final layer 1 output)
    for (int i = 0; i < hs; i++) ref_x[i] += ref_xb[i];
    printf("Ref layer 1 final (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", ref_x[i]);
    printf("\n");

    // Compare Verilog vs reference (ALL 576 elements)
    printf("\nVerilog layer 1 final (first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", bits_to_float(sim.read_mem(BUF_BASE + i * 4)));
    printf("\n");

    float max_l1_final_diff = 0;
    int max_diff_idx_l1 = 0;
    int mismatch_count_l1 = 0;
    for (int i = 0; i < hs; i++) {
        float vx = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
        float diff = fabsf(vx - ref_x[i]);
        if (diff > max_l1_final_diff) {
            max_l1_final_diff = diff;
            max_diff_idx_l1 = i;
        }
        if (diff > 0.01f) mismatch_count_l1++;
    }
    printf("Max diff - Layer 1 final (ALL %d elements): %.6e at index %d\n",
           hs, max_l1_final_diff, max_diff_idx_l1);
    printf("Mismatches (>0.01) in layer 1: %d / %d\n", mismatch_count_l1, hs);

    // Also compare with x_per_layer[1] to debug the pre-computed reference
    printf("\nx_per_layer[1] (pre-computed, first 8): ");
    for (int i = 0; i < 8; i++) printf("%.4f ", x_per_layer[1][i]);
    printf("\n");

    float max_xpl_diff = 0;
    for (int i = 0; i < 8; i++) {
        float diff = fabsf(ref_x[i] - x_per_layer[1][i]);
        if (diff > max_xpl_diff) max_xpl_diff = diff;
    }
    printf("Max diff - Ref vs x_per_layer[1] (first 8): %.6e\n", max_xpl_diff);

    if (max_l1_final_diff > 0.1f) {
        printf("*** LAYER 1 FINAL MISMATCH ***\n");
        printf("  Verilog x[%d] = %.6f\n", max_diff_idx_l1,
               bits_to_float(sim.read_mem(BUF_BASE + max_diff_idx_l1 * 4)));
        printf("  Ref     x[%d] = %.6f\n", max_diff_idx_l1, ref_x[max_diff_idx_l1]);
        return 1;
    }

    printf("\n=== Running remaining layers with periodic checks... ===\n");

    // Run forward pass (up to 5B cycles - full 30-layer pass is ~3-5B cycles)
    // Layer 0 finished at ~24M cycles, each subsequent layer takes ~30M cycles
    uint64_t max_cycles = 5000000000ULL;
    uint64_t layer0_end = sim.cycle_count;  // Current cycle count after layer 0
    uint64_t cycles_per_layer = 30000000;   // Approximately 30M cycles per layer
    int current_layer = 1;  // We just finished layer 0
    uint64_t next_layer_check = layer0_end + cycles_per_layer;

    while (!sim.top->halted && sim.cycle_count < max_cycles) {
        sim.tick();

        // Check for layer completion (approximately)
        if (sim.cycle_count >= next_layer_check && current_layer < 30) {
            // Check BUF_X and compare with reference for current layer
            float max_layer_diff = 0;
            for (int i = 0; i < 8; i++) {
                float vx = bits_to_float(sim.read_mem(BUF_BASE + i * 4));
                float rx = x_per_layer[current_layer][i];  // current layer reference
                float diff = fabsf(vx - rx);
                if (diff > max_layer_diff) max_layer_diff = diff;
            }

            // Read current layer output
            printf("  Layer %d checkpoint (cycle %luM): max_diff=%.6e\n",
                   current_layer, sim.cycle_count / 1000000, max_layer_diff);
            printf("    Verilog x[0:3]=[%.4f,%.4f,%.4f,%.4f]\n",
                   bits_to_float(sim.read_mem(BUF_BASE)),
                   bits_to_float(sim.read_mem(BUF_BASE + 4)),
                   bits_to_float(sim.read_mem(BUF_BASE + 8)),
                   bits_to_float(sim.read_mem(BUF_BASE + 12)));
            printf("    Ref     x[0:3]=[%.4f,%.4f,%.4f,%.4f]\n",
                   x_per_layer[current_layer][0],
                   x_per_layer[current_layer][1],
                   x_per_layer[current_layer][2],
                   x_per_layer[current_layer][3]);

            if (max_layer_diff > 0.1f) {
                printf("    *** DIVERGENCE DETECTED at layer %d! ***\n", current_layer);
            }
            fflush(stdout);

            current_layer++;
            next_layer_check += cycles_per_layer;
        }
    }

    printf("Completed: %lu cycles, halted=%d\n", sim.cycle_count, sim.top->halted);

    if (!sim.top->halted) {
        printf("FAILED - did not halt within cycle limit\n");
        return 1;
    }

    // Read logits from output buffer
    uint32_t logits_addr = BUF_BASE + 0x80000;
    float* verilog_logits = (float*)malloc(cfg.vocab_size * sizeof(float));
    for (int i = 0; i < cfg.vocab_size; i++) {
        verilog_logits[i] = bits_to_float(sim.read_mem(logits_addr + i * 4));
    }

    // Compare results
    printf("\nComparing Verilog vs Reference...\n");
    float max_diff = 0;
    int mismatches = 0;
    double sum_diff = 0;
    for (int i = 0; i < cfg.vocab_size; i++) {
        float diff = fabsf(verilog_logits[i] - ref_logits[i]);
        sum_diff += diff;
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.1f) mismatches++;
    }
    float avg_diff = sum_diff / cfg.vocab_size;

    printf("Max logit difference: %.6e\n", max_diff);
    printf("Avg logit difference: %.6e\n", avg_diff);
    printf("Mismatches (>0.1): %d / %d\n", mismatches, cfg.vocab_size);

    // Show top-5 predictions from both
    printf("\nTop-5 (Verilog):\n");
    float* vlog_copy = (float*)malloc(cfg.vocab_size * sizeof(float));
    memcpy(vlog_copy, verilog_logits, cfg.vocab_size * sizeof(float));
    for (int kk = 0; kk < 5; kk++) {
        int best = 0;
        for (int i = 1; i < cfg.vocab_size; i++)
            if (vlog_copy[i] > vlog_copy[best]) best = i;
        printf("  [%d] logit=%.4f\n", best, vlog_copy[best]);
        vlog_copy[best] = -1e9f;
    }
    free(vlog_copy);

    printf("Top-5 (Reference):\n");
    float* ref_copy = (float*)malloc(cfg.vocab_size * sizeof(float));
    memcpy(ref_copy, ref_logits, cfg.vocab_size * sizeof(float));
    for (int kk = 0; kk < 5; kk++) {
        int best = 0;
        for (int i = 1; i < cfg.vocab_size; i++)
            if (ref_copy[i] > ref_copy[best]) best = i;
        printf("  [%d] logit=%.4f\n", best, ref_copy[best]);
        ref_copy[best] = -1e9f;
    }
    free(ref_copy);

    // Cleanup
    free(embed_data);
    free(final_norm);
    free(rope_cos);
    free(rope_sin);
    free(layer_addrs);
    for (int l = 0; l < cfg.num_layers; l++) {
        free(layers[l].input_ln);
        free(layers[l].q_data);
        free(layers[l].k_data);
        free(layers[l].v_data);
        free(layers[l].o_data);
        free(layers[l].post_ln);
        free(layers[l].gate_data);
        free(layers[l].up_data);
        free(layers[l].down_data);
        free(k_caches[l]);
        free(v_caches[l]);
    }
    free(layers);
    free(ref_logits);
    free(verilog_logits);

    int passed = (mismatches == 0);
    printf("\n%s\n", passed ? "PASSED - Verilog matches reference!" : "DIFFERENCES FOUND");

    return passed ? 0 : 1;
}

// ============================================================================
// Embed Kernel Test (minimal debugging test)
// ============================================================================
int test_embed_kernel() {
    printf("\n=== Testing embed kernel with Verilator ===\n");

    SmolSimulator sim(false);
    sim.hold_reset();
    sim.tick();

    // Place HALT trap at address 0
    sim.write_mem(0x0000, 0xE0000000);

    // Load embed kernel
    if (sim.load_kernel("../kernels/embed.bin", 0x1000) != 0) {
        printf("SKIPPED - kernel binary not found\n");
        return 0;
    }

    // Test with simple INT8 data
    const int n = 16;
    const float scale = 0.5f;

    // Input: INT8 values [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8]
    int8_t input[16] = {1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8};

    // Write INT8 data to memory (packed as 32-bit words)
    for (int i = 0; i < 16; i += 4) {
        uint32_t word = 0;
        for (int j = 0; j < 4; j++) {
            word |= ((uint32_t)(uint8_t)input[i + j]) << (j * 8);
        }
        sim.write_mem(BUF_W + i, word);
        printf("  Input word at 0x%X: 0x%08X (bytes: %d, %d, %d, %d)\n",
               BUF_W + i, word, input[i], input[i+1], input[i+2], input[i+3]);
    }

    // Clear output buffer
    for (int i = 0; i < n; i++) {
        sim.write_mem(BUF_XB + i * 4, 0);
    }

    // Compute expected output
    float expected[16];
    for (int i = 0; i < n; i++) {
        expected[i] = (float)input[i] * scale;
        printf("  Expected[%d] = %d * %.2f = %.4f\n", i, input[i], scale, expected[i]);
    }

    // Release reset and set up registers
    sim.release_reset();
    sim.top->eval();

    // Set up registers:
    // R3 = output pointer
    // R4 = input pointer (int8_t*)
    // R5 = n (number of elements)
    // F1 = scale
    sim.write_reg(1, 0);                    // RA = 0 (halt on return)
    sim.write_reg(3, BUF_XB);               // output
    sim.write_reg(4, BUF_W);                // input (int8)
    sim.write_reg(5, n);                    // n
    sim.write_fp_reg(1, float_to_bits(scale));  // F1 = scale
    sim.top->eval();

    printf("\nRunning embed kernel...\n");

    // Run until halt
    int max_cycles = 1000;
    while (!sim.top->halted && sim.cycle_count < (uint64_t)max_cycles) {
        sim.tick();
    }

    printf("Completed: %lu cycles, halted=%d\n", sim.cycle_count, sim.top->halted);

    // Read and verify output
    printf("\nResults:\n");
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float got = bits_to_float(sim.read_mem(BUF_XB + i * 4));
        float diff = fabsf(got - expected[i]);
        printf("  [%2d] got=%.4f, expected=%.4f, diff=%.6f %s\n",
               i, got, expected[i], diff, diff > 0.001f ? "ERROR" : "");
        if (diff > 0.001f) errors++;
    }

    printf("\n%s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);
    return errors == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    int failures = 0;
    bool run_forward = false;
    bool run_embed = false;

    // Check for flags
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--forward") == 0) {
            run_forward = true;
        }
        if (strcmp(argv[i], "--embed") == 0) {
            run_embed = true;
        }
    }

    if (run_embed) {
        return test_embed_kernel();
    }

    if (run_forward) {
        // Only run the full forward pass test
        return test_forward_pass();
    }

    // Default: run kernel tests
    // Test 1: Simple instruction
    int result = test_simple();
    if (result != 0) {
        printf("Simple test failed, aborting\n");
        return 1;
    }

    // Test 2: matmul_q8 kernel (inline instructions)
    result = test_matmul_kernel();
    if (result != 0) failures++;

    // Test 3: rmsnorm kernel (from binary)
    result = test_rmsnorm_kernel();
    if (result != 0) failures++;

    // Test 4: residual kernel
    result = test_residual_kernel();
    if (result != 0) failures++;

    // Test 5: exp/silu numerical accuracy
    result = test_exp_silu_accuracy();
    if (result != 0) failures++;

    // Test 6: silu_mul kernel
    result = test_silu_mul_kernel();
    if (result != 0) failures++;

    printf("\n=== Summary: %d failures ===\n", failures);
    printf("Run with --forward to test full forward pass\n");
    return failures;
}
