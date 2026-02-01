/**
 * SMOL-32 Kernel Test: matmul_q8
 * Tests the Q8 matrix-vector multiply kernel with small data
 */

`timescale 1ns/1ps

module tb_kernel_matmul;

    //=========================================================================
    // Parameters
    //=========================================================================

    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter MEM_SIZE = 32'h00100000;  // 1MB for testing
    parameter MAX_CYCLES = 100000;

    //=========================================================================
    // Signals
    //=========================================================================

    reg         clk;
    reg         rst_n;
    wire        halted;
    wire [31:0] pc;
    wire [3:0]  state;

    // Debug interface
    reg         dbg_mem_re;
    reg  [31:0] dbg_mem_addr;
    wire [31:0] dbg_mem_rdata;

    //=========================================================================
    // Clock Generation
    //=========================================================================

    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    //=========================================================================
    // DUT Instantiation
    //=========================================================================

    smol32_top #(
        .MEM_SIZE(MEM_SIZE),
        .MEM_INIT_FILE("")
    ) dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .halted       (halted),
        .pc           (pc),
        .state        (state),
        .dbg_mem_re   (dbg_mem_re),
        .dbg_mem_addr (dbg_mem_addr),
        .dbg_mem_rdata(dbg_mem_rdata)
    );

    //=========================================================================
    // Test Setup
    //=========================================================================

    // matmul_q8 kernel calling convention:
    //   R3 = output pointer (float*)
    //   R4 = weight data pointer (int8_t*)
    //   R5 = scale pointer (float*)
    //   R6 = input pointer (float*)
    //   R7 = rows
    //   R8 = cols
    //   RA = return address (0 for halt)
    //
    // Memory layout:
    //   0x0000: HALT trap
    //   0x1000: matmul_q8 kernel code
    //   0x2000: output buffer (8 bytes = 2 floats)
    //   0x2100: weight data (32 bytes = 2 rows * 16 cols Q8)
    //   0x2200: scale (4 bytes = 1 float = 1.0)
    //   0x2300: input (64 bytes = 16 floats = 1.0 each)
    //
    // Test case: 2 rows, 16 cols
    //   weights[0] = [1, 1, 1, ...] (all 1s) -> sum = 16
    //   weights[1] = [2, 2, 2, ...] (all 2s) -> sum = 32
    //   scale = 1.0
    //   input = [1.0, 1.0, ...]
    //   expected output = [16.0, 32.0]

    initial begin
        // Wait for reset
        @(negedge rst_n);
        @(posedge rst_n);

        // Place HALT trap at address 0
        dut.memory[0] = 32'hE0000000;

        // Load matmul_q8 kernel at 0x1000
        // 17 instructions from matmul_q8.bin
        dut.memory[32'h1000/4 + 0]  = 32'h08250000;  // LF F1, 0(R5)
        dut.memory[32'h1000/4 + 1]  = 32'h70010000;  // QSETSCALE F1
        dut.memory[32'h1000/4 + 2]  = 32'h70060080;  // FSETBASE R6
        dut.memory[32'h1000/4 + 3]  = 32'h11270000;  // MV R9, R7 (ADDI R9, R7, 0)
        dut.memory[32'h1000/4 + 4]  = 32'h70040040;  // QSETBASE R4
        dut.memory[32'h1000/4 + 5]  = 32'h70060080;  // FSETBASE R6
        dut.memory[32'h1000/4 + 6]  = 32'h74000000;  // ACCZERO
        dut.memory[32'h1000/4 + 7]  = 32'h11480000;  // MV R10, R8 (ADDI R10, R8, 0)
        dut.memory[32'h1000/4 + 8]  = 32'h114a0244;  // SRLI R10, R10, 4
        dut.memory[32'h1000/4 + 9]  = 32'h740000d0;  // Q8MACINC 16
        dut.memory[32'h1000/4 + 10] = 32'hcd40ffff;  // LOOP R10, -1
        dut.memory[32'h1000/4 + 11] = 32'h74400040;  // ACCREAD F2
        dut.memory[32'h1000/4 + 12] = 32'h0c430000;  // SF F2, 0(R3)
        dut.memory[32'h1000/4 + 13] = 32'h14630004;  // ADDI R3, R3, 4
        dut.memory[32'h1000/4 + 14] = 32'h10844000;  // ADD R4, R4, R8
        dut.memory[32'h1000/4 + 15] = 32'hcd20fff5;  // LOOP R9, -11
        dut.memory[32'h1000/4 + 16] = 32'hc8010000;  // RET (JALR R0, RA, 0)

        // Output buffer at 0x2000 (initialized to 0)
        dut.memory[32'h2000/4] = 32'h00000000;
        dut.memory[32'h2004/4] = 32'h00000000;

        // Weight data at 0x2100: 2 rows x 16 cols of Q8
        // Row 0: all 1s, Row 1: all 2s
        // Packed as 4 bytes per word (little-endian)
        dut.memory[32'h2100/4 + 0] = 32'h01010101;  // [1,1,1,1]
        dut.memory[32'h2100/4 + 1] = 32'h01010101;  // [1,1,1,1]
        dut.memory[32'h2100/4 + 2] = 32'h01010101;  // [1,1,1,1]
        dut.memory[32'h2100/4 + 3] = 32'h01010101;  // [1,1,1,1]
        dut.memory[32'h2100/4 + 4] = 32'h02020202;  // [2,2,2,2]
        dut.memory[32'h2100/4 + 5] = 32'h02020202;  // [2,2,2,2]
        dut.memory[32'h2100/4 + 6] = 32'h02020202;  // [2,2,2,2]
        dut.memory[32'h2100/4 + 7] = 32'h02020202;  // [2,2,2,2]

        // Scale at 0x2200: 1.0 = 0x3F800000
        dut.memory[32'h2200/4] = 32'h3F800000;

        // Input at 0x2300: 16 floats, all 1.0
        dut.memory[32'h2300/4 + 0]  = 32'h3F800000;  // 1.0
        dut.memory[32'h2300/4 + 1]  = 32'h3F800000;
        dut.memory[32'h2300/4 + 2]  = 32'h3F800000;
        dut.memory[32'h2300/4 + 3]  = 32'h3F800000;
        dut.memory[32'h2300/4 + 4]  = 32'h3F800000;
        dut.memory[32'h2300/4 + 5]  = 32'h3F800000;
        dut.memory[32'h2300/4 + 6]  = 32'h3F800000;
        dut.memory[32'h2300/4 + 7]  = 32'h3F800000;
        dut.memory[32'h2300/4 + 8]  = 32'h3F800000;
        dut.memory[32'h2300/4 + 9]  = 32'h3F800000;
        dut.memory[32'h2300/4 + 10] = 32'h3F800000;
        dut.memory[32'h2300/4 + 11] = 32'h3F800000;
        dut.memory[32'h2300/4 + 12] = 32'h3F800000;
        dut.memory[32'h2300/4 + 13] = 32'h3F800000;
        dut.memory[32'h2300/4 + 14] = 32'h3F800000;
        dut.memory[32'h2300/4 + 15] = 32'h3F800000;

        // Set up registers for kernel call
        // R3 = 0x2000 (output)
        // R4 = 0x2100 (weights)
        // R5 = 0x2200 (scale)
        // R6 = 0x2300 (input)
        // R7 = 2 (rows)
        // R8 = 16 (cols)
        // RA (R1) = 0 (halt on return)
        dut.core.regfile.regs[1]  = 32'h00000000;  // RA = 0 (halt trap)
        dut.core.regfile.regs[3]  = 32'h00002000;  // output ptr
        dut.core.regfile.regs[4]  = 32'h00002100;  // weight ptr
        dut.core.regfile.regs[5]  = 32'h00002200;  // scale ptr
        dut.core.regfile.regs[6]  = 32'h00002300;  // input ptr
        dut.core.regfile.regs[7]  = 32'h00000002;  // rows = 2
        dut.core.regfile.regs[8]  = 32'h00000010;  // cols = 16
    end

    //=========================================================================
    // Test Execution
    //=========================================================================

    integer cycle_count;
    integer test_passed;

    initial begin
        $dumpfile("tb_kernel_matmul.vcd");
        $dumpvars(0, tb_kernel_matmul);

        // Initialize
        rst_n = 0;
        dbg_mem_re = 0;
        dbg_mem_addr = 0;
        cycle_count = 0;
        test_passed = 1;

        // Reset
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        $display("=== SMOL-32 Kernel Test: matmul_q8 ===");
        $display("Starting execution at PC=0x%08X", pc);

        // Run until halted or timeout
        while (!halted && cycle_count < MAX_CYCLES) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end

        // Check results
        $display("");
        $display("=== Execution Complete ===");
        $display("Cycles: %0d", cycle_count);
        $display("Halted: %0d", halted);

        // Check output buffer
        $display("");
        $display("=== Output Check ===");

        // output[0] should be 16.0 (sum of 16 ones times scale 1.0 times input 1.0)
        // 16.0 = 0x41800000
        dbg_mem_addr = 32'h2000;
        dbg_mem_re = 1;
        #1;
        $display("output[0] = 0x%08X (expected 0x41800000 = 16.0)", dbg_mem_rdata);
        if (dbg_mem_rdata != 32'h41800000) begin
            $display("  ERROR: output[0] mismatch!");
            test_passed = 0;
        end

        // output[1] should be 32.0 (sum of 16 twos times scale 1.0 times input 1.0)
        // 32.0 = 0x42000000
        dbg_mem_addr = 32'h2004;
        #1;
        $display("output[1] = 0x%08X (expected 0x42000000 = 32.0)", dbg_mem_rdata);
        if (dbg_mem_rdata != 32'h42000000) begin
            $display("  ERROR: output[1] mismatch!");
            test_passed = 0;
        end

        // Final result
        $display("");
        if (test_passed) begin
            $display("=== KERNEL TEST PASSED! ===");
        end else begin
            $display("=== KERNEL TEST FAILED! ===");
        end

        #(CLK_PERIOD * 10);
        $finish;
    end

    //=========================================================================
    // Timeout Watchdog
    //=========================================================================

    initial begin
        #(CLK_PERIOD * MAX_CYCLES * 10);
        $display("TIMEOUT!");
        $finish;
    end

endmodule
