/**
 * SMOL-32 Testbench
 * Tests integer and floating-point operations
 */

`timescale 1ns/1ps

module tb_smol32_top;

    //=========================================================================
    // Parameters
    //=========================================================================

    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter MEM_SIZE = 32'h00100000;  // 1MB for testing
    parameter MAX_CYCLES = 10000;

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
    // Test Program Loading
    //=========================================================================

    // Test program with integer and FP operations:
    //
    // Integer test:
    // 0x1000: ADDI R3, R0, 10       -> R3 = 10
    // 0x1004: ADDI R4, R0, 20       -> R4 = 20
    // 0x1008: ADD R5, R3, R4        -> R5 = 30
    // 0x100C: SW R5, 0(R0)          -> mem[0] = 30
    //
    // FP load/store test:
    // 0x1010: ADDI R6, R0, 100      -> R6 = 100 (address for FP data)
    // 0x1014: LF F1, 0(R6)          -> F1 = mem[100] (load 4.0)
    // 0x1018: SF F1, 4(R6)          -> mem[104] = F1 (store back)
    //
    // FP arithmetic test:
    // 0x101C: LF F2, 8(R6)          -> F2 = mem[108] (load 2.0)
    // 0x1020: FADD F3, F1, F2       -> F3 = 4.0 + 2.0 = 6.0
    // 0x1024: FMUL F4, F1, F2       -> F4 = 4.0 * 2.0 = 8.0
    // 0x1028: SF F3, 12(R6)         -> mem[112] = F3
    // 0x102C: SF F4, 16(R6)         -> mem[116] = F4
    //
    // FP special test:
    // 0x1030: FSQRT F5, F1          -> F5 = sqrt(4.0) = 2.0
    // 0x1034: FRSQRT F6, F1         -> F6 = 1/sqrt(4.0) = 0.5
    // 0x1038: LF F7, 20(R6)         -> F7 = mem[120] = 0.0 (for FSILU test)
    // 0x103C: FSILU F8, F7          -> F8 = SiLU(0.0) = 0.0
    // 0x1040: SF F5, 24(R6)         -> mem[124] = F5
    // 0x1044: SF F6, 28(R6)         -> mem[128] = F6
    //
    // Vector test (V0, V1 pre-loaded with 1.0, 2.0 for all 16 lanes):
    // 0x1048: VADD V2, V0, V1       -> V2[i] = 1.0 + 2.0 = 3.0
    // 0x104C: VREDSUM F9, V2        -> F9 = sum of V2 = 16 * 3.0 = 48.0
    //
    // 0x1050: JALR R0, R0, 0        -> HALT

    // Encoding reference:
    // I-type: [op:6][rd:5][rs1:5][imm:16]
    // R-type: [op:6][rd:5][rs1:5][rs2:5][func:5][ext:6]
    // FP R-type: [op:6][fd:5][fs1:5][fs2:5][func:5][ext:6]
    // FSPEC:  [op:6][fd:5][fs1:5][00000][func:5][ext:6] - op=0x0C

    initial begin
        // Wait for reset
        @(negedge rst_n);
        @(posedge rst_n);

        // --- Integer Test ---
        // ADDI R3, R0, 10:  op=05, rd=3, rs1=0, imm=10
        dut.memory[32'h1000/4] = 32'h1460000A;

        // ADDI R4, R0, 20:  op=05, rd=4, rs1=0, imm=20
        dut.memory[32'h1004/4] = 32'h14800014;

        // ADD R5, R3, R4:   op=04, rd=5, rs1=3, rs2=4, func=0, ext=0
        dut.memory[32'h1008/4] = 32'h10A32000;

        // SW R5, 0(R0):     op=01, rd=5, rs1=0, imm=0
        dut.memory[32'h100C/4] = 32'h04A00000;

        // --- FP Load/Store Test ---
        // ADDI R6, R0, 100: op=05, rd=6, rs1=0, imm=100
        // Binary: 000101 00110 00000 0000000001100100 = 0x14C00064
        dut.memory[32'h1010/4] = 32'h14C00064;

        // LF F1, 0(R6):     op=02, fd=1, rs1=6, imm=0
        // Binary: 000010 00001 00110 0000000000000000 = 0x08260000
        dut.memory[32'h1014/4] = 32'h08260000;

        // SF F1, 4(R6):     op=03, fd=1, rs1=6, imm=4
        // Binary: 000011 00001 00110 0000000000000100 = 0x0C260004
        dut.memory[32'h1018/4] = 32'h0C260004;

        // --- FP Arithmetic Test ---
        // LF F2, 8(R6):     op=02, fd=2, rs1=6, imm=8
        // Binary: 000010 00010 00110 0000000000001000 = 0x08460008
        dut.memory[32'h101C/4] = 32'h08460008;

        // FADD F3, F1, F2:  op=08, fd=3, fs1=1, fs2=2, func=0 (FADD), ext=0
        // Binary: 001000 00011 00001 00010 00000 000000 = 0x20611000
        dut.memory[32'h1020/4] = 32'h20611000;

        // FMUL F4, F1, F2:  op=08, fd=4, fs1=1, fs2=2, func=2 (FMUL), ext=0
        // Binary: 001000 00100 00001 00010 00010 000000 = 0x20811080
        dut.memory[32'h1024/4] = 32'h20811080;

        // SF F3, 12(R6):    op=03, fd=3, rs1=6, imm=12
        // Binary: 000011 00011 00110 0000000000001100 = 0x0C66000C
        dut.memory[32'h1028/4] = 32'h0C66000C;

        // SF F4, 16(R6):    op=03, fd=4, rs1=6, imm=16
        // Binary: 000011 00100 00110 0000000000010000 = 0x0C860010
        dut.memory[32'h102C/4] = 32'h0C860010;

        // --- FP Special Functions Test ---
        // FSQRT F5, F1:     op=0C, fd=5, fs1=1, fs2=0, func=0 (FSQRT), ext=0
        // Binary: 001100 00101 00001 00000 00000 000000 = 0x30A10000
        dut.memory[32'h1030/4] = 32'h30A10000;

        // FRSQRT F6, F1:    op=0C, fd=6, fs1=1, fs2=0, func=1 (FRSQRT), ext=0
        // Binary: 001100 00110 00001 00000 00001 000000 = 0x30C10040
        dut.memory[32'h1034/4] = 32'h30C10040;

        // LF F7, 20(R6):    op=02, fd=7, rs1=6, imm=20
        // Binary: 000010 00111 00110 0000000000010100 = 0x08E60014
        dut.memory[32'h1038/4] = 32'h08E60014;

        // FSILU F8, F7:     op=0C, fd=8, fs1=7, fs2=0, func=7 (FSILU), ext=0
        // Binary: 001100 01000 00111 00000 00111 000000 = 0x310700C0
        dut.memory[32'h103C/4] = 32'h310701C0;

        // SF F5, 24(R6):    op=03, fd=5, rs1=6, imm=24
        // Binary: 000011 00101 00110 0000000000011000 = 0x0CA60018
        dut.memory[32'h1040/4] = 32'h0CA60018;

        // SF F6, 28(R6):    op=03, fd=6, rs1=6, imm=28
        // Binary: 000011 00110 00110 0000000000011100 = 0x0CC6001C
        dut.memory[32'h1044/4] = 32'h0CC6001C;

        // --- Vector Test ---
        // VADD V2, V0, V1:   op=10, vd=2, vs1=0, vs2=1, func=0, ext=0 (VFUNC_ADD)
        // Binary: 010000 00010 00000 00001 00000 000000 = 0x40400800
        dut.memory[32'h1048/4] = 32'h40400800;

        // VREDSUM F9, V2:    op=12, fd=9, vs1=2, vs2=0, func=0, ext=0 (VRED_SUM)
        // Binary: 010010 01001 00010 00000 00000 000000 = 0x49220000
        dut.memory[32'h104C/4] = 32'h49220000;

        // --- Q8 MAC Test ---
        // Test: Set scale from F1 (4.0), zero accumulator, read accumulator to F10
        //
        // QSETSCALE F1:     op=1C, rd=0, fs1=1, rs2=0, func=0 (Q8_SETSCALE), ext=0
        // Binary: 011100 00000 00001 00000 00000 000000 = 0x70010000
        //         [op:6 ][rd:5 ][rs1:5][rs2:5][func:5][ext:6]
        dut.memory[32'h1050/4] = 32'h70010000;

        // ACCZERO:          op=1D, rd=0, rs1=0, rs2=0, func=0 (Q8_ACCZERO), ext=0
        // Binary: 011101 00000 00000 00000 00000 000000 = 0x74000000
        dut.memory[32'h1054/4] = 32'h74000000;

        // ACCREAD F10:      op=1D, fd=10, rs1=0, rs2=0, func=1 (Q8_ACCREAD), ext=0
        // Binary: 011101 01010 00000 00000 00001 000000 = 0x75400040
        dut.memory[32'h1058/4] = 32'h75400040;

        // --- Q8MACINC Test ---
        // Test: Q8MACINC with 4 elements
        // Q8 bytes at 0x200: [1, 2, 3, 4]
        // FP32 values at 0x400: [1.0, 2.0, 3.0, 4.0]
        // Scale: 0.5 (from F6)
        // Expected: ACC = (1*0.5*1.0) + (2*0.5*2.0) + (3*0.5*3.0) + (4*0.5*4.0)
        //               = 0.5 + 2.0 + 4.5 + 8.0 = 15.0

        // ADDI R7, R0, 0x200:   op=05, rd=7, rs1=0, imm=0x200
        // Binary: 000101 00111 00000 0000001000000000 = 0x14E00200
        dut.memory[32'h105C/4] = 32'h14E00200;

        // ADDI R8, R0, 0x400:   op=05, rd=8, rs1=0, imm=0x400
        // Binary: 000101 01000 00000 0000010000000000 = 0x15000400
        dut.memory[32'h1060/4] = 32'h15000400;

        // QSETQBASE R7:     op=1C, rd=0, rs1=7, rs2=0, func=1 (Q8_SETQBASE), ext=0
        // Binary: 011100 00000 00111 00000 00001 000000 = 0x70070040
        dut.memory[32'h1064/4] = 32'h70070040;

        // QSETFBASE R8:     op=1C, rd=0, rs1=8, rs2=0, func=2 (Q8_SETFBASE), ext=0
        // Binary: 011100 00000 01000 00000 00010 000000 = 0x70080080
        dut.memory[32'h1068/4] = 32'h70080080;

        // QSETSCALE F6:     op=1C, rd=0, fs1=6, rs2=0, func=0 (Q8_SETSCALE), ext=0
        // F6 has 0.5 from rsqrt test
        // Binary: 011100 00000 00110 00000 00000 000000 = 0x70060000
        dut.memory[32'h106C/4] = 32'h70060000;

        // ACCZERO:          op=1D, rd=0, rs1=0, rs2=0, func=0 (Q8_ACCZERO), ext=0
        // Binary: 011101 00000 00000 00000 00000 000000 = 0x74000000
        dut.memory[32'h1070/4] = 32'h74000000;

        // Q8MACINC n=4:     op=1D, rd=0, rs1=0, rs2=0, func=3 (Q8_MACINC), ext=4
        // Binary: 011101 00000 00000 00000 00011 000100 = 0x740000C4
        dut.memory[32'h1074/4] = 32'h740000C4;

        // ACCREAD F11:      op=1D, fd=11, rs1=0, rs2=0, func=1 (Q8_ACCREAD), ext=0
        // Binary: 011101 01011 00000 00000 00001 000000 = 0x75600040
        dut.memory[32'h1078/4] = 32'h75600040;

        // --- Vector Load/Store Test (LVF, SVF) ---
        // Test: Load 16 FP32 values from memory into V3, store V3 to different location
        // Test data at 0x600: 16 FP32 values (1.0 to 16.0)
        // Store to 0x640 (64 bytes = 16 floats offset)

        // ADDI R9, R0, 0x600:  op=05, rd=9, rs1=0, imm=0x600
        // Binary: 000101 01001 00000 0000011000000000 = 0x15200600
        dut.memory[32'h107C/4] = 32'h15200600;

        // LVF V3, 0(R9):       op=19, vd=3, rs1=9, imm=0
        // Binary: 011001 00011 01001 0000000000000000 = 0x64690000
        dut.memory[32'h1080/4] = 32'h64690000;

        // SVF V3, 64(R9):      op=1A, vs=3, rs1=9, imm=64
        // Binary: 011010 00011 01001 0000000001000000 = 0x68690040
        dut.memory[32'h1084/4] = 32'h68690040;

        // JALR R0, R0, 0:   op=32, rd=0, rs1=0, imm=0 (jump to 0 = halt)
        dut.memory[32'h1088/4] = 32'hC8000000;

        // Test data at address 100: IEEE 754 for 4.0 (for sqrt test)
        // 4.0 = 0x40800000
        dut.memory[32'd100/4] = 32'h40800000;

        // Test data at address 108: IEEE 754 for 2.0
        // 2.0 = 0x40000000
        dut.memory[32'd108/4] = 32'h40000000;

        // Test data at address 120: IEEE 754 for 0.0 (for SiLU test)
        // 0.0 = 0x00000000
        dut.memory[32'd120/4] = 32'h00000000;

        // Place HALT trap at address 0
        dut.memory[0] = 32'hE0000000;

        // Pre-load vector registers V0 and V1 for vector test
        // V0 = all 1.0 (0x3F800000), V1 = all 2.0 (0x40000000)
        dut.core.vec_regfile.vregs[0] = {16{32'h3F800000}};  // V0 = 1.0 × 16
        dut.core.vec_regfile.vregs[1] = {16{32'h40000000}};  // V1 = 2.0 × 16

        // Q8MACINC test data
        // Q8 bytes at 0x200: [1, 2, 3, 4] packed as little-endian word
        // Byte 0x200=1, 0x201=2, 0x202=3, 0x203=4
        dut.memory[32'h200/4] = 32'h04030201;

        // FP32 values at 0x400: 1.0, 2.0, 3.0, 4.0
        dut.memory[32'h400/4] = 32'h3F800000;  // 1.0
        dut.memory[32'h404/4] = 32'h40000000;  // 2.0
        dut.memory[32'h408/4] = 32'h40400000;  // 3.0
        dut.memory[32'h40C/4] = 32'h40800000;  // 4.0

        // Vector load/store test data at 0x600: 16 FP32 values (1.0 to 16.0)
        dut.memory[32'h600/4] = 32'h3F800000;   // 1.0
        dut.memory[32'h604/4] = 32'h40000000;   // 2.0
        dut.memory[32'h608/4] = 32'h40400000;   // 3.0
        dut.memory[32'h60C/4] = 32'h40800000;   // 4.0
        dut.memory[32'h610/4] = 32'h40A00000;   // 5.0
        dut.memory[32'h614/4] = 32'h40C00000;   // 6.0
        dut.memory[32'h618/4] = 32'h40E00000;   // 7.0
        dut.memory[32'h61C/4] = 32'h41000000;   // 8.0
        dut.memory[32'h620/4] = 32'h41100000;   // 9.0
        dut.memory[32'h624/4] = 32'h41200000;   // 10.0
        dut.memory[32'h628/4] = 32'h41300000;   // 11.0
        dut.memory[32'h62C/4] = 32'h41400000;   // 12.0
        dut.memory[32'h630/4] = 32'h41500000;   // 13.0
        dut.memory[32'h634/4] = 32'h41600000;   // 14.0
        dut.memory[32'h638/4] = 32'h41700000;   // 15.0
        dut.memory[32'h63C/4] = 32'h41800000;   // 16.0
    end

    //=========================================================================
    // Test Execution
    //=========================================================================

    integer cycle_count;
    integer insn_count;
    integer test_passed;

    initial begin
        $dumpfile("smol32_test.vcd");
        $dumpvars(0, tb_smol32_top);

        // Initialize
        rst_n = 0;
        dbg_mem_re = 0;
        dbg_mem_addr = 0;
        cycle_count = 0;
        insn_count = 0;
        test_passed = 1;

        // Reset
        #(CLK_PERIOD * 5);
        rst_n = 1;
        #(CLK_PERIOD * 2);

        $display("=== SMOL-32 Verilog Testbench ===");
        $display("Starting execution at PC=0x%08X", pc);

        // Run until halted or timeout
        while (!halted && cycle_count < MAX_CYCLES) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;

            // Count instructions (when we return to FETCH state)
            if (state == 4'h0) begin
                insn_count = insn_count + 1;
            end

            // Debug output
            if (state == 4'h4) begin  // WRITEBACK
                $display("Cycle %0d: PC=0x%08X", cycle_count, pc);
            end
        end

        // Check results
        $display("");
        $display("=== Execution Complete ===");
        $display("Cycles: %0d", cycle_count);
        $display("Instructions: %0d", insn_count);
        $display("Halted: %0d", halted);

        // --- Integer Test Check ---
        $display("");
        $display("=== Integer Test ===");
        dbg_mem_addr = 0;
        dbg_mem_re = 1;
        #1;
        $display("mem[0] = %0d (expected: 30)", dbg_mem_rdata);
        if (dbg_mem_rdata != 32'd30) test_passed = 0;

        $display("R3 = %0d (expected: 10)", dut.core.regfile.regs[3]);
        if (dut.core.regfile.regs[3] != 32'd10) test_passed = 0;

        $display("R4 = %0d (expected: 20)", dut.core.regfile.regs[4]);
        if (dut.core.regfile.regs[4] != 32'd20) test_passed = 0;

        $display("R5 = %0d (expected: 30)", dut.core.regfile.regs[5]);
        if (dut.core.regfile.regs[5] != 32'd30) test_passed = 0;

        // --- FP Load/Store Test Check ---
        $display("");
        $display("=== FP Load/Store Test ===");
        $display("F1 = 0x%08X (expected: 0x40800000 = 4.0)", dut.core.fp_regfile.fregs[1]);
        if (dut.core.fp_regfile.fregs[1] != 32'h40800000) test_passed = 0;

        dbg_mem_addr = 104;
        #1;
        $display("mem[104] = 0x%08X (expected: 0x40800000)", dbg_mem_rdata);
        if (dbg_mem_rdata != 32'h40800000) test_passed = 0;

        // --- FP Arithmetic Test Check ---
        $display("");
        $display("=== FP Arithmetic Test ===");
        $display("F2 = 0x%08X (expected: 0x40000000 = 2.0)", dut.core.fp_regfile.fregs[2]);
        if (dut.core.fp_regfile.fregs[2] != 32'h40000000) test_passed = 0;

        // F3 = F1 + F2 = 4.0 + 2.0 = 6.0 = 0x40C00000
        $display("F3 = 0x%08X (FADD result, expected 0x40C00000 = 6.0)", dut.core.fp_regfile.fregs[3]);
        if (dut.core.fp_regfile.fregs[3] != 32'h40C00000) begin
            $display("  ERROR: FADD result mismatch!");
            test_passed = 0;
        end

        // F4 = F1 * F2 = 4.0 * 2.0 = 8.0 = 0x41000000
        $display("F4 = 0x%08X (FMUL result, expected 0x41000000 = 8.0)", dut.core.fp_regfile.fregs[4]);
        if (dut.core.fp_regfile.fregs[4] != 32'h41000000) begin
            $display("  ERROR: FMUL result mismatch!");
            test_passed = 0;
        end

        // Verify stored results
        dbg_mem_addr = 112;
        #1;
        $display("mem[112] = 0x%08X (stored F3)", dbg_mem_rdata);

        dbg_mem_addr = 116;
        #1;
        $display("mem[116] = 0x%08X (stored F4)", dbg_mem_rdata);

        // --- FP Special Functions Test Check ---
        $display("");
        $display("=== FP Special Functions Test ===");

        // F5 = sqrt(4.0) = 2.0 = 0x40000000
        $display("F5 = 0x%08X (FSQRT result, expected 0x40000000 = 2.0)", dut.core.fp_regfile.fregs[5]);
        if (dut.core.fp_regfile.fregs[5] != 32'h40000000) begin
            $display("  ERROR: FSQRT result mismatch!");
            test_passed = 0;
        end

        // F6 = 1/sqrt(4.0) = 0.5 = 0x3F000000
        $display("F6 = 0x%08X (FRSQRT result, expected 0x3F000000 = 0.5)", dut.core.fp_regfile.fregs[6]);
        if (dut.core.fp_regfile.fregs[6] != 32'h3F000000) begin
            $display("  ERROR: FRSQRT result mismatch!");
            test_passed = 0;
        end

        // F8 = SiLU(0.0) = 0.0 = 0x00000000
        $display("F8 = 0x%08X (FSILU result, expected 0x00000000 = 0.0)", dut.core.fp_regfile.fregs[8]);
        if (dut.core.fp_regfile.fregs[8] != 32'h00000000) begin
            $display("  ERROR: FSILU result mismatch!");
            test_passed = 0;
        end

        // Verify stored sqrt and rsqrt results
        dbg_mem_addr = 124;
        #1;
        $display("mem[124] = 0x%08X (stored F5 = sqrt(4.0))", dbg_mem_rdata);

        dbg_mem_addr = 128;
        #1;
        $display("mem[128] = 0x%08X (stored F6 = rsqrt(4.0))", dbg_mem_rdata);

        // --- Vector Test Check ---
        $display("");
        $display("=== Vector Test ===");

        // Check V2[0] = V0[0] + V1[0] = 1.0 + 2.0 = 3.0 = 0x40400000
        $display("V2[0] = 0x%08X (expected 0x40400000 = 3.0)", dut.core.vec_regfile.vregs[2][31:0]);
        if (dut.core.vec_regfile.vregs[2][31:0] != 32'h40400000) begin
            $display("  ERROR: VADD result mismatch!");
            test_passed = 0;
        end

        // F9 = VREDSUM(V2) = 16 × 3.0 = 48.0 = 0x42400000
        $display("F9 = 0x%08X (VREDSUM result, expected 0x42400000 = 48.0)", dut.core.fp_regfile.fregs[9]);
        if (dut.core.fp_regfile.fregs[9] != 32'h42400000) begin
            $display("  ERROR: VREDSUM result mismatch!");
            test_passed = 0;
        end

        // --- Q8 MAC Test Check ---
        $display("");
        $display("=== Q8 MAC Test ===");

        // Note: Scale is overwritten by Q8MACINC test to F6 (0.5)
        $display("Q8 SCALE = 0x%08X (expected 0x3F000000 = 0.5)", dut.core.q8mac.scale_reg);
        if (dut.core.q8mac.scale_reg != 32'h3F000000) begin
            $display("  ERROR: QSETSCALE result mismatch!");
            test_passed = 0;
        end

        // F10 = ACCREAD after ACCZERO = 0.0 = 0x00000000
        $display("F10 = 0x%08X (ACCREAD after ACCZERO, expected 0x00000000 = 0.0)", dut.core.fp_regfile.fregs[10]);
        if (dut.core.fp_regfile.fregs[10] != 32'h00000000) begin
            $display("  ERROR: ACCREAD result mismatch!");
            test_passed = 0;
        end

        // --- Q8MACINC Test Check ---
        $display("");
        $display("=== Q8MACINC Test ===");

        // Check QBASE was set and auto-incremented by 4
        $display("Q8 QBASE = 0x%08X (expected 0x00000204 after MACINC)", dut.core.q8mac.qbase_reg);
        if (dut.core.q8mac.qbase_reg != 32'h00000204) begin
            $display("  ERROR: QBASE auto-increment mismatch!");
            test_passed = 0;
        end

        // Check FBASE was set and auto-incremented by 16 (4 * 4)
        $display("Q8 FBASE = 0x%08X (expected 0x00000410 after MACINC)", dut.core.q8mac.fbase_reg);
        if (dut.core.q8mac.fbase_reg != 32'h00000410) begin
            $display("  ERROR: FBASE auto-increment mismatch!");
            test_passed = 0;
        end

        // F11 = ACCREAD after Q8MACINC = 15.0 = 0x41700000
        // Calculation: (1*0.5*1.0) + (2*0.5*2.0) + (3*0.5*3.0) + (4*0.5*4.0)
        //            = 0.5 + 2.0 + 4.5 + 8.0 = 15.0
        $display("F11 = 0x%08X (Q8MACINC result, expected 0x41700000 = 15.0)", dut.core.fp_regfile.fregs[11]);
        if (dut.core.fp_regfile.fregs[11] != 32'h41700000) begin
            $display("  ERROR: Q8MACINC result mismatch!");
            test_passed = 0;
        end

        // --- Vector Load/Store (LVF, SVF) Test Check ---
        $display("");
        $display("=== Vector Load/Store Test ===");

        // Check V3[0] = loaded 1.0 = 0x3F800000
        $display("V3[0] = 0x%08X (LVF result, expected 0x3F800000 = 1.0)", dut.core.vec_regfile.vregs[3][31:0]);
        if (dut.core.vec_regfile.vregs[3][31:0] != 32'h3F800000) begin
            $display("  ERROR: LVF V3[0] mismatch!");
            test_passed = 0;
        end

        // Check V3[7] = loaded 8.0 = 0x41000000
        $display("V3[7] = 0x%08X (LVF result, expected 0x41000000 = 8.0)", dut.core.vec_regfile.vregs[3][255:224]);
        if (dut.core.vec_regfile.vregs[3][255:224] != 32'h41000000) begin
            $display("  ERROR: LVF V3[7] mismatch!");
            test_passed = 0;
        end

        // Check V3[15] = loaded 16.0 = 0x41800000
        $display("V3[15] = 0x%08X (LVF result, expected 0x41800000 = 16.0)", dut.core.vec_regfile.vregs[3][511:480]);
        if (dut.core.vec_regfile.vregs[3][511:480] != 32'h41800000) begin
            $display("  ERROR: LVF V3[15] mismatch!");
            test_passed = 0;
        end

        // Check SVF stored to 0x640
        dbg_mem_addr = 32'h640;
        #1;
        $display("mem[0x640] = 0x%08X (SVF result, expected 0x3F800000 = 1.0)", dbg_mem_rdata);
        if (dbg_mem_rdata != 32'h3F800000) begin
            $display("  ERROR: SVF mem[0x640] mismatch!");
            test_passed = 0;
        end

        dbg_mem_addr = 32'h65C;
        #1;
        $display("mem[0x65C] = 0x%08X (SVF result, expected 0x41000000 = 8.0)", dbg_mem_rdata);
        if (dbg_mem_rdata != 32'h41000000) begin
            $display("  ERROR: SVF mem[0x65C] mismatch!");
            test_passed = 0;
        end

        dbg_mem_addr = 32'h67C;
        #1;
        $display("mem[0x67C] = 0x%08X (SVF result, expected 0x41800000 = 16.0)", dbg_mem_rdata);
        if (dbg_mem_rdata != 32'h41800000) begin
            $display("  ERROR: SVF mem[0x67C] mismatch!");
            test_passed = 0;
        end

        // Final result
        $display("");
        if (test_passed) begin
            $display("=== ALL TESTS PASSED! ===");
        end else begin
            $display("=== SOME TESTS FAILED! ===");
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
