/**
 * SMOL-32 Control Unit
 * Multi-cycle FSM: FETCH -> DECODE -> EXECUTE -> MEMORY -> WRITEBACK
 * Synthesizable design
 */

`include "smol32_defines.vh"

module smol32_control (
    input  wire        clk,
    input  wire        rst_n,

    // Current PC (for halt detection)
    input  wire [31:0] pc,

    // Decoded instruction info
    input  wire [5:0]  opcode,
    input  wire [4:0]  func,
    input  wire        is_load,
    input  wire        is_store,
    input  wire        is_alu_r,
    input  wire        is_alu_i,
    input  wire        is_branch,
    input  wire        is_jal,
    input  wire        is_jalr,
    input  wire        is_loop,
    input  wire        is_fp_load,
    input  wire        is_fp_store,
    input  wire        is_fpu,
    input  wire        is_fpu_special,
    input  wire        is_vector,
    input  wire        is_vec_arith,
    input  wire        is_vec_scalar,
    input  wire        is_vec_reduce,
    input  wire        is_vec_special,
    input  wire        is_vec_load,
    input  wire        is_vec_store,
    input  wire        is_q8,
    input  wire        is_system,

    // Branch decision from datapath
    input  wire        branch_taken,

    // Q8 MAC stall signals
    input  wire        mac_busy,
    input  wire        mac_done,

    // Vector memory stall signals
    input  wire        vec_mem_busy,
    input  wire        vec_mem_done,

    // Control outputs
    output reg  [3:0]  state,
    output reg         pc_we,
    output reg  [1:0]  pc_src,       // 0=PC+4, 1=branch, 2=jal, 3=jalr
    output reg         ir_we,        // Instruction register write enable
    output reg         regfile_we,
    output reg  [1:0]  rd_src,       // 0=ALU, 1=memory, 2=PC+4, 3=FP
    output reg  [3:0]  alu_op,
    output reg         alu_src_b,    // 0=rs2, 1=immediate
    output reg         mem_we,
    output reg         mem_re,
    output reg         fp_regfile_we,
    output reg         vec_regfile_we,
    output reg         halted
);

    // State register
    reg [3:0] next_state;

    // State machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= `STATE_FETCH;
            halted <= 1'b0;
        end else begin
            state <= next_state;
            if (next_state == `STATE_HALT)
                halted <= 1'b1;
        end
    end

    // Next state logic
    always @(*) begin
        next_state = state;

        case (state)
            `STATE_FETCH: begin
                next_state = `STATE_DECODE;
            end

            `STATE_DECODE: begin
                // Check for HALT (PC == 0 is the trap address)
                if (pc == 32'b0) begin
                    next_state = `STATE_HALT;
                end else begin
                    next_state = `STATE_EXECUTE;
                end
            end

            `STATE_EXECUTE: begin
                if (is_load || is_store || is_fp_load || is_fp_store) begin
                    next_state = `STATE_MEMORY;
                end else if (is_vec_load || is_vec_store) begin
                    // Vector load/store, go to vector memory wait state
                    next_state = `STATE_VEC_MEM;
                end else if (is_q8 && (opcode == `OP_Q8MAC) && (func == `Q8_MAC || func == `Q8_MACINC)) begin
                    // Q8MACINC/Q8MAC instruction, go to wait state
                    next_state = `STATE_MAC_WAIT;
                end else begin
                    next_state = `STATE_WRITEBACK;
                end
            end

            `STATE_MAC_WAIT: begin
                // Wait for MAC operation to complete
                if (mac_done) begin
                    next_state = `STATE_WRITEBACK;
                end else begin
                    next_state = `STATE_MAC_WAIT;
                end
            end

            `STATE_VEC_MEM: begin
                // Wait for vector memory operation to complete
                if (vec_mem_done) begin
                    next_state = `STATE_WRITEBACK;
                end else begin
                    next_state = `STATE_VEC_MEM;
                end
            end

            `STATE_MEMORY: begin
                next_state = `STATE_WRITEBACK;
            end

            `STATE_WRITEBACK: begin
                next_state = `STATE_FETCH;
            end

            `STATE_HALT: begin
                next_state = `STATE_HALT;  // Stay halted
            end

            default: begin
                next_state = `STATE_FETCH;
            end
        endcase
    end

    // Control signal generation
    always @(*) begin
        // Default values
        pc_we        = 1'b0;
        pc_src       = 2'b00;
        ir_we        = 1'b0;
        regfile_we   = 1'b0;
        rd_src       = 2'b00;
        alu_op       = `ALU_ADD;
        alu_src_b    = 1'b0;
        mem_we       = 1'b0;
        mem_re       = 1'b0;
        fp_regfile_we = 1'b0;
        vec_regfile_we = 1'b0;

        case (state)
            `STATE_FETCH: begin
                ir_we  = 1'b1;   // Capture instruction
                mem_re = 1'b1;   // Read from memory
            end

            `STATE_DECODE: begin
                // Just decode, no actions needed here
            end

            `STATE_EXECUTE: begin
                // ALU R-type
                if (is_alu_r) begin
                    case (func)
                        `FUNC_ADD:  alu_op = `ALU_ADD;
                        `FUNC_SUB:  alu_op = `ALU_SUB;
                        `FUNC_MUL:  alu_op = `ALU_MUL;
                        `FUNC_AND:  alu_op = `ALU_AND;
                        `FUNC_OR:   alu_op = `ALU_OR;
                        `FUNC_XOR:  alu_op = `ALU_XOR;
                        `FUNC_SLL:  alu_op = `ALU_SLL;
                        `FUNC_SRL:  alu_op = `ALU_SRL;
                        `FUNC_SRA:  alu_op = `ALU_SRA;
                        `FUNC_SLT:  alu_op = `ALU_SLT;
                        `FUNC_SLTU: alu_op = `ALU_SLTU;
                        default:    alu_op = `ALU_ADD;
                    endcase
                    alu_src_b = 1'b0;  // Use rs2
                end

                // ALU I-type (ADDI, shifts with immediate)
                if (is_alu_i) begin
                    if (opcode == `OP_ALUI)
                        alu_op = `ALU_ADD;
                    else if (opcode == `OP_SRLI)
                        alu_op = `ALU_SRL;
                    else if (opcode == `OP_SRAI)
                        alu_op = `ALU_SRA;
                    alu_src_b = 1'b1;  // Use immediate
                end

                // Load/Store address calculation (including vector)
                if (is_load || is_store || is_fp_load || is_fp_store || is_vec_load || is_vec_store) begin
                    alu_op    = `ALU_ADD;
                    alu_src_b = 1'b1;  // rs1 + immediate
                end

                // Branch
                if (is_branch) begin
                    alu_op    = `ALU_SUB;  // Compare rs1 - rs2
                    alu_src_b = 1'b0;
                    if (branch_taken) begin
                        pc_src = 2'b01;  // Branch target
                        pc_we  = 1'b1;
                    end
                end

                // JAL
                if (is_jal) begin
                    pc_src = 2'b10;
                    pc_we  = 1'b1;
                end

                // JALR
                if (is_jalr) begin
                    alu_op    = `ALU_ADD;  // rs1 + offset
                    alu_src_b = 1'b1;
                    pc_src    = 2'b11;
                    pc_we     = 1'b1;
                end

                // LOOP
                if (is_loop) begin
                    // Loop logic: rd--, if rd > 0 branch
                    alu_op    = `ALU_SUB;
                    alu_src_b = 1'b1;  // Will use 1 as immediate
                    if (branch_taken) begin
                        pc_src = 2'b01;
                        pc_we  = 1'b1;
                    end
                end
            end

            `STATE_MEMORY: begin
                if (is_load || is_fp_load) begin
                    mem_re = 1'b1;
                end
                if (is_store || is_fp_store) begin
                    mem_we = 1'b1;
                end
            end

            `STATE_WRITEBACK: begin
                // Update PC if not already done
                if (!is_branch && !is_jal && !is_jalr && !is_loop) begin
                    pc_we  = 1'b1;
                    pc_src = 2'b00;  // PC + 4
                end
                // Handle loop PC update when not taken
                if (is_loop && !branch_taken) begin
                    pc_we  = 1'b1;
                    pc_src = 2'b00;
                end
                // Handle branch PC update when not taken
                if (is_branch && !branch_taken) begin
                    pc_we  = 1'b1;
                    pc_src = 2'b00;  // PC + 4
                end

                // Register writeback
                if (is_alu_r || is_alu_i) begin
                    regfile_we = 1'b1;
                    rd_src     = 2'b00;  // ALU result
                end
                if (is_load) begin
                    regfile_we = 1'b1;
                    rd_src     = 2'b01;  // Memory data
                end
                if (is_jal || is_jalr) begin
                    regfile_we = 1'b1;
                    rd_src     = 2'b10;  // PC + 4
                end
                if (is_loop) begin
                    regfile_we = 1'b1;
                    rd_src     = 2'b00;  // Decremented value
                end
                if (is_fp_load) begin
                    fp_regfile_we = 1'b1;
                end
                // FPU arithmetic writeback
                if (is_fpu || is_fpu_special) begin
                    fp_regfile_we = 1'b1;
                end
                // Vector arithmetic writeback
                if (is_vec_arith || is_vec_scalar || is_vec_special) begin
                    vec_regfile_we = 1'b1;
                end
                // Vector load writeback
                if (is_vec_load) begin
                    vec_regfile_we = 1'b1;
                end
                // Vector reduction writeback (result goes to FP register)
                if (is_vec_reduce) begin
                    fp_regfile_we = 1'b1;
                end
                // Q8 ACCREAD writeback (accumulator to FP register)
                if (is_q8 && func == `Q8_ACCREAD) begin
                    fp_regfile_we = 1'b1;
                end
                // VSETVL writeback (vl to integer register)
                if (is_system && func == 5'h10) begin
                    regfile_we = 1'b1;
                    rd_src     = 2'b00;  // rd_data is overridden in core for VSETVL
                end
            end

            `STATE_HALT: begin
                // Do nothing
            end

            default: begin
                // Do nothing
            end
        endcase
    end

endmodule
