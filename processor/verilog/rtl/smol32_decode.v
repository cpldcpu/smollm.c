/**
 * SMOL-32 Instruction Decoder
 * Decodes 32-bit instruction into control signals
 * Synthesizable design
 */

`include "smol32_defines.vh"

module smol32_decode (
    input  wire [31:0] insn,        // Instruction word

    // Decoded fields
    output wire [5:0]  opcode,
    output wire [4:0]  rd,
    output wire [4:0]  rs1,
    output wire [4:0]  rs2,
    output wire [4:0]  func,
    output wire [5:0]  ext,
    output wire [15:0] imm16,
    output wire [31:0] imm_sext,    // Sign-extended immediate

    // Branch fields
    output wire [2:0]  br_cond,
    output wire [12:0] br_offset,
    output wire [31:0] br_offset_sext,

    // V-type (vector) instruction fields
    output wire [2:0]  vd,       // Vector dest [25:23]
    output wire [2:0]  vs1,      // Vector src1 [22:20]
    output wire [2:0]  vs2,      // Vector src2 [19:17]
    output wire [2:0]  vfunc,    // Vector function [16:14]
    output wire [13:0] vext,     // Vector extension [13:0]

    // Instruction type flags
    output wire        is_load,
    output wire        is_store,
    output wire        is_alu_r,
    output wire        is_alu_i,
    output wire        is_branch,
    output wire        is_jal,
    output wire        is_jalr,
    output wire        is_loop,
    output wire        is_fp_load,
    output wire        is_fp_store,
    output wire        is_fpu,
    output wire        is_fpu_special,
    output wire        is_vector,
    output wire        is_vec_arith,
    output wire        is_vec_scalar,
    output wire        is_vec_reduce,
    output wire        is_vec_special,
    output wire        is_vec_load,
    output wire        is_vec_store,
    output wire        is_q8,
    output wire        is_system
);

    // Extract fields
    assign opcode = insn[`OPCODE_HI:`OPCODE_LO];
    assign rd     = insn[`RD_HI:`RD_LO];
    assign rs1    = insn[`RS1_HI:`RS1_LO];
    assign rs2    = insn[`RS2_HI:`RS2_LO];
    assign func   = insn[`FUNC_HI:`FUNC_LO];
    assign ext    = insn[`EXT_HI:`EXT_LO];
    assign imm16  = insn[`IMM_HI:`IMM_LO];

    // Sign-extend 16-bit immediate
    assign imm_sext = {{16{imm16[15]}}, imm16};

    // Branch fields
    assign br_cond   = insn[`BR_COND_HI:`BR_COND_LO];
    assign br_offset = insn[`BR_OFF_HI:`BR_OFF_LO];
    // Sign-extend 13-bit branch offset and shift left by 2
    assign br_offset_sext = {{17{br_offset[12]}}, br_offset, 2'b00};

    // V-type (vector) fields - different layout from R-type
    assign vd    = insn[25:23];
    assign vs1   = insn[22:20];
    assign vs2   = insn[19:17];
    assign vfunc = insn[16:14];
    assign vext  = insn[13:0];

    // Instruction type decoding
    assign is_load      = (opcode == `OP_LW);
    assign is_store     = (opcode == `OP_SW);
    assign is_fp_load   = (opcode == `OP_LF);
    assign is_fp_store  = (opcode == `OP_SF);

    assign is_alu_r     = (opcode == `OP_ALU);
    assign is_alu_i     = (opcode == `OP_ALUI) ||
                          (opcode == `OP_SRLI) ||
                          (opcode == `OP_SRAI);

    assign is_branch    = (opcode == `OP_BRANCH);
    assign is_jal       = (opcode == `OP_JAL);
    assign is_jalr      = (opcode == `OP_JALR);
    assign is_loop      = (opcode == `OP_LOOP);

    assign is_fpu       = (opcode == `OP_FPU);
    assign is_fpu_special = (opcode == `OP_FSPEC);

    assign is_vector    = (opcode == `OP_VARITH) ||
                          (opcode == `OP_VSCALAR) ||
                          (opcode == `OP_VRED) ||
                          (opcode == `OP_VSPEC) ||
                          (opcode == `OP_VLOAD) ||
                          (opcode == `OP_VSTORE);

    // More specific vector flags
    assign is_vec_arith   = (opcode == `OP_VARITH);
    assign is_vec_scalar  = (opcode == `OP_VSCALAR);
    assign is_vec_reduce  = (opcode == `OP_VRED);
    assign is_vec_special = (opcode == `OP_VSPEC);
    assign is_vec_load    = (opcode == `OP_VLOAD);
    assign is_vec_store   = (opcode == `OP_VSTORE);

    assign is_q8        = (opcode == `OP_Q8SET) ||
                          (opcode == `OP_Q8MAC) ||
                          (opcode == `OP_VQ8);

    assign is_system    = (opcode == `OP_SYSTEM) ||
                          (opcode == `OP_CSR) ||
                          (opcode == `OP_FENCE);

endmodule
