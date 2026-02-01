/**
 * SMOL-32 Core
 * Integrates: Register files (int + FP), ALU, FPU, Decoder, Control Unit
 * Multi-cycle implementation
 * Synthesizable design
 */

`include "smol32_defines.vh"

module smol32_core (
    input  wire        clk,
    input  wire        rst_n,

    // Memory interface
    output wire [31:0] mem_addr,
    output wire [31:0] mem_wdata,
    input  wire [31:0] mem_rdata,
    output wire        mem_we,
    output wire        mem_re,

    // Status
    output wire        halted,
    output wire [31:0] pc_out,
    output wire [3:0]  state_out
);

    //=========================================================================
    // Internal Registers
    //=========================================================================

    reg [31:0] pc;           // Program counter
    reg [31:0] ir;           // Instruction register
    reg [31:0] alu_out_reg;  // ALU output register
    reg [31:0] mem_data_reg; // Memory data register

    //=========================================================================
    // Decoder Signals
    //=========================================================================

    wire [5:0]  opcode;
    wire [4:0]  rd, rs1, rs2, func;
    wire [5:0]  ext;
    wire [15:0] imm16;
    wire [31:0] imm_sext;
    wire [2:0]  br_cond;
    wire [12:0] br_offset;
    wire [31:0] br_offset_sext;

    // V-type (vector) instruction fields
    wire [2:0]  vd_field;    // Vector dest from V-type encoding
    wire [2:0]  vs1_field;   // Vector src1 from V-type encoding
    wire [2:0]  vs2_field;   // Vector src2 from V-type encoding
    wire [2:0]  vfunc;       // Vector function
    wire [13:0] vext;        // Vector extension

    wire is_load, is_store, is_alu_r, is_alu_i;
    wire is_branch, is_jal, is_jalr, is_loop;
    wire is_fp_load, is_fp_store, is_fpu, is_fpu_special;
    wire is_vector, is_q8, is_system;
    wire is_vec_arith, is_vec_scalar, is_vec_reduce, is_vec_special;
    wire is_vec_load, is_vec_store;

    //=========================================================================
    // Control Signals
    //=========================================================================

    wire [3:0]  state;
    wire        pc_we;
    wire [1:0]  pc_src;
    wire        ir_we;
    wire        regfile_we;
    wire [1:0]  rd_src;
    wire [3:0]  alu_op;
    wire        alu_src_b;
    wire        ctrl_mem_we, ctrl_mem_re;
    wire        fp_regfile_we;

    //=========================================================================
    // Register File Signals
    //=========================================================================

    wire [31:0] rs1_data, rs2_data;
    reg  [31:0] rd_data;
    wire        rd_we_final;

    //=========================================================================
    // FP Register File Signals
    //=========================================================================

    wire [31:0] fs1_data, fs2_data;
    reg  [31:0] fd_data;
    wire        fd_we_final;

    //=========================================================================
    // ALU Signals
    //=========================================================================

    wire [31:0] alu_a, alu_b;
    wire [31:0] alu_result;
    wire        alu_zero, alu_negative, alu_overflow;

    //=========================================================================
    // FPU Signals
    //=========================================================================

    wire [31:0] fpu_result;
    wire        fpu_zero, fpu_negative, fpu_nan, fpu_inf;
    reg  [31:0] fpu_out_reg;  // FPU output register

    //=========================================================================
    // Vector Signals
    //=========================================================================

    wire [511:0] vs1_data, vs2_data;
    wire [511:0] vec_result;
    wire [31:0]  vec_reduce_result;
    reg  [511:0] vec_out_reg;
    reg  [31:0]  vec_reduce_reg;
    wire [4:0]   vl;               // Vector length (from vector regfile)
    wire         vec_regfile_we;

    // VSETVL detection: SYSTEM opcode, func=0x10
    wire is_vsetvl = (opcode == `OP_SYSTEM) && (func == 5'h10);

    // VSETVL: vl_new = min(rs1, 16)
    wire [4:0] vl_new = (rs1_data[4:0] > 5'd16) ? 5'd16 :
                        (rs1_data[4:0] == 5'd0) ? 5'd16 : rs1_data[4:0];
    wire vl_we = (state == `STATE_EXECUTE) && is_vsetvl;

    //=========================================================================
    // Q8 MAC Signals
    //=========================================================================

    wire [31:0] q8_acc_out;
    wire [31:0] q8_qbase_out;
    wire [31:0] q8_fbase_out;
    wire [31:0] q8_scale_out;
    wire        q8_mac_busy;
    wire        q8_mac_done;
    wire        is_q8set = (opcode == `OP_Q8SET);
    wire        is_q8mac_op = (opcode == `OP_Q8MAC);
    wire [31:0] q8_mem_addr;
    wire        q8_mem_re;

    //=========================================================================
    // Vector Memory Signals
    //=========================================================================

    wire [31:0]  vec_mem_addr;
    wire [31:0]  vec_mem_wdata;
    wire         vec_mem_we;
    wire         vec_mem_re;
    wire [511:0] vec_load_data;
    wire         vec_mem_busy;
    wire         vec_mem_done;

    //=========================================================================
    // Branch Logic
    //=========================================================================

    reg branch_taken;

    always @(*) begin
        branch_taken = 1'b0;
        if (is_branch) begin
            case (br_cond)
                `BR_EQ:  branch_taken = (rs1_data == rs2_data);
                `BR_NE:  branch_taken = (rs1_data != rs2_data);
                `BR_LT:  branch_taken = ($signed(rs1_data) < $signed(rs2_data));
                `BR_GE:  branch_taken = ($signed(rs1_data) >= $signed(rs2_data));
                `BR_LTU: branch_taken = (rs1_data < rs2_data);
                `BR_GEU: branch_taken = (rs1_data >= rs2_data);
                `BR_GTZ: branch_taken = ($signed(rs1_data) > 0);
                `BR_LEZ: branch_taken = ($signed(rs1_data) <= 0);
                default: branch_taken = 1'b0;
            endcase
        end
        // LOOP: branch if (rd - 1) > 0, i.e., rd > 1
        if (is_loop) begin
            branch_taken = ($signed(rs1_data) > 1);
        end
    end

    //=========================================================================
    // Instruction Decoder
    //=========================================================================

    smol32_decode decoder (
        .insn           (ir),
        .opcode         (opcode),
        .rd             (rd),
        .rs1            (rs1),
        .rs2            (rs2),
        .func           (func),
        .ext            (ext),
        .imm16          (imm16),
        .imm_sext       (imm_sext),
        .br_cond        (br_cond),
        .br_offset      (br_offset),
        .br_offset_sext (br_offset_sext),
        .vd             (vd_field),
        .vs1            (vs1_field),
        .vs2            (vs2_field),
        .vfunc          (vfunc),
        .vext           (vext),
        .is_load        (is_load),
        .is_store       (is_store),
        .is_alu_r       (is_alu_r),
        .is_alu_i       (is_alu_i),
        .is_branch      (is_branch),
        .is_jal         (is_jal),
        .is_jalr        (is_jalr),
        .is_loop        (is_loop),
        .is_fp_load     (is_fp_load),
        .is_fp_store    (is_fp_store),
        .is_fpu         (is_fpu),
        .is_fpu_special (is_fpu_special),
        .is_vector      (is_vector),
        .is_vec_arith   (is_vec_arith),
        .is_vec_scalar  (is_vec_scalar),
        .is_vec_reduce  (is_vec_reduce),
        .is_vec_special (is_vec_special),
        .is_vec_load    (is_vec_load),
        .is_vec_store   (is_vec_store),
        .is_q8          (is_q8),
        .is_system      (is_system)
    );

    //=========================================================================
    // Control Unit
    //=========================================================================

    smol32_control control (
        .clk            (clk),
        .rst_n          (rst_n),
        .pc             (pc),
        .opcode         (opcode),
        .func           (func),
        .is_load        (is_load),
        .is_store       (is_store),
        .is_alu_r       (is_alu_r),
        .is_alu_i       (is_alu_i),
        .is_branch      (is_branch),
        .is_jal         (is_jal),
        .is_jalr        (is_jalr),
        .is_loop        (is_loop),
        .is_fp_load     (is_fp_load),
        .is_fp_store    (is_fp_store),
        .is_fpu         (is_fpu),
        .is_fpu_special (is_fpu_special),
        .is_vector      (is_vector),
        .is_vec_arith   (is_vec_arith),
        .is_vec_scalar  (is_vec_scalar),
        .is_vec_reduce  (is_vec_reduce),
        .is_vec_special (is_vec_special),
        .is_vec_load    (is_vec_load),
        .is_vec_store   (is_vec_store),
        .is_q8          (is_q8),
        .is_system      (is_system),
        .branch_taken   (branch_taken),
        .mac_busy       (q8_mac_busy),
        .mac_done       (q8_mac_done),
        .vec_mem_busy   (vec_mem_busy),
        .vec_mem_done   (vec_mem_done),
        .state          (state),
        .pc_we          (pc_we),
        .pc_src         (pc_src),
        .ir_we          (ir_we),
        .regfile_we     (regfile_we),
        .rd_src         (rd_src),
        .alu_op         (alu_op),
        .alu_src_b      (alu_src_b),
        .mem_we         (ctrl_mem_we),
        .mem_re         (ctrl_mem_re),
        .fp_regfile_we  (fp_regfile_we),
        .vec_regfile_we (vec_regfile_we),
        .halted         (halted)
    );

    //=========================================================================
    // Register File
    //=========================================================================

    // For LOOP and BRANCH instructions, first operand is from rd field
    // (Encoding: branch uses rd for first compare, rs1 for second compare)
    wire [4:0] rf_rs1_addr = (is_loop || is_branch) ? rd : rs1;

    // For store instructions, rs2 port reads the register to store (rd field)
    // For BRANCH instructions, rs2 port reads rs1 field (second compare operand)
    // For other instructions, rs2 port reads rs2 field
    wire [4:0] rf_rs2_addr = (is_store || is_fp_store) ? rd : (is_branch ? rs1 : rs2);

    smol32_regfile regfile (
        .clk      (clk),
        .rst_n    (rst_n),
        .rs1_addr (rf_rs1_addr),
        .rs1_data (rs1_data),
        .rs2_addr (rf_rs2_addr),
        .rs2_data (rs2_data),
        .rd_addr  (rd),
        .rd_data  (rd_data),
        .rd_we    (rd_we_final)
    );

    // Don't write to R0
    assign rd_we_final = regfile_we && (rd != 5'b0);

    //=========================================================================
    // FP Register File
    //=========================================================================

    // For FP store, read the FP register to store (rd field = fd)
    wire [4:0] fp_fs2_addr = is_fp_store ? rd : rs2;

    smol32_regfile_fp fp_regfile (
        .clk      (clk),
        .rst_n    (rst_n),
        .fs1_addr (rs1),         // fs1 uses rs1 field
        .fs1_data (fs1_data),
        .fs2_addr (fp_fs2_addr), // fs2 or fd for store
        .fs2_data (fs2_data),
        .fd_addr  (rd),          // fd uses rd field
        .fd_data  (fd_data),
        .fd_we    (fd_we_final)
    );

    // Don't write to F0
    assign fd_we_final = fp_regfile_we && (rd != 5'b0);

    //=========================================================================
    // ALU
    //=========================================================================

    // ALU operand selection
    assign alu_a = rs1_data;

    // ALU operand B: register or immediate
    // For LOOP: subtract 1
    // For shifts with ext field: use ext[4:0] as shift amount
    wire [31:0] shift_amount = {27'b0, ext[4:0]};
    assign alu_b = is_loop ? 32'd1 :
                   (is_alu_r && (func == `FUNC_SLL || func == `FUNC_SRL || func == `FUNC_SRA)) ? shift_amount :
                   alu_src_b ? imm_sext : rs2_data;

    smol32_alu alu (
        .a        (alu_a),
        .b        (alu_b),
        .op       (alu_op),
        .result   (alu_result),
        .zero     (alu_zero),
        .negative (alu_negative),
        .overflow (alu_overflow)
    );

    //=========================================================================
    // FPU
    //=========================================================================

    smol32_fpu fpu (
        .a          (fs1_data),       // FP operand from fs1
        .b          (fs2_data),       // FP operand from fs2
        .int_a      (rs1_data),       // Integer operand for FCVT.S.W
        .op         (func),           // FPU operation from func field
        .is_special (is_fpu_special), // 1 for FSPEC operations
        .result     (fpu_result),
        .zero       (fpu_zero),
        .negative   (fpu_negative),
        .nan_out    (fpu_nan),
        .inf_out    (fpu_inf)
    );

    //=========================================================================
    // Vector Register File
    //=========================================================================

    // Encoding types:
    // - VARITH (0x10): V-type, uses vd_field, vs1_field, vs2_field, vfunc
    // - VSCALAR (0x11): R-type, uses rd[2:0]=vd, rs1[2:0]=vs1, rs2=fs, func
    // - VRED (0x12): R-type, uses rs1[2:0]=vs, func (fd in rd)
    // - VSPEC (0x18): V-type, uses vd_field, vs1_field, vfunc
    // - VLOAD/VSTORE: I-type, uses rd[2:0]=vd/vs, rs1=base, imm=stride
    wire is_vec_vtype = is_vec_arith || is_vec_special;  // V-type encoding
    wire is_vec_rtype = is_vec_scalar || is_vec_reduce;  // R-type encoding

    wire [2:0] vd_addr  = is_vec_vtype ? vd_field : rd[2:0];
    wire [2:0] vs1_addr = is_vec_vtype ? vs1_field :
                          (is_vec_store ? rd[2:0] : rs1[2:0]);
    // For SVF (vector store), source vector is in rd[2:0] (I-type encoding)
    wire [2:0] vs2_addr = is_vec_vtype ? vs2_field :
                          (is_vec_store ? rd[2:0] : rs2[2:0]);

    smol32_regfile_vec vec_regfile (
        .clk      (clk),
        .rst_n    (rst_n),
        .vs1_addr (vs1_addr),
        .vs1_data (vs1_data),
        .vs2_addr (vs2_addr),
        .vs2_data (vs2_data),
        .vd_addr  (vd_addr),
        .vd_data  (vec_out_reg),
        .vd_we    (vec_regfile_we),
        .vl_in    (vl_new),          // New vl from VSETVL
        .vl_we    (vl_we),           // Write enable from VSETVL
        .vl_out   (vl)
    );

    //=========================================================================
    // Vector Unit
    //=========================================================================

    // Vector operation: use vfunc for V-type, func[2:0] for R-type
    wire [2:0] vec_op = is_vec_vtype ? vfunc : func[2:0];

    smol32_vecunit vecunit (
        .vs1        (vs1_data),
        .vs2        (vs2_data),
        .fs         (fs2_data),        // Scalar operand from F[rs2] (per emulator)
        .vl         (vl),
        .op         (vec_op),
        .is_scalar  (is_vec_scalar),
        .is_reduce  (is_vec_reduce),
        .is_special (is_vec_special),
        .vd         (vec_result),
        .fd         (vec_reduce_result)
    );

    //=========================================================================
    // Q8 MAC Unit
    //=========================================================================

    smol32_q8mac q8mac (
        .clk         (clk),
        .rst_n       (rst_n),
        .func        (func),
        .is_q8set    (is_q8set),
        .is_q8mac    (is_q8mac_op),
        .execute     (state == `STATE_EXECUTE),
        .rs1_data    (rs1_data),
        .fs1_data    (fs1_data),
        .ext         (ext),
        .q8_mem_addr (q8_mem_addr),
        .q8_mem_rdata(mem_rdata),     // Share memory data bus
        .q8_mem_re   (q8_mem_re),
        .acc_out     (q8_acc_out),
        .qbase_out   (q8_qbase_out),
        .fbase_out   (q8_fbase_out),
        .scale_out   (q8_scale_out),
        .mac_busy    (q8_mac_busy),
        .mac_done    (q8_mac_done)
    );

    //=========================================================================
    // Vector Memory Unit
    //=========================================================================

    smol32_vecmem vecmem (
        .clk          (clk),
        .rst_n        (rst_n),
        .is_vec_load  (is_vec_load),
        .is_vec_store (is_vec_store),
        .execute      (state == `STATE_EXECUTE),
        .vl           (vl),
        .base_addr    (rs1_data),      // Base address is rs1 only (NOT rs1+imm)
        .stride       (imm16),         // Stride between elements from immediate
        .vs_data      (vs2_data),      // Source vector for store (from rd field via vs2_addr mux)
        .vec_mem_addr (vec_mem_addr),
        .vec_mem_wdata(vec_mem_wdata),
        .vec_mem_rdata(mem_rdata),
        .vec_mem_we   (vec_mem_we),
        .vec_mem_re   (vec_mem_re),
        .vd_data      (vec_load_data),
        .vec_busy     (vec_mem_busy),
        .vec_done     (vec_mem_done)
    );

    //=========================================================================
    // Register Write Data Selection
    //=========================================================================

    always @(*) begin
        if (is_vsetvl) begin
            // VSETVL returns actual vl to rd (the new value being set)
            rd_data = {27'b0, vl_new};
        end else begin
            case (rd_src)
                2'b00:   rd_data = alu_out_reg;    // ALU result
                2'b01:   rd_data = mem_data_reg;   // Memory data
                2'b10:   rd_data = pc_at_fetch + 4;// Link address (PC at fetch + 4 for JAL/JALR)
                2'b11:   rd_data = 32'b0;          // Reserved (FP later)
                default: rd_data = alu_out_reg;
            endcase
        end
    end

    //=========================================================================
    // Program Counter
    //=========================================================================

    wire [31:0] pc_plus_4 = pc + 4;
    wire [31:0] branch_target = pc + br_offset_sext;
    wire [31:0] jal_target = pc + {{15{imm16[15]}}, imm16, 2'b00}; // Sign-extend and shift
    wire [31:0] jalr_target = alu_result & 32'hFFFFFFFE; // Clear LSB

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pc <= 32'h00001000;  // Start at first kernel entry
        end else if (pc_we) begin
            case (pc_src)
                2'b00:   pc <= pc_plus_4;
                2'b01:   pc <= branch_target;
                2'b10:   pc <= jal_target;
                2'b11:   pc <= jalr_target;
                default: pc <= pc_plus_4;
            endcase
        end
    end

    //=========================================================================
    // Instruction Register
    //=========================================================================

    reg [31:0] pc_at_fetch;  // Save PC when instruction is fetched (for JAL/JALR link)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ir <= 32'b0;
            pc_at_fetch <= 32'b0;
        end else if (ir_we) begin
            ir <= mem_rdata;
            pc_at_fetch <= pc;  // Save the PC at fetch time
        end
    end

    //=========================================================================
    // Pipeline Registers
    //=========================================================================

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            alu_out_reg    <= 32'b0;
            mem_data_reg   <= 32'b0;
            fpu_out_reg    <= 32'b0;
            vec_out_reg    <= 512'b0;
            vec_reduce_reg <= 32'b0;
        end else begin
            // Capture ALU result at end of execute
            if (state == `STATE_EXECUTE) begin
                alu_out_reg    <= alu_result;
                fpu_out_reg    <= fpu_result;  // Capture FPU result too
                vec_out_reg    <= vec_result;  // Capture vector result
                vec_reduce_reg <= vec_reduce_result;
            end
            // Capture memory data at end of memory
            if (state == `STATE_MEMORY) begin
                mem_data_reg <= mem_rdata;
            end
            // Capture vector load data when vec_mem_done
            if (vec_mem_done && is_vec_load) begin
                vec_out_reg <= vec_load_data;
            end
        end
    end

    //=========================================================================
    // FP Register Write Data Selection
    //=========================================================================

    always @(*) begin
        if (is_fpu || is_fpu_special) begin
            // FPU arithmetic: write FPU result to FP register
            fd_data = fpu_out_reg;
        end else if (is_vec_reduce) begin
            // Vector reduction: write scalar result to FP register
            fd_data = vec_reduce_reg;
        end else if (is_q8mac_op && func == `Q8_ACCREAD) begin
            // ACCREAD: write accumulator to FP register
            fd_data = q8_acc_out;
        end else begin
            // FP load: write memory data to FP register
            fd_data = mem_data_reg;
        end
    end

    //=========================================================================
    // Memory Interface
    //=========================================================================

    // Address muxing: Vector memory and Q8 MAC take priority when busy
    // Priority: vec_mem > q8_mac > CPU
    assign mem_addr = vec_mem_busy ? vec_mem_addr :
                      q8_mac_busy  ? q8_mem_addr :
                      (state == `STATE_FETCH) ? pc : alu_out_reg;

    // Write data: vector data, FP store, or integer store
    assign mem_wdata = vec_mem_busy ? vec_mem_wdata :
                       is_fp_store  ? fs2_data : rs2_data;

    // Control signals: Vector and Q8 MAC take priority when busy
    assign mem_we = vec_mem_busy ? vec_mem_we : ctrl_mem_we;
    assign mem_re = vec_mem_busy ? vec_mem_re :
                    q8_mac_busy  ? q8_mem_re : ctrl_mem_re;

    //=========================================================================
    // Debug Outputs
    //=========================================================================

    assign pc_out = pc;
    assign state_out = state;

endmodule
