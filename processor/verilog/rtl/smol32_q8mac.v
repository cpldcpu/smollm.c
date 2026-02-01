/**
 * SMOL-32 Q8 MAC Unit
 * Handles INT8 dequantize-multiply-accumulate operations
 *
 * Special registers:
 * - ACC: 64-bit floating-point accumulator (as double)
 * - SCALE: FP32 dequantization scale
 * - QBASE: Byte address for Q8 weights
 * - FBASE: Word address for FP32 activations
 *
 * Instructions:
 * - QSETSCALE: Set scale register from FP register
 * - QSETQBASE: Set QBASE from integer register
 * - QSETFBASE: Set FBASE from integer register
 * - Q8MACINC n: ACC += sum(Q8[qbase:qbase+n] * scale * FP32[fbase:fbase+n*4]); auto-increment
 * - ACCZERO: ACC = 0
 * - ACCREAD fd: fd = (float)ACC
 *
 * Multi-cycle implementation with proper memory sequencing.
 */

`include "smol32_defines.vh"

module smol32_q8mac (
    input  wire        clk,
    input  wire        rst_n,

    // Control
    input  wire [4:0]  func,        // Q8 operation
    input  wire        is_q8set,    // OP_Q8SET instruction
    input  wire        is_q8mac,    // OP_Q8MAC instruction
    input  wire        execute,     // Execute strobe

    // Data inputs
    input  wire [31:0] rs1_data,    // Integer register data
    input  wire [31:0] fs1_data,    // FP register data (for scale)
    input  wire [5:0]  ext,         // Extension field (n for Q8MACINC)

    // Memory interface (directly connected to memory)
    output reg  [31:0] q8_mem_addr, // Memory address
    input  wire [31:0] q8_mem_rdata,// Memory data
    output reg         q8_mem_re,   // Memory read enable

    // Outputs
    output wire [31:0] acc_out,     // Accumulator as FP32
    output wire [31:0] qbase_out,   // Current QBASE
    output wire [31:0] fbase_out,   // Current FBASE
    output wire [31:0] scale_out,   // Current SCALE

    // MAC state for multi-cycle operation
    output reg         mac_busy,    // MAC operation in progress
    output reg         mac_done     // MAC operation complete
);

    // Internal registers
    reg [31:0] scale_reg;    // FP32 scale
    reg [31:0] qbase_reg;    // Q8 byte address
    reg [31:0] fbase_reg;    // FP32 word address
    real       acc_real;     // Accumulator (simulation uses real)

    // MAC state machine
    reg [3:0]  mac_state;
    reg [5:0]  mac_count;    // Elements remaining
    reg [31:0] mac_qaddr;    // Current Q8 address
    reg [31:0] mac_faddr;    // Current FP32 address
    reg [31:0] q8_word;      // Fetched word containing Q8 bytes
    reg [31:0] fp_word;      // Fetched FP32 value
    reg        is_macinc;    // Remember if MACINC vs MAC
    reg [5:0]  total_count;  // Total elements to process

    // State encoding
    localparam MAC_IDLE       = 4'd0;
    localparam MAC_FETCH_Q8   = 4'd1;   // Issue Q8 word read
    localparam MAC_WAIT_Q8    = 4'd2;   // Wait for Q8 data
    localparam MAC_FETCH_FP   = 4'd3;   // Issue FP32 word read
    localparam MAC_WAIT_FP    = 4'd4;   // Wait for FP32 data
    localparam MAC_COMPUTE    = 4'd5;   // Dequantize, multiply, accumulate
    localparam MAC_NEXT       = 4'd6;   // Move to next element
    localparam MAC_FINISH     = 4'd7;   // Update pointers if MACINC
    localparam MAC_DONE       = 4'd8;   // Signal completion

    // FP conversion: IEEE 754 single to Verilog real
    function real single_to_real;
        input [31:0] single;
        reg [63:0] double_bits;
        reg [7:0] s_exp;
        reg [22:0] s_mant;
        reg s_sign;
        reg [10:0] d_exp;
        begin
            s_sign = single[31];
            s_exp  = single[30:23];
            s_mant = single[22:0];

            if (s_exp == 8'h00) begin
                double_bits = {s_sign, 63'b0};
            end else if (s_exp == 8'hFF) begin
                double_bits = {s_sign, 11'h7FF, s_mant, 29'b0};
            end else begin
                d_exp = {3'b0, s_exp} + 11'd896;
                double_bits = {s_sign, d_exp, s_mant, 29'b0};
            end
            single_to_real = $bitstoreal(double_bits);
        end
    endfunction

    // FP conversion: Verilog real to IEEE 754 single
    function [31:0] real_to_single;
        input real r;
        reg [63:0] double_bits;
        reg d_sign;
        reg [10:0] d_exp;
        reg [51:0] d_mant;
        reg [10:0] s_exp_wide;
        begin
            double_bits = $realtobits(r);
            d_sign = double_bits[63];
            d_exp  = double_bits[62:52];
            d_mant = double_bits[51:0];

            if (d_exp == 11'h000) begin
                real_to_single = {d_sign, 31'b0};
            end else if (d_exp == 11'h7FF) begin
                real_to_single = {d_sign, 8'hFF, d_mant[51:29]};
            end else if (d_exp > 11'd1150) begin
                real_to_single = {d_sign, 8'hFF, 23'b0};
            end else if (d_exp < 11'd897) begin
                real_to_single = {d_sign, 31'b0};
            end else begin
                s_exp_wide = d_exp - 11'd896;
                real_to_single = {d_sign, s_exp_wide[7:0], d_mant[51:29]};
            end
        end
    endfunction

    // Extract signed 8-bit value from word based on byte offset
    function signed [7:0] extract_q8;
        input [31:0] word;
        input [1:0] byte_pos;
        begin
            case (byte_pos)
                2'd0: extract_q8 = word[7:0];
                2'd1: extract_q8 = word[15:8];
                2'd2: extract_q8 = word[23:16];
                2'd3: extract_q8 = word[31:24];
            endcase
        end
    endfunction

    // Computation variables
    reg signed [7:0] q8_val;
    real scale_r;
    real fp_r;
    real dequant;
    real product;

    // Output assignments
    assign acc_out   = real_to_single(acc_real);
    assign qbase_out = qbase_reg;
    assign fbase_out = fbase_reg;
    assign scale_out = scale_reg;

    // State machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            scale_reg <= 32'h3F800000;  // 1.0
            qbase_reg <= 32'b0;
            fbase_reg <= 32'b0;
            acc_real  <= 0.0;
            mac_state <= MAC_IDLE;
            mac_count <= 6'b0;
            mac_qaddr <= 32'b0;
            mac_faddr <= 32'b0;
            mac_busy  <= 1'b0;
            mac_done  <= 1'b0;
            q8_mem_re <= 1'b0;
            q8_mem_addr <= 32'b0;
            is_macinc <= 1'b0;
            total_count <= 6'b0;
            q8_word <= 32'b0;
            fp_word <= 32'b0;
        end else begin
            mac_done <= 1'b0;  // Clear done flag each cycle

            // Handle Q8SET instructions
            if (execute && is_q8set && mac_state == MAC_IDLE) begin
                case (func)
                    `Q8_SETSCALE: scale_reg <= fs1_data;
                    `Q8_SETQBASE: qbase_reg <= rs1_data;
                    `Q8_SETFBASE: fbase_reg <= rs1_data;
                    default: ;
                endcase
            end

            // Handle Q8MAC instructions
            if (execute && is_q8mac && mac_state == MAC_IDLE) begin
                case (func)
                    `Q8_ACCZERO: begin
                        acc_real <= 0.0;
                    end
                    `Q8_ACCREAD: begin
                        // Output is handled by acc_out
                    end
                    `Q8_MAC, `Q8_MACINC: begin
                        if (ext > 0) begin
                            // Start MAC operation
                            mac_state <= MAC_FETCH_Q8;
                            mac_count <= ext;
                            total_count <= ext;
                            mac_qaddr <= qbase_reg;
                            mac_faddr <= fbase_reg;
                            mac_busy  <= 1'b1;
                            is_macinc <= (func == `Q8_MACINC);
                            // Issue first Q8 word read (aligned)
                            q8_mem_addr <= {qbase_reg[31:2], 2'b00};
                            q8_mem_re <= 1'b1;
                        end
                    end
                    default: ;
                endcase
            end

            // MAC state machine
            case (mac_state)
                MAC_IDLE: begin
                    // Idle, waiting for command
                end

                MAC_FETCH_Q8: begin
                    // Wait one cycle for memory read
                    mac_state <= MAC_WAIT_Q8;
                end

                MAC_WAIT_Q8: begin
                    // Capture Q8 word
                    q8_word <= q8_mem_rdata;
                    // Issue FP32 read
                    q8_mem_addr <= mac_faddr;
                    mac_state <= MAC_FETCH_FP;
                end

                MAC_FETCH_FP: begin
                    // Wait one cycle for memory read
                    mac_state <= MAC_WAIT_FP;
                end

                MAC_WAIT_FP: begin
                    // Capture FP32 word
                    fp_word <= q8_mem_rdata;
                    mac_state <= MAC_COMPUTE;
                end

                MAC_COMPUTE: begin
                    // Extract Q8 byte from word
                    q8_val = extract_q8(q8_word, mac_qaddr[1:0]);

                    // Dequantize and multiply-accumulate
                    scale_r = single_to_real(scale_reg);
                    fp_r = single_to_real(fp_word);
                    dequant = $itor(q8_val) * scale_r;
                    product = dequant * fp_r;
                    acc_real <= acc_real + product;

                    mac_state <= MAC_NEXT;
                end

                MAC_NEXT: begin
                    mac_count <= mac_count - 1;

                    if (mac_count == 1) begin
                        // Last element done
                        mac_state <= MAC_FINISH;
                        q8_mem_re <= 1'b0;
                    end else begin
                        // Advance to next element
                        mac_qaddr <= mac_qaddr + 1;
                        mac_faddr <= mac_faddr + 4;
                        // Issue next Q8 word read (aligned to word boundary)
                        // If we're at byte 3, move to next word, otherwise stay on same word
                        if (mac_qaddr[1:0] == 2'd3) begin
                            q8_mem_addr <= {mac_qaddr[31:2] + 30'd1, 2'b00};
                        end else begin
                            q8_mem_addr <= {mac_qaddr[31:2], 2'b00};
                        end
                        mac_state <= MAC_FETCH_Q8;
                    end
                end

                MAC_FINISH: begin
                    // Update base registers for MACINC
                    if (is_macinc) begin
                        qbase_reg <= qbase_reg + {26'b0, total_count};
                        fbase_reg <= fbase_reg + {24'b0, total_count, 2'b00};
                    end
                    mac_state <= MAC_DONE;
                end

                MAC_DONE: begin
                    mac_busy  <= 1'b0;
                    mac_done  <= 1'b1;
                    mac_state <= MAC_IDLE;
                end

                default: begin
                    mac_state <= MAC_IDLE;
                end
            endcase
        end
    end

endmodule
