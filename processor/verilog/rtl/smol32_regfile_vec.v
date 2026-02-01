/**
 * SMOL-32 Vector Register File
 * 8 registers × 16 lanes × 32-bit = 512 bits per register
 * V0 is NOT hardwired to zero (unlike scalar registers)
 * Synthesizable design
 */

`include "smol32_defines.vh"

module smol32_regfile_vec (
    input  wire        clk,
    input  wire        rst_n,

    // Read port 1 (vs1)
    input  wire [2:0]  vs1_addr,
    output wire [511:0] vs1_data,

    // Read port 2 (vs2)
    input  wire [2:0]  vs2_addr,
    output wire [511:0] vs2_data,

    // Write port (vd)
    input  wire [2:0]  vd_addr,
    input  wire [511:0] vd_data,
    input  wire        vd_we,

    // Vector length register
    input  wire [4:0]  vl_in,       // Vector length (1-16)
    input  wire        vl_we,
    output reg  [4:0]  vl_out
);

    // 8 vector registers × 512 bits (16 × 32-bit lanes)
    reg [511:0] vregs [0:7];

    integer i;

    // Reset
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < 8; i = i + 1) begin
                vregs[i] <= 512'b0;
            end
            vl_out <= 5'd16;  // Default to full vector length
        end else begin
            if (vd_we) begin
                vregs[vd_addr] <= vd_data;
            end
            if (vl_we) begin
                vl_out <= vl_in;
            end
        end
    end

    // Read ports (combinatorial)
    assign vs1_data = vregs[vs1_addr];
    assign vs2_data = vregs[vs2_addr];

endmodule
