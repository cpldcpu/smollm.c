/**
 * SMOL-32 Floating-Point Register File
 * 32 x 32-bit registers, F0 hardwired to 0.0
 * Dual read ports, single write port
 * IEEE 754 single-precision format
 * Synthesizable design
 */

`include "smol32_defines.vh"

module smol32_regfile_fp (
    input  wire        clk,
    input  wire        rst_n,

    // Read port 1
    input  wire [4:0]  fs1_addr,
    output wire [31:0] fs1_data,

    // Read port 2
    input  wire [4:0]  fs2_addr,
    output wire [31:0] fs2_data,

    // Write port
    input  wire [4:0]  fd_addr,
    input  wire [31:0] fd_data,
    input  wire        fd_we
);

    // Register array (F1-F31, F0 is hardwired to 0.0)
    reg [31:0] fregs [1:31];

    integer i;

    // IEEE 754 representation of 0.0 is all zeros
    localparam [31:0] FP_ZERO = 32'h00000000;

    // Write logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 1; i < 32; i = i + 1) begin
                fregs[i] <= FP_ZERO;
            end
        end else if (fd_we && fd_addr != 5'b0) begin
            fregs[fd_addr] <= fd_data;
        end
    end

    // Read port 1 - combinational with F0 = 0.0
    assign fs1_data = (fs1_addr == 5'b0) ? FP_ZERO : fregs[fs1_addr];

    // Read port 2 - combinational with F0 = 0.0
    assign fs2_data = (fs2_addr == 5'b0) ? FP_ZERO : fregs[fs2_addr];

endmodule
