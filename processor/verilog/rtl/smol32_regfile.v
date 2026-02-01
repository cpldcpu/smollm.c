/**
 * SMOL-32 Integer Register File
 * 32 x 32-bit registers, R0 hardwired to 0
 * Dual read ports, single write port
 * Synthesizable design
 */

`include "smol32_defines.vh"

module smol32_regfile (
    input  wire        clk,
    input  wire        rst_n,

    // Read port 1
    input  wire [4:0]  rs1_addr,
    output wire [31:0] rs1_data,

    // Read port 2
    input  wire [4:0]  rs2_addr,
    output wire [31:0] rs2_data,

    // Write port
    input  wire [4:0]  rd_addr,
    input  wire [31:0] rd_data,
    input  wire        rd_we
);

    // Register array (R1-R31, R0 is hardwired to 0)
    reg [31:0] regs [1:31];

    integer i;

    // Write logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 1; i < 32; i = i + 1) begin
                regs[i] <= 32'b0;
            end
        end else if (rd_we && rd_addr != 5'b0) begin
            regs[rd_addr] <= rd_data;
        end
    end

    // Read port 1 - combinational with R0 = 0
    assign rs1_data = (rs1_addr == 5'b0) ? 32'b0 : regs[rs1_addr];

    // Read port 2 - combinational with R0 = 0
    assign rs2_data = (rs2_addr == 5'b0) ? 32'b0 : regs[rs2_addr];

endmodule
