/**
 * SMOL-32 Integer ALU
 * Supports: ADD, SUB, MUL, AND, OR, XOR, SLL, SRL, SRA, SLT, SLTU
 * Synthesizable design
 */

`include "smol32_defines.vh"

module smol32_alu (
    input  wire [31:0] a,           // Operand A
    input  wire [31:0] b,           // Operand B
    input  wire [3:0]  op,          // ALU operation
    output reg  [31:0] result,      // Result
    output wire        zero,        // Result is zero
    output wire        negative,    // Result is negative (MSB)
    output wire        overflow     // Overflow occurred
);

    // Internal signals for overflow detection
    wire [31:0] sum;
    wire [31:0] diff;
    wire        add_overflow;
    wire        sub_overflow;

    // Addition and subtraction
    assign sum = a + b;
    assign diff = a - b;

    // Overflow detection for signed operations
    assign add_overflow = (a[31] == b[31]) && (sum[31] != a[31]);
    assign sub_overflow = (a[31] != b[31]) && (diff[31] != a[31]);

    // Main ALU operation
    always @(*) begin
        case (op)
            `ALU_ADD:    result = sum;
            `ALU_SUB:    result = diff;
            `ALU_MUL:    result = a * b;  // Lower 32 bits
            `ALU_AND:    result = a & b;
            `ALU_OR:     result = a | b;
            `ALU_XOR:    result = a ^ b;
            `ALU_SLL:    result = a << b[4:0];
            `ALU_SRL:    result = a >> b[4:0];
            `ALU_SRA:    result = $signed(a) >>> b[4:0];
            `ALU_SLT:    result = ($signed(a) < $signed(b)) ? 32'd1 : 32'd0;
            `ALU_SLTU:   result = (a < b) ? 32'd1 : 32'd0;
            `ALU_PASS_A: result = a;
            `ALU_PASS_B: result = b;
            default:     result = 32'b0;
        endcase
    end

    // Status flags
    assign zero = (result == 32'b0);
    assign negative = result[31];
    assign overflow = (op == `ALU_ADD) ? add_overflow :
                      (op == `ALU_SUB) ? sub_overflow : 1'b0;

endmodule
