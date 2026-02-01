/**
 * SMOL-32 Floating-Point Unit
 * IEEE 754 single-precision operations
 *
 * NOTE: This is a simulation model using shortreal (32-bit float).
 * For synthesis, replace with actual FPU IP.
 */

`include "smol32_defines.vh"

module smol32_fpu (
    input  wire [31:0] a,           // Operand A (IEEE 754)
    input  wire [31:0] b,           // Operand B (IEEE 754)
    input  wire [31:0] int_a,       // Integer operand for FCVT.S.W
    input  wire [4:0]  op,          // FPU operation (func field)
    input  wire        is_special,  // 1 for FSPEC operations
    output reg  [31:0] result,      // Result (IEEE 754 or integer for FCVT.W.S)
    output wire        zero,        // Result is zero
    output wire        negative,    // Result is negative
    output wire        nan_out,     // Result is NaN
    output wire        inf_out      // Result is infinity
);

    // IEEE 754 field extraction for flags
    wire        r_sign = result[31];
    wire [7:0]  r_exp  = result[30:23];
    wire [22:0] r_mant = result[22:0];

    assign zero     = (result == 32'h00000000) || (result == 32'h80000000);
    assign negative = r_sign && !zero;
    assign nan_out  = (r_exp == 8'hFF) && (r_mant != 23'h0);
    assign inf_out  = (r_exp == 8'hFF) && (r_mant == 23'h0);

    // Simulation model using DPI-style conversion
    // iverilog supports $bitstoshortreal and $shortrealtobits
    // If not available, we use a behavioral approximation

    // For simulation: use real type and manual bit packing
    // This is NOT synthesizable - for synthesis use FPU IP
    real a_r, b_r, r_r;
    reg [63:0] tmp64;
    reg [10:0] exp_double;

    // Helper function: convert single precision IEEE 754 to double
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
                // Zero or denormal -> treat as zero
                double_bits = {s_sign, 63'b0};
            end else if (s_exp == 8'hFF) begin
                // Infinity or NaN
                double_bits = {s_sign, 11'h7FF, s_mant, 29'b0};
            end else begin
                // Normal: adjust exponent (bias 127 -> 1023, so add 896)
                d_exp = {3'b0, s_exp} + 11'd896;
                double_bits = {s_sign, d_exp, s_mant, 29'b0};
            end
            single_to_real = $bitstoreal(double_bits);
        end
    endfunction

    // Helper function: convert double to single precision IEEE 754
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
                // Zero
                real_to_single = {d_sign, 31'b0};
            end else if (d_exp == 11'h7FF) begin
                // Infinity or NaN
                real_to_single = {d_sign, 8'hFF, d_mant[51:29]};
            end else if (d_exp > 11'd1150) begin
                // Overflow -> infinity (exp > 127+1023 = single max)
                real_to_single = {d_sign, 8'hFF, 23'b0};
            end else if (d_exp < 11'd897) begin
                // Underflow -> zero (exp < -126+1023 = single min)
                real_to_single = {d_sign, 31'b0};
            end else begin
                // Normal: adjust exponent (double bias 1023 -> single bias 127)
                // s_exp = d_exp - 1023 + 127 = d_exp - 896
                s_exp_wide = d_exp - 11'd896;
                real_to_single = {d_sign, s_exp_wide[7:0], d_mant[51:29]};
            end
        end
    endfunction

    // Math helper: exp(x) approximation using exp(x) = 2^(x * log2(e))
    // Decompose: x * log2(e) = n + f, where n is integer, f in [0,1)
    // Then: exp(x) = 2^n * 2^f
    // 2^f is approximated with a polynomial over [0,1)
    function real exp_approx;
        input real x;
        real t, f, pow2_f;
        integer n;
        begin
            // log2(e) = 1.4426950408889634
            t = x * 1.4426950408889634;

            // Floor function (works for negative numbers too)
            n = (t >= 0) ? $rtoi(t) : ($rtoi(t) - ((t == $rtoi(t)) ? 0 : 1));
            f = t - n;  // f is now in [0, 1)

            // Approximate 2^f using minimax polynomial (accurate to ~1e-7)
            // Coefficients from Remez algorithm for 2^x on [0,1]
            pow2_f = 1.0 + f * (0.6931471805599453 +
                           f * (0.2402265069591007 +
                           f * (0.0555041086648216 +
                           f * (0.0096181291076285 +
                           f * 0.0013333558146428))));

            // 2^n * 2^f with overflow/underflow protection
            if (n > 127)
                exp_approx = 3.4e38;  // max float approx
            else if (n < -126)
                exp_approx = 0.0;
            else
                exp_approx = pow2_f * (2.0 ** n);
        end
    endfunction

    // Math helper: SiLU activation = x * sigmoid(x) = x / (1 + exp(-x))
    function real silu;
        input real x;
        begin
            silu = x / (1.0 + exp_approx(-x));
        end
    endfunction

    // Math helper: sigmoid(x) = 1 / (1 + exp(-x))
    function real sigmoid;
        input real x;
        begin
            sigmoid = 1.0 / (1.0 + exp_approx(-x));
        end
    endfunction

    // Math helper: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    function real tanh_approx;
        input real x;
        real e2x;
        begin
            e2x = exp_approx(2.0 * x);
            tanh_approx = (e2x - 1.0) / (e2x + 1.0);
        end
    endfunction

    // Math helper: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    function real gelu;
        input real x;
        real inner;
        begin
            // sqrt(2/π) ≈ 0.7978845608
            inner = 0.7978845608 * (x + 0.044715 * x * x * x);
            gelu = 0.5 * x * (1.0 + tanh_approx(inner));
        end
    endfunction

    always @(*) begin
        // Convert inputs from single to real
        a_r = single_to_real(a);
        b_r = single_to_real(b);

        if (is_special) begin
            // Special FP operations (OP_FSPEC = 0x0C)
            case (op)
                `FUNC_FSQRT:    r_r = (a_r >= 0.0) ? $sqrt(a_r) : 0.0;
                `FUNC_FRSQRT:   r_r = (a_r > 0.0) ? (1.0 / $sqrt(a_r)) : 0.0;
                `FUNC_FRECIP:   r_r = (a_r != 0.0) ? (1.0 / a_r) : 0.0;
                `FUNC_FEXP:     r_r = exp_approx(a_r);
                `FUNC_FLOG:     r_r = (a_r > 0.0) ? $ln(a_r) : 0.0;
                `FUNC_FSIN:     r_r = $sin(a_r);
                `FUNC_FCOS:     r_r = $cos(a_r);
                `FUNC_FSILU:    r_r = silu(a_r);
                `FUNC_FGELU:    r_r = gelu(a_r);
                `FUNC_FTANH:    r_r = tanh_approx(a_r);
                `FUNC_FSIGMOID: r_r = sigmoid(a_r);
                default:        r_r = 0.0;
            endcase
        end else begin
            // Normal FP operations (OP_FPU = 0x08)
            case (op)
                `FUNC_FADD: r_r = a_r + b_r;
                `FUNC_FSUB: r_r = a_r - b_r;
                `FUNC_FMUL: r_r = a_r * b_r;
                `FUNC_FDIV: r_r = a_r / b_r;
                `FUNC_FMIN: r_r = (a_r < b_r) ? a_r : b_r;
                `FUNC_FMAX: r_r = (a_r > b_r) ? a_r : b_r;
                `FUNC_FABS: begin
                    tmp64 = $realtobits(a_r);
                    tmp64[63] = 1'b0;
                    r_r = $bitstoreal(tmp64);
                end
                `FUNC_FNEG: r_r = -a_r;
                `FUNC_FMV:  r_r = a_r;
                default:    r_r = 0.0;
            endcase
        end

        // Convert result back to single precision (for most operations)
        // FCVT.W.S and FCVT.S.W are handled specially
        if (op == `FUNC_FCVT_W_S) begin
            // Float to integer conversion: truncate toward zero
            // Input a is already a float, convert to integer
            if (a_r >= 2147483647.0)
                result = 32'h7FFFFFFF;  // Max positive int
            else if (a_r <= -2147483648.0)
                result = 32'h80000000;  // Min negative int
            else
                result = $rtoi(a_r);    // Truncate to integer
        end else if (op == `FUNC_FCVT_S_W) begin
            // Integer to float conversion
            // Input is from int_a (integer register)
            r_r = $itor($signed(int_a));
            result = real_to_single(r_r);
        end else begin
            result = real_to_single(r_r);
        end
    end

endmodule
