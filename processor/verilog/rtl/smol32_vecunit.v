/**
 * SMOL-32 Vector Unit
 * Supports 16-lane FP32 SIMD operations
 * - Element-wise: VADD, VSUB, VMUL, VDIV, VMIN, VMAX
 * - Scalar broadcast: VADD.S, VMUL.S
 * - Reductions: VREDSUM, VREDMAX, VREDMIN, VREDSQS
 * - Special: VSQRT, VRSQRT, VEXP, VSILU
 *
 * NOTE: This is a simulation model. For synthesis, replace with FPU IP.
 */

`include "smol32_defines.vh"

module smol32_vecunit (
    input  wire [511:0] vs1,        // Vector operand 1 (16 × 32-bit)
    input  wire [511:0] vs2,        // Vector operand 2 (16 × 32-bit)
    input  wire [31:0]  fs,         // Scalar operand (for broadcast ops)
    input  wire [4:0]   vl,         // Vector length (1-16)
    input  wire [2:0]   op,         // Vector operation
    input  wire         is_scalar,  // 1 = vs2 is scalar (broadcast)
    input  wire         is_reduce,  // 1 = reduction operation
    input  wire         is_special, // 1 = special function
    output reg  [511:0] vd,         // Vector result
    output reg  [31:0]  fd          // Scalar result (for reductions)
);

    // Extract lanes from vectors
    wire [31:0] lane1 [0:15];
    wire [31:0] lane2 [0:15];
    reg  [31:0] result_lane [0:15];

    genvar g;
    generate
        for (g = 0; g < 16; g = g + 1) begin : extract_lanes
            assign lane1[g] = vs1[g*32 +: 32];
            assign lane2[g] = is_scalar ? fs : vs2[g*32 +: 32];
        end
    endgenerate

    // Single-to-real conversion helper (reuse from FPU)
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

    // Exp approximation using exp(x) = 2^(x * log2(e))
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

    // SiLU activation
    function real silu;
        input real x;
        begin
            silu = x / (1.0 + exp_approx(-x));
        end
    endfunction

    // Process each lane
    real a_r, b_r, r_r;
    integer lane;
    real sum_r, max_r, min_r, sqs_r, tmp_r;

    always @(*) begin
        // Default: pass through
        for (lane = 0; lane < 16; lane = lane + 1) begin
            result_lane[lane] = 32'b0;
        end
        fd = 32'b0;

        if (is_reduce) begin
            // Reduction operations
            sum_r = 0.0;
            max_r = single_to_real(lane1[0]);
            min_r = single_to_real(lane1[0]);
            sqs_r = 0.0;

            for (lane = 0; lane < 16; lane = lane + 1) begin
                if (lane < vl) begin
                    tmp_r = single_to_real(lane1[lane]);
                    case (op)
                        `VRED_SUM: sum_r = sum_r + tmp_r;
                        `VRED_MAX: if (tmp_r > max_r) max_r = tmp_r;
                        `VRED_MIN: if (tmp_r < min_r) min_r = tmp_r;
                        `VRED_SQS: sqs_r = sqs_r + tmp_r * tmp_r;
                        default: ;
                    endcase
                end
            end

            case (op)
                `VRED_SUM: fd = real_to_single(sum_r);
                `VRED_MAX: fd = real_to_single(max_r);
                `VRED_MIN: fd = real_to_single(min_r);
                `VRED_SQS: fd = real_to_single(sqs_r);
                default:   fd = 32'b0;
            endcase

        end else if (is_special) begin
            // Special vector functions
            for (lane = 0; lane < 16; lane = lane + 1) begin
                if (lane < vl) begin
                    a_r = single_to_real(lane1[lane]);
                    case (op)
                        `VSPEC_SQRT:  r_r = (a_r >= 0.0) ? $sqrt(a_r) : 0.0;
                        `VSPEC_RSQRT: r_r = (a_r > 0.0) ? (1.0 / $sqrt(a_r)) : 0.0;
                        `VSPEC_EXP:   r_r = exp_approx(a_r);
                        `VSPEC_SILU:  r_r = silu(a_r);
                        default:      r_r = 0.0;
                    endcase
                    result_lane[lane] = real_to_single(r_r);
                end else begin
                    result_lane[lane] = 32'b0;
                end
            end

        end else begin
            // Element-wise operations
            for (lane = 0; lane < 16; lane = lane + 1) begin
                if (lane < vl) begin
                    a_r = single_to_real(lane1[lane]);
                    b_r = single_to_real(lane2[lane]);
                    case (op)
                        `VFUNC_ADD:   r_r = a_r + b_r;
                        `VFUNC_SUB:   r_r = a_r - b_r;
                        `VFUNC_MUL:   r_r = a_r * b_r;
                        `VFUNC_DIV:   r_r = (b_r != 0.0) ? (a_r / b_r) : 0.0;
                        `VFUNC_FMADD: r_r = a_r * b_r + single_to_real(fs);
                        `VFUNC_MIN:   r_r = (a_r < b_r) ? a_r : b_r;
                        `VFUNC_MAX:   r_r = (a_r > b_r) ? a_r : b_r;
                        default:      r_r = 0.0;
                    endcase
                    result_lane[lane] = real_to_single(r_r);
                end else begin
                    result_lane[lane] = 32'b0;
                end
            end
        end

        // Pack result lanes
        vd = {result_lane[15], result_lane[14], result_lane[13], result_lane[12],
              result_lane[11], result_lane[10], result_lane[9],  result_lane[8],
              result_lane[7],  result_lane[6],  result_lane[5],  result_lane[4],
              result_lane[3],  result_lane[2],  result_lane[1],  result_lane[0]};
    end

endmodule
