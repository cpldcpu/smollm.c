# SMOL-32 Assembly: Residual Addition (element-wise vector add)
#
# void residual(float *dst, float *src, int n)
#
# Arguments:
#   R3 = dst pointer (also first source, result stored here)
#   R4 = src pointer (second source)
#   R5 = n (number of elements, must be multiple of 16)
#
# Computes: dst[i] += src[i] for i = 0..n-1

    .text
    .global residual

residual:
    VSETVL  R10, R5             # VL = min(n, 16)

residual_loop:
    LVF     V0, R3, 4          # load dst[0:16]
    LVF     V1, R4, 4          # load src[0:16]
    VADD    V0, V0, V1         # dst += src
    SVF     V0, R3, 4          # store result

    ADDI    R3, R3, 64         # advance dst (16 * 4)
    ADDI    R4, R4, 64         # advance src
    ADDI    R5, R5, -16
    BGTZ    R5, residual_loop

    RET
