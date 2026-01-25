# SMOL-32 Assembly: Float Memory Copy
#
# void memcpy_f(float *dst, float *src, int n)
#
# Arguments:
#   R3 = dst pointer
#   R4 = src pointer
#   R5 = n (number of floats to copy, must be multiple of 16)
#
# Copies n floats from src to dst using vector load/store.

    .text
    .global memcpy_f

memcpy_f:
    VSETVL  R10, R5             # VL = min(n, 16)

copy_loop:
    LVF     V0, R4, 4          # load 16 floats from src
    SVF     V0, R3, 4          # store to dst

    ADDI    R3, R3, 64         # dst += 16 floats
    ADDI    R4, R4, 64         # src += 16 floats
    ADDI    R5, R5, -16
    BGTZ    R5, copy_loop

    RET
