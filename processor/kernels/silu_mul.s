# SMOL-32 Assembly: SiLU-Gate (silu(gate) * up, element-wise)
#
# void silu_mul(float *gate, float *up, int n)
#
# Arguments:
#   R3 = gate pointer (input/output: result stored here)
#   R4 = up pointer
#   R5 = n (number of elements, must be multiple of 16)
#
# Computes: gate[i] = silu(gate[i]) * up[i] for i = 0..n-1

    .text
    .global silu_mul

silu_mul:
    VSETVL  R10, R5             # VL = min(n, 16)

silu_loop:
    LVF     V0, R3, 4          # load gate[0:16]
    VSILU   V0, V0             # silu(gate)
    LVF     V1, R4, 4          # load up[0:16]
    VMUL    V0, V0, V1         # silu(gate) * up
    SVF     V0, R3, 4          # store result

    ADDI    R3, R3, 64         # advance gate (16 * 4)
    ADDI    R4, R4, 64         # advance up
    ADDI    R5, R5, -16
    BGTZ    R5, silu_loop

    RET
