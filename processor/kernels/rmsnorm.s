# SMOL-32 Assembly: RMS Normalization
#
# void rmsnorm(float *o, float *x, float *w, int n, float eps)
#
# Arguments:
#   R3 = output pointer
#   R4 = input pointer (x)
#   R5 = weight pointer (w)
#   R6 = n (dimension)
#   F1 = eps
#
# Computes: o[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]

    .text
    .global rmsnorm

rmsnorm:
    # Save input pointer for second pass
    MV      R7, R4              # R7 = saved x pointer

    # Set vector length
    VSETVL  R10, R6             # VL = min(n, 16)

    # Initialize sum of squares
    FMOV    F2, F0              # F2 = 0.0 (sum)

    # R8 = element counter
    MV      R8, R6

    # ===== Pass 1: Compute sum of squares =====
sum_sq_loop:
    LVF     V0, R4, 4           # Load 16 floats from x
    VREDSQS F3, V0              # F3 = sum of squares of V0
    FADD    F2, F2, F3          # Accumulate into F2
    ADDI    R4, R4, 64          # Advance pointer (16 * 4 bytes)
    ADDI    R8, R8, -16         # Decrement counter
    BGTZ    R8, sum_sq_loop

    # ===== Compute normalization factor =====
    # F2 = sum of squares
    # Need: 1 / sqrt(F2/n + eps)

    # Convert n to float
    FCVT.S.W F3, R6             # F3 = (float)n

    # F2 = sum / n
    FDIV    F2, F2, F3

    # F2 = sum/n + eps
    FADD    F2, F2, F1

    # F2 = 1/sqrt(sum/n + eps)
    FRSQRT  F2, F2

    # ===== Pass 2: Scale by weights =====
    # Restore x pointer
    MV      R4, R7

    # Reset counter
    MV      R8, R6

scale_loop:
    LVF     V0, R4, 4           # Load x[i:i+16]
    LVF     V1, R5, 4           # Load w[i:i+16]
    VMULS   V0, V0, F2          # V0 = x * scale
    VMUL    V0, V0, V1          # V0 = (x * scale) * w
    SVF     V0, R3, 4           # Store to output
    ADDI    R4, R4, 64          # Advance x
    ADDI    R5, R5, 64          # Advance w
    ADDI    R3, R3, 64          # Advance output
    ADDI    R8, R8, -16
    BGTZ    R8, scale_loop

    RET


# Alternative: In-place RMSNorm (output = input)
    .global rmsnorm_inplace

rmsnorm_inplace:
    # R3 = x (input/output)
    # R4 = w (weights)
    # R5 = n
    # F1 = eps

    MV      R6, R3              # Save x pointer
    VSETVL  R10, R5
    FMOV    F2, F0              # sum = 0
    MV      R7, R5              # counter

sum_loop_ip:
    LVF     V0, R3, 4
    VREDSQS F3, V0
    FADD    F2, F2, F3
    ADDI    R3, R3, 64
    ADDI    R7, R7, -16
    BGTZ    R7, sum_loop_ip

    # Compute scale
    FCVT.S.W F3, R5
    FDIV    F2, F2, F3
    FADD    F2, F2, F1
    FRSQRT  F2, F2

    # Restore and scale
    MV      R3, R6
    MV      R7, R5

scale_loop_ip:
    LVF     V0, R3, 4
    LVF     V1, R4, 4
    VMULS   V0, V0, F2
    VMUL    V0, V0, V1
    SVF     V0, R3, 4
    ADDI    R3, R3, 64
    ADDI    R4, R4, 64
    ADDI    R7, R7, -16
    BGTZ    R7, scale_loop_ip

    RET
