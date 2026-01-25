# SMOL-32 Assembly: Q8 Matrix-Vector Multiply
#
# void matmul_q8(float *o, Q8Tensor *W, float *x)
#
# Arguments:
#   R3 = output pointer (float*)
#   R4 = weight data pointer (int8_t*)
#   R5 = scale pointer (float*)
#   R6 = input pointer (float*)
#   R7 = rows
#   R8 = cols
#
# This computes: o[i] = sum_j(W[i,j] * scale * x[j])

    .text
    .global matmul_q8

matmul_q8:
    # Load scale into FP register and set up Q8 unit
    LF      F1, 0(R5)           # F1 = scale
    QSETSCALE F1                # Set Q8 dequant scale
    FSETBASE R6                 # Set activation base pointer

    # R9 = row counter
    MV      R9, R7

row_loop:
    # Set Q8 weight base to current row
    QSETBASE R4

    # Reset activation base (Q8MACINC advances it)
    FSETBASE R6

    # Clear accumulator
    ACCZERO

    # R10 = number of 16-element chunks
    MV      R10, R8
    SRLI    R10, R10, 4         # cols / 16

col_loop:
    # Fused Q8 MAC: ACC += sum(Q8[0:16] * scale * FP32[0:16])
    # Auto-increments QBASE by 16 and FBASE by 64
    Q8MACINC 16

    # Decrement chunk counter and loop
    LOOP    R10, col_loop

    # Read accumulated result
    ACCREAD F2                  # F2 = sum

    # Store to output
    SF      F2, 0(R3)

    # Advance output pointer
    ADDI    R3, R3, 4

    # Advance weight pointer to next row
    ADD     R4, R4, R8

    # Decrement row counter and loop
    LOOP    R9, row_loop

    RET


# Optimized version using software pipelining
# Overlaps MAC with pointer arithmetic

    .global matmul_q8_opt

matmul_q8_opt:
    LF      F1, 0(R5)
    QSETSCALE F1
    FSETBASE R6
    MV      R9, R7

row_loop_opt:
    QSETBASE R4
    FSETBASE R6
    ACCZERO

    # Compute chunks and remainder
    MV      R10, R8
    SRLI    R10, R10, 4         # chunks = cols / 16

    # Prefetch first chunk (if hardware supports)
    # PREFETCH R4, 0

col_loop_opt:
    Q8MACINC 16                 # MAC and increment QBASE+FBASE
    LOOP    R10, col_loop_opt

    ACCREAD F2
    SF      F2, 0(R3)
    ADDI    R3, R3, 4
    ADD     R4, R4, R8
    LOOP    R9, row_loop_opt

    RET
