# SMOL-32 Assembly: Embedding Dequantization (int8 -> float * scale)
#
# void embed_dequant(float *output, int8_t *input, int n, float scale)
#
# Arguments:
#   R3 = output pointer (float*)
#   R4 = input pointer (int8_t*)
#   R5 = n (number of elements, must be multiple of 4)
#   F1 = scale factor
#
# Computes: output[i] = (float)input[i] * scale for i = 0..n-1
# Processes 4 bytes per iteration using LW + byte extraction.

    .text
    .global embed_dequant

embed_dequant:

embed_loop:
    # Load 4 bytes as a word
    LW      R10, 0(R4)

    # Byte 0 (bits 7:0): sign-extend lowest byte
    SLLI    R11, R10, 24
    SRAI    R11, R11, 24
    FCVT.S.W F2, R11
    FMUL    F2, F2, F1
    SF      F2, 0(R3)

    # Byte 1 (bits 15:8)
    SLLI    R11, R10, 16
    SRAI    R11, R11, 24
    FCVT.S.W F2, R11
    FMUL    F2, F2, F1
    SF      F2, 4(R3)

    # Byte 2 (bits 23:16)
    SLLI    R11, R10, 8
    SRAI    R11, R11, 24
    FCVT.S.W F2, R11
    FMUL    F2, F2, F1
    SF      F2, 8(R3)

    # Byte 3 (bits 31:24): arithmetic right shift
    SRAI    R11, R10, 24
    FCVT.S.W F2, R11
    FMUL    F2, F2, F1
    SF      F2, 12(R3)

    # Advance pointers
    ADDI    R3, R3, 16         # output += 4 floats
    ADDI    R4, R4, 4          # input += 4 bytes
    ADDI    R5, R5, -4
    BGTZ    R5, embed_loop

    RET
