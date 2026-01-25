# SMOL-32 Assembly: Multi-Head Grouped-Query Attention
#
# void attention(float *q, float *k_cache, float *v_cache,
#                float *output, float *att, int slen, int max_seq)
#
# Arguments:
#   R3 = q pointer         [nh * hd] floats
#   R4 = k_cache pointer   [nkv][max_seq][hd] floats (already updated with current k)
#   R5 = v_cache pointer   [nkv][max_seq][hd] floats (already updated with current v)
#   R6 = output pointer    [nh * hd] floats (must be pre-zeroed by caller)
#   R7 = att scratch        [nh * max_seq] floats
#   R8 = slen              (pos + 1, number of tokens in cache)
#   R9 = max_seq_len       (stride for kv cache indexing)
#
# Hardcoded constants: hd=64, nh=9, nkv=3, ng=3
# Scale factor: 1/sqrt(64) = 0.125

    .text
    .global attention

attention:
    ADDI    SP, SP, -56
    SW      R16, 0(SP)
    SW      R17, 4(SP)
    SW      R18, 8(SP)
    SW      R19, 12(SP)
    SW      R20, 16(SP)
    SW      R21, 20(SP)
    SW      R22, 24(SP)
    SW      R23, 28(SP)
    SW      R24, 32(SP)
    SW      R25, 36(SP)
    SW      R26, 40(SP)
    SW      R27, 44(SP)
    SW      RA, 48(SP)

    # Save arguments
    MV      R16, R3             # q
    MV      R17, R4             # k_cache
    MV      R18, R5             # v_cache
    MV      R19, R6             # output (xb2)
    MV      R20, R7             # att scratch
    MV      R21, R8             # slen
    MV      R22, R9             # max_seq_len

    # Compute kv_head_stride_bytes = max_seq * hd * 4
    LI      R3, 64              # hd
    MUL     R23, R22, R3        # max_seq * hd (floats)
    SLLI    R23, R23, 2         # * 4 = bytes

    # Compute 1/sqrt(hd) = 1/sqrt(64) = 0.125
    LI      R3, 64
    FCVT.S.W F5, R3
    FRSQRT  F5, F5              # F5 = 0.125

    # att_stride_bytes = max_seq * 4
    SLLI    R24, R22, 2         # R24 = max_seq_len * 4

    # ===== Loop over heads =====
    LI      R25, 9              # num_heads counter
    MV      R26, R16            # current q head pointer
    MV      R27, R20            # current att pointer

head_loop:
    # Compute kvh = (9 - heads_remaining) / 3
    # head_index = 9 - R25
    LI      R3, 9
    SUB     R3, R3, R25         # head_index
    LI      R4, 3
    # Integer division by 3 (quotient in R10)
    LI      R10, 0
div3:
    SLT     R11, R3, R4
    BNEZ    R11, div3_done
    SUB     R3, R3, R4
    ADDI    R10, R10, 1
    BEQ     R0, R0, div3
div3_done:
    # R10 = kvh (0, 1, or 2)

    # kc = k_cache + kvh * kv_head_stride_bytes
    MUL     R11, R10, R23
    ADD     R11, R17, R11       # R11 = kc

    # vc = v_cache + kvh * kv_head_stride_bytes
    MUL     R12, R10, R23
    ADD     R12, R18, R12       # R12 = vc

    # ----- Compute attention scores -----
    # For t = 0..slen-1: att[t] = dot(q_h, kc[t]) * scale
    MV      R13, R21            # t counter = slen
    MV      R14, R27            # att output pointer
    MV      R15, R11            # current kt pointer

score_loop:
    # Dot product: 4 chunks of 16 floats
    FMOV    F2, F0              # accumulator = 0
    MV      R3, R26             # q_h pointer
    MV      R4, R15             # kt pointer
    LI      R10, 4
dot_chunk:
    LVF     V0, R3, 4
    LVF     V1, R4, 4
    VMUL    V2, V0, V1
    VREDSUM F3, V2
    FADD    F2, F2, F3
    ADDI    R3, R3, 64
    ADDI    R4, R4, 64
    LOOP    R10, dot_chunk

    # Scale and store
    FMUL    F2, F2, F5
    SF      F2, 0(R14)

    ADDI    R14, R14, 4         # next att slot
    ADDI    R15, R15, 256       # next cached key (hd * 4)
    LOOP    R13, score_loop

    # ----- Softmax over att[0..slen-1] -----
    # Find max
    MV      R3, R27
    LF      F2, 0(R3)           # max = att[0]
    MV      R13, R21
    ADDI    R13, R13, -1
    ADDI    R3, R3, 4
sm_max:
    BEQZ    R13, sm_exp
    LF      F3, 0(R3)
    FMAX    F2, F2, F3
    ADDI    R3, R3, 4
    ADDI    R13, R13, -1
    BEQ     R0, R0, sm_max

sm_exp:
    # exp(att[i] - max) and sum
    MV      R3, R27
    MV      R13, R21
    FMOV    F4, F0              # sum = 0
sm_exp_loop:
    LF      F3, 0(R3)
    FSUB    F3, F3, F2
    FEXP    F3, F3
    SF      F3, 0(R3)
    FADD    F4, F4, F3
    ADDI    R3, R3, 4
    LOOP    R13, sm_exp_loop

    # Divide by sum
    FRECIP  F4, F4
    MV      R3, R27
    MV      R13, R21
sm_div:
    LF      F3, 0(R3)
    FMUL    F3, F3, F4
    SF      F3, 0(R3)
    ADDI    R3, R3, 4
    LOOP    R13, sm_div

    # ----- Weighted sum of values -----
    # Compute oh pointer: output + head_index * hd * 4
    LI      R3, 9
    SUB     R3, R3, R25         # head_index
    SLLI    R3, R3, 8           # * 256 (hd * 4)
    ADD     R3, R19, R3         # R3 = oh

    MV      R13, R21            # t counter = slen
    MV      R14, R27            # att pointer
    MV      R15, R12            # vc pointer

value_loop:
    LF      F2, 0(R14)          # att[t]

    # oh[d] += att[t] * vc[t][d], 4 chunks of 16
    MV      R4, R3              # oh (preserve across inner loop)
    MV      R10, R15            # vt
    LI      R11, 4
val_chunk:
    LVF     V0, R4, 4           # load oh
    LVF     V1, R10, 4          # load vt
    VMULS   V1, V1, F2          # vt * att[t]
    VADD    V0, V0, V1          # oh += ...
    SVF     V0, R4, 4           # store oh
    ADDI    R4, R4, 64
    ADDI    R10, R10, 64
    LOOP    R11, val_chunk

    ADDI    R14, R14, 4         # next att[t]
    ADDI    R15, R15, 256       # next vt (hd * 4)
    LOOP    R13, value_loop

    # ----- Next head -----
    ADDI    R26, R26, 256       # next q head (hd * 4)
    ADD     R27, R27, R24       # next att row (max_seq * 4)
    LOOP    R25, head_loop

    # Restore and return
    LW      R16, 0(SP)
    LW      R17, 4(SP)
    LW      R18, 8(SP)
    LW      R19, 12(SP)
    LW      R20, 16(SP)
    LW      R21, 20(SP)
    LW      R22, 24(SP)
    LW      R23, 28(SP)
    LW      R24, 32(SP)
    LW      R25, 36(SP)
    LW      R26, 40(SP)
    LW      R27, 44(SP)
    LW      RA, 48(SP)
    ADDI    SP, SP, 56
    RET
