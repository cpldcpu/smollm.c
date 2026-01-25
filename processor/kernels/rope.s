# SMOL-32 Assembly: Rotary Position Embedding
#
# void apply_rope(float *v, int head_dim, float *cos, float *sin)
#
# Arguments:
#   R3 = v pointer (query or key vector)
#   R4 = head_dim (e.g., 64)
#   R5 = cos pointer
#   R6 = sin pointer
#
# Computes:
#   v[i]     = v[i] * cos[i] - v[i+h] * sin[i]
#   v[i+h]   = v[i+h] * cos[i+h] + v[i] * sin[i+h]
# where h = head_dim / 2

    .text
    .global apply_rope

apply_rope:
    # h = head_dim / 2
    SRLI    R7, R4, 1           # R7 = h = head_dim / 2

    # R8 = byte offset to second half = h * 4
    SLLI    R8, R7, 2

    # Set vector length (process 16 elements at a time)
    VSETVL  R10, R7             # VL = min(h, 16)

    # R9 = counter
    MV      R9, R7

rope_loop:
    # Load v[0:16] (first half)
    LVF     V0, R3, 4

    # Load v[h:h+16] (second half)
    ADD     R11, R3, R8
    LVF     V1, R11, 4

    # Load cos and sin for this position
    LVF     V2, R5, 4           # cos[0:16]
    LVF     V3, R6, 4           # sin[0:16]

    # Method 1: Using fused ROPE instruction (if available)
    # ROPE V4, V5, V2, V3      # V4 = V0*cos - V1*sin, V5 = V1*cos + V0*sin

    # Method 2: Manual computation (more portable)
    # First half: v[i] = v0 * cos - v1 * sin
    VMUL    V4, V0, V2          # V4 = v0 * cos
    VMUL    V5, V1, V3          # V5 = v1 * sin
    VSUB    V4, V4, V5          # V4 = v0*cos - v1*sin

    # Second half: v[i+h] = v1 * cos + v0 * sin
    VMUL    V5, V1, V2          # V5 = v1 * cos
    VMUL    V6, V0, V3          # V6 = v0 * sin
    VADD    V5, V5, V6          # V5 = v1*cos + v0*sin

    # Store results
    SVF     V4, R3, 4           # Store first half
    SVF     V5, R11, 4          # Store second half

    # Advance pointers
    ADDI    R3, R3, 64          # v += 16
    ADDI    R5, R5, 64          # cos += 16
    ADDI    R6, R6, 64          # sin += 16

    # Decrement counter
    ADDI    R9, R9, -16
    BGTZ    R9, rope_loop

    RET


# Apply RoPE to all heads in Q and K
# void apply_rope_all(float *q, float *k, int num_heads, int num_kv_heads,
#                     int head_dim, float *cos, float *sin)

    .global apply_rope_qk

apply_rope_qk:
    # Arguments:
    # R3 = q pointer
    # R4 = k pointer
    # R5 = num_heads
    # R6 = num_kv_heads
    # R7 = head_dim
    # Stack: cos, sin pointers

    # Save callee-saved registers
    ADDI    SP, SP, -32
    SW      R16, 0(SP)
    SW      R17, 4(SP)
    SW      R18, 8(SP)
    SW      R19, 12(SP)
    SW      R20, 16(SP)
    SW      RA, 20(SP)

    # Load cos/sin from stack
    LW      R16, 32(SP)         # cos
    LW      R17, 36(SP)         # sin

    MV      R18, R3             # Save q
    MV      R19, R4             # Save k
    MV      R20, R7             # Save head_dim

    # Apply RoPE to Q heads
    MV      R9, R5              # counter = num_heads

q_head_loop:
    MV      R3, R18             # v = current q head
    MV      R4, R20             # head_dim
    MV      R5, R16             # cos
    MV      R6, R17             # sin
    JAL     RA, apply_rope      # Apply RoPE

    # Advance to next head
    SLLI    R10, R20, 2         # head_dim * 4
    ADD     R18, R18, R10

    LOOP    R9, q_head_loop

    # Apply RoPE to K heads (num_kv_heads)
    LW      R9, -24(SP)         # Reload num_kv_heads (passed differently)
    # ... similar loop for K

    # Restore and return
    LW      R16, 0(SP)
    LW      R17, 4(SP)
    LW      R18, 8(SP)
    LW      R19, 12(SP)
    LW      R20, 16(SP)
    LW      RA, 20(SP)
    ADDI    SP, SP, 32
    RET
