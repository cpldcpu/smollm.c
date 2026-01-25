# SMOL-32 Assembly: Full Transformer Forward Pass
#
# void forward(int token, int position, uint32_t desc_base)
#
# Arguments:
#   R3 = token ID
#   R4 = position (0-based)
#   R5 = model descriptor base address
#
# Returns: logits written to 0x180000
#
# Model Descriptor Layout (at desc_base):
#   +0x00: embed_data_addr
#   +0x04: embed_scale_addr
#   +0x08: final_norm_addr
#   +0x0C: kv_cache_addr
#   +0x10: hidden_size (576)
#   +0x14: head_dim (64)
#   +0x18: num_heads (9)
#   +0x1C: num_kv_heads (3)
#   +0x20: intermediate_size (1536)
#   +0x24: num_layers (30)
#   +0x28: max_seq_len (8192)
#   +0x2C: rms_norm_eps (float)
#   +0x30: vocab_size (49152)
#   +0x34: rope_cos_base
#   +0x38: kv_head_stride_bytes (max_seq * hd * 4)
#   +0x3C: (reserved)
#
# Per-layer descriptor (+0x40, 64 bytes each):
#   +0x00: input_ln_addr
#   +0x04: q_data_addr
#   +0x08: q_scale_addr
#   +0x0C: k_data_addr
#   +0x10: k_scale_addr
#   +0x14: v_data_addr
#   +0x18: v_scale_addr
#   +0x1C: o_data_addr
#   +0x20: o_scale_addr
#   +0x24: post_ln_addr
#   +0x28: gate_data_addr
#   +0x2C: gate_scale_addr
#   +0x30: up_data_addr
#   +0x34: up_scale_addr
#   +0x38: down_data_addr
#   +0x3C: down_scale_addr
#
# Kernel entry points (fixed addresses):
#   0x1000 = matmul_q8
#   0x2000 = rmsnorm
#   0x3000 = rope (apply_rope)
#   0x4000 = attention
#   0x5000 = silu_mul
#   0x6000 = residual
#   0x7000 = embed_dequant
#   0x8000 = memcpy_f
#
# Buffer addresses (fixed):
#   BUF_X   = 0x100000
#   BUF_XB  = 0x101000
#   BUF_XB2 = 0x102000
#   BUF_Q   = 0x103000
#   BUF_K   = 0x104000
#   BUF_V   = 0x105000
#   BUF_ATT = 0x106000
#   BUF_HB  = 0x150000
#   BUF_HB2 = 0x152000
#   LOGITS  = 0x180000

    .text
    .global forward

forward:
    # ===== PROLOGUE =====
    ADDI    SP, SP, -72
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
    SW      R28, 48(SP)
    SW      R29, 52(SP)
    SW      RA, 56(SP)

    # Save arguments into callee-saved registers
    MV      R16, R5             # R16 = descriptor base
    MV      R17, R4             # R17 = position

    # Load constants from descriptor
    LW      R28, 16(R16)        # hidden_size (576)
    LW      R22, 56(R16)        # kv_head_stride_bytes
    LW      R26, 40(R16)        # max_seq_len (8192)

    # Compute BUF_X base address = 0x100000
    LI      R27, 16
    SLLI    R27, R27, 16        # R27 = 0x100000

    # slen = position + 1
    ADDI    R25, R17, 1

    # ===== EMBEDDING DEQUANT =====
    # output[i] = (float)embed_data[token*hs + i] * scale
    LW      R10, 0(R16)         # embed_data_addr
    MUL     R11, R3, R28        # token * hidden_size
    ADD     R4, R10, R11        # input = embed_data + token*hs
    MV      R3, R27             # output = BUF_X
    MV      R5, R28             # n = hidden_size
    LW      R10, 4(R16)         # embed_scale_addr
    LF      F1, 0(R10)          # F1 = embed_scale
    LI      R10, 0x7000         # EMBED_ENTRY
    JALR    RA, R10, 0

    # ===== COMPUTE ROPE ADDRESSES =====
    # cos_addr = rope_cos_base + pos * head_dim * 4
    # sin_addr = cos_addr + kv_head_stride_bytes (= max_seq * hd * 4)
    LW      R20, 52(R16)        # rope_cos_base
    SLLI    R10, R17, 8         # pos * 256 (pos * hd * 4)
    ADD     R20, R20, R10       # R20 = cos_addr
    ADD     R21, R20, R22       # R21 = sin_addr

    # ===== SETUP KV CACHE POINTERS =====
    LW      R23, 12(R16)        # R23 = kv_cache_addr = layer_k_cache[0]
    # layer_v_cache = layer_k_cache + nkv * kv_head_stride_bytes (nkv=3)
    MV      R10, R22
    SLLI    R10, R10, 1         # 2 * stride
    ADD     R10, R10, R22       # 3 * stride
    ADD     R24, R23, R10       # R24 = layer_v_cache[0]

    # ===== LAYER LOOP SETUP =====
    LW      R19, 36(R16)        # num_layers (30)
    MV      R18, R16
    ADDI    R18, R18, 64        # R18 = first layer descriptor

    # Load eps into F10 (preserved across all kernel calls)
    LF      F10, 44(R16)        # rms_norm_eps

layer_loop:

    # ===== INPUT RMSNORM: BUF_X -> BUF_XB =====
    MV      R3, R27
    ADDI    R3, R3, 0x1000     # output = BUF_XB
    MV      R4, R27             # input = BUF_X
    LW      R5, 0(R18)          # weight = input_ln_addr
    MV      R6, R28             # n = hidden_size
    FMOV    F1, F10             # eps
    LI      R10, 0x2000         # RMSNORM_ENTRY
    JALR    RA, R10, 0

    # ===== Q PROJECTION: BUF_XB -> BUF_Q (576x576) =====
    MV      R3, R27
    ADDI    R3, R3, 0x3000     # output = BUF_Q
    LW      R4, 4(R18)          # q_data_addr
    LW      R5, 8(R18)          # q_scale_addr
    MV      R6, R27
    ADDI    R6, R6, 0x1000     # input = BUF_XB
    MV      R7, R28             # rows = 576
    MV      R8, R28             # cols = 576
    LI      R10, 0x1000         # MATMUL_ENTRY
    JALR    RA, R10, 0

    # ===== K PROJECTION: BUF_XB -> BUF_K (192x576) =====
    MV      R3, R27
    ADDI    R3, R3, 0x4000     # output = BUF_K
    LW      R4, 12(R18)         # k_data_addr
    LW      R5, 16(R18)         # k_scale_addr
    MV      R6, R27
    ADDI    R6, R6, 0x1000     # input = BUF_XB
    LI      R7, 192             # rows = nkv*hd
    MV      R8, R28             # cols = 576
    LI      R10, 0x1000
    JALR    RA, R10, 0

    # ===== V PROJECTION: BUF_XB -> BUF_V (192x576) =====
    MV      R3, R27
    ADDI    R3, R3, 0x5000     # output = BUF_V
    LW      R4, 20(R18)         # v_data_addr
    LW      R5, 24(R18)         # v_scale_addr
    MV      R6, R27
    ADDI    R6, R6, 0x1000     # input = BUF_XB
    LI      R7, 192
    MV      R8, R28
    LI      R10, 0x1000
    JALR    RA, R10, 0

    # ===== ROPE ON Q (9 heads) =====
    MV      R29, R27
    ADDI    R29, R29, 0x3000   # R29 = current Q head pointer
    LI      R10, 9
    SW      R10, 60(SP)         # head counter on stack

rope_q_loop:
    MV      R3, R29             # vec addr
    LI      R4, 64              # head_dim
    MV      R5, R20             # cos_addr
    MV      R6, R21             # sin_addr
    LI      R10, 0x3000         # ROPE_ENTRY
    JALR    RA, R10, 0
    ADDI    R29, R29, 256       # next head (hd*4 = 256 bytes)
    LW      R10, 60(SP)
    ADDI    R10, R10, -1
    SW      R10, 60(SP)
    BNEZ    R10, rope_q_loop

    # ===== ROPE ON K (3 heads) =====
    MV      R29, R27
    ADDI    R29, R29, 0x4000   # R29 = BUF_K
    LI      R10, 3
    SW      R10, 60(SP)

rope_k_loop:
    MV      R3, R29
    LI      R4, 64
    MV      R5, R20
    MV      R6, R21
    LI      R10, 0x3000
    JALR    RA, R10, 0
    ADDI    R29, R29, 256
    LW      R10, 60(SP)
    ADDI    R10, R10, -1
    SW      R10, 60(SP)
    BNEZ    R10, rope_k_loop

    # ===== KV CACHE SCATTER (3 heads) =====
    # Precompute pos_offset = pos * hd * 4 = pos * 256
    SLLI    R10, R17, 8
    SW      R10, 64(SP)         # save pos_offset

    # -- Head 0 --
    ADD     R3, R23, R10        # k_dst = k_cache + pos_offset
    MV      R4, R27
    ADDI    R4, R4, 0x4000     # src = BUF_K + 0
    LI      R5, 64              # n = hd
    LI      R10, 0x7FFF
    ADDI    R10, R10, 1         # MEMCPY_ENTRY = 0x8000
    JALR    RA, R10, 0

    LW      R10, 64(SP)
    ADD     R3, R24, R10        # v_dst = v_cache + pos_offset
    MV      R4, R27
    ADDI    R4, R4, 0x5000     # src = BUF_V + 0
    LI      R5, 64
    LI      R10, 0x7FFF
    ADDI    R10, R10, 1
    JALR    RA, R10, 0

    # -- Head 1 --
    LW      R10, 64(SP)
    ADD     R10, R10, R22       # pos_offset + 1*kv_head_stride
    ADD     R3, R23, R10
    MV      R4, R27
    ADDI    R4, R4, 0x4000
    ADDI    R4, R4, 256         # BUF_K + hd*4
    LI      R5, 64
    LI      R11, 0x7FFF
    ADDI    R11, R11, 1
    JALR    RA, R11, 0

    LW      R10, 64(SP)
    ADD     R10, R10, R22
    ADD     R3, R24, R10
    MV      R4, R27
    ADDI    R4, R4, 0x5000
    ADDI    R4, R4, 256         # BUF_V + hd*4
    LI      R5, 64
    LI      R11, 0x7FFF
    ADDI    R11, R11, 1
    JALR    RA, R11, 0

    # -- Head 2 --
    LW      R10, 64(SP)
    ADD     R10, R10, R22
    ADD     R10, R10, R22       # pos_offset + 2*kv_head_stride
    ADD     R3, R23, R10
    MV      R4, R27
    ADDI    R4, R4, 0x4000
    ADDI    R4, R4, 512         # BUF_K + 2*hd*4
    LI      R5, 64
    LI      R11, 0x7FFF
    ADDI    R11, R11, 1
    JALR    RA, R11, 0

    LW      R10, 64(SP)
    ADD     R10, R10, R22
    ADD     R10, R10, R22
    ADD     R3, R24, R10
    MV      R4, R27
    ADDI    R4, R4, 0x5000
    ADDI    R4, R4, 512         # BUF_V + 2*hd*4
    LI      R5, 64
    LI      R11, 0x7FFF
    ADDI    R11, R11, 1
    JALR    RA, R11, 0

    # ===== ZERO BUF_XB2 (attention output accumulator) =====
    MV      R3, R27
    ADDI    R3, R3, 0x2000     # BUF_XB2
    VSUB    V0, V0, V0          # V0 = all zeros
    LI      R10, 36             # 576/16
zero_xb2:
    SVF     V0, R3, 4
    ADDI    R3, R3, 64
    LOOP    R10, zero_xb2

    # ===== MULTI-HEAD ATTENTION =====
    MV      R3, R27
    ADDI    R3, R3, 0x3000     # q = BUF_Q
    MV      R4, R23             # k_cache
    MV      R5, R24             # v_cache
    MV      R6, R27
    ADDI    R6, R6, 0x2000     # output = BUF_XB2
    MV      R7, R27
    ADDI    R7, R7, 0x6000     # att scratch = BUF_ATT
    MV      R8, R25             # slen = pos + 1
    MV      R9, R26             # max_seq_len
    LI      R10, 0x4000         # ATTN_ENTRY
    JALR    RA, R10, 0

    # ===== O PROJECTION: BUF_XB2 -> BUF_XB (576x576) =====
    MV      R3, R27
    ADDI    R3, R3, 0x1000     # output = BUF_XB
    LW      R4, 28(R18)         # o_data_addr
    LW      R5, 32(R18)         # o_scale_addr
    MV      R6, R27
    ADDI    R6, R6, 0x2000     # input = BUF_XB2
    MV      R7, R28             # rows = 576
    MV      R8, R28             # cols = 576
    LI      R10, 0x1000
    JALR    RA, R10, 0

    # ===== RESIDUAL: BUF_X += BUF_XB =====
    MV      R3, R27             # dst = BUF_X
    MV      R4, R27
    ADDI    R4, R4, 0x1000     # src = BUF_XB
    MV      R5, R28             # n = hidden_size
    LI      R10, 0x6000         # RESIDUAL_ENTRY
    JALR    RA, R10, 0

    # ===== POST-ATTENTION RMSNORM: BUF_X -> BUF_XB =====
    MV      R3, R27
    ADDI    R3, R3, 0x1000     # output = BUF_XB
    MV      R4, R27             # input = BUF_X
    LW      R5, 36(R18)         # weight = post_ln_addr
    MV      R6, R28             # n = hidden_size
    FMOV    F1, F10             # eps
    LI      R10, 0x2000
    JALR    RA, R10, 0

    # ===== GATE PROJECTION: BUF_XB -> BUF_HB (576 -> 1536) =====
    LI      R3, 21
    SLLI    R3, R3, 16          # output = 0x150000 = BUF_HB
    LW      R4, 40(R18)         # gate_data_addr
    LW      R5, 44(R18)         # gate_scale_addr
    MV      R6, R27
    ADDI    R6, R6, 0x1000     # input = BUF_XB
    LW      R7, 32(R16)         # rows = intermediate_size (1536)
    MV      R8, R28             # cols = hidden_size (576)
    LI      R10, 0x1000
    JALR    RA, R10, 0

    # ===== UP PROJECTION: BUF_XB -> BUF_HB2 (576 -> 1536) =====
    LI      R3, 21
    SLLI    R3, R3, 16
    ADDI    R3, R3, 0x2000     # output = 0x152000 = BUF_HB2
    LW      R4, 48(R18)         # up_data_addr
    LW      R5, 52(R18)         # up_scale_addr
    MV      R6, R27
    ADDI    R6, R6, 0x1000     # input = BUF_XB
    LW      R7, 32(R16)         # rows = intermediate_size
    MV      R8, R28             # cols = hidden_size
    LI      R10, 0x1000
    JALR    RA, R10, 0

    # ===== SILU-GATE: BUF_HB = silu(BUF_HB) * BUF_HB2 =====
    LI      R3, 21
    SLLI    R3, R3, 16          # gate = BUF_HB
    LI      R4, 21
    SLLI    R4, R4, 16
    ADDI    R4, R4, 0x2000     # up = BUF_HB2
    LW      R5, 32(R16)         # n = intermediate_size
    LI      R10, 0x5000         # SILU_ENTRY
    JALR    RA, R10, 0

    # ===== DOWN PROJECTION: BUF_HB -> BUF_XB (1536 -> 576) =====
    MV      R3, R27
    ADDI    R3, R3, 0x1000     # output = BUF_XB
    LW      R4, 56(R18)         # down_data_addr
    LW      R5, 60(R18)         # down_scale_addr
    LI      R6, 21
    SLLI    R6, R6, 16          # input = BUF_HB (0x150000)
    MV      R7, R28             # rows = hidden_size (576)
    LW      R8, 32(R16)         # cols = intermediate_size (1536)
    LI      R10, 0x1000
    JALR    RA, R10, 0

    # ===== RESIDUAL: BUF_X += BUF_XB =====
    MV      R3, R27             # dst = BUF_X
    MV      R4, R27
    ADDI    R4, R4, 0x1000     # src = BUF_XB
    MV      R5, R28             # n = hidden_size
    LI      R10, 0x6000
    JALR    RA, R10, 0

    # ===== ADVANCE TO NEXT LAYER =====
    ADDI    R18, R18, 64        # next layer descriptor

    # Advance KV cache: stride = nkv*2*kv_head_stride = 6*R22
    MV      R10, R22
    SLLI    R10, R10, 1         # 2*stride
    ADD     R11, R10, R22       # 3*stride (= nkv*stride = v_offset)
    ADD     R10, R11, R11       # 6*stride (= layer stride)
    ADD     R23, R23, R10       # next layer k_cache
    ADD     R24, R23, R11       # v_cache = k_cache + v_offset

    LOOP    R19, layer_loop

    # ===== FINAL RMSNORM (in-place on BUF_X) =====
    MV      R3, R27             # output = BUF_X
    MV      R4, R27             # input = BUF_X
    LW      R5, 8(R16)          # weight = final_norm_addr
    MV      R6, R28             # n = hidden_size
    FMOV    F1, F10             # eps
    LI      R10, 0x2000
    JALR    RA, R10, 0

    # ===== LM HEAD: BUF_X -> LOGITS (576 -> 49152) =====
    LI      R3, 24
    SLLI    R3, R3, 16          # output = 0x180000
    LW      R4, 0(R16)          # weight = embed_data (tied)
    LW      R5, 4(R16)          # scale = embed_scale_addr
    MV      R6, R27             # input = BUF_X
    LW      R7, 48(R16)         # rows = vocab_size (49152)
    MV      R8, R28             # cols = hidden_size (576)
    LI      R10, 0x1000
    JALR    RA, R10, 0

    # ===== EPILOGUE =====
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
    LW      R28, 48(SP)
    LW      R29, 52(SP)
    LW      RA, 56(SP)
    ADDI    SP, SP, 72
    RET
