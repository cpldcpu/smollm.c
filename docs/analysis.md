# Computational Analysis: C Code → SMOL-32 ISA

## Operation Frequency Analysis

Profiling the SmolLM2 inference reveals the following operation distribution:

```
┌─────────────────────────────────────────────────────────────────┐
│ Operation Breakdown (per token, 576 hidden, 30 layers)         │
├─────────────────────┬──────────────┬───────────┬───────────────┤
│ Operation           │ Count        │ FLOPs     │ % of Total    │
├─────────────────────┼──────────────┼───────────┼───────────────┤
│ Q8 Matmul (QKV)     │ 90           │ 59.7M     │ 22.4%         │
│ Q8 Matmul (O proj)  │ 30           │ 9.9M      │ 3.7%          │
│ Q8 Matmul (MLP)     │ 90           │ 159.2M    │ 59.8%         │
│ Q8 Embedding        │ 1            │ 28.3M     │ 10.6%         │
│ Attention scores    │ 270 heads    │ 4.4M      │ 1.7%          │
│ Attention weighted  │ 270 heads    │ 4.4M      │ 1.7%          │
│ RMSNorm             │ 61           │ 105K      │ <0.1%         │
│ RoPE                │ 360 heads    │ 46K       │ <0.1%         │
│ SiLU + elemwise     │ 30           │ 92K       │ <0.1%         │
│ Softmax             │ 270          │ ~varies   │ <0.1%         │
├─────────────────────┼──────────────┼───────────┼───────────────┤
│ TOTAL               │              │ ~266M     │ 100%          │
└─────────────────────┴──────────────┴───────────┴───────────────┘
```

**Key insight:** 96% of compute is Q8 matrix-vector multiplication.

---

## C Code to ISA Mapping

### 1. Q8 Matrix-Vector Multiply (96% of compute)

**C Code:**
```c
static void matmul_q8(float *o, Q8Tensor *W, float *x) {
    int rows = W->rows, cols = W->cols;
    float s = W->scale;
    int8_t *d = W->data;
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        int8_t *row = d + i * cols;
        for (int j = 0; j < cols; j++)
            sum += row[j] * s * x[j];
        o[i] = sum;
    }
}
```

**Computational Pattern:**
```
For each row i:
    acc = 0
    For each col j (unroll by 16):
        acc += (int8)W[i,j] * scale * (float)x[j]
    o[i] = acc
```

**ISA Optimization:**
```
┌────────────────────────────────────────────────────────────┐
│ Q8MACINC Instruction                                       │
│                                                            │
│ Single cycle operation:                                    │
│ 1. Load 16× INT8 from QBASE                               │
│ 2. Convert INT8 → FP32                                    │
│ 3. Multiply by SCALE (broadcast)                          │
│ 4. Load 16× FP32 from FBASE                               │
│ 5. Element-wise multiply                                  │
│ 6. Horizontal sum → ACC                                   │
│ 7. QBASE += 16, FBASE += 64                               │
│                                                            │
│ Replaces: 16 loads + 16 converts + 32 muls + 15 adds     │
│ Speedup: ~80× per inner loop iteration                    │
└────────────────────────────────────────────────────────────┘
```

### 2. RMSNorm

**C Code:**
```c
static void rmsnorm(float *o, float *x, float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}
```

**ISA Mapping:**
```
Pass 1: VREDSQS (sum of squares reduction)
        VSETVL R10, R6          # VL = 16
        FMOV   F2, F0           # sum = 0
.loop1: LVF    V0, R7, 4        # Load x[i:i+16]
        VREDSQS F3, V0          # F3 = Σ x[i]²
        FADD   F2, F2, F3       # Accumulate
        ...loop...

Compute: FDIV + FADD + FRSQRT
        FDIV   F2, F2, F3       # ss/n
        FADD   F2, F2, F1       # + eps
        FRSQRT F2, F2           # 1/sqrt(...)

Pass 2: Vector scale
.loop2: LVF    V0, R7, 4        # Load x
        LVF    V1, R5, 4        # Load w
        VMULS  V0, V0, F2       # x * scale
        VMUL   V0, V0, V1       # * w
        SVF    V0, R3, 4        # Store
        ...loop...
```

**Alternative (fused):**
```
VRMS vd, vx, vw, feps    # Single instruction for full RMSNorm
```

### 3. RoPE (Rotary Position Embedding)

**C Code:**
```c
static void apply_rope(float *v, int hd, float *c, float *s) {
    int h = hd / 2;
    for (int i = 0; i < h; i++) {
        float v0 = v[i], v1 = v[i + h];
        v[i] = v0 * c[i] - v1 * s[i];
        v[i + h] = v1 * c[i + h] + v0 * s[i + h];
    }
}
```

**Pattern:** Complex rotation
```
[v0']   [cos  -sin] [v0]
[v1'] = [sin   cos] [v1]
```

**ISA Mapping:**
```
ROPE V4, V5, V2, V3
# V4 = V0 * V2 - V1 * V3  (first half)
# V5 = V1 * V2 + V0 * V3  (second half)

Without fused instruction:
    VMUL   V4, V0, V2       # v0 * cos
    VMUL   V5, V1, V3       # v1 * sin
    VSUB   V4, V4, V5       # v0*cos - v1*sin
    VMUL   V5, V1, V2       # v1 * cos
    VMUL   V6, V0, V3       # v0 * sin
    VADD   V5, V5, V6       # v1*cos + v0*sin
```

### 4. Attention Score Computation

**C Code:**
```c
for (int t = 0; t < slen; t++) {
    float score = 0;
    float *kt = kc + t * hd;
    for (int d = 0; d < hd; d++)
        score += qh[d] * kt[d];
    att[t] = score * sc;
}
```

**ISA Mapping:** Dot product with broadcast scale
```
    LVF    V0, R_Q, 4        # Load query vector (16 floats)
    FMOV   F10, F0           # score = 0

.t_loop:
    LVF    V1, R_KT, 4       # Load key[t] (strided by hd)
    VMUL   V2, V0, V1        # Element-wise multiply
    VREDSUM F11, V2          # Horizontal sum
    FADD   F10, F10, F11     # Accumulate (for hd > 16)
    ... (repeat for full head_dim)

    FMUL   F10, F10, F_SCALE # * 1/sqrt(hd)
    SF     F10, R_ATT        # Store attention score
```

### 5. Softmax

**C Code:**
```c
static void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < n; i++)
        x[i] /= sum;
}
```

**ISA Mapping:**
```
# Pass 1: Find max
    FMOV   F1, F0
.max: LVF    V0, R3, 4
      VREDMAX F2, V0
      FMAX   F1, F1, F2
      ...loop...

# Pass 2: exp(x - max) and sum
    FMOV   F3, F0           # sum = 0
.exp: LVF    V0, R3, 4
      VADDS  V0, V0, F1_NEG # x - max (F1_NEG = -max)
      VEXP   V0, V0         # exp(x - max)
      SVF    V0, R3, 4      # Store back
      VREDSUM F2, V0        # Sum
      FADD   F3, F3, F2
      ...loop...

# Pass 3: Divide by sum
    FRECIP F3, F3           # 1/sum
.div: LVF    V0, R3, 4
      VMULS  V0, V0, F3     # x / sum
      SVF    V0, R3, 4
      ...loop...
```

**Alternative (fused):**
```
VSOFTMAX Vd, Vs, len    # Full softmax in hardware
```

### 6. SiLU Activation with Element-wise Multiply

**C Code:**
```c
static float silu(float x) { return x / (1.0f + expf(-x)); }
// ...
for (int i = 0; i < c->intermediate_size; i++)
    m->hb[i] = silu(m->hb[i]) * m->hb2[i];
```

**ISA Mapping:**
```
.silu_mul:
    LVF    V0, R_GATE, 4    # Load gate (to apply SiLU)
    LVF    V1, R_UP, 4      # Load up projection
    VSILU  V0, V0           # V0 = SiLU(V0)
    VMUL   V0, V0, V1       # V0 = SiLU(gate) * up
    SVF    V0, R_OUT, 4     # Store
    ...loop...
```

---

## Data Flow Through Transformer Layer

```
                Input: x[576]
                      │
              ┌───────▼───────┐
              │   RMSNorm     │  VRMS instruction
              │  (61 cycles)  │
              └───────┬───────┘
                      │ xb[576]
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
    │ Q_proj  │  │ K_proj  │  │ V_proj  │  Q8MAC × 3
    │ (1296)  │  │ (432)   │  │ (432)   │
    └────┬────┘  └────┬────┘  └────┬────┘
         │ q[576]     │ k[192]     │ v[192]
         │            │            │
    ┌────▼────┐  ┌────▼────┐       │
    │  RoPE   │  │  RoPE   │       │  ROPE instruction
    │  (36)   │  │  (12)   │       │
    └────┬────┘  └────┬────┘       │
         │            │            │
         │       ┌────▼────────────▼────┐
         │       │     KV Cache         │  Vector stores
         │       │  k_cache, v_cache    │
         │       └──────────┬───────────┘
         │                  │
    ┌────▼──────────────────▼────┐
    │      Attention (9 heads)   │  Dot products + softmax
    │      Score: Q × K^T        │
    │      Softmax               │
    │      Output: Att × V       │
    │      (~550 cycles/head)    │
    └─────────────┬──────────────┘
                  │ xb2[576]
             ┌────▼────┐
             │ O_proj  │  Q8MAC
             │ (1296)  │
             └────┬────┘
                  │
             ┌────▼────┐
             │ Residual│  Vector add
             │  x += xb│
             └────┬────┘
                  │
              ┌───▼───┐
              │RMSNorm│
              └───┬───┘
                  │
         ┌────────┼────────┐
         │        │        │
    ┌────▼────┐ ┌─▼──┐     │
    │Gate_proj│ │Up  │     │  Q8MAC × 2
    │ (2592)  │ │proj│     │
    └────┬────┘ └─┬──┘     │
         │        │        │
    ┌────▼────────▼────┐   │
    │ SiLU(gate) × up  │   │  VSILU + VMUL
    └────────┬─────────┘   │
             │             │
        ┌────▼────┐        │
        │Down_proj│        │  Q8MAC
        │ (2592)  │        │
        └────┬────┘        │
             │             │
        ┌────▼─────────────▼────┐
        │     Residual          │  Vector add
        │     x += mlp_out      │
        └───────────┬───────────┘
                    │
                Output: x[576]
```

---

## Cycle Estimates per Layer

| Component | Operations | SMOL-32 Cycles |
|-----------|------------|----------------|
| RMSNorm (input) | 576 sum_sq + rsqrt + 576 mul | ~80 |
| Q/K/V proj | 3 × (576×576) Q8MAC | ~3,888 |
| RoPE | 9+3 heads × 64 dim | ~48 |
| KV cache write | 192 × 2 stores | ~24 |
| Attention (9 heads) | 9 × (dot + softmax + weighted) | ~4,500 |
| O projection | 576×576 Q8MAC | ~1,296 |
| Residual add | 576 adds | ~36 |
| RMSNorm (post) | Same as input | ~80 |
| Gate + Up proj | 2 × (1536×576) Q8MAC | ~5,184 |
| SiLU × elemwise | 1536 ops | ~96 |
| Down proj | 576×1536 Q8MAC | ~2,592 |
| Residual add | 576 adds | ~36 |
| **Layer total** | | **~17,860** |

**30 layers total:** ~535,800 cycles

**Full token (+ embedding + LM head):** ~600,000 cycles

At 500 MHz: **~1.2 ms per token** (~830 tokens/second)

---

## Memory Bandwidth Requirements

| Access Type | Bytes/Token | Description |
|-------------|-------------|-------------|
| Weights (streaming) | 129 MB | Full model read once |
| Activations (R/W) | ~50 KB | Layer activations |
| KV Cache (growing) | ~74 KB/tok | Cached keys/values |

**Required bandwidth:** 129 MB / 1.2 ms ≈ **107 GB/s**

This is achievable with:
- HBM2 (256+ GB/s)
- Multiple LPDDR5 channels
- On-chip weight buffer with prefetching

---

## Summary

The SMOL-32 ISA achieves efficiency through:

1. **Q8MAC fusion** - 80× speedup on inner loop
2. **Vector processing** - 16× parallelism on activations
3. **Special functions** - Single-cycle exp/sqrt/rsqrt
4. **Transformer primitives** - ROPE, VRMS reduce instruction count
5. **Streaming memory** - Prefetch and burst access for weights

**Estimated performance:** 35-40× faster than scalar implementation.
