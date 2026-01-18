# SmolLM2 C Performance Optimization Proposal

## Executive Summary

This document outlines performance optimization opportunities for the SmolLM2 C inference engine. The current implementation is clean and functional but uses scalar operations throughout. Based on code analysis, we can expect **3-10x speedup** through systematic optimizations.

## Current Performance Characteristics

### Hotspots (by estimated compute time):
1. **Matrix multiplications (matmul_q8)**: ~70-80% of runtime
   - Called 8 times per layer (Q/K/V/O + gate/up/down + final LM head)
   - 30 layers × 7 matmuls/layer = 210+ matmuls per token
2. **Attention computation**: ~10-15% of runtime
3. **Normalization & activations**: ~5-10% of runtime
4. **Everything else**: ~5% of runtime

---

## Optimization Roadmap

### 🔴 **Critical Priority** - Expected 2-5x speedup

#### 1. SIMD Vectorization for matmul_q8 (lines 20-27)

**Current code:**
```c
static void matmul_q8(float *o, Q8Tensor *W, float *x) {
    int rows = W->rows, cols = W->cols; float s = W->scale; int8_t *d = W->data;
    for (int i = 0; i < rows; i++) {
        float sum = 0; int8_t *row = d + i * cols;
        for (int j = 0; j < cols; j++) sum += row[j] * s * x[j];
        o[i] = sum;
    }
}
```

**Issues:**
- Scalar inner loop - no vectorization
- Scale factor multiplied inside loop (wasteful)
- No loop unrolling
- Cache inefficient for large matrices

**Proposed solution:**
- Use AVX2/AVX-512 intrinsics for 8x/16x parallelism
- Use `_mm256_dpbusd_epi32` (VNNI) for int8×int8→int32 if available
- Implement multi-threading for outer loop (OpenMP)
- Add loop tiling/blocking for cache efficiency
- Hoist scale multiplication outside accumulation

**Expected impact:** 3-8x speedup on this function alone

**Implementation approach:**
```c
// Pseudo-code for AVX2 version
void matmul_q8_avx2(float *o, Q8Tensor *W, float *x) {
    // Convert int8 weights and float input to enable SIMD
    // Use __m256i for int8, __m256 for float
    // Process 32 elements per iteration (256-bit / 8-bit)
    // Horizontal sum at end of each row
}
```

#### 2. Optimize Attention Q·K Scoring (lines 183-187)

**Current code:**
```c
for (int t = 0; t < slen; t++) {
    float score = 0; float *kt = kc + t * hd;
    for (int d = 0; d < hd; d++) score += qh[d] * kt[d];
    att[t] = score * sc;
}
```

**Issues:**
- Scalar dot product
- Called for every head at every position
- head_dim=64, so 64 scalar multiplies per score

**Proposed solution:**
- AVX2/AVX-512 vectorized dot product
- Fuse scale multiplication

**Expected impact:** 4-8x speedup on attention scoring

#### 3. Vectorize Attention Value Accumulation (line 190)

**Current code:**
```c
for (int t = 0; t < slen; t++) {
    float a = att[t];
    float *vt = vc + t * hd;
    for (int d = 0; d < hd; d++) oh[d] += a * vt[d];
}
```

**Proposed solution:**
- SIMD broadcast of attention weight
- Vectorized multiply-add

**Expected impact:** 3-6x speedup

---

### 🟡 **High Priority** - Expected 1.5-3x additional speedup

#### 4. Vectorize RMSNorm (lines 14-18)

**Current code:**
```c
static void rmsnorm(float *o, float *x, float *w, int n, float eps) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}
```

**Issues:**
- Two separate scalar loops
- Called 3 times per layer (61 times total per token)

**Proposed solution:**
- AVX2 horizontal sum for variance computation
- Vectorized multiply in second loop
- Consider fusing with subsequent operations

**Expected impact:** 3-5x speedup on normalization

#### 5. Vectorize Softmax (lines 31-35)

**Proposed solution:**
- SIMD max reduction
- Vectorized exp (use fast approximation or `_mm256_exp_ps`)
- SIMD sum reduction

**Expected impact:** 3-5x speedup

#### 6. Optimize LM Head (lines 207-211)

**Current code:**
```c
for (int i = 0; i < c->vocab_size; i++) {
    float sum = 0; int8_t *row = w->embed_tokens.q8.data + i * hs;
    for (int j = 0; j < hs; j++) sum += m->x[j] * row[j] * s;
    m->logits[i] = sum;
}
```

**Issues:**
- This is essentially matmul_q8 but written separately
- vocab_size=49,152 × hidden_size=576 = massive computation

**Proposed solution:**
- Use same optimized matmul_q8 infrastructure
- Consider computing only top-k logits for sampling
- Add early termination for greedy decoding

**Expected impact:** 3-8x speedup on LM head

#### 7. Optimize Embedding Lookup (lines 156-157)

**Current code:**
```c
float s = w->embed_tokens.q8.scale; int8_t *emb = w->embed_tokens.q8.data + tok * hs;
for (int i = 0; i < hs; i++) m->x[i] = emb[i] * s;
```

**Proposed solution:**
- SIMD int8→float conversion with scaling
- Use `_mm256_cvtepi8_epi32` + `_mm256_cvtepi32_ps`

**Expected impact:** 4-8x speedup

---

### 🟢 **Medium Priority** - Expected 1.2-2x additional speedup

#### 8. Memory Layout Optimization

**Issues:**
- KV cache layout may cause cache misses
- Current: `[num_kv_heads, max_seq_len, head_dim]`

**Proposed solution:**
- Experiment with different layouts
- Consider head_dim innermost for better vectorization
- Align allocations to cache lines (64 bytes)

**Expected impact:** 10-30% improvement

#### 9. Fused Operations

**Opportunities:**
- Fuse SiLU activation with element-wise multiply (line 198)
- Fuse scale multiplication into matmul inner loops
- Fuse residual additions with subsequent operations

**Expected impact:** 15-25% improvement

#### 10. Multi-threading

**Proposed solution:**
- Parallelize matmul outer loop with OpenMP
- Parallelize across attention heads
- Be careful with small batch sizes (overhead vs. benefit)

**Expected impact:** 1.5-3x on multi-core CPUs (diminishing returns with SIMD)

#### 11. Better Compiler Flags

**Current:** `-O3 -march=native`

**Additional flags to try:**
```makefile
CFLAGS = -O3 -march=native -mtune=native -ffast-math \
         -funroll-loops -fomit-frame-pointer \
         -flto -fno-stack-protector
```

**Expected impact:** 5-15% improvement

---

### 🔵 **Low Priority / Advanced** - Expected 1.1-1.5x additional speedup

#### 12. Fast Math Approximations

- Fast exp() for softmax (polynomial approximation)
- Fast rsqrt for RMSNorm
- Fast sqrt via SSE intrinsics

**Expected impact:** 10-20% improvement

#### 13. Algorithmic Improvements

- Flash Attention style algorithm (reduce memory accesses)
- KV cache pruning for long sequences
- Speculative decoding
- Quantized KV cache (store as int8)

**Expected impact:** 1.2-2x for long sequences

#### 14. Platform-Specific Optimizations

- ARM NEON intrinsics for ARM CPUs
- Separate code paths for AVX2 vs AVX-512
- GPU acceleration with CUDA/Metal/Vulkan

**Expected impact:** Highly variable

---

## Implementation Strategy

### Phase 1: Foundation (Week 1)
1. Add SIMD-optimized matmul_q8 with AVX2
2. Add runtime CPU feature detection
3. Create performance benchmarking harness
4. Expected: 3-5x total speedup

### Phase 2: Attention Optimization (Week 2)
1. Vectorize attention Q·K and V accumulation
2. Vectorize RMSNorm and Softmax
3. Expected: Additional 2-3x speedup (6-15x total)

### Phase 3: Polish (Week 3)
1. Optimize LM head and embedding
2. Memory layout experiments
3. Fused operations
4. Multi-threading
5. Expected: Additional 1.5-2x speedup (9-30x total)

### Phase 4: Advanced (Optional)
1. Fast math approximations
2. Algorithmic improvements
3. Platform-specific code

---

## Estimated Total Impact

| Optimization Level | Expected Speedup | Implementation Effort |
|-------------------|------------------|----------------------|
| Phase 1 (SIMD matmul) | 3-5x | Medium |
| Phase 2 (Attention + norms) | 2-3x additional | Medium |
| Phase 3 (Polish) | 1.5-2x additional | Medium-High |
| **Total (Phases 1-3)** | **9-30x** | **High** |
| Phase 4 (Advanced) | 1.2-2x additional | Very High |

---

## Benchmarking Plan

Before/after measurements for:
1. **Tokens per second** (primary metric)
2. Per-function timing breakdown
3. Cache miss rates (`perf stat`)
4. FLOPS utilization
5. Memory bandwidth utilization

Test conditions:
- Single token generation (prompt=1 token)
- Batch generation (30 tokens)
- Long context (100+ tokens)

---

## Code Structure Recommendations

```
smolc/
├── smolc.c              # High-level API (unchanged)
├── smolc_scalar.c       # Current scalar fallback
├── smolc_avx2.c         # AVX2 optimized kernels
├── smolc_avx512.c       # AVX-512 optimized kernels
├── smolc_neon.c         # ARM NEON kernels
├── cpu_detect.c         # Runtime CPU feature detection
└── bench.c              # Benchmarking harness
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Numerical differences from SIMD | Medium | Validation against reference, tolerance testing |
| Increased code complexity | High | Keep scalar fallback, good testing |
| Portability issues | Medium | Runtime dispatch, multiple backends |
| Maintenance burden | High | Comprehensive test suite |

---

## References

- [BLAS Level 1-3 operations](https://netlib.org/blas/)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- llama.cpp, ggml, whisper.cpp for reference implementations

---

## Conclusion

The current SmolLM2 C implementation is well-structured but entirely scalar. By systematically applying SIMD vectorization, multi-threading, and algorithmic improvements, we can achieve **9-30x speedup** with reasonable engineering effort.

The highest ROI optimizations are:
1. ✅ SIMD matmul_q8 (3-8x alone)
2. ✅ SIMD attention (2-4x additional)
3. ✅ SIMD norms/activations (1.5-2x additional)

These three account for 90%+ of the total potential speedup.
