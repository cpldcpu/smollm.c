/**
 * SMOL-32 Full Model Runner
 * Loads SmolLM2-135M and runs inference on the SMOL-32 emulator,
 * comparing output with the reference C implementation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "emulator.h"

/* ===== Memory Layout =====
 * 0x0000_0000 - 0x0000_FFFF : Kernel code (64KB)
 * 0x0010_0000 - 0x003F_FFFF : State buffers (3MB)
 * 0x0040_0000 - 0x08FF_FFFF : Model weights (~135MB for Q8)
 * 0x0900_0000 - 0x0EFF_FFFF : KV caches (~90MB)
 * 0x0F00_0000 - 0x0F1F_FFFF : RoPE tables (~1MB)
 * 0x0FF0_0000 - 0x0FFF_FFFF : Stack (1MB, grows down)
 */

#define CODE_BASE       0x00000000
#define MATMUL_ENTRY    0x00001000
#define RMSNORM_ENTRY   0x00002000
#define ROPE_ENTRY      0x00003000
#define ATTN_ENTRY      0x00004000
#define SILU_ENTRY      0x00005000
#define RESIDUAL_ENTRY  0x00006000
#define EMBED_ENTRY     0x00007000
#define MEMCPY_ENTRY    0x00008000
#define FORWARD_ENTRY   0x00009000

#define DESC_BASE       0x000E0000

#define BUF_BASE        0x00100000
#define BUF_X           (BUF_BASE + 0x00000)   /* [576] = 2304 bytes */
#define BUF_XB          (BUF_BASE + 0x01000)   /* [576] */
#define BUF_XB2         (BUF_BASE + 0x02000)   /* [576] */
#define BUF_Q           (BUF_BASE + 0x03000)   /* [576] */
#define BUF_K           (BUF_BASE + 0x04000)   /* [192] */
#define BUF_V           (BUF_BASE + 0x05000)   /* [192] */
#define BUF_ATT         (BUF_BASE + 0x06000)   /* [9 * 2048] = 73728 bytes */
#define BUF_LOGITS      (BUF_BASE + 0x20000)   /* [49152] = 196608 bytes */
#define BUF_HB          (BUF_BASE + 0x50000)   /* [1536] = 6144 bytes */
#define BUF_HB2         (BUF_BASE + 0x52000)   /* [1536] */

#define WEIGHT_BASE     0x00400000
#define KV_BASE         0x09000000
#define ROPE_BASE       0x0F000000

/* Model config */
typedef struct {
    int hidden_size, intermediate_size, num_layers;
    int num_heads, num_kv_heads, vocab_size, max_seq_len;
    float rope_theta, rms_norm_eps;
    int head_dim;
} ModelConfig;

/* Layer weight locations in emulator memory */
typedef struct {
    uint32_t input_ln;       /* FP32 layernorm weights addr */
    uint32_t q_data, q_scale_addr;
    uint32_t k_data, k_scale_addr;
    uint32_t v_data, v_scale_addr;
    uint32_t o_data, o_scale_addr;
    uint32_t post_ln;
    uint32_t gate_data, gate_scale_addr;
    uint32_t up_data, up_scale_addr;
    uint32_t down_data, down_scale_addr;
} LayerAddrs;

/* Reference implementation */
static void ref_rmsnorm(float *o, float *x, float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}

static void ref_matmul_q8(float *o, int8_t *data, float scale, float *x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        int8_t *row = data + i * cols;
        for (int j = 0; j < cols; j++) sum += row[j] * scale * x[j];
        o[i] = sum;
    }
}

static float ref_silu(float x) { return x / (1.0f + expf(-x)); }

static void ref_softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void ref_apply_rope(float *v, int hd, float *c, float *s) {
    int h = hd / 2;
    for (int i = 0; i < h; i++) {
        float v0 = v[i], v1 = v[i + h];
        v[i] = v0 * c[i] - v1 * s[i];
        v[i + h] = v1 * c[i + h] + v0 * s[i + h];
    }
}

/* ===== Load assembled kernel binaries ===== */

/* HALT instruction encoding (SYSTEM func=0x1F) */
#define HALT_INSN 0xE00007C0

static int load_kernel(SmolCPU *cpu, const char *path, uint32_t addr) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Can't open kernel: %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = malloc(sz);
    if (fread(buf, 1, sz, f) != (size_t)sz) { fclose(f); free(buf); return -1; }
    fclose(f);
    cpu_load_data(cpu, addr, buf, sz);
    free(buf);
    printf("  Loaded kernel %s (%ld bytes) at 0x%08x\n", path, sz, addr);
    return 0;
}

/* Reset CPU state for a kernel call.
 * Sets RA (R1) to 0x0000 where a HALT lives, so RET stops the CPU. */
static void cpu_reset_for_call(SmolCPU *cpu, uint32_t entry) {
    memset(cpu->r, 0, sizeof(cpu->r));
    memset(cpu->f, 0, sizeof(cpu->f));
    cpu->r[1] = CODE_BASE;       /* RA -> HALT trap at addr 0 */
    cpu->r[2] = MEM_SIZE - 4096; /* SP */
    cpu->pc = entry;
    cpu->halted = 0;
    cpu->insn_count = 0;
    cpu->cycle_count = 0;
    cpu->acc = 0;
    cpu->vl = VEC_LEN;
}

/* Run a matmul on the emulator */
static void emu_matmul_q8(SmolCPU *cpu, uint32_t out_addr, uint32_t weight_addr,
                           uint32_t scale_addr, uint32_t input_addr, int rows, int cols) {
    cpu_reset_for_call(cpu, MATMUL_ENTRY);
    cpu->r[3] = out_addr;
    cpu->r[4] = weight_addr;
    cpu->r[5] = scale_addr;
    cpu->r[6] = input_addr;
    cpu->r[7] = rows;
    cpu->r[8] = cols;
    cpu_run(cpu, 500000000ULL);
}

/* Run rmsnorm on the emulator */
static void emu_rmsnorm(SmolCPU *cpu, uint32_t out_addr, uint32_t in_addr,
                         uint32_t weight_addr, int n, float eps) {
    cpu_reset_for_call(cpu, RMSNORM_ENTRY);
    cpu->r[3] = out_addr;
    cpu->r[4] = in_addr;
    cpu->r[5] = weight_addr;
    cpu->r[6] = n;
    cpu->f[1] = eps;
    cpu_run(cpu, 100000000ULL);
}

/* Run RoPE on a single head vector in emulator memory */
static void emu_apply_rope(SmolCPU *cpu, uint32_t vec_addr, int head_dim,
                            uint32_t cos_addr, uint32_t sin_addr) {
    cpu_reset_for_call(cpu, ROPE_ENTRY);
    cpu->r[3] = vec_addr;
    cpu->r[4] = head_dim;
    cpu->r[5] = cos_addr;
    cpu->r[6] = sin_addr;
    cpu_run(cpu, 10000000ULL);
}

/* Run multi-head attention on the emulator.
 * Output buffer must be pre-zeroed in emulator memory. */
static void emu_attention(SmolCPU *cpu, uint32_t q_addr, uint32_t k_cache_addr,
                           uint32_t v_cache_addr, uint32_t out_addr,
                           uint32_t att_addr, int slen, int max_seq) {
    cpu_reset_for_call(cpu, ATTN_ENTRY);
    cpu->r[3] = q_addr;
    cpu->r[4] = k_cache_addr;
    cpu->r[5] = v_cache_addr;
    cpu->r[6] = out_addr;
    cpu->r[7] = att_addr;
    cpu->r[8] = slen;
    cpu->r[9] = max_seq;
    cpu_run(cpu, 2000000000ULL);
}

/* Run SiLU-gate: gate[i] = silu(gate[i]) * up[i] */
static void emu_silu_mul(SmolCPU *cpu, uint32_t gate_addr, uint32_t up_addr, int n) {
    cpu_reset_for_call(cpu, SILU_ENTRY);
    cpu->r[3] = gate_addr;
    cpu->r[4] = up_addr;
    cpu->r[5] = n;
    cpu_run(cpu, 100000000ULL);
}

/* Run residual add: dst[i] += src[i] */
static void emu_residual(SmolCPU *cpu, uint32_t dst_addr, uint32_t src_addr, int n) {
    cpu_reset_for_call(cpu, RESIDUAL_ENTRY);
    cpu->r[3] = dst_addr;
    cpu->r[4] = src_addr;
    cpu->r[5] = n;
    cpu_run(cpu, 10000000ULL);
}

/* Run embedding dequant: output[i] = (float)input[i] * scale */
static void emu_embed_dequant(SmolCPU *cpu, uint32_t out_addr, uint32_t in_addr,
                               int n, float scale) {
    cpu_reset_for_call(cpu, EMBED_ENTRY);
    cpu->r[3] = out_addr;
    cpu->r[4] = in_addr;
    cpu->r[5] = n;
    cpu->f[1] = scale;
    cpu_run(cpu, 50000000ULL);
}

/* Copy n floats within emulator memory */
static void emu_memcpy_f(SmolCPU *cpu, uint32_t dst_addr, uint32_t src_addr, int n) {
    cpu_reset_for_call(cpu, MEMCPY_ENTRY);
    cpu->r[3] = dst_addr;
    cpu->r[4] = src_addr;
    cpu->r[5] = n;
    cpu_run(cpu, 10000000ULL);
}

/* Helpers to read/write float arrays to emulator memory */
static void load_floats(SmolCPU *cpu, uint32_t addr, float *data, int n) {
    cpu_load_data(cpu, addr, data, n * sizeof(float));
}

static void read_floats(SmolCPU *cpu, uint32_t addr, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = mem_readf(cpu, addr + i * 4);
}

/* ===== Model loading ===== */

static int rd_u32(FILE *f) { uint32_t v; return fread(&v, 4, 1, f) == 1 ? (int)v : -1; }
static float rd_f32(FILE *f) { float v; return fread(&v, 4, 1, f) == 1 ? v : 0; }
static void rd_bytes(FILE *f, void *buf, size_t n) {
    if (fread(buf, 1, n, f) != n) { fprintf(stderr, "Read error\n"); exit(1); }
}

typedef struct {
    ModelConfig config;
    /* Emulator addresses */
    LayerAddrs *layer_addrs;
    uint32_t embed_data_addr, embed_scale_addr;
    uint32_t final_norm_addr;
    uint32_t rope_cos_addr, rope_sin_addr;
    uint32_t kv_cache_addr; /* Base of all KV caches */
    int *cache_lens;
    /* Host-side copies for reference */
    int8_t *embed_data;
    float embed_scale;
    float *final_norm;
    float *rope_cos, *rope_sin;
    /* Per-layer host data for reference */
    struct {
        float *input_ln;
        int8_t *q_data; float q_scale;
        int8_t *k_data; float k_scale;
        int8_t *v_data; float v_scale;
        int8_t *o_data; float o_scale;
        float *post_ln;
        int8_t *gate_data; float gate_scale;
        int8_t *up_data; float up_scale;
        int8_t *down_data; float down_scale;
    } *layers;
} Model;

static void precompute_rope(Model *m) {
    int hd = m->config.head_dim, maxlen = m->config.max_seq_len;
    float theta = m->config.rope_theta;
    m->rope_cos = malloc(maxlen * hd * sizeof(float));
    m->rope_sin = malloc(maxlen * hd * sizeof(float));
    for (int p = 0; p < maxlen; p++) {
        for (int i = 0; i < hd / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / hd);
            float c = cosf(p * freq), s = sinf(p * freq);
            m->rope_cos[p * hd + i] = m->rope_cos[p * hd + hd/2 + i] = c;
            m->rope_sin[p * hd + i] = m->rope_sin[p * hd + hd/2 + i] = s;
        }
    }
}

static int load_model(Model *m, SmolCPU *cpu, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", path); return -1; }

    char magic[5] = {0};
    rd_bytes(f, magic, 4);
    if (strcmp(magic, "SMOL")) { fprintf(stderr, "Bad magic\n"); fclose(f); return -1; }
    int ver = rd_u32(f);
    if (ver == 2) { rd_u32(f); rd_u32(f); } /* skip quant_type, group_size */

    ModelConfig *c = &m->config;
    c->hidden_size = rd_u32(f);
    c->intermediate_size = rd_u32(f);
    c->num_layers = rd_u32(f);
    c->num_heads = rd_u32(f);
    c->num_kv_heads = rd_u32(f);
    c->vocab_size = rd_u32(f);
    c->max_seq_len = rd_u32(f);
    c->rope_theta = rd_f32(f);
    c->rms_norm_eps = rd_f32(f);
    c->head_dim = c->hidden_size / c->num_heads;

    printf("Config: hs=%d, intermediate=%d, layers=%d, heads=%d, kv=%d, vocab=%d\n",
           c->hidden_size, c->intermediate_size, c->num_layers,
           c->num_heads, c->num_kv_heads, c->vocab_size);
    printf("  head_dim=%d, max_seq=%d, theta=%.1f, eps=%.1e\n",
           c->head_dim, c->max_seq_len, c->rope_theta, c->rms_norm_eps);

    /* Skip tokenizer */
    int nv = rd_u32(f), nm = rd_u32(f);
    for (int i = 0; i < nv; i++) { int len = rd_u32(f); fseek(f, len, SEEK_CUR); }
    for (int i = 0; i < nm; i++) { int len = rd_u32(f); fseek(f, len, SEEK_CUR); }

    int hs = c->hidden_size, hd = c->head_dim;
    int nh = c->num_heads, nkv = c->num_kv_heads;

    /* ===== Load embedding ===== */
    m->embed_scale = rd_f32(f);
    size_t embed_sz = (size_t)c->vocab_size * hs;
    m->embed_data = malloc(embed_sz);
    rd_bytes(f, m->embed_data, embed_sz);

    uint32_t wptr = WEIGHT_BASE;
    m->embed_data_addr = wptr;
    cpu_load_data(cpu, wptr, m->embed_data, embed_sz);
    wptr += embed_sz;
    /* Align to 4 bytes */
    wptr = (wptr + 3) & ~3u;
    m->embed_scale_addr = wptr;
    mem_writef(cpu, wptr, m->embed_scale);
    wptr += 4;

    /* ===== Load layers ===== */
    m->layers = calloc(c->num_layers, sizeof(*m->layers));
    m->layer_addrs = calloc(c->num_layers, sizeof(LayerAddrs));

    for (int l = 0; l < c->num_layers; l++) {
        LayerAddrs *la = &m->layer_addrs[l];

        /* input_layernorm: FP32 [hs] */
        m->layers[l].input_ln = malloc(hs * sizeof(float));
        rd_bytes(f, m->layers[l].input_ln, hs * sizeof(float));
        la->input_ln = wptr;
        cpu_load_data(cpu, wptr, m->layers[l].input_ln, hs * sizeof(float));
        wptr += hs * sizeof(float);

        /* q_proj: scale + data[nh*hd, hs] */
        int q_rows = nh * hd, q_cols = hs;
        m->layers[l].q_scale = rd_f32(f);
        m->layers[l].q_data = malloc(q_rows * q_cols);
        rd_bytes(f, m->layers[l].q_data, q_rows * q_cols);
        la->q_scale_addr = wptr; mem_writef(cpu, wptr, m->layers[l].q_scale); wptr += 4;
        la->q_data = wptr;
        cpu_load_data(cpu, wptr, m->layers[l].q_data, q_rows * q_cols);
        wptr += q_rows * q_cols;
        wptr = (wptr + 3) & ~3u;

        /* k_proj: [nkv*hd, hs] */
        int k_rows = nkv * hd, k_cols = hs;
        m->layers[l].k_scale = rd_f32(f);
        m->layers[l].k_data = malloc(k_rows * k_cols);
        rd_bytes(f, m->layers[l].k_data, k_rows * k_cols);
        la->k_scale_addr = wptr; mem_writef(cpu, wptr, m->layers[l].k_scale); wptr += 4;
        la->k_data = wptr;
        cpu_load_data(cpu, wptr, m->layers[l].k_data, k_rows * k_cols);
        wptr += k_rows * k_cols;
        wptr = (wptr + 3) & ~3u;

        /* v_proj: [nkv*hd, hs] */
        m->layers[l].v_scale = rd_f32(f);
        m->layers[l].v_data = malloc(k_rows * k_cols);
        rd_bytes(f, m->layers[l].v_data, k_rows * k_cols);
        la->v_scale_addr = wptr; mem_writef(cpu, wptr, m->layers[l].v_scale); wptr += 4;
        la->v_data = wptr;
        cpu_load_data(cpu, wptr, m->layers[l].v_data, k_rows * k_cols);
        wptr += k_rows * k_cols;
        wptr = (wptr + 3) & ~3u;

        /* o_proj: [hs, nh*hd] */
        int o_rows = hs, o_cols = nh * hd;
        m->layers[l].o_scale = rd_f32(f);
        m->layers[l].o_data = malloc(o_rows * o_cols);
        rd_bytes(f, m->layers[l].o_data, o_rows * o_cols);
        la->o_scale_addr = wptr; mem_writef(cpu, wptr, m->layers[l].o_scale); wptr += 4;
        la->o_data = wptr;
        cpu_load_data(cpu, wptr, m->layers[l].o_data, o_rows * o_cols);
        wptr += o_rows * o_cols;
        wptr = (wptr + 3) & ~3u;

        /* post_attention_layernorm: FP32 [hs] */
        m->layers[l].post_ln = malloc(hs * sizeof(float));
        rd_bytes(f, m->layers[l].post_ln, hs * sizeof(float));
        la->post_ln = wptr;
        cpu_load_data(cpu, wptr, m->layers[l].post_ln, hs * sizeof(float));
        wptr += hs * sizeof(float);

        /* gate_proj: [intermediate, hs] */
        int g_rows = c->intermediate_size, g_cols = hs;
        m->layers[l].gate_scale = rd_f32(f);
        m->layers[l].gate_data = malloc(g_rows * g_cols);
        rd_bytes(f, m->layers[l].gate_data, g_rows * g_cols);
        la->gate_scale_addr = wptr; mem_writef(cpu, wptr, m->layers[l].gate_scale); wptr += 4;
        la->gate_data = wptr;
        cpu_load_data(cpu, wptr, m->layers[l].gate_data, g_rows * g_cols);
        wptr += g_rows * g_cols;
        wptr = (wptr + 3) & ~3u;

        /* up_proj: [intermediate, hs] */
        m->layers[l].up_scale = rd_f32(f);
        m->layers[l].up_data = malloc(g_rows * g_cols);
        rd_bytes(f, m->layers[l].up_data, g_rows * g_cols);
        la->up_scale_addr = wptr; mem_writef(cpu, wptr, m->layers[l].up_scale); wptr += 4;
        la->up_data = wptr;
        cpu_load_data(cpu, wptr, m->layers[l].up_data, g_rows * g_cols);
        wptr += g_rows * g_cols;
        wptr = (wptr + 3) & ~3u;

        /* down_proj: [hs, intermediate] */
        int d_rows = hs, d_cols = c->intermediate_size;
        m->layers[l].down_scale = rd_f32(f);
        m->layers[l].down_data = malloc(d_rows * d_cols);
        rd_bytes(f, m->layers[l].down_data, d_rows * d_cols);
        la->down_scale_addr = wptr; mem_writef(cpu, wptr, m->layers[l].down_scale); wptr += 4;
        la->down_data = wptr;
        cpu_load_data(cpu, wptr, m->layers[l].down_data, d_rows * d_cols);
        wptr += d_rows * d_cols;
        wptr = (wptr + 3) & ~3u;

        if (l == 0) printf("  Layer 0 weights loaded, wptr=0x%08x\n", wptr);
    }

    /* final_norm: FP32 [hs] */
    m->final_norm = malloc(hs * sizeof(float));
    rd_bytes(f, m->final_norm, hs * sizeof(float));
    m->final_norm_addr = wptr;
    cpu_load_data(cpu, wptr, m->final_norm, hs * sizeof(float));
    wptr += hs * sizeof(float);

    fclose(f);
    printf("  Total weight memory: %u bytes (%.1f MB)\n", wptr - WEIGHT_BASE, (wptr - WEIGHT_BASE) / 1048576.0f);

    /* ===== Precompute RoPE and load to emulator ===== */
    precompute_rope(m);
    m->rope_cos_addr = ROPE_BASE;
    m->rope_sin_addr = ROPE_BASE + c->max_seq_len * hd * sizeof(float);
    cpu_load_data(cpu, m->rope_cos_addr, m->rope_cos, c->max_seq_len * hd * sizeof(float));
    cpu_load_data(cpu, m->rope_sin_addr, m->rope_sin, c->max_seq_len * hd * sizeof(float));

    /* ===== Set up KV cache addresses ===== */
    /* Cap effective sequence length to fit in emulator memory */
    int eff_seq = 64; /* Only need a few positions for testing */
    m->kv_cache_addr = KV_BASE;
    m->cache_lens = calloc(c->num_layers, sizeof(int));
    printf("  KV cache: %d layers x %d kv_heads x %d seq x %d hd = %.1f MB\n",
           c->num_layers, nkv, eff_seq, hd,
           (float)c->num_layers * nkv * 2 * eff_seq * hd * 4 / 1048576.0f);
    /* Store effective seq len for use in forward pass */
    c->max_seq_len = eff_seq;

    printf("Model loaded successfully.\n");
    return 0;
}

/* ===== Emulated forward pass ===== */

static void emu_forward(Model *m, SmolCPU *cpu, int token, int pos, float *logits_out) {
    ModelConfig *c = &m->config;
    int hs = c->hidden_size, hd = c->head_dim;
    int nh = c->num_heads, nkv = c->num_kv_heads;
    int kv_head_stride = c->max_seq_len * hd;

    /* Embedding dequant on emulator */
    uint32_t emb_addr = m->embed_data_addr + (uint32_t)token * hs;
    emu_embed_dequant(cpu, BUF_X, emb_addr, hs, m->embed_scale);

    uint32_t cos_addr = m->rope_cos_addr + pos * hd * sizeof(float);
    uint32_t sin_addr = m->rope_sin_addr + pos * hd * sizeof(float);

    static const float zeros[576] = {0};

    for (int l = 0; l < c->num_layers; l++) {
        LayerAddrs *la = &m->layer_addrs[l];

        /* RMSNorm: BUF_X -> BUF_XB */
        emu_rmsnorm(cpu, BUF_XB, BUF_X, la->input_ln, hs, c->rms_norm_eps);

        /* Q/K/V projections (BUF_XB persists across matmuls - read-only input) */
        emu_matmul_q8(cpu, BUF_Q, la->q_data, la->q_scale_addr, BUF_XB, nh * hd, hs);
        emu_matmul_q8(cpu, BUF_K, la->k_data, la->k_scale_addr, BUF_XB, nkv * hd, hs);
        emu_matmul_q8(cpu, BUF_V, la->v_data, la->v_scale_addr, BUF_XB, nkv * hd, hs);

        /* RoPE on emulator */
        for (int h = 0; h < nh; h++)
            emu_apply_rope(cpu, BUF_Q + h * hd * 4, hd, cos_addr, sin_addr);
        for (int h = 0; h < nkv; h++)
            emu_apply_rope(cpu, BUF_K + h * hd * 4, hd, cos_addr, sin_addr);

        /* Update KV cache (scatter within emulator memory) */
        uint32_t layer_kv = m->kv_cache_addr +
            (uint32_t)l * nkv * 2 * kv_head_stride * sizeof(float);
        uint32_t layer_k_cache = layer_kv;
        uint32_t layer_v_cache = layer_kv + nkv * kv_head_stride * sizeof(float);

        for (int h = 0; h < nkv; h++) {
            uint32_t k_dst = layer_k_cache + (h * kv_head_stride + pos * hd) * sizeof(float);
            uint32_t v_dst = layer_v_cache + (h * kv_head_stride + pos * hd) * sizeof(float);
            emu_memcpy_f(cpu, k_dst, BUF_K + h * hd * sizeof(float), hd);
            emu_memcpy_f(cpu, v_dst, BUF_V + h * hd * sizeof(float), hd);
        }
        int slen = m->cache_lens[l] + 1;

        /* Attention on emulator */
        load_floats(cpu, BUF_XB2, (float *)zeros, hs);
        emu_attention(cpu, BUF_Q, layer_k_cache, layer_v_cache,
                      BUF_XB2, BUF_ATT, slen, c->max_seq_len);
        m->cache_lens[l] = slen;

        /* O projection + residual */
        emu_matmul_q8(cpu, BUF_XB, la->o_data, la->o_scale_addr, BUF_XB2, hs, nh * hd);
        emu_residual(cpu, BUF_X, BUF_XB, hs);

        /* Post-attention norm + MLP */
        emu_rmsnorm(cpu, BUF_XB, BUF_X, la->post_ln, hs, c->rms_norm_eps);
        emu_matmul_q8(cpu, BUF_HB, la->gate_data, la->gate_scale_addr, BUF_XB, c->intermediate_size, hs);
        emu_matmul_q8(cpu, BUF_HB2, la->up_data, la->up_scale_addr, BUF_XB, c->intermediate_size, hs);
        emu_silu_mul(cpu, BUF_HB, BUF_HB2, c->intermediate_size);
        emu_matmul_q8(cpu, BUF_XB, la->down_data, la->down_scale_addr, BUF_HB, hs, c->intermediate_size);
        emu_residual(cpu, BUF_X, BUF_XB, hs);

        if (l % 10 == 0) printf("  Layer %d/%d done\n", l + 1, c->num_layers);
    }

    /* Final norm + LM head */
    emu_rmsnorm(cpu, BUF_X, BUF_X, m->final_norm_addr, hs, c->rms_norm_eps);
    emu_matmul_q8(cpu, BUF_LOGITS, m->embed_data_addr, m->embed_scale_addr,
                   BUF_X, c->vocab_size, hs);
    read_floats(cpu, BUF_LOGITS, logits_out, c->vocab_size);
}

/* ===== Assembly forward pass ===== */

static void populate_descriptor(Model *m, SmolCPU *cpu) {
    ModelConfig *c = &m->config;
    uint32_t d = DESC_BASE;

    mem_write32(cpu, d + 0x00, m->embed_data_addr);
    mem_write32(cpu, d + 0x04, m->embed_scale_addr);
    mem_write32(cpu, d + 0x08, m->final_norm_addr);
    mem_write32(cpu, d + 0x0C, m->kv_cache_addr);
    mem_write32(cpu, d + 0x10, c->hidden_size);
    mem_write32(cpu, d + 0x14, c->head_dim);
    mem_write32(cpu, d + 0x18, c->num_heads);
    mem_write32(cpu, d + 0x1C, c->num_kv_heads);
    mem_write32(cpu, d + 0x20, c->intermediate_size);
    mem_write32(cpu, d + 0x24, c->num_layers);
    mem_write32(cpu, d + 0x28, c->max_seq_len);
    mem_writef(cpu, d + 0x2C, c->rms_norm_eps);
    mem_write32(cpu, d + 0x30, c->vocab_size);
    mem_write32(cpu, d + 0x34, ROPE_BASE);
    mem_write32(cpu, d + 0x38, c->max_seq_len * c->head_dim * 4);
    mem_write32(cpu, d + 0x3C, 0);

    for (int l = 0; l < c->num_layers; l++) {
        uint32_t ld = d + 0x40 + l * 64;
        LayerAddrs *la = &m->layer_addrs[l];
        mem_write32(cpu, ld + 0x00, la->input_ln);
        mem_write32(cpu, ld + 0x04, la->q_data);
        mem_write32(cpu, ld + 0x08, la->q_scale_addr);
        mem_write32(cpu, ld + 0x0C, la->k_data);
        mem_write32(cpu, ld + 0x10, la->k_scale_addr);
        mem_write32(cpu, ld + 0x14, la->v_data);
        mem_write32(cpu, ld + 0x18, la->v_scale_addr);
        mem_write32(cpu, ld + 0x1C, la->o_data);
        mem_write32(cpu, ld + 0x20, la->o_scale_addr);
        mem_write32(cpu, ld + 0x24, la->post_ln);
        mem_write32(cpu, ld + 0x28, la->gate_data);
        mem_write32(cpu, ld + 0x2C, la->gate_scale_addr);
        mem_write32(cpu, ld + 0x30, la->up_data);
        mem_write32(cpu, ld + 0x34, la->up_scale_addr);
        mem_write32(cpu, ld + 0x38, la->down_data);
        mem_write32(cpu, ld + 0x3C, la->down_scale_addr);
    }
}

static void emu_forward_asm(Model *m, SmolCPU *cpu, int token, int pos, float *logits_out) {
    cpu_reset_for_call(cpu, FORWARD_ENTRY);
    cpu->r[3] = token;
    cpu->r[4] = pos;
    cpu->r[5] = DESC_BASE;
    cpu_run(cpu, 500000000000ULL);

    uint32_t logits_addr = BUF_BASE + 0x80000;
    read_floats(cpu, logits_addr, logits_out, m->config.vocab_size);
    printf("  Assembly forward: %llu instructions executed\n",
           (unsigned long long)cpu->insn_count);
}

/* ===== Reference forward pass ===== */

static void ref_forward(Model *m, int token, int pos, float *logits_out) {
    ModelConfig *c = &m->config;
    int hs = c->hidden_size, hd = c->head_dim;
    int nh = c->num_heads, nkv = c->num_kv_heads, ng = nh / nkv;

    static float x[576], xb[576], xb2[576], q[576], k[192], v[192];
    static float hb[1536], hb2[1536];
    static float att[9 * 2048];
    static float *k_caches[30], *v_caches[30];
    static int cache_lens[30];
    static int initialized = 0;

    if (!initialized) {
        for (int l = 0; l < c->num_layers; l++) {
            k_caches[l] = calloc(nkv * c->max_seq_len * hd, sizeof(float));
            v_caches[l] = calloc(nkv * c->max_seq_len * hd, sizeof(float));
            cache_lens[l] = 0;
        }
        initialized = 1;
    }

    /* Embedding */
    int8_t *emb = m->embed_data + token * hs;
    for (int i = 0; i < hs; i++) x[i] = emb[i] * m->embed_scale;

    float *rc = m->rope_cos + pos * hd;
    float *rs = m->rope_sin + pos * hd;

    for (int l = 0; l < c->num_layers; l++) {
        ref_rmsnorm(xb, x, m->layers[l].input_ln, hs, c->rms_norm_eps);
        ref_matmul_q8(q, m->layers[l].q_data, m->layers[l].q_scale, xb, nh * hd, hs);
        ref_matmul_q8(k, m->layers[l].k_data, m->layers[l].k_scale, xb, nkv * hd, hs);
        ref_matmul_q8(v, m->layers[l].v_data, m->layers[l].v_scale, xb, nkv * hd, hs);

        for (int h = 0; h < nh; h++) ref_apply_rope(q + h * hd, hd, rc, rs);
        for (int h = 0; h < nkv; h++) ref_apply_rope(k + h * hd, hd, rc, rs);

        int off = cache_lens[l] * hd;
        for (int h = 0; h < nkv; h++) {
            memcpy(k_caches[l] + h * c->max_seq_len * hd + off, k + h * hd, hd * sizeof(float));
            memcpy(v_caches[l] + h * c->max_seq_len * hd + off, v + h * hd, hd * sizeof(float));
        }
        int slen = cache_lens[l] + 1;

        memset(xb2, 0, hs * sizeof(float));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float *qh = q + h * hd;
            float *kc = k_caches[l] + kvh * c->max_seq_len * hd;
            float *vc = v_caches[l] + kvh * c->max_seq_len * hd;
            float sc = 1.0f / sqrtf((float)hd);

            for (int t = 0; t < slen; t++) {
                float score = 0;
                float *kt = kc + t * hd;
                for (int d = 0; d < hd; d++) score += qh[d] * kt[d];
                att[h * c->max_seq_len + t] = score * sc;
            }
            ref_softmax(att + h * c->max_seq_len, slen);

            float *oh = xb2 + h * hd;
            for (int t = 0; t < slen; t++) {
                float a = att[h * c->max_seq_len + t];
                float *vt = vc + t * hd;
                for (int d = 0; d < hd; d++) oh[d] += a * vt[d];
            }
        }
        cache_lens[l] = slen;

        ref_matmul_q8(xb, m->layers[l].o_data, m->layers[l].o_scale, xb2, hs, nh * hd);
        for (int i = 0; i < hs; i++) x[i] += xb[i];

        ref_rmsnorm(xb, x, m->layers[l].post_ln, hs, c->rms_norm_eps);
        ref_matmul_q8(hb, m->layers[l].gate_data, m->layers[l].gate_scale, xb, c->intermediate_size, hs);
        ref_matmul_q8(hb2, m->layers[l].up_data, m->layers[l].up_scale, xb, c->intermediate_size, hs);
        for (int i = 0; i < c->intermediate_size; i++) hb[i] = ref_silu(hb[i]) * hb2[i];
        ref_matmul_q8(xb, m->layers[l].down_data, m->layers[l].down_scale, hb, hs, c->intermediate_size);
        for (int i = 0; i < hs; i++) x[i] += xb[i];
    }

    ref_rmsnorm(x, x, m->final_norm, hs, c->rms_norm_eps);
    for (int i = 0; i < c->vocab_size; i++) {
        float sum = 0;
        int8_t *row = m->embed_data + i * hs;
        for (int j = 0; j < hs; j++) sum += x[j] * row[j] * m->embed_scale;
        logits_out[i] = sum;
    }
}

/* ===== Main ===== */

int main(int argc, char **argv) {
    const char *model_path = "../models/smollm2-135m-q8.bin";
    int token = 1; /* Default test token */
    int pos = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) token = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) pos = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h")) {
            printf("Usage: %s [-m model] [-t token] [-p pos]\n", argv[0]);
            return 0;
        }
    }

    printf("SMOL-32 Model Runner\n");
    printf("====================\n\n");

    /* Initialize emulator */
    SmolCPU cpu;
    cpu_init(&cpu);

    /* Place HALT trap at address 0 (kernels RET to here) */
    uint32_t halt = HALT_INSN;
    cpu_load_data(&cpu, CODE_BASE, &halt, 4);

    /* Load assembled kernel binaries */
    if (load_kernel(&cpu, "kernels/matmul_q8.bin", MATMUL_ENTRY) < 0 ||
        load_kernel(&cpu, "kernels/rmsnorm.bin", RMSNORM_ENTRY) < 0 ||
        load_kernel(&cpu, "kernels/rope.bin", ROPE_ENTRY) < 0 ||
        load_kernel(&cpu, "kernels/attention.bin", ATTN_ENTRY) < 0 ||
        load_kernel(&cpu, "kernels/silu_mul.bin", SILU_ENTRY) < 0 ||
        load_kernel(&cpu, "kernels/residual.bin", RESIDUAL_ENTRY) < 0 ||
        load_kernel(&cpu, "kernels/embed.bin", EMBED_ENTRY) < 0 ||
        load_kernel(&cpu, "kernels/memcpy.bin", MEMCPY_ENTRY) < 0 ||
        load_kernel(&cpu, "kernels/forward.bin", FORWARD_ENTRY) < 0) {
        fprintf(stderr, "Failed to load kernels. Run 'make kernels' first.\n");
        cpu_free(&cpu);
        return 1;
    }

    /* Load model */
    Model model;
    memset(&model, 0, sizeof(model));
    printf("Loading model: %s\n", model_path);
    if (load_model(&model, &cpu, model_path) != 0) {
        cpu_free(&cpu);
        return 1;
    }

    /* Populate descriptor for assembly forward pass */
    populate_descriptor(&model, &cpu);

    printf("\nRunning forward pass: token=%d, pos=%d\n", token, pos);

    /* Run reference forward pass */
    float *ref_logits = malloc(model.config.vocab_size * sizeof(float));
    printf("\n--- Reference forward pass ---\n");
    ref_forward(&model, token, pos, ref_logits);

    /* Run assembly forward pass */
    float *asm_logits = malloc(model.config.vocab_size * sizeof(float));
    printf("\n--- Assembly forward pass ---\n");
    emu_forward_asm(&model, &cpu, token, pos, asm_logits);

    /* Compare assembly vs reference */
    printf("\n--- Comparison: Assembly vs Reference ---\n");
    float max_diff = 0;
    int mismatches = 0;
    double sum_diff = 0;
    for (int i = 0; i < model.config.vocab_size; i++) {
        float diff = fabsf(asm_logits[i] - ref_logits[i]);
        sum_diff += diff;
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.1f) mismatches++;
    }
    float avg_diff = sum_diff / model.config.vocab_size;

    printf("  Max logit difference: %.6e\n", max_diff);
    printf("  Avg logit difference: %.6e\n", avg_diff);
    printf("  Mismatches (>0.1): %d / %d\n", mismatches, model.config.vocab_size);

    /* Show top-5 predictions from both */
    printf("\n  Top-5 (Assembly):\n");
    for (int k = 0; k < 5; k++) {
        int best = 0;
        for (int i = 1; i < model.config.vocab_size; i++)
            if (asm_logits[i] > asm_logits[best]) best = i;
        printf("    [%d] logit=%.4f\n", best, asm_logits[best]);
        asm_logits[best] = -1e9f;
    }

    printf("  Top-5 (Reference):\n");
    for (int k = 0; k < 5; k++) {
        int best = 0;
        for (int i = 1; i < model.config.vocab_size; i++)
            if (ref_logits[i] > ref_logits[best]) best = i;
        printf("    [%d] logit=%.4f\n", best, ref_logits[best]);
        ref_logits[best] = -1e9f;
    }

    printf("\n  Result: %s\n", mismatches == 0 ? "PASS - Assembly matches reference!" : "DIFFERENCES FOUND");

    free(asm_logits);
    free(ref_logits);
    free(model.layer_addrs);
    free(model.cache_lens);
    cpu_free(&cpu);
    return mismatches > 0 ? 1 : 0;
}
