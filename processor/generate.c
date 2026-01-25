/**
 * SMOL-32 Text Generator
 * Runs SmolLM2-135M inference on the SMOL-32 emulator with full
 * tokenization, generation loop, and text decoding.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "emulator.h"

/* ===== Memory Layout ===== */
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
#define BUF_X           (BUF_BASE + 0x00000)
#define BUF_XB          (BUF_BASE + 0x01000)
#define BUF_XB2         (BUF_BASE + 0x02000)
#define BUF_Q           (BUF_BASE + 0x03000)
#define BUF_K           (BUF_BASE + 0x04000)
#define BUF_V           (BUF_BASE + 0x05000)
#define BUF_ATT         (BUF_BASE + 0x06000)
#define BUF_HB          (BUF_BASE + 0x50000)
#define BUF_HB2         (BUF_BASE + 0x52000)

#define WEIGHT_BASE     0x00400000
#define KV_BASE         0x09000000
#define ROPE_BASE       0x0F000000

#define HALT_INSN       0xE00007C0

/* ===== Model types ===== */

typedef struct {
    int hidden_size, intermediate_size, num_layers;
    int num_heads, num_kv_heads, vocab_size, max_seq_len;
    float rope_theta, rms_norm_eps;
    int head_dim;
} ModelConfig;

typedef struct {
    uint32_t input_ln;
    uint32_t q_data, q_scale_addr;
    uint32_t k_data, k_scale_addr;
    uint32_t v_data, v_scale_addr;
    uint32_t o_data, o_scale_addr;
    uint32_t post_ln;
    uint32_t gate_data, gate_scale_addr;
    uint32_t up_data, up_scale_addr;
    uint32_t down_data, down_scale_addr;
} LayerAddrs;

typedef struct {
    char **vocab;
    int vocab_size;
    char **merges;
    int num_merges;
} Tokenizer;

typedef struct {
    ModelConfig config;
    LayerAddrs *layer_addrs;
    uint32_t embed_data_addr, embed_scale_addr;
    uint32_t final_norm_addr;
    uint32_t kv_cache_addr;
    int *cache_lens;
    int8_t *embed_data;
    float embed_scale;
    float *rope_cos, *rope_sin;
    Tokenizer tokenizer;
} Model;

/* ===== Helpers ===== */

static void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* ===== Kernel loading and emulator calls ===== */

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
    return 0;
}

static void cpu_reset_for_call(SmolCPU *cpu, uint32_t entry) {
    memset(cpu->r, 0, sizeof(cpu->r));
    memset(cpu->f, 0, sizeof(cpu->f));
    cpu->r[1] = CODE_BASE;
    cpu->r[2] = MEM_SIZE - 4096;
    cpu->pc = entry;
    cpu->halted = 0;
    cpu->insn_count = 0;
    cpu->cycle_count = 0;
    cpu->acc = 0;
    cpu->vl = VEC_LEN;
}

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

static int load_model(Model *m, SmolCPU *cpu, const char *path, int max_seq) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Can't open %s\n", path); return -1; }

    char magic[5] = {0};
    rd_bytes(f, magic, 4);
    if (strcmp(magic, "SMOL")) { fprintf(stderr, "Bad magic\n"); fclose(f); return -1; }
    int ver = rd_u32(f);
    if (ver == 2) { rd_u32(f); rd_u32(f); }

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

    fprintf(stderr, "Config: hs=%d, layers=%d, heads=%d, kv=%d, vocab=%d\n",
           c->hidden_size, c->num_layers, c->num_heads, c->num_kv_heads, c->vocab_size);

    /* Load tokenizer */
    Tokenizer *tok = &m->tokenizer;
    tok->vocab_size = rd_u32(f);
    tok->num_merges = rd_u32(f);
    tok->vocab = malloc(tok->vocab_size * sizeof(char *));
    tok->merges = malloc(tok->num_merges * sizeof(char *));
    for (int i = 0; i < tok->vocab_size; i++) {
        int len = rd_u32(f);
        tok->vocab[i] = malloc(len + 1);
        rd_bytes(f, tok->vocab[i], len);
        tok->vocab[i][len] = 0;
    }
    for (int i = 0; i < tok->num_merges; i++) {
        int len = rd_u32(f);
        tok->merges[i] = malloc(len + 1);
        rd_bytes(f, tok->merges[i], len);
        tok->merges[i][len] = 0;
    }
    fprintf(stderr, "Tokenizer: %d vocab, %d merges\n", tok->vocab_size, tok->num_merges);

    int hs = c->hidden_size, hd = c->head_dim;
    int nh = c->num_heads, nkv = c->num_kv_heads;

    /* Load embedding */
    m->embed_scale = rd_f32(f);
    size_t embed_sz = (size_t)c->vocab_size * hs;
    m->embed_data = malloc(embed_sz);
    rd_bytes(f, m->embed_data, embed_sz);

    uint32_t wptr = WEIGHT_BASE;
    m->embed_data_addr = wptr;
    cpu_load_data(cpu, wptr, m->embed_data, embed_sz);
    wptr += embed_sz;
    wptr = (wptr + 3) & ~3u;
    m->embed_scale_addr = wptr;
    mem_writef(cpu, wptr, m->embed_scale);
    wptr += 4;

    /* Load layers */
    m->layer_addrs = calloc(c->num_layers, sizeof(LayerAddrs));
    for (int l = 0; l < c->num_layers; l++) {
        LayerAddrs *la = &m->layer_addrs[l];

        /* input_layernorm */
        float *ln = malloc(hs * sizeof(float));
        rd_bytes(f, ln, hs * sizeof(float));
        la->input_ln = wptr;
        cpu_load_data(cpu, wptr, ln, hs * sizeof(float));
        wptr += hs * sizeof(float);
        free(ln);

        /* q_proj */
        int q_rows = nh * hd, q_cols = hs;
        la->q_scale_addr = wptr; mem_writef(cpu, wptr, rd_f32(f)); wptr += 4;
        la->q_data = wptr;
        { void *buf = malloc(q_rows * q_cols); rd_bytes(f, buf, q_rows * q_cols);
          cpu_load_data(cpu, wptr, buf, q_rows * q_cols); free(buf); }
        wptr += q_rows * q_cols; wptr = (wptr + 3) & ~3u;

        /* k_proj */
        int k_rows = nkv * hd, k_cols = hs;
        la->k_scale_addr = wptr; mem_writef(cpu, wptr, rd_f32(f)); wptr += 4;
        la->k_data = wptr;
        { void *buf = malloc(k_rows * k_cols); rd_bytes(f, buf, k_rows * k_cols);
          cpu_load_data(cpu, wptr, buf, k_rows * k_cols); free(buf); }
        wptr += k_rows * k_cols; wptr = (wptr + 3) & ~3u;

        /* v_proj */
        la->v_scale_addr = wptr; mem_writef(cpu, wptr, rd_f32(f)); wptr += 4;
        la->v_data = wptr;
        { void *buf = malloc(k_rows * k_cols); rd_bytes(f, buf, k_rows * k_cols);
          cpu_load_data(cpu, wptr, buf, k_rows * k_cols); free(buf); }
        wptr += k_rows * k_cols; wptr = (wptr + 3) & ~3u;

        /* o_proj */
        int o_rows = hs, o_cols = nh * hd;
        la->o_scale_addr = wptr; mem_writef(cpu, wptr, rd_f32(f)); wptr += 4;
        la->o_data = wptr;
        { void *buf = malloc(o_rows * o_cols); rd_bytes(f, buf, o_rows * o_cols);
          cpu_load_data(cpu, wptr, buf, o_rows * o_cols); free(buf); }
        wptr += o_rows * o_cols; wptr = (wptr + 3) & ~3u;

        /* post_attention_layernorm */
        ln = malloc(hs * sizeof(float));
        rd_bytes(f, ln, hs * sizeof(float));
        la->post_ln = wptr;
        cpu_load_data(cpu, wptr, ln, hs * sizeof(float));
        wptr += hs * sizeof(float);
        free(ln);

        /* gate_proj */
        int g_rows = c->intermediate_size, g_cols = hs;
        la->gate_scale_addr = wptr; mem_writef(cpu, wptr, rd_f32(f)); wptr += 4;
        la->gate_data = wptr;
        { void *buf = malloc(g_rows * g_cols); rd_bytes(f, buf, g_rows * g_cols);
          cpu_load_data(cpu, wptr, buf, g_rows * g_cols); free(buf); }
        wptr += g_rows * g_cols; wptr = (wptr + 3) & ~3u;

        /* up_proj */
        la->up_scale_addr = wptr; mem_writef(cpu, wptr, rd_f32(f)); wptr += 4;
        la->up_data = wptr;
        { void *buf = malloc(g_rows * g_cols); rd_bytes(f, buf, g_rows * g_cols);
          cpu_load_data(cpu, wptr, buf, g_rows * g_cols); free(buf); }
        wptr += g_rows * g_cols; wptr = (wptr + 3) & ~3u;

        /* down_proj */
        int d_rows = hs, d_cols = c->intermediate_size;
        la->down_scale_addr = wptr; mem_writef(cpu, wptr, rd_f32(f)); wptr += 4;
        la->down_data = wptr;
        { void *buf = malloc(d_rows * d_cols); rd_bytes(f, buf, d_rows * d_cols);
          cpu_load_data(cpu, wptr, buf, d_rows * d_cols); free(buf); }
        wptr += d_rows * d_cols; wptr = (wptr + 3) & ~3u;
    }

    /* final_norm */
    float *fn = malloc(hs * sizeof(float));
    rd_bytes(f, fn, hs * sizeof(float));
    m->final_norm_addr = wptr;
    cpu_load_data(cpu, wptr, fn, hs * sizeof(float));
    wptr += hs * sizeof(float);
    free(fn);

    fclose(f);
    fprintf(stderr, "Weights: %.1f MB\n", (wptr - WEIGHT_BASE) / 1048576.0f);

    /* RoPE - precompute for effective sequence length */
    int orig_max = c->max_seq_len;
    if (max_seq > 0 && max_seq < orig_max) c->max_seq_len = max_seq;
    precompute_rope(m);
    cpu_load_data(cpu, ROPE_BASE, m->rope_cos, c->max_seq_len * hd * sizeof(float));
    cpu_load_data(cpu, ROPE_BASE + c->max_seq_len * hd * sizeof(float),
                  m->rope_sin, c->max_seq_len * hd * sizeof(float));

    /* KV cache */
    m->kv_cache_addr = KV_BASE;
    m->cache_lens = calloc(c->num_layers, sizeof(int));
    fprintf(stderr, "KV cache: %d positions, %.1f MB\n", c->max_seq_len,
           (float)c->num_layers * nkv * 2 * c->max_seq_len * hd * 4 / 1048576.0f);

    fprintf(stderr, "Ready.\n\n");
    return 0;
}

/* ===== Model Descriptor for Assembly Forward Pass ===== */

static void populate_descriptor(Model *m, SmolCPU *cpu) {
    ModelConfig *c = &m->config;
    uint32_t d = DESC_BASE;

    /* Header */
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
    mem_write32(cpu, d + 0x38, c->max_seq_len * c->head_dim * 4); /* kv_head_stride_bytes */
    mem_write32(cpu, d + 0x3C, 0); /* reserved */

    /* Per-layer descriptors (64 bytes each, matching LayerAddrs layout) */
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

/* ===== Forward pass (assembly kernel) ===== */

static void forward(Model *m, SmolCPU *cpu, int token, int pos, float *logits_out) {
    (void)m;
    cpu_reset_for_call(cpu, FORWARD_ENTRY);
    cpu->r[3] = token;
    cpu->r[4] = pos;
    cpu->r[5] = DESC_BASE;
    cpu_run(cpu, 500000000000ULL);

    uint32_t logits_addr = BUF_BASE + 0x80000;
    read_floats(cpu, logits_addr, logits_out, m->config.vocab_size);
}

/* ===== Tokenizer ===== */

static int find_tok(Tokenizer *t, const char *s, int len) {
    for (int i = 0; i < t->vocab_size; i++)
        if (strlen(t->vocab[i]) == (size_t)len && !strncmp(t->vocab[i], s, len))
            return i;
    return -1;
}

static int tokenize(Model *m, const char *text, int *toks, int max) {
    Tokenizer *t = &m->tokenizer;
    char enc[4096], *o = enc;
    const char *p = text;
    int start = 1;
    while (*p && o < enc + 4090) {
        if (*p == ' ') { if (!start) { *o++ = 0xC4; *o++ = 0xA0; } p++; }
        else if (*p == '\n') { *o++ = 0xC4; *o++ = 0x8A; p++; start = 1; }
        else { *o++ = *p++; start = 0; }
    }
    *o = 0;

    int n = 0;
    p = enc;
    int elen = strlen(enc);
    while (*p && n < max) {
        int best_len = 0, best_id = -1;
        for (int len = 1; len <= 32 && p + len <= enc + elen; len++) {
            int id = find_tok(t, p, len);
            if (id >= 0) { best_len = len; best_id = id; }
        }
        if (best_id >= 0) { toks[n++] = best_id; p += best_len; }
        else { int id = find_tok(t, p, 1); if (id >= 0) toks[n++] = id; p++; }
    }

    /* BPE merges */
    for (int changed = 1, iter = 0; changed && n > 1 && iter < 1000; iter++) {
        changed = 0;
        for (int i = 0; i < t->num_merges && !changed; i++) {
            char *merge = t->merges[i], *sp = strchr(merge, ' ');
            if (!sp) continue;
            int l1 = sp - merge, l2 = strlen(sp + 1);
            for (int j = 0; j < n - 1; j++) {
                const char *t1 = t->vocab[toks[j]], *t2 = t->vocab[toks[j + 1]];
                if (strlen(t1) == (size_t)l1 && !strncmp(t1, merge, l1) &&
                    strlen(t2) == (size_t)l2 && !strcmp(t2, sp + 1)) {
                    char merged[256];
                    snprintf(merged, 256, "%s%s", t1, t2);
                    int mid = find_tok(t, merged, strlen(merged));
                    if (mid >= 0) {
                        toks[j] = mid;
                        for (int k = j + 1; k < n - 1; k++) toks[k] = toks[k + 1];
                        n--;
                        changed = 1;
                        break;
                    }
                }
            }
        }
    }
    return n;
}

static char dec_buf[256];
static const char *decode_token(Model *m, int tok) {
    if (tok < 0 || tok >= m->tokenizer.vocab_size) return "";
    const char *r = m->tokenizer.vocab[tok];
    char *o = dec_buf;
    while (*r && o < dec_buf + 254) {
        if ((unsigned char)r[0] == 0xC4 && (unsigned char)r[1] == 0xA0) { *o++ = ' '; r += 2; }
        else if ((unsigned char)r[0] == 0xC4 && (unsigned char)r[1] == 0x8A) { *o++ = '\n'; r += 2; }
        else *o++ = *r++;
    }
    *o = 0;
    return dec_buf;
}

static int sample(float *logits, int vocab, float temp) {
    if (temp <= 0) {
        int mi = 0;
        for (int i = 1; i < vocab; i++) if (logits[i] > logits[mi]) mi = i;
        return mi;
    }
    for (int i = 0; i < vocab; i++) logits[i] /= temp;
    softmax(logits, vocab);
    float r = (float)rand() / RAND_MAX, cs = 0;
    for (int i = 0; i < vocab; i++) { cs += logits[i]; if (cs >= r) return i; }
    return vocab - 1;
}

/* ===== Main ===== */

int main(int argc, char **argv) {
    const char *model_path = "../models/smollm2-135m-q8.bin";
    const char *prompt = "The capital of France is";
    int max_tok = 50;
    float temp = 0.0f;
    int max_seq = 128; /* Effective max sequence for KV cache */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_tok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) max_seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h")) {
            printf("Usage: %s [-m model] [-p prompt] [-n max_tokens] [-t temp] [-s max_seq]\n", argv[0]);
            return 0;
        }
    }

    srand(time(NULL));

    /* Initialize emulator */
    SmolCPU cpu;
    cpu_init(&cpu);

    uint32_t halt = HALT_INSN;
    cpu_load_data(&cpu, CODE_BASE, &halt, 4);

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
    if (load_model(&model, &cpu, model_path, max_seq) != 0) {
        cpu_free(&cpu);
        return 1;
    }

    /* Populate model descriptor for assembly forward pass */
    populate_descriptor(&model, &cpu);

    /* Tokenize prompt */
    int toks[512];
    int n = tokenize(&model, prompt, toks, 512);
    if (!n) { fprintf(stderr, "Tokenize failed\n"); cpu_free(&cpu); return 1; }

    fprintf(stderr, "Prompt tokens (%d):", n);
    for (int i = 0; i < n; i++) fprintf(stderr, " %d", toks[i]);
    fprintf(stderr, "\n\n");

    /* Print prompt */
    printf("%s", prompt);
    fflush(stdout);

    /* Prefill: run forward pass on all prompt tokens */
    float *logits = malloc(model.config.vocab_size * sizeof(float));
    for (int i = 0; i < n; i++) {
        forward(&model, &cpu, toks[i], i, logits);
        fprintf(stderr, "\rPrefill: %d/%d", i + 1, n);
    }
    fprintf(stderr, "\rPrefill: done     \n");

    /* Generate */
    for (int i = 0; i < max_tok; i++) {
        int next = sample(logits, model.config.vocab_size, temp);
        if (next == 2) break; /* EOS */
        printf("%s", decode_token(&model, next));
        fflush(stdout);
        if (n + i >= model.config.max_seq_len - 1) break; /* KV cache full */
        forward(&model, &cpu, next, n + i, logits);
    }
    printf("\n");

    free(logits);
    cpu_free(&cpu);
    return 0;
}
