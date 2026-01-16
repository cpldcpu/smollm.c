/* SmolLM2 Q8 Inference - Compact C implementation */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "smolc.h"

static void *xmalloc(size_t n) { void *p = malloc(n); if (!p) { fprintf(stderr, "OOM\n"); exit(1); } return p; }
static int rd_u32(FILE *f) { uint32_t v; return fread(&v, 4, 1, f) == 1 ? (int)v : -1; }
static float rd_f32(FILE *f) { float v; return fread(&v, 4, 1, f) == 1 ? v : 0; }
static void rd_bytes(FILE *f, void *buf, size_t n) { if (fread(buf, 1, n, f) != n) { fprintf(stderr, "Read error\n"); exit(1); } }

static void rmsnorm(float *o, float *x, float *w, int n, float eps) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}

static void matmul_q8(float *o, Q8Tensor *W, float *x) {
    int rows = W->rows, cols = W->cols; float s = W->scale; int8_t *d = W->data;
    for (int i = 0; i < rows; i++) {
        float sum = 0; int8_t *row = d + i * cols;
        for (int j = 0; j < cols; j++) sum += row[j] * s * x[j];
        o[i] = sum;
    }
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

static void softmax(float *x, int n) {
    float max = x[0]; for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static void precompute_rope(SmolLM2 *m) {
    int hd = m->config.head_dim, maxlen = m->config.max_seq_len;
    float theta = m->config.rope_theta;
    m->rope_cos = xmalloc(maxlen * hd * sizeof(float));
    m->rope_sin = xmalloc(maxlen * hd * sizeof(float));
    for (int p = 0; p < maxlen; p++) {
        for (int i = 0; i < hd / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / hd);
            float c = cosf(p * freq), s = sinf(p * freq);
            m->rope_cos[p * hd + i] = m->rope_cos[p * hd + hd/2 + i] = c;
            m->rope_sin[p * hd + i] = m->rope_sin[p * hd + hd/2 + i] = s;
        }
    }
}

static void apply_rope(float *v, int hd, float *c, float *s) {
    int h = hd / 2;
    for (int i = 0; i < h; i++) {
        float v0 = v[i], v1 = v[i + h];
        v[i] = v0 * c[i] - v1 * s[i];
        v[i + h] = v1 * c[i + h] + v0 * s[i + h];
    }
}

static void read_q8(FILE *f, QuantTensor *t, int rows, int cols) {
    t->quant_type = QUANT_Q8; t->rows = t->q8.rows = rows; t->cols = t->q8.cols = cols;
    t->q8.scale = rd_f32(f);
    size_t sz = (size_t)rows * cols;
    t->q8.data = xmalloc(sz); rd_bytes(f, t->q8.data, sz);
}

static FP32Tensor read_fp32(FILE *f, int n) {
    FP32Tensor t = { .size = n, .data = xmalloc(n * sizeof(float)) };
    rd_bytes(f, t.data, n * sizeof(float)); return t;
}

int smolc_load(SmolLM2 *m, const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "Can't open %s\n", path); return -1; }
    char magic[5] = {0}; rd_bytes(f, magic, 4);
    if (strcmp(magic, "SMOL")) { fprintf(stderr, "Bad magic\n"); fclose(f); return -1; }
    int ver = rd_u32(f); if (ver != 1 && ver != 2) { fprintf(stderr, "Bad version\n"); fclose(f); return -1; }
    Config *c = &m->config;
    if (ver == 2) { int qt = rd_u32(f); rd_u32(f); if (qt != QUANT_Q8) { fprintf(stderr, "Q8 only\n"); fclose(f); return -1; } }
    c->quant_type = QUANT_Q8; c->group_size = 0;
    c->hidden_size = rd_u32(f); c->intermediate_size = rd_u32(f); c->num_layers = rd_u32(f);
    c->num_heads = rd_u32(f); c->num_kv_heads = rd_u32(f); c->vocab_size = rd_u32(f);
    c->max_seq_len = rd_u32(f); c->rope_theta = rd_f32(f); c->rms_norm_eps = rd_f32(f);
    c->head_dim = c->hidden_size / c->num_heads;
    printf("Config: hidden=%d layers=%d heads=%d kv=%d vocab=%d\n", c->hidden_size, c->num_layers, c->num_heads, c->num_kv_heads, c->vocab_size);

    int nv = rd_u32(f), nm = rd_u32(f);
    Tokenizer *tok = &m->tokenizer; tok->vocab_size = nv; tok->num_merges = nm;
    tok->vocab = xmalloc(nv * sizeof(char*)); tok->merges = xmalloc(nm * sizeof(char*));
    for (int i = 0; i < nv; i++) { int len = rd_u32(f); tok->vocab[i] = xmalloc(len + 1); rd_bytes(f, tok->vocab[i], len); tok->vocab[i][len] = 0; }
    for (int i = 0; i < nm; i++) { int len = rd_u32(f); tok->merges[i] = xmalloc(len + 1); rd_bytes(f, tok->merges[i], len); tok->merges[i][len] = 0; }
    printf("Tokenizer: %d vocab, %d merges\n", nv, nm);

    ModelWeights *w = &m->weights; int hd = c->head_dim;
    read_q8(f, &w->embed_tokens, c->vocab_size, c->hidden_size);
    w->layers = xmalloc(c->num_layers * sizeof(LayerWeights));
    for (int l = 0; l < c->num_layers; l++) {
        LayerWeights *lw = &w->layers[l];
        lw->input_layernorm = read_fp32(f, c->hidden_size);
        read_q8(f, &lw->q_proj, c->num_heads * hd, c->hidden_size);
        read_q8(f, &lw->k_proj, c->num_kv_heads * hd, c->hidden_size);
        read_q8(f, &lw->v_proj, c->num_kv_heads * hd, c->hidden_size);
        read_q8(f, &lw->o_proj, c->hidden_size, c->num_heads * hd);
        lw->post_attention_layernorm = read_fp32(f, c->hidden_size);
        read_q8(f, &lw->gate_proj, c->intermediate_size, c->hidden_size);
        read_q8(f, &lw->up_proj, c->intermediate_size, c->hidden_size);
        read_q8(f, &lw->down_proj, c->hidden_size, c->intermediate_size);
    }
    w->final_norm = read_fp32(f, c->hidden_size);
    fclose(f);

    precompute_rope(m);
    m->kv_caches = xmalloc(c->num_layers * sizeof(KVCache));
    for (int l = 0; l < c->num_layers; l++) {
        m->kv_caches[l].k_cache = xmalloc(c->num_kv_heads * c->max_seq_len * hd * sizeof(float));
        m->kv_caches[l].v_cache = xmalloc(c->num_kv_heads * c->max_seq_len * hd * sizeof(float));
        m->kv_caches[l].cache_len = 0;
    }
    m->x = xmalloc(c->hidden_size * sizeof(float)); m->xb = xmalloc(c->hidden_size * sizeof(float));
    m->xb2 = xmalloc(c->hidden_size * sizeof(float)); m->q = xmalloc(c->num_heads * hd * sizeof(float));
    m->k = xmalloc(c->num_kv_heads * hd * sizeof(float)); m->v = xmalloc(c->num_kv_heads * hd * sizeof(float));
    m->att = xmalloc(c->num_heads * c->max_seq_len * sizeof(float));
    m->logits = xmalloc(c->vocab_size * sizeof(float));
    m->hb = xmalloc(c->intermediate_size * sizeof(float)); m->hb2 = xmalloc(c->intermediate_size * sizeof(float));
    printf("Model loaded\n"); return 0;
}

void smolc_free(SmolLM2 *m) {
    Config *c = &m->config;
    for (int i = 0; i < m->tokenizer.vocab_size; i++) free(m->tokenizer.vocab[i]);
    free(m->tokenizer.vocab);
    for (int i = 0; i < m->tokenizer.num_merges; i++) free(m->tokenizer.merges[i]);
    free(m->tokenizer.merges);
    free(m->weights.embed_tokens.q8.data);
    for (int l = 0; l < c->num_layers; l++) {
        LayerWeights *lw = &m->weights.layers[l];
        free(lw->input_layernorm.data); free(lw->post_attention_layernorm.data);
        free(lw->q_proj.q8.data); free(lw->k_proj.q8.data); free(lw->v_proj.q8.data); free(lw->o_proj.q8.data);
        free(lw->gate_proj.q8.data); free(lw->up_proj.q8.data); free(lw->down_proj.q8.data);
    }
    free(m->weights.layers); free(m->weights.final_norm.data);
    free(m->rope_cos); free(m->rope_sin);
    for (int l = 0; l < c->num_layers; l++) { free(m->kv_caches[l].k_cache); free(m->kv_caches[l].v_cache); }
    free(m->kv_caches);
    free(m->x); free(m->xb); free(m->xb2); free(m->q); free(m->k); free(m->v);
    free(m->att); free(m->logits); free(m->hb); free(m->hb2);
}

void smolc_reset_cache(SmolLM2 *m) { for (int l = 0; l < m->config.num_layers; l++) m->kv_caches[l].cache_len = 0; }

float* smolc_forward(SmolLM2 *m, int tok, int pos) {
    Config *c = &m->config; ModelWeights *w = &m->weights;
    int hs = c->hidden_size, hd = c->head_dim, nh = c->num_heads, nkv = c->num_kv_heads, ng = nh / nkv;

    /* Embedding */
    float s = w->embed_tokens.q8.scale; int8_t *emb = w->embed_tokens.q8.data + tok * hs;
    for (int i = 0; i < hs; i++) m->x[i] = emb[i] * s;

    float *rc = m->rope_cos + pos * hd, *rs = m->rope_sin + pos * hd;

    for (int l = 0; l < c->num_layers; l++) {
        LayerWeights *lw = &w->layers[l]; KVCache *kv = &m->kv_caches[l];
        rmsnorm(m->xb, m->x, lw->input_layernorm.data, hs, c->rms_norm_eps);
        matmul_q8(m->q, &lw->q_proj.q8, m->xb);
        matmul_q8(m->k, &lw->k_proj.q8, m->xb);
        matmul_q8(m->v, &lw->v_proj.q8, m->xb);
        for (int h = 0; h < nh; h++) apply_rope(m->q + h * hd, hd, rc, rs);
        for (int h = 0; h < nkv; h++) apply_rope(m->k + h * hd, hd, rc, rs);

        int off = kv->cache_len * hd;
        for (int h = 0; h < nkv; h++) {
            memcpy(kv->k_cache + h * c->max_seq_len * hd + off, m->k + h * hd, hd * sizeof(float));
            memcpy(kv->v_cache + h * c->max_seq_len * hd + off, m->v + h * hd, hd * sizeof(float));
        }
        int slen = kv->cache_len + 1;

        memset(m->xb2, 0, hs * sizeof(float));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float *qh = m->q + h * hd, *kc = kv->k_cache + kvh * c->max_seq_len * hd;
            float *vc = kv->v_cache + kvh * c->max_seq_len * hd, *att = m->att + h * c->max_seq_len;
            float sc = 1.0f / sqrtf((float)hd);
            for (int t = 0; t < slen; t++) {
                float score = 0; float *kt = kc + t * hd;
                for (int d = 0; d < hd; d++) score += qh[d] * kt[d];
                att[t] = score * sc;
            }
            softmax(att, slen);
            float *oh = m->xb2 + h * hd;
            for (int t = 0; t < slen; t++) { float a = att[t]; float *vt = vc + t * hd; for (int d = 0; d < hd; d++) oh[d] += a * vt[d]; }
        }

        matmul_q8(m->xb, &lw->o_proj.q8, m->xb2);
        for (int i = 0; i < hs; i++) m->x[i] += m->xb[i];
        rmsnorm(m->xb, m->x, lw->post_attention_layernorm.data, hs, c->rms_norm_eps);
        matmul_q8(m->hb, &lw->gate_proj.q8, m->xb);
        matmul_q8(m->hb2, &lw->up_proj.q8, m->xb);
        for (int i = 0; i < c->intermediate_size; i++) m->hb[i] = silu(m->hb[i]) * m->hb2[i];
        matmul_q8(m->xb, &lw->down_proj.q8, m->hb);
        for (int i = 0; i < hs; i++) m->x[i] += m->xb[i];
        kv->cache_len = slen;
    }

    rmsnorm(m->x, m->x, w->final_norm.data, hs, c->rms_norm_eps);
    /* LM head (tied weights) */
    s = w->embed_tokens.q8.scale;
    for (int i = 0; i < c->vocab_size; i++) {
        float sum = 0; int8_t *row = w->embed_tokens.q8.data + i * hs;
        for (int j = 0; j < hs; j++) sum += m->x[j] * row[j] * s;
        m->logits[i] = sum;
    }
    return m->logits;
}

/* Tokenizer */
static int find_tok(Tokenizer *t, const char *s, int len) {
    for (int i = 0; i < t->vocab_size; i++) if (strlen(t->vocab[i]) == (size_t)len && !strncmp(t->vocab[i], s, len)) return i;
    return -1;
}

int smolc_tokenize(SmolLM2 *m, const char *text, int *toks, int max) {
    Tokenizer *t = &m->tokenizer;
    char enc[1024], *o = enc; const char *p = text; int start = 1;
    while (*p && o < enc + 1020) {
        if (*p == ' ') { if (!start) { *o++ = 0xC4; *o++ = 0xA0; } p++; }
        else if (*p == '\n') { *o++ = 0xC4; *o++ = 0x8A; p++; start = 1; }
        else { *o++ = *p++; start = 0; }
    } *o = 0;

    int n = 0; p = enc; int elen = strlen(enc);
    while (*p && n < max) {
        int best_len = 0, best_id = -1;
        for (int len = 1; len <= 32 && p + len <= enc + elen; len++) { int id = find_tok(t, p, len); if (id >= 0) { best_len = len; best_id = id; } }
        if (best_id >= 0) { toks[n++] = best_id; p += best_len; }
        else { int id = find_tok(t, p, 1); if (id >= 0) toks[n++] = id; p++; }
    }

    for (int changed = 1, iter = 0; changed && n > 1 && iter < 1000; iter++) {
        changed = 0;
        for (int i = 0; i < t->num_merges && !changed; i++) {
            char *merge = t->merges[i], *sp = strchr(merge, ' '); if (!sp) continue;
            int l1 = sp - merge, l2 = strlen(sp + 1);
            for (int j = 0; j < n - 1; j++) {
                const char *t1 = t->vocab[toks[j]], *t2 = t->vocab[toks[j + 1]];
                if (strlen(t1) == (size_t)l1 && !strncmp(t1, merge, l1) && strlen(t2) == (size_t)l2 && !strcmp(t2, sp + 1)) {
                    char merged[256]; snprintf(merged, 256, "%s%s", t1, t2);
                    int mid = find_tok(t, merged, strlen(merged));
                    if (mid >= 0) { toks[j] = mid; for (int k = j + 1; k < n - 1; k++) toks[k] = toks[k + 1]; n--; changed = 1; break; }
                }
            }
        }
    }
    return n;
}

static char dec_buf[256];
const char* smolc_decode(SmolLM2 *m, int tok) {
    if (tok < 0 || tok >= m->tokenizer.vocab_size) return "";
    const char *r = m->tokenizer.vocab[tok]; char *o = dec_buf;
    while (*r && o < dec_buf + 254) {
        if ((unsigned char)r[0] == 0xC4 && (unsigned char)r[1] == 0xA0) { *o++ = ' '; r += 2; }
        else if ((unsigned char)r[0] == 0xC4 && (unsigned char)r[1] == 0x8A) { *o++ = '\n'; r += 2; }
        else *o++ = *r++;
    } *o = 0; return dec_buf;
}

int smolc_sample(float *logits, int vocab, float temp) {
    if (temp <= 0) { int mi = 0; for (int i = 1; i < vocab; i++) if (logits[i] > logits[mi]) mi = i; return mi; }
    for (int i = 0; i < vocab; i++) logits[i] /= temp;
    softmax(logits, vocab);
    float r = (float)rand() / RAND_MAX, cs = 0;
    for (int i = 0; i < vocab; i++) { cs += logits[i]; if (cs >= r) return i; }
    return vocab - 1;
}

void smolc_generate(SmolLM2 *m, const char *prompt, int max_tok, float temp) {
    int toks[512], n = smolc_tokenize(m, prompt, toks, 512);
    if (!n) { printf("Tokenize failed\n"); return; }
    printf("Tokens: "); for (int i = 0; i < n; i++) printf("%d ", toks[i]); printf("\n\n%s", prompt); fflush(stdout);
    smolc_reset_cache(m);
    float *logits = NULL; for (int i = 0; i < n; i++) logits = smolc_forward(m, toks[i], i);
    for (int i = 0; i < max_tok; i++) {
        int next = smolc_sample(logits, m->config.vocab_size, temp);
        if (next == 2) break;
        printf("%s", smolc_decode(m, next)); fflush(stdout);
        logits = smolc_forward(m, next, n + i);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    const char *model = "../models/smollm2-135m-q8.bin", *prompt = "The capital of France is";
    int max_tok = 50; float temp = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_tok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "-h")) { printf("Usage: %s [-m model] [-p prompt] [-n tokens] [-t temp]\n", argv[0]); return 0; }
    }
    srand(time(NULL));
    SmolLM2 m; printf("Loading %s...\n", model);
    if (smolc_load(&m, model)) return 1;
    smolc_generate(&m, prompt, max_tok, temp);
    smolc_free(&m); return 0;
}
