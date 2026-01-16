/**
 * SmolLM2 C Inference Engine (Q4/Q8)
 * ANSI C implementation of SmolLM2-135M with Q8/Q4 quantization
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "smolc.h"

/* ============== Memory helpers ============== */

static void* safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Failed to allocate %zu bytes\n", size);
        exit(1);
    }
    return ptr;
}

/* ============== Tensor operations ============== */

/* RMSNorm: x = x * rsqrt(mean(x^2) + eps) * weight */
static void rmsnorm(float *out, float *x, float *weight, int size, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / size + eps;
    ss = 1.0f / sqrtf(ss);
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * ss * weight[i];
    }
}

/* Matrix-vector multiply with Q8 dequantization: out = W * x */
static void matmul_q8(float *out, Q8Tensor *W, float *x) {
    int out_size = W->rows;
    int in_size = W->cols;
    float scale = W->scale;
    int8_t *data = W->data;

    for (int i = 0; i < out_size; i++) {
        float sum = 0.0f;
        int8_t *row = data + i * in_size;
        for (int j = 0; j < in_size; j++) {
            sum += ((float)row[j] * scale) * x[j];
        }
        out[i] = sum;
    }
}

/* Sign extension lookup table for 4-bit values */
static const int8_t q4_sign_extend[16] = {
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1
};

/* Optimized Q4 matmul: bit shifts + 2-element unrolling + lookup table */
static void matmul_q4(float *out, Q4Tensor *W, float *x) {
    const int out_size = W->rows;
    const int in_size = W->cols;
    const int group_size = W->group_size;
    const float *scales = W->scales;
    const uint8_t *data = W->data;

    /* Compute log2(group_size) for bit shifting (group_size=32 -> shift=5) */
    int shift = 0;
    for (int gs = group_size; gs > 1; gs >>= 1) shift++;

    const int groups_per_row = (in_size + group_size - 1) >> shift;

    for (int i = 0; i < out_size; i++) {
        float sum = 0.0f;
        const int row_group_base = i * groups_per_row;
        const int row_byte_base = (i * in_size) >> 1;

        int byte_idx = row_byte_base;
        int j = 0;

        /* Process 2 elements at a time (packed in one byte) */
        for (; j + 1 < in_size; j += 2, byte_idx++) {
            uint8_t packed = data[byte_idx];

            int group_idx0 = row_group_base + (j >> shift);
            float scale0 = scales[group_idx0];
            int8_t val0 = q4_sign_extend[packed & 0x0F];

            int group_idx1 = row_group_base + ((j + 1) >> shift);
            float scale1 = scales[group_idx1];
            int8_t val1 = q4_sign_extend[packed >> 4];

            sum += (float)val0 * scale0 * x[j];
            sum += (float)val1 * scale1 * x[j + 1];
        }

        /* Handle odd trailing element */
        if (j < in_size) {
            uint8_t packed = data[byte_idx];
            int group_idx = row_group_base + (j >> shift);
            float scale = scales[group_idx];
            sum += (float)q4_sign_extend[packed & 0x0F] * scale * x[j];
        }

        out[i] = sum;
    }
}

/* Generic matmul dispatcher */
static void matmul_quant(float *out, QuantTensor *W, float *x) {
    if (W->quant_type == QUANT_Q8) {
        matmul_q8(out, &W->q8, x);
    } else {
        matmul_q4(out, &W->q4, x);
    }
}

/* SiLU activation: x * sigmoid(x) */
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* Softmax in place */
static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

/* ============== RoPE ============== */

static void precompute_rope(SmolLM2 *model) {
    int head_dim = model->config.head_dim;
    int max_seq_len = model->config.max_seq_len;
    float theta = model->config.rope_theta;

    model->rope_cos = safe_malloc(max_seq_len * head_dim * sizeof(float));
    model->rope_sin = safe_malloc(max_seq_len * head_dim * sizeof(float));

    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float angle = pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            model->rope_cos[pos * head_dim + i] = cos_val;
            model->rope_cos[pos * head_dim + head_dim / 2 + i] = cos_val;
            model->rope_sin[pos * head_dim + i] = sin_val;
            model->rope_sin[pos * head_dim + head_dim / 2 + i] = sin_val;
        }
    }
}

static void apply_rope(float *vec, int head_dim, float *cos, float *sin) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float v0 = vec[i];
        float v1 = vec[i + half];
        vec[i] = v0 * cos[i] - v1 * sin[i];
        vec[i + half] = v1 * cos[i + half] + v0 * sin[i + half];
    }
}

/* ============== Model loading ============== */

static int read_uint32(FILE *f) {
    uint32_t val;
    if (fread(&val, sizeof(uint32_t), 1, f) != 1) return -1;
    return (int)val;
}

static float read_float32(FILE *f) {
    float val;
    if (fread(&val, sizeof(float), 1, f) != 1) return 0.0f;
    return val;
}

static void read_quant_tensor(FILE *f, QuantTensor *t, int rows, int cols, int quant_type, int group_size) {
    t->quant_type = quant_type;
    t->rows = rows;
    t->cols = cols;

    if (quant_type == QUANT_Q8) {
        t->q8.rows = rows;
        t->q8.cols = cols;
        t->q8.scale = read_float32(f);
        size_t size = (size_t)rows * cols;
        t->q8.data = safe_malloc(size);
        if (fread(t->q8.data, 1, size, f) != size) {
            fprintf(stderr, "Failed to read Q8 tensor data\n");
            exit(1);
        }
    } else {
        /* Q4 */
        t->q4.rows = rows;
        t->q4.cols = cols;
        t->q4.group_size = group_size;
        t->q4.num_groups = read_uint32(f);
        t->q4.scales = safe_malloc(t->q4.num_groups * sizeof(float));
        if (fread(t->q4.scales, sizeof(float), t->q4.num_groups, f) != (size_t)t->q4.num_groups) {
            fprintf(stderr, "Failed to read Q4 scales\n");
            exit(1);
        }
        /* Packed data: numel/2 bytes (padded) */
        int numel = rows * cols;
        int padded = ((numel + group_size - 1) / group_size) * group_size;
        size_t packed_size = padded / 2;
        t->q4.data = safe_malloc(packed_size);
        if (fread(t->q4.data, 1, packed_size, f) != packed_size) {
            fprintf(stderr, "Failed to read Q4 data\n");
            exit(1);
        }
    }
}

static FP32Tensor read_fp32_tensor(FILE *f, int size) {
    FP32Tensor t;
    t.size = size;
    t.data = safe_malloc(size * sizeof(float));
    if (fread(t.data, sizeof(float), size, f) != (size_t)size) {
        fprintf(stderr, "Failed to read FP32 tensor data\n");
        exit(1);
    }
    return t;
}

int smolc_load(SmolLM2 *model, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open model file: %s\n", filepath);
        return -1;
    }

    /* Read magic */
    char magic[5] = {0};
    if (fread(magic, 1, 4, f) != 4 || strcmp(magic, "SMOL") != 0) {
        fprintf(stderr, "Invalid magic: %s\n", magic);
        fclose(f);
        return -1;
    }

    /* Read version */
    int version = read_uint32(f);
    if (version != 1 && version != 2) {
        fprintf(stderr, "Unsupported version: %d\n", version);
        fclose(f);
        return -1;
    }

    Config *cfg = &model->config;

    /* Version 2 has quant_type and group_size */
    if (version == 2) {
        cfg->quant_type = read_uint32(f);
        cfg->group_size = read_uint32(f);
    } else {
        cfg->quant_type = QUANT_Q8;
        cfg->group_size = 32;
    }

    cfg->hidden_size = read_uint32(f);
    cfg->intermediate_size = read_uint32(f);
    cfg->num_layers = read_uint32(f);
    cfg->num_heads = read_uint32(f);
    cfg->num_kv_heads = read_uint32(f);
    cfg->vocab_size = read_uint32(f);
    cfg->max_seq_len = read_uint32(f);
    cfg->rope_theta = read_float32(f);
    cfg->rms_norm_eps = read_float32(f);
    cfg->head_dim = cfg->hidden_size / cfg->num_heads;

    printf("Config: hidden=%d, layers=%d, heads=%d, kv_heads=%d, vocab=%d\n",
           cfg->hidden_size, cfg->num_layers, cfg->num_heads, cfg->num_kv_heads, cfg->vocab_size);
    printf("Quantization: %s", cfg->quant_type == QUANT_Q4 ? "Q4" : "Q8");
    if (cfg->quant_type == QUANT_Q4) {
        printf(" (group_size=%d)", cfg->group_size);
    }
    printf("\n");

    /* Read tokenizer */
    int num_vocab = read_uint32(f);
    int num_merges = read_uint32(f);

    model->tokenizer.vocab_size = num_vocab;
    model->tokenizer.num_merges = num_merges;
    model->tokenizer.vocab = safe_malloc(num_vocab * sizeof(char*));
    model->tokenizer.merges = safe_malloc(num_merges * sizeof(char*));

    for (int i = 0; i < num_vocab; i++) {
        int len = read_uint32(f);
        model->tokenizer.vocab[i] = safe_malloc(len + 1);
        if (fread(model->tokenizer.vocab[i], 1, len, f) != (size_t)len) {
            fprintf(stderr, "Failed to read vocab entry\n");
            fclose(f);
            return -1;
        }
        model->tokenizer.vocab[i][len] = '\0';
    }

    for (int i = 0; i < num_merges; i++) {
        int len = read_uint32(f);
        model->tokenizer.merges[i] = safe_malloc(len + 1);
        if (fread(model->tokenizer.merges[i], 1, len, f) != (size_t)len) {
            fprintf(stderr, "Failed to read merge entry\n");
            fclose(f);
            return -1;
        }
        model->tokenizer.merges[i][len] = '\0';
    }

    printf("Tokenizer: %d vocab, %d merges\n", num_vocab, num_merges);

    /* Read weights */
    ModelWeights *w = &model->weights;
    int head_dim = cfg->head_dim;
    int quant_type = cfg->quant_type;
    int group_size = cfg->group_size;

    /* Embedding */
    read_quant_tensor(f, &w->embed_tokens, cfg->vocab_size, cfg->hidden_size, quant_type, group_size);

    /* Layers */
    w->layers = safe_malloc(cfg->num_layers * sizeof(LayerWeights));

    for (int l = 0; l < cfg->num_layers; l++) {
        LayerWeights *lw = &w->layers[l];
        lw->input_layernorm = read_fp32_tensor(f, cfg->hidden_size);
        read_quant_tensor(f, &lw->q_proj, cfg->num_heads * head_dim, cfg->hidden_size, quant_type, group_size);
        read_quant_tensor(f, &lw->k_proj, cfg->num_kv_heads * head_dim, cfg->hidden_size, quant_type, group_size);
        read_quant_tensor(f, &lw->v_proj, cfg->num_kv_heads * head_dim, cfg->hidden_size, quant_type, group_size);
        read_quant_tensor(f, &lw->o_proj, cfg->hidden_size, cfg->num_heads * head_dim, quant_type, group_size);
        lw->post_attention_layernorm = read_fp32_tensor(f, cfg->hidden_size);
        read_quant_tensor(f, &lw->gate_proj, cfg->intermediate_size, cfg->hidden_size, quant_type, group_size);
        read_quant_tensor(f, &lw->up_proj, cfg->intermediate_size, cfg->hidden_size, quant_type, group_size);
        read_quant_tensor(f, &lw->down_proj, cfg->hidden_size, cfg->intermediate_size, quant_type, group_size);
    }

    /* Final norm */
    w->final_norm = read_fp32_tensor(f, cfg->hidden_size);

    fclose(f);

    /* Precompute RoPE */
    precompute_rope(model);

    /* Allocate KV caches */
    model->kv_caches = safe_malloc(cfg->num_layers * sizeof(KVCache));
    for (int l = 0; l < cfg->num_layers; l++) {
        model->kv_caches[l].k_cache = safe_malloc(cfg->num_kv_heads * cfg->max_seq_len * head_dim * sizeof(float));
        model->kv_caches[l].v_cache = safe_malloc(cfg->num_kv_heads * cfg->max_seq_len * head_dim * sizeof(float));
        model->kv_caches[l].cache_len = 0;
    }

    /* Allocate scratch buffers */
    model->x = safe_malloc(cfg->hidden_size * sizeof(float));
    model->xb = safe_malloc(cfg->hidden_size * sizeof(float));
    model->xb2 = safe_malloc(cfg->hidden_size * sizeof(float));
    model->q = safe_malloc(cfg->num_heads * head_dim * sizeof(float));
    model->k = safe_malloc(cfg->num_kv_heads * head_dim * sizeof(float));
    model->v = safe_malloc(cfg->num_kv_heads * head_dim * sizeof(float));
    model->att = safe_malloc(cfg->num_heads * cfg->max_seq_len * sizeof(float));
    model->logits = safe_malloc(cfg->vocab_size * sizeof(float));
    model->hb = safe_malloc(cfg->intermediate_size * sizeof(float));
    model->hb2 = safe_malloc(cfg->intermediate_size * sizeof(float));

    printf("Model loaded successfully\n");
    return 0;
}

static void free_quant_tensor(QuantTensor *t) {
    if (t->quant_type == QUANT_Q8) {
        free(t->q8.data);
    } else {
        free(t->q4.scales);
        free(t->q4.data);
    }
}

void smolc_free(SmolLM2 *model) {
    Config *cfg = &model->config;

    /* Free tokenizer */
    for (int i = 0; i < model->tokenizer.vocab_size; i++) {
        free(model->tokenizer.vocab[i]);
    }
    free(model->tokenizer.vocab);
    for (int i = 0; i < model->tokenizer.num_merges; i++) {
        free(model->tokenizer.merges[i]);
    }
    free(model->tokenizer.merges);

    /* Free weights */
    free_quant_tensor(&model->weights.embed_tokens);
    for (int l = 0; l < cfg->num_layers; l++) {
        LayerWeights *lw = &model->weights.layers[l];
        free(lw->input_layernorm.data);
        free_quant_tensor(&lw->q_proj);
        free_quant_tensor(&lw->k_proj);
        free_quant_tensor(&lw->v_proj);
        free_quant_tensor(&lw->o_proj);
        free(lw->post_attention_layernorm.data);
        free_quant_tensor(&lw->gate_proj);
        free_quant_tensor(&lw->up_proj);
        free_quant_tensor(&lw->down_proj);
    }
    free(model->weights.layers);
    free(model->weights.final_norm.data);

    /* Free RoPE */
    free(model->rope_cos);
    free(model->rope_sin);

    /* Free KV caches */
    for (int l = 0; l < cfg->num_layers; l++) {
        free(model->kv_caches[l].k_cache);
        free(model->kv_caches[l].v_cache);
    }
    free(model->kv_caches);

    /* Free scratch buffers */
    free(model->x);
    free(model->xb);
    free(model->xb2);
    free(model->q);
    free(model->k);
    free(model->v);
    free(model->att);
    free(model->logits);
    free(model->hb);
    free(model->hb2);
}

void smolc_reset_cache(SmolLM2 *model) {
    for (int l = 0; l < model->config.num_layers; l++) {
        model->kv_caches[l].cache_len = 0;
    }
}

/* ============== Forward pass ============== */

/* Get embedding for a token from quantized embedding matrix */
static void get_embedding(SmolLM2 *model, int token, float *out) {
    QuantTensor *emb = &model->weights.embed_tokens;
    int hidden_size = model->config.hidden_size;

    if (emb->quant_type == QUANT_Q8) {
        float scale = emb->q8.scale;
        int8_t *row = emb->q8.data + token * hidden_size;
        for (int i = 0; i < hidden_size; i++) {
            out[i] = (float)row[i] * scale;
        }
    } else {
        /* Q4 - optimized */
        int group_size = emb->q4.group_size;
        int shift = 0;
        for (int gs = group_size; gs > 1; gs >>= 1) shift++;

        int groups_per_row = (hidden_size + group_size - 1) >> shift;
        int row_group_base = token * groups_per_row;
        int byte_idx = (token * hidden_size) >> 1;

        int i = 0;
        for (; i + 1 < hidden_size; i += 2, byte_idx++) {
            uint8_t packed = emb->q4.data[byte_idx];
            out[i] = (float)q4_sign_extend[packed & 0x0F] * emb->q4.scales[row_group_base + (i >> shift)];
            out[i + 1] = (float)q4_sign_extend[packed >> 4] * emb->q4.scales[row_group_base + ((i + 1) >> shift)];
        }
        if (i < hidden_size) {
            uint8_t packed = emb->q4.data[byte_idx];
            out[i] = (float)q4_sign_extend[packed & 0x0F] * emb->q4.scales[row_group_base + (i >> shift)];
        }
    }
}

/* Compute logits: out = x @ embed_tokens.T */
static void compute_logits(SmolLM2 *model, float *x) {
    QuantTensor *emb = &model->weights.embed_tokens;
    int vocab_size = model->config.vocab_size;
    int hidden_size = model->config.hidden_size;

    if (emb->quant_type == QUANT_Q8) {
        float scale = emb->q8.scale;
        for (int i = 0; i < vocab_size; i++) {
            float sum = 0.0f;
            int8_t *row = emb->q8.data + i * hidden_size;
            for (int j = 0; j < hidden_size; j++) {
                sum += x[j] * ((float)row[j] * scale);
            }
            model->logits[i] = sum;
        }
    } else {
        /* Q4 - optimized */
        int group_size = emb->q4.group_size;
        int shift = 0;
        for (int gs = group_size; gs > 1; gs >>= 1) shift++;

        int groups_per_row = (hidden_size + group_size - 1) >> shift;

        for (int i = 0; i < vocab_size; i++) {
            float sum = 0.0f;
            int row_group_base = i * groups_per_row;
            int byte_idx = (i * hidden_size) >> 1;

            int j = 0;
            for (; j + 1 < hidden_size; j += 2, byte_idx++) {
                uint8_t packed = emb->q4.data[byte_idx];
                float w0 = (float)q4_sign_extend[packed & 0x0F] * emb->q4.scales[row_group_base + (j >> shift)];
                float w1 = (float)q4_sign_extend[packed >> 4] * emb->q4.scales[row_group_base + ((j + 1) >> shift)];
                sum += x[j] * w0 + x[j + 1] * w1;
            }
            if (j < hidden_size) {
                uint8_t packed = emb->q4.data[byte_idx];
                sum += x[j] * (float)q4_sign_extend[packed & 0x0F] * emb->q4.scales[row_group_base + (j >> shift)];
            }
            model->logits[i] = sum;
        }
    }
}

float* smolc_forward(SmolLM2 *model, int token, int pos) {
    Config *cfg = &model->config;
    ModelWeights *w = &model->weights;
    int hidden_size = cfg->hidden_size;
    int head_dim = cfg->head_dim;
    int num_heads = cfg->num_heads;
    int num_kv_heads = cfg->num_kv_heads;
    int num_kv_groups = num_heads / num_kv_heads;

    /* Embedding */
    get_embedding(model, token, model->x);

    /* RoPE cos/sin for this position */
    float *rope_cos = model->rope_cos + pos * head_dim;
    float *rope_sin = model->rope_sin + pos * head_dim;

    /* Forward through layers */
    for (int l = 0; l < cfg->num_layers; l++) {
        LayerWeights *lw = &w->layers[l];
        KVCache *cache = &model->kv_caches[l];

        /* Input layernorm */
        rmsnorm(model->xb, model->x, lw->input_layernorm.data, hidden_size, cfg->rms_norm_eps);

        /* Q, K, V projections */
        matmul_quant(model->q, &lw->q_proj, model->xb);
        matmul_quant(model->k, &lw->k_proj, model->xb);
        matmul_quant(model->v, &lw->v_proj, model->xb);

        /* Apply RoPE to Q and K */
        for (int h = 0; h < num_heads; h++) {
            apply_rope(model->q + h * head_dim, head_dim, rope_cos, rope_sin);
        }
        for (int h = 0; h < num_kv_heads; h++) {
            apply_rope(model->k + h * head_dim, head_dim, rope_cos, rope_sin);
        }

        /* Store K, V in cache */
        int cache_offset = cache->cache_len * head_dim;
        for (int h = 0; h < num_kv_heads; h++) {
            memcpy(cache->k_cache + h * cfg->max_seq_len * head_dim + cache_offset,
                   model->k + h * head_dim, head_dim * sizeof(float));
            memcpy(cache->v_cache + h * cfg->max_seq_len * head_dim + cache_offset,
                   model->v + h * head_dim, head_dim * sizeof(float));
        }
        int seq_len = cache->cache_len + 1;

        /* Multi-head attention */
        memset(model->xb2, 0, hidden_size * sizeof(float));

        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / num_kv_groups;
            float *q_head = model->q + h * head_dim;
            float *k_cache = cache->k_cache + kv_h * cfg->max_seq_len * head_dim;
            float *v_cache = cache->v_cache + kv_h * cfg->max_seq_len * head_dim;
            float *att = model->att + h * cfg->max_seq_len;

            float scale = 1.0f / sqrtf((float)head_dim);
            for (int t = 0; t < seq_len; t++) {
                float score = 0.0f;
                float *k_t = k_cache + t * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    score += q_head[d] * k_t[d];
                }
                att[t] = score * scale;
            }

            softmax(att, seq_len);

            float *out_head = model->xb2 + h * head_dim;
            for (int t = 0; t < seq_len; t++) {
                float a = att[t];
                float *v_t = v_cache + t * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    out_head[d] += a * v_t[d];
                }
            }
        }

        /* Output projection */
        matmul_quant(model->xb, &lw->o_proj, model->xb2);

        /* Residual connection */
        for (int i = 0; i < hidden_size; i++) {
            model->x[i] += model->xb[i];
        }

        /* Post-attention layernorm */
        rmsnorm(model->xb, model->x, lw->post_attention_layernorm.data, hidden_size, cfg->rms_norm_eps);

        /* MLP: SwiGLU */
        matmul_quant(model->hb, &lw->gate_proj, model->xb);
        matmul_quant(model->hb2, &lw->up_proj, model->xb);

        for (int i = 0; i < cfg->intermediate_size; i++) {
            model->hb[i] = silu(model->hb[i]) * model->hb2[i];
        }

        matmul_quant(model->xb, &lw->down_proj, model->hb);

        for (int i = 0; i < hidden_size; i++) {
            model->x[i] += model->xb[i];
        }

        cache->cache_len = seq_len;
    }

    /* Final norm */
    rmsnorm(model->x, model->x, w->final_norm.data, hidden_size, cfg->rms_norm_eps);

    /* LM head */
    compute_logits(model, model->x);

    return model->logits;
}

/* ============== Tokenizer ============== */

static int find_token(Tokenizer *tok, const char *str, int len) {
    for (int i = 0; i < tok->vocab_size; i++) {
        if (strlen(tok->vocab[i]) == (size_t)len && strncmp(tok->vocab[i], str, len) == 0) {
            return i;
        }
    }
    return -1;
}

static int convert_to_gpt2_bytes(const char *text, char *out, int max_out) {
    char *o = out;
    const char *p = text;
    int is_start = 1;

    while (*p && o < out + max_out - 3) {
        if (*p == ' ') {
            if (!is_start) {
                *o++ = (char)0xC4;
                *o++ = (char)0xA0;
            }
            p++;
        } else if (*p == '\n') {
            *o++ = (char)0xC4;
            *o++ = (char)0x8A;
            p++;
            is_start = 1;
        } else {
            *o++ = *p++;
            is_start = 0;
        }
    }
    *o = '\0';
    return (int)(o - out);
}

int smolc_tokenize(SmolLM2 *model, const char *text, int *tokens, int max_tokens) {
    Tokenizer *tok = &model->tokenizer;

    char encoded[1024];
    convert_to_gpt2_bytes(text, encoded, sizeof(encoded));

    int n = 0;
    const char *p = encoded;
    int encoded_len = strlen(encoded);

    while (*p && n < max_tokens) {
        int best_len = 0;
        int best_id = -1;

        for (int len = 1; len <= 32 && p + len <= encoded + encoded_len; len++) {
            int id = find_token(tok, p, len);
            if (id >= 0) {
                best_len = len;
                best_id = id;
            }
        }

        if (best_id >= 0) {
            tokens[n++] = best_id;
            p += best_len;
        } else {
            int id = find_token(tok, p, 1);
            if (id >= 0) {
                tokens[n++] = id;
            } else {
                fprintf(stderr, "Warning: unknown byte 0x%02x\n", (unsigned char)*p);
            }
            p++;
        }
    }

    int changed = 1;
    int iterations = 0;
    while (changed && n > 1 && iterations < 1000) {
        changed = 0;
        iterations++;

        for (int m = 0; m < tok->num_merges && !changed; m++) {
            char *merge = tok->merges[m];
            char *space = strchr(merge, ' ');
            if (!space) continue;

            int len1 = space - merge;
            int len2 = strlen(space + 1);

            for (int i = 0; i < n - 1; i++) {
                const char *t1 = tok->vocab[tokens[i]];
                const char *t2 = tok->vocab[tokens[i + 1]];

                if (strlen(t1) == (size_t)len1 && strncmp(t1, merge, len1) == 0 &&
                    strlen(t2) == (size_t)len2 && strcmp(t2, space + 1) == 0) {
                    char merged[256];
                    snprintf(merged, sizeof(merged), "%s%s", t1, t2);
                    int merged_id = find_token(tok, merged, strlen(merged));
                    if (merged_id >= 0) {
                        tokens[i] = merged_id;
                        for (int j = i + 1; j < n - 1; j++) {
                            tokens[j] = tokens[j + 1];
                        }
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

static char decode_buf[256];

const char* smolc_decode(SmolLM2 *model, int token) {
    if (token < 0 || token >= model->tokenizer.vocab_size) {
        return "";
    }
    const char *raw = model->tokenizer.vocab[token];

    char *out = decode_buf;
    while (*raw && out < decode_buf + sizeof(decode_buf) - 1) {
        if ((unsigned char)raw[0] == 0xC4 && (unsigned char)raw[1] == 0xA0) {
            *out++ = ' ';
            raw += 2;
        }
        else if ((unsigned char)raw[0] == 0xC4 && (unsigned char)raw[1] == 0x8A) {
            *out++ = '\n';
            raw += 2;
        }
        else {
            *out++ = *raw++;
        }
    }
    *out = '\0';
    return decode_buf;
}

/* ============== Sampling ============== */

int smolc_sample(float *logits, int vocab_size, float temperature) {
    if (temperature <= 0.0f) {
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }

    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }
    softmax(logits, vocab_size);

    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += logits[i];
        if (cumsum >= r) {
            return i;
        }
    }
    return vocab_size - 1;
}

/* ============== Generation ============== */

void smolc_generate(SmolLM2 *model, const char *prompt, int max_tokens, float temperature) {
    int tokens[512];
    int n_tokens = smolc_tokenize(model, prompt, tokens, 512);

    if (n_tokens == 0) {
        printf("Failed to tokenize prompt\n");
        return;
    }

    printf("Prompt tokens: ");
    for (int i = 0; i < n_tokens; i++) {
        printf("%d ", tokens[i]);
    }
    printf("\n\nGenerating:\n");
    printf("%s", prompt);
    fflush(stdout);

    smolc_reset_cache(model);

    float *logits = NULL;
    for (int i = 0; i < n_tokens; i++) {
        logits = smolc_forward(model, tokens[i], i);
    }

    for (int i = 0; i < max_tokens; i++) {
        int next_token = smolc_sample(logits, model->config.vocab_size, temperature);

        if (next_token == 2) {
            break;
        }

        const char *token_str = smolc_decode(model, next_token);
        printf("%s", token_str);
        fflush(stdout);

        logits = smolc_forward(model, next_token, n_tokens + i);
    }

    printf("\n");
}

/* ============== Main ============== */

int main(int argc, char **argv) {
    const char *model_path = "../models/smollm2-135m-q8.bin";
    const char *prompt = "The capital of France is";
    int max_tokens = 50;
    float temperature = 0.0f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  -m <path>   Model file path (default: %s)\n", model_path);
            printf("  -p <text>   Prompt text\n");
            printf("  -n <num>    Max tokens to generate (default: %d)\n", max_tokens);
            printf("  -t <temp>   Temperature (default: %.1f, 0 = greedy)\n", temperature);
            return 0;
        }
    }

    srand(time(NULL));

    SmolLM2 model;
    printf("Loading model from %s...\n", model_path);
    if (smolc_load(&model, model_path) != 0) {
        return 1;
    }

    smolc_generate(&model, prompt, max_tokens, temperature);

    smolc_free(&model);
    return 0;
}
