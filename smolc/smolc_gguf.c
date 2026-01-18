/* SmolLM2 GGUF Q8_0 Inference - GGUF format support */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "smolc_gguf.h"

/* Helper macros and functions */
static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { fprintf(stderr, "OOM\n"); exit(1); }
    return p;
}

static uint32_t rd_u32(FILE *f) {
    uint32_t v;
    return fread(&v, 4, 1, f) == 1 ? v : 0;
}

static uint64_t rd_u64(FILE *f) {
    uint64_t v;
    return fread(&v, 8, 1, f) == 1 ? v : 0;
}

static float rd_f32(FILE *f) {
    float v;
    return fread(&v, 4, 1, f) == 1 ? v : 0;
}

static void rd_bytes(FILE *f, void *buf, size_t n) {
    if (fread(buf, 1, n, f) != n) {
        fprintf(stderr, "Read error\n");
        exit(1);
    }
}

static char *rd_string(FILE *f) {
    uint64_t len = rd_u64(f);
    if (len > 1024 * 1024) { fprintf(stderr, "String too long\n"); exit(1); }
    char *s = xmalloc(len + 1);
    rd_bytes(f, s, len);
    s[len] = 0;
    return s;
}

/* FP16 to FP32 conversion */
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = (h & 0x03FF);

    if (exponent == 0) {
        if (mantissa == 0) {
            /* Zero */
            return *(float*)&sign;
        }
        /* Denormalized */
        exponent = 1;
        while ((mantissa & 0x0400) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x03FF;
    } else if (exponent == 31) {
        /* Inf or NaN */
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        return *(float*)&result;
    }

    /* Normalized */
    exponent = exponent - 15 + 127;
    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    return *(float*)&result;
}

/* Dequantize Q8_0 block to FP32 */
void dequantize_q8_0(const block_q8_0 *block, float *out) {
    float scale = fp16_to_fp32(block->d);
    for (int i = 0; i < QK8_0; i++) {
        out[i] = block->qs[i] * scale;
    }
}

/* Core math operations */
static void rmsnorm(float *o, float *x, float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}

/* Optimized Q8_0 matrix multiplication */
static void matmul_q8_0(float *o, Q8_0_Tensor *W, float *x) {
    int rows = W->rows, cols = W->cols;
    int blocks_per_row = cols / QK8_0;

    for (int i = 0; i < rows; i++) {
        float sum = 0;
        block_q8_0 *row_blocks = W->blocks + i * blocks_per_row;

        for (int b = 0; b < blocks_per_row; b++) {
            block_q8_0 *block = &row_blocks[b];
            float scale = fp16_to_fp32(block->d);
            float block_sum = 0;

            /* Accumulate over block */
            for (int j = 0; j < QK8_0; j++) {
                block_sum += block->qs[j] * x[b * QK8_0 + j];
            }
            sum += block_sum * scale;
        }
        o[i] = sum;
    }
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max) max = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* RoPE precomputation */
static void precompute_rope(SmolLM2_GGUF *m) {
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

/* Skip GGUF metadata value */
static void skip_metadata_value(FILE *f, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:   fseek(f, 1, SEEK_CUR); break;
        case GGUF_TYPE_INT8:    fseek(f, 1, SEEK_CUR); break;
        case GGUF_TYPE_UINT16:  fseek(f, 2, SEEK_CUR); break;
        case GGUF_TYPE_INT16:   fseek(f, 2, SEEK_CUR); break;
        case GGUF_TYPE_UINT32:  fseek(f, 4, SEEK_CUR); break;
        case GGUF_TYPE_INT32:   fseek(f, 4, SEEK_CUR); break;
        case GGUF_TYPE_FLOAT32: fseek(f, 4, SEEK_CUR); break;
        case GGUF_TYPE_UINT64:  fseek(f, 8, SEEK_CUR); break;
        case GGUF_TYPE_INT64:   fseek(f, 8, SEEK_CUR); break;
        case GGUF_TYPE_FLOAT64: fseek(f, 8, SEEK_CUR); break;
        case GGUF_TYPE_BOOL:    fseek(f, 1, SEEK_CUR); break;
        case GGUF_TYPE_STRING: {
            uint64_t len = rd_u64(f);
            fseek(f, len, SEEK_CUR);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t arr_type = rd_u32(f);
            uint64_t arr_len = rd_u64(f);
            for (uint64_t i = 0; i < arr_len; i++) {
                skip_metadata_value(f, arr_type);
            }
            break;
        }
    }
}

/* Read GGUF metadata value */
static int read_metadata_u32(FILE *f, uint32_t type) {
    if (type == GGUF_TYPE_UINT32 || type == GGUF_TYPE_INT32) {
        return (int)rd_u32(f);
    } else if (type == GGUF_TYPE_UINT64 || type == GGUF_TYPE_INT64) {
        return (int)rd_u64(f);
    }
    skip_metadata_value(f, type);
    return 0;
}

static float read_metadata_f32(FILE *f, uint32_t type) {
    if (type == GGUF_TYPE_FLOAT32) {
        return rd_f32(f);
    }
    skip_metadata_value(f, type);
    return 0.0f;
}

/* Load GGUF model */
int smolc_gguf_load(SmolLM2_GGUF *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Can't open %s\n", path);
        return -1;
    }

    /* Read header */
    uint32_t magic = rd_u32(f);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: 0x%08x\n", magic);
        fclose(f);
        return -1;
    }

    uint32_t version = rd_u32(f);
    if (version != GGUF_VERSION) {
        fprintf(stderr, "Unsupported GGUF version: %u (expected %d)\n", version, GGUF_VERSION);
        fclose(f);
        return -1;
    }

    uint64_t tensor_count = rd_u64(f);
    uint64_t metadata_kv_count = rd_u64(f);

    printf("GGUF: version=%u, tensors=%llu, metadata=%llu\n",
           version, (unsigned long long)tensor_count, (unsigned long long)metadata_kv_count);

    /* Parse metadata */
    Config_GGUF *c = &m->config;
    memset(c, 0, sizeof(Config_GGUF));
    c->rope_theta = 10000.0f;  /* Default */
    c->rms_norm_eps = 1e-5f;   /* Default */

    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        char *key = rd_string(f);
        uint32_t value_type = rd_u32(f);

        /* Parse relevant keys */
        if (!strcmp(key, "llama.embedding_length") || !strcmp(key, "llama.embed_length")) {
            c->hidden_size = read_metadata_u32(f, value_type);
        } else if (!strcmp(key, "llama.feed_forward_length")) {
            c->intermediate_size = read_metadata_u32(f, value_type);
        } else if (!strcmp(key, "llama.block_count")) {
            c->num_layers = read_metadata_u32(f, value_type);
        } else if (!strcmp(key, "llama.attention.head_count")) {
            c->num_heads = read_metadata_u32(f, value_type);
        } else if (!strcmp(key, "llama.attention.head_count_kv")) {
            c->num_kv_heads = read_metadata_u32(f, value_type);
        } else if (!strcmp(key, "llama.context_length")) {
            c->max_seq_len = read_metadata_u32(f, value_type);
        } else if (!strcmp(key, "llama.rope.freq_base")) {
            c->rope_theta = read_metadata_f32(f, value_type);
        } else if (!strcmp(key, "llama.attention.layer_norm_rms_epsilon")) {
            c->rms_norm_eps = read_metadata_f32(f, value_type);
        } else if (!strcmp(key, "tokenizer.ggml.model")) {
            char *tok_model = rd_string(f);
            free(tok_model);
        } else {
            skip_metadata_value(f, value_type);
        }
        free(key);
    }

    /* Read tensor info */
    typedef struct {
        char *name;
        uint32_t n_dims;
        uint64_t *dims;
        uint32_t type;
        uint64_t offset;
        size_t size;
    } TensorInfo;

    TensorInfo *tensors = xmalloc(tensor_count * sizeof(TensorInfo));
    uint64_t max_offset = 0;

    for (uint64_t i = 0; i < tensor_count; i++) {
        tensors[i].name = rd_string(f);
        tensors[i].n_dims = rd_u32(f);
        tensors[i].dims = xmalloc(tensors[i].n_dims * sizeof(uint64_t));
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            tensors[i].dims[d] = rd_u64(f);
        }
        tensors[i].type = rd_u32(f);
        tensors[i].offset = rd_u64(f);

        /* Calculate tensor size */
        size_t n_elems = 1;
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            n_elems *= tensors[i].dims[d];
        }

        if (tensors[i].type == GGML_TYPE_Q8_0) {
            tensors[i].size = (n_elems / QK8_0) * sizeof(block_q8_0);
        } else if (tensors[i].type == GGML_TYPE_F32) {
            tensors[i].size = n_elems * sizeof(float);
        } else {
            fprintf(stderr, "Unsupported tensor type: %u\n", tensors[i].type);
            fclose(f);
            return -1;
        }

        if (tensors[i].offset + tensors[i].size > max_offset) {
            max_offset = tensors[i].offset + tensors[i].size;
        }
    }

    /* Align to get tensor data start */
    long metadata_end = ftell(f);
    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t padding = (alignment - (metadata_end % alignment)) % alignment;
    fseek(f, padding, SEEK_CUR);
    long tensor_data_start = ftell(f);

    printf("Tensor data starts at offset: %ld\n", tensor_data_start);

    /* Derive vocab size from embedding tensor */
    for (uint64_t i = 0; i < tensor_count; i++) {
        if (strstr(tensors[i].name, "token_embd.weight") ||
            strstr(tensors[i].name, "output.weight")) {
            if (tensors[i].n_dims >= 2) {
                c->vocab_size = tensors[i].dims[1];
                break;
            }
        }
    }

    c->head_dim = c->hidden_size / c->num_heads;

    printf("Config: hidden=%d, intermediate=%d, layers=%d, heads=%d, kv_heads=%d, vocab=%d\n",
           c->hidden_size, c->intermediate_size, c->num_layers,
           c->num_heads, c->num_kv_heads, c->vocab_size);
    printf("        max_seq=%d, head_dim=%d, rope_theta=%.1f, rms_eps=%e\n",
           c->max_seq_len, c->head_dim, c->rope_theta, c->rms_norm_eps);

    /* Allocate model weights */
    ModelWeights_GGUF *w = &m->weights;
    w->layers = xmalloc(c->num_layers * sizeof(LayerWeights_GGUF));

    /* Load tensors */
    for (uint64_t i = 0; i < tensor_count; i++) {
        fseek(f, tensor_data_start + tensors[i].offset, SEEK_SET);

        Q8_0_Tensor *target_q8 = NULL;
        FP32Tensor_GGUF *target_fp32 = NULL;
        int layer_idx = -1;

        /* Parse tensor name and find target */
        if (strstr(tensors[i].name, "token_embd.weight")) {
            target_q8 = &w->embed_tokens;
        } else if (strstr(tensors[i].name, "output_norm.weight")) {
            target_fp32 = &w->final_norm;
        } else {
            /* Layer-specific tensors */
            for (int l = 0; l < c->num_layers; l++) {
                char prefix[64];
                snprintf(prefix, sizeof(prefix), "blk.%d.", l);
                if (strstr(tensors[i].name, prefix)) {
                    layer_idx = l;
                    break;
                }
            }

            if (layer_idx >= 0) {
                LayerWeights_GGUF *lw = &w->layers[layer_idx];
                if (strstr(tensors[i].name, "attn_norm.weight")) {
                    target_fp32 = &lw->input_layernorm;
                } else if (strstr(tensors[i].name, "attn_q.weight")) {
                    target_q8 = &lw->q_proj;
                } else if (strstr(tensors[i].name, "attn_k.weight")) {
                    target_q8 = &lw->k_proj;
                } else if (strstr(tensors[i].name, "attn_v.weight")) {
                    target_q8 = &lw->v_proj;
                } else if (strstr(tensors[i].name, "attn_output.weight")) {
                    target_q8 = &lw->o_proj;
                } else if (strstr(tensors[i].name, "ffn_norm.weight")) {
                    target_fp32 = &lw->post_attention_layernorm;
                } else if (strstr(tensors[i].name, "ffn_gate.weight")) {
                    target_q8 = &lw->gate_proj;
                } else if (strstr(tensors[i].name, "ffn_up.weight")) {
                    target_q8 = &lw->up_proj;
                } else if (strstr(tensors[i].name, "ffn_down.weight")) {
                    target_q8 = &lw->down_proj;
                }
            }
        }

        /* Load tensor data */
        if (target_q8 && tensors[i].type == GGML_TYPE_Q8_0) {
            target_q8->rows = (tensors[i].n_dims >= 2) ? tensors[i].dims[1] : 1;
            target_q8->cols = tensors[i].dims[0];
            target_q8->n_blocks = (target_q8->rows * target_q8->cols) / QK8_0;
            target_q8->blocks = xmalloc(tensors[i].size);
            rd_bytes(f, target_q8->blocks, tensors[i].size);
        } else if (target_fp32 && tensors[i].type == GGML_TYPE_F32) {
            target_fp32->size = tensors[i].dims[0];
            target_fp32->data = xmalloc(tensors[i].size);
            rd_bytes(f, target_fp32->data, tensors[i].size);
        }
    }

    /* Cleanup tensor info */
    for (uint64_t i = 0; i < tensor_count; i++) {
        free(tensors[i].name);
        free(tensors[i].dims);
    }
    free(tensors);
    fclose(f);

    /* Initialize tokenizer (minimal GPT2-style) */
    /* For now, use a simple placeholder - full GGUF tokenizer parsing can be added */
    Tokenizer_GGUF *tok = &m->tokenizer;
    tok->vocab_size = c->vocab_size;
    tok->num_merges = 0;
    tok->vocab = xmalloc(c->vocab_size * sizeof(char*));
    for (int i = 0; i < c->vocab_size; i++) {
        tok->vocab[i] = xmalloc(16);
        snprintf(tok->vocab[i], 16, "[%d]", i);
    }
    tok->merges = NULL;

    /* Precompute RoPE */
    precompute_rope(m);

    /* Allocate KV caches */
    int hd = c->head_dim;
    m->kv_caches = xmalloc(c->num_layers * sizeof(KVCache_GGUF));
    for (int l = 0; l < c->num_layers; l++) {
        m->kv_caches[l].k_cache = xmalloc(c->num_kv_heads * c->max_seq_len * hd * sizeof(float));
        m->kv_caches[l].v_cache = xmalloc(c->num_kv_heads * c->max_seq_len * hd * sizeof(float));
        m->kv_caches[l].cache_len = 0;
    }

    /* Allocate scratch buffers */
    m->x = xmalloc(c->hidden_size * sizeof(float));
    m->xb = xmalloc(c->hidden_size * sizeof(float));
    m->xb2 = xmalloc(c->hidden_size * sizeof(float));
    m->q = xmalloc(c->num_heads * hd * sizeof(float));
    m->k = xmalloc(c->num_kv_heads * hd * sizeof(float));
    m->v = xmalloc(c->num_kv_heads * hd * sizeof(float));
    m->att = xmalloc(c->num_heads * c->max_seq_len * sizeof(float));
    m->logits = xmalloc(c->vocab_size * sizeof(float));
    m->hb = xmalloc(c->intermediate_size * sizeof(float));
    m->hb2 = xmalloc(c->intermediate_size * sizeof(float));

    printf("Model loaded successfully\n");
    return 0;
}

void smolc_gguf_free(SmolLM2_GGUF *m) {
    Config_GGUF *c = &m->config;

    /* Free tokenizer */
    for (int i = 0; i < m->tokenizer.vocab_size; i++)
        free(m->tokenizer.vocab[i]);
    free(m->tokenizer.vocab);
    if (m->tokenizer.merges) {
        for (int i = 0; i < m->tokenizer.num_merges; i++)
            free(m->tokenizer.merges[i]);
        free(m->tokenizer.merges);
    }

    /* Free weights */
    free(m->weights.embed_tokens.blocks);
    for (int l = 0; l < c->num_layers; l++) {
        LayerWeights_GGUF *lw = &m->weights.layers[l];
        free(lw->input_layernorm.data);
        free(lw->post_attention_layernorm.data);
        free(lw->q_proj.blocks);
        free(lw->k_proj.blocks);
        free(lw->v_proj.blocks);
        free(lw->o_proj.blocks);
        free(lw->gate_proj.blocks);
        free(lw->up_proj.blocks);
        free(lw->down_proj.blocks);
    }
    free(m->weights.layers);
    free(m->weights.final_norm.data);

    /* Free caches and buffers */
    free(m->rope_cos);
    free(m->rope_sin);
    for (int l = 0; l < c->num_layers; l++) {
        free(m->kv_caches[l].k_cache);
        free(m->kv_caches[l].v_cache);
    }
    free(m->kv_caches);
    free(m->x); free(m->xb); free(m->xb2);
    free(m->q); free(m->k); free(m->v);
    free(m->att); free(m->logits);
    free(m->hb); free(m->hb2);
}

void smolc_gguf_reset_cache(SmolLM2_GGUF *m) {
    for (int l = 0; l < m->config.num_layers; l++)
        m->kv_caches[l].cache_len = 0;
}

float* smolc_gguf_forward(SmolLM2_GGUF *m, int tok, int pos) {
    Config_GGUF *c = &m->config;
    ModelWeights_GGUF *w = &m->weights;
    int hs = c->hidden_size, hd = c->head_dim;
    int nh = c->num_heads, nkv = c->num_kv_heads, ng = nh / nkv;

    /* Embedding lookup (dequantize Q8_0) */
    int blocks_per_row = hs / QK8_0;
    block_q8_0 *emb_blocks = w->embed_tokens.blocks + tok * blocks_per_row;
    for (int b = 0; b < blocks_per_row; b++) {
        dequantize_q8_0(&emb_blocks[b], m->x + b * QK8_0);
    }

    float *rc = m->rope_cos + pos * hd;
    float *rs = m->rope_sin + pos * hd;

    /* Transformer layers */
    for (int l = 0; l < c->num_layers; l++) {
        LayerWeights_GGUF *lw = &w->layers[l];
        KVCache_GGUF *kv = &m->kv_caches[l];

        /* Attention */
        rmsnorm(m->xb, m->x, lw->input_layernorm.data, hs, c->rms_norm_eps);
        matmul_q8_0(m->q, &lw->q_proj, m->xb);
        matmul_q8_0(m->k, &lw->k_proj, m->xb);
        matmul_q8_0(m->v, &lw->v_proj, m->xb);

        /* Apply RoPE */
        for (int h = 0; h < nh; h++)
            apply_rope(m->q + h * hd, hd, rc, rs);
        for (int h = 0; h < nkv; h++)
            apply_rope(m->k + h * hd, hd, rc, rs);

        /* Update KV cache */
        int off = kv->cache_len * hd;
        for (int h = 0; h < nkv; h++) {
            memcpy(kv->k_cache + h * c->max_seq_len * hd + off,
                   m->k + h * hd, hd * sizeof(float));
            memcpy(kv->v_cache + h * c->max_seq_len * hd + off,
                   m->v + h * hd, hd * sizeof(float));
        }
        int slen = kv->cache_len + 1;

        /* Multi-head attention */
        memset(m->xb2, 0, hs * sizeof(float));
        for (int h = 0; h < nh; h++) {
            int kvh = h / ng;
            float *qh = m->q + h * hd;
            float *kc = kv->k_cache + kvh * c->max_seq_len * hd;
            float *vc = kv->v_cache + kvh * c->max_seq_len * hd;
            float *att = m->att + h * c->max_seq_len;
            float sc = 1.0f / sqrtf((float)hd);

            /* Compute attention scores */
            for (int t = 0; t < slen; t++) {
                float score = 0;
                float *kt = kc + t * hd;
                for (int d = 0; d < hd; d++)
                    score += qh[d] * kt[d];
                att[t] = score * sc;
            }
            softmax(att, slen);

            /* Weighted sum of values */
            float *oh = m->xb2 + h * hd;
            for (int t = 0; t < slen; t++) {
                float a = att[t];
                float *vt = vc + t * hd;
                for (int d = 0; d < hd; d++)
                    oh[d] += a * vt[d];
            }
        }

        /* Attention output projection */
        matmul_q8_0(m->xb, &lw->o_proj, m->xb2);
        for (int i = 0; i < hs; i++)
            m->x[i] += m->xb[i];

        /* FFN */
        rmsnorm(m->xb, m->x, lw->post_attention_layernorm.data, hs, c->rms_norm_eps);
        matmul_q8_0(m->hb, &lw->gate_proj, m->xb);
        matmul_q8_0(m->hb2, &lw->up_proj, m->xb);
        for (int i = 0; i < c->intermediate_size; i++)
            m->hb[i] = silu(m->hb[i]) * m->hb2[i];
        matmul_q8_0(m->xb, &lw->down_proj, m->hb);
        for (int i = 0; i < hs; i++)
            m->x[i] += m->xb[i];

        kv->cache_len = slen;
    }

    /* Final norm */
    rmsnorm(m->x, m->x, w->final_norm.data, hs, c->rms_norm_eps);

    /* LM head (using tied embeddings) */
    for (int i = 0; i < c->vocab_size; i++) {
        float sum = 0;
        block_q8_0 *row_blocks = w->embed_tokens.blocks + i * blocks_per_row;
        for (int b = 0; b < blocks_per_row; b++) {
            block_q8_0 *block = &row_blocks[b];
            float scale = fp16_to_fp32(block->d);
            for (int j = 0; j < QK8_0; j++) {
                sum += m->x[b * QK8_0 + j] * block->qs[j] * scale;
            }
        }
        m->logits[i] = sum;
    }

    return m->logits;
}

/* Simplified tokenizer functions */
int smolc_gguf_tokenize(SmolLM2_GGUF *m, const char *text, int *toks, int max) {
    /* Placeholder: return single token per character */
    int n = 0;
    while (*text && n < max) {
        toks[n++] = (unsigned char)*text;
        text++;
    }
    return n;
}

static char dec_buf[256];
const char* smolc_gguf_decode(SmolLM2_GGUF *m, int tok) {
    if (tok < 0 || tok >= m->tokenizer.vocab_size)
        return "";
    return m->tokenizer.vocab[tok];
}

int smolc_gguf_sample(float *logits, int vocab, float temp) {
    if (temp <= 0) {
        int mi = 0;
        for (int i = 1; i < vocab; i++)
            if (logits[i] > logits[mi]) mi = i;
        return mi;
    }
    for (int i = 0; i < vocab; i++)
        logits[i] /= temp;
    softmax(logits, vocab);
    float r = (float)rand() / RAND_MAX, cs = 0;
    for (int i = 0; i < vocab; i++) {
        cs += logits[i];
        if (cs >= r) return i;
    }
    return vocab - 1;
}

void smolc_gguf_generate(SmolLM2_GGUF *m, const char *prompt, int max_tok, float temp) {
    int toks[512];
    int n = smolc_gguf_tokenize(m, prompt, toks, 512);
    if (!n) {
        printf("Tokenize failed\n");
        return;
    }

    printf("Generating... (tokens: %d)\n", n);
    smolc_gguf_reset_cache(m);

    float *logits = NULL;
    for (int i = 0; i < n; i++)
        logits = smolc_gguf_forward(m, toks[i], i);

    for (int i = 0; i < max_tok; i++) {
        int next = smolc_gguf_sample(logits, m->config.vocab_size, temp);
        if (next == 2) break;  /* EOS */
        printf("%s", smolc_gguf_decode(m, next));
        fflush(stdout);
        logits = smolc_gguf_forward(m, next, n + i);
    }
    printf("\n");
}

/* Main */
int main(int argc, char **argv) {
    const char *model = "../models/smollm2-135m-q8_0.gguf";
    const char *prompt = "Hello";
    int max_tok = 50;
    float temp = 0.8f;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) max_tok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "-h")) {
            printf("Usage: %s [-m model.gguf] [-p prompt] [-n tokens] [-t temp]\n", argv[0]);
            return 0;
        }
    }

    srand(time(NULL));
    SmolLM2_GGUF m;

    printf("Loading GGUF model: %s\n", model);
    if (smolc_gguf_load(&m, model)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    smolc_gguf_generate(&m, prompt, max_tok, temp);
    smolc_gguf_free(&m);

    return 0;
}
