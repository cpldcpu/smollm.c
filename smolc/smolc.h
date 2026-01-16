/**
 * SmolLM2 C Inference Engine
 * ANSI C implementation of SmolLM2-135M with Q8/Q4 quantization
 */

#ifndef SMOLC_H
#define SMOLC_H

#include <stdint.h>
#include <stddef.h>

/* Quantization types */
#define QUANT_Q8 0
#define QUANT_Q4 1

/* Model configuration */
typedef struct {
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int vocab_size;
    int max_seq_len;
    float rope_theta;
    float rms_norm_eps;
    int head_dim;
    int quant_type;   /* QUANT_Q8 or QUANT_Q4 */
    int group_size;   /* For Q4 quantization */
} Config;

/* Q8 tensor: per-tensor scale + int8 data */
typedef struct {
    float scale;
    int8_t *data;
    int rows;
    int cols;
} Q8Tensor;

/* Q4 tensor: group-wise scales + packed int4 data */
typedef struct {
    int num_groups;
    int group_size;
    float *scales;      /* [num_groups] */
    uint8_t *data;      /* Packed: 2 int4 values per byte */
    int rows;
    int cols;
} Q4Tensor;

/* Generic quantized tensor (can be Q8 or Q4) */
typedef struct {
    int quant_type;
    union {
        Q8Tensor q8;
        Q4Tensor q4;
    };
    int rows;
    int cols;
} QuantTensor;

/* FP32 tensor */
typedef struct {
    float *data;
    int size;
} FP32Tensor;

/* Transformer layer weights */
typedef struct {
    FP32Tensor input_layernorm;
    QuantTensor q_proj;
    QuantTensor k_proj;
    QuantTensor v_proj;
    QuantTensor o_proj;
    FP32Tensor post_attention_layernorm;
    QuantTensor gate_proj;
    QuantTensor up_proj;
    QuantTensor down_proj;
} LayerWeights;

/* Full model weights */
typedef struct {
    QuantTensor embed_tokens;
    LayerWeights *layers;
    FP32Tensor final_norm;
} ModelWeights;

/* KV cache for one layer */
typedef struct {
    float *k_cache;  /* [num_kv_heads, cache_len, head_dim] */
    float *v_cache;  /* [num_kv_heads, cache_len, head_dim] */
    int cache_len;
} KVCache;

/* Tokenizer */
typedef struct {
    char **vocab;       /* Token strings indexed by ID */
    int vocab_size;
    char **merges;      /* Merge rules */
    int num_merges;
} Tokenizer;

/* Runtime state */
typedef struct {
    Config config;
    ModelWeights weights;
    Tokenizer tokenizer;
    KVCache *kv_caches;  /* One per layer */

    /* Precomputed RoPE */
    float *rope_cos;     /* [max_seq_len, head_dim] */
    float *rope_sin;     /* [max_seq_len, head_dim] */

    /* Scratch buffers */
    float *x;            /* [hidden_size] - current hidden state */
    float *xb;           /* [hidden_size] - buffer for residual */
    float *xb2;          /* [hidden_size] - another buffer */
    float *q;            /* [num_heads * head_dim] */
    float *k;            /* [num_kv_heads * head_dim] */
    float *v;            /* [num_kv_heads * head_dim] */
    float *att;          /* [num_heads, max_seq_len] - attention scores */
    float *logits;       /* [vocab_size] */
    float *hb;           /* [intermediate_size] - MLP buffer */
    float *hb2;          /* [intermediate_size] - MLP buffer */
} SmolLM2;

/* Public API */

/* Load model from binary file */
int smolc_load(SmolLM2 *model, const char *filepath);

/* Free model resources */
void smolc_free(SmolLM2 *model);

/* Reset KV cache (for new sequence) */
void smolc_reset_cache(SmolLM2 *model);

/* Forward pass for single token, returns logits */
float* smolc_forward(SmolLM2 *model, int token, int pos);

/* Tokenize text to token IDs (returns count, fills tokens array) */
int smolc_tokenize(SmolLM2 *model, const char *text, int *tokens, int max_tokens);

/* Decode token ID to string (returns pointer to internal buffer) */
const char* smolc_decode(SmolLM2 *model, int token);

/* Sample next token from logits */
int smolc_sample(float *logits, int vocab_size, float temperature);

/* Generate text */
void smolc_generate(SmolLM2 *model, const char *prompt, int max_tokens, float temperature);

#endif /* SMOLC_H */
