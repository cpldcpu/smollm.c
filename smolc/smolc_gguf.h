/**
 * SmolLM2 GGUF Inference Engine
 * GGUF format loader with Q8_0 block quantization support
 */

#ifndef SMOLC_GGUF_H
#define SMOLC_GGUF_H

#include <stdint.h>
#include <stddef.h>

/* GGUF constants */
#define GGUF_MAGIC 0x46554747  /* "GGUF" in little-endian */
#define GGUF_VERSION 3
#define GGUF_DEFAULT_ALIGNMENT 32

/* GGML quantization types */
#define GGML_TYPE_F32  0
#define GGML_TYPE_F16  1
#define GGML_TYPE_Q4_0 2
#define GGML_TYPE_Q4_1 3
#define GGML_TYPE_Q5_0 6
#define GGML_TYPE_Q5_1 7
#define GGML_TYPE_Q8_0 8
#define GGML_TYPE_Q8_1 9
#define GGML_TYPE_I8   24
#define GGML_TYPE_I16  25
#define GGML_TYPE_I32  26

/* GGUF metadata value types */
#define GGUF_TYPE_UINT8   0
#define GGUF_TYPE_INT8    1
#define GGUF_TYPE_UINT16  2
#define GGUF_TYPE_INT16   3
#define GGUF_TYPE_UINT32  4
#define GGUF_TYPE_INT32   5
#define GGUF_TYPE_FLOAT32 6
#define GGUF_TYPE_BOOL    7
#define GGUF_TYPE_STRING  8
#define GGUF_TYPE_ARRAY   9
#define GGUF_TYPE_UINT64  10
#define GGUF_TYPE_INT64   11
#define GGUF_TYPE_FLOAT64 12

/* Q8_0 block: 32 int8 values + 1 fp16 scale */
#define QK8_0 32
typedef struct {
    uint16_t d;           /* Scale factor (FP16 format) */
    int8_t qs[QK8_0];     /* Quantized values */
} block_q8_0;

/* Q8_0 tensor: array of blocks */
typedef struct {
    int rows;
    int cols;
    int n_blocks;         /* Total blocks = (rows * cols) / QK8_0 */
    block_q8_0 *blocks;
} Q8_0_Tensor;

/* FP32 tensor */
typedef struct {
    float *data;
    int size;
} FP32Tensor_GGUF;

/* Transformer layer weights (GGUF version) */
typedef struct {
    FP32Tensor_GGUF input_layernorm;
    Q8_0_Tensor q_proj;
    Q8_0_Tensor k_proj;
    Q8_0_Tensor v_proj;
    Q8_0_Tensor o_proj;
    FP32Tensor_GGUF post_attention_layernorm;
    Q8_0_Tensor gate_proj;
    Q8_0_Tensor up_proj;
    Q8_0_Tensor down_proj;
} LayerWeights_GGUF;

/* Full model weights (GGUF version) */
typedef struct {
    Q8_0_Tensor embed_tokens;
    LayerWeights_GGUF *layers;
    FP32Tensor_GGUF final_norm;
} ModelWeights_GGUF;

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
} Config_GGUF;

/* KV cache for one layer */
typedef struct {
    float *k_cache;  /* [num_kv_heads, cache_len, head_dim] */
    float *v_cache;  /* [num_kv_heads, cache_len, head_dim] */
    int cache_len;
} KVCache_GGUF;

/* Tokenizer */
typedef struct {
    char **vocab;       /* Token strings indexed by ID */
    int vocab_size;
    char **merges;      /* Merge rules */
    int num_merges;
} Tokenizer_GGUF;

/* Runtime state */
typedef struct {
    Config_GGUF config;
    ModelWeights_GGUF weights;
    Tokenizer_GGUF tokenizer;
    KVCache_GGUF *kv_caches;  /* One per layer */

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
} SmolLM2_GGUF;

/* Public API */

/* Load model from GGUF file */
int smolc_gguf_load(SmolLM2_GGUF *model, const char *filepath);

/* Free model resources */
void smolc_gguf_free(SmolLM2_GGUF *model);

/* Reset KV cache (for new sequence) */
void smolc_gguf_reset_cache(SmolLM2_GGUF *model);

/* Forward pass for single token, returns logits */
float* smolc_gguf_forward(SmolLM2_GGUF *model, int token, int pos);

/* Tokenize text to token IDs (returns count, fills tokens array) */
int smolc_gguf_tokenize(SmolLM2_GGUF *model, const char *text, int *tokens, int max_tokens);

/* Decode token ID to string (returns pointer to internal buffer) */
const char* smolc_gguf_decode(SmolLM2_GGUF *model, int token);

/* Sample next token from logits */
int smolc_gguf_sample(float *logits, int vocab_size, float temperature);

/* Generate text */
void smolc_gguf_generate(SmolLM2_GGUF *model, const char *prompt, int max_tokens, float temperature);

/* Utility: Convert FP16 to FP32 */
float fp16_to_fp32(uint16_t h);

/* Utility: Dequantize Q8_0 block to FP32 */
void dequantize_q8_0(const block_q8_0 *block, float *out);

#endif /* SMOLC_GGUF_H */
