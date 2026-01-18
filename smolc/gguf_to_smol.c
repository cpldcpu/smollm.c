/* GGUF to SMOL Converter - Convert GGUF Q8_0 models to SMOL Q8 format */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* GGUF constants */
#define GGUF_MAGIC 0x46554747  /* "GGUF" */
#define GGUF_VERSION 3
#define GGUF_DEFAULT_ALIGNMENT 32
#define QK8_0 32

/* GGML types */
#define GGML_TYPE_F32  0
#define GGML_TYPE_Q8_0 8

/* GGUF metadata types */
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

/* Q8_0 block structure */
typedef struct {
    uint16_t d;           /* FP16 scale */
    int8_t qs[QK8_0];     /* Quantized values */
} block_q8_0;

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
} Config;

static void die(const char *msg) {
    fprintf(stderr, "Error: %s\n", msg);
    exit(1);
}

static uint32_t rd_u32(FILE *f) {
    uint32_t v;
    if (fread(&v, 4, 1, f) != 1) die("Read error");
    return v;
}

static uint64_t rd_u64(FILE *f) {
    uint64_t v;
    if (fread(&v, 8, 1, f) != 1) die("Read error");
    return v;
}

static float rd_f32(FILE *f) {
    float v;
    if (fread(&v, 4, 1, f) != 1) die("Read error");
    return v;
}

static void rd_bytes(FILE *f, void *buf, size_t n) {
    if (fread(buf, 1, n, f) != n) die("Read error");
}

static char *rd_string(FILE *f) {
    uint64_t len = rd_u64(f);
    if (len > 1024 * 1024) die("String too long");
    char *s = malloc(len + 1);
    if (!s) die("Out of memory");
    rd_bytes(f, s, len);
    s[len] = 0;
    return s;
}

/* FP16 to FP32 conversion */
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = (h & 0x03FF);

    if (exponent == 0) {
        if (mantissa == 0) return *(float*)&sign;
        exponent = 1;
        while ((mantissa & 0x0400) == 0) {
            mantissa <<= 1;
            exponent--;
        }
        mantissa &= 0x03FF;
    } else if (exponent == 31) {
        uint32_t result = sign | 0x7F800000 | (mantissa << 13);
        return *(float*)&result;
    }

    exponent = exponent - 15 + 127;
    uint32_t result = sign | (exponent << 23) | (mantissa << 13);
    return *(float*)&result;
}

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

/* Convert Q8_0 blocks to per-tensor Q8 with single scale */
static void convert_q8_0_to_q8(FILE *in, FILE *out, int rows, int cols) {
    int blocks_per_row = cols / QK8_0;
    int total_blocks = rows * blocks_per_row;

    /* Read all Q8_0 blocks */
    block_q8_0 *blocks = malloc(total_blocks * sizeof(block_q8_0));
    if (!blocks) die("Out of memory");
    rd_bytes(in, blocks, total_blocks * sizeof(block_q8_0));

    /* Compute average scale across all blocks */
    double scale_sum = 0;
    int scale_count = 0;
    for (int i = 0; i < total_blocks; i++) {
        float s = fp16_to_fp32(blocks[i].d);
        if (s > 0) {
            scale_sum += s;
            scale_count++;
        }
    }
    float avg_scale = scale_count > 0 ? (float)(scale_sum / scale_count) : 1.0f;

    /* Write SMOL Q8 format: single float32 scale + all int8 values */
    fwrite(&avg_scale, sizeof(float), 1, out);

    /* Rescale and write int8 values */
    for (int i = 0; i < total_blocks; i++) {
        float block_scale = fp16_to_fp32(blocks[i].d);
        float scale_ratio = block_scale / avg_scale;

        for (int j = 0; j < QK8_0; j++) {
            /* Rescale each value to use the unified scale */
            int8_t rescaled = (int8_t)(blocks[i].qs[j] * scale_ratio);
            fwrite(&rescaled, 1, 1, out);
        }
    }

    free(blocks);
}

/* Copy FP32 tensor as-is */
static void copy_fp32_tensor(FILE *in, FILE *out, int size) {
    float *data = malloc(size * sizeof(float));
    if (!data) die("Out of memory");
    rd_bytes(in, data, size * sizeof(float));
    fwrite(data, sizeof(float), size, out);
    free(data);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: %s <input.gguf> <output.bin>\n", argv[0]);
        printf("\nConverts GGUF Q8_0 models to SMOL Q8 format for smolc.\n");
        return 1;
    }

    FILE *in = fopen(argv[1], "rb");
    if (!in) die("Can't open input file");

    /* Parse GGUF header */
    uint32_t magic = rd_u32(in);
    if (magic != GGUF_MAGIC) die("Invalid GGUF magic");

    uint32_t version = rd_u32(in);
    if (version != GGUF_VERSION) die("Unsupported GGUF version");

    uint64_t tensor_count = rd_u64(in);
    uint64_t metadata_kv_count = rd_u64(in);

    printf("GGUF: version=%u, tensors=%llu, metadata=%llu\n",
           version, (unsigned long long)tensor_count, (unsigned long long)metadata_kv_count);

    /* Parse metadata */
    Config cfg = {0};
    cfg.rope_theta = 10000.0f;
    cfg.rms_norm_eps = 1e-5f;

    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        char *key = rd_string(in);
        uint32_t value_type = rd_u32(in);

        if (!strcmp(key, "llama.embedding_length") || !strcmp(key, "llama.embed_length")) {
            cfg.hidden_size = read_metadata_u32(in, value_type);
        } else if (!strcmp(key, "llama.feed_forward_length")) {
            cfg.intermediate_size = read_metadata_u32(in, value_type);
        } else if (!strcmp(key, "llama.block_count")) {
            cfg.num_layers = read_metadata_u32(in, value_type);
        } else if (!strcmp(key, "llama.attention.head_count")) {
            cfg.num_heads = read_metadata_u32(in, value_type);
        } else if (!strcmp(key, "llama.attention.head_count_kv")) {
            cfg.num_kv_heads = read_metadata_u32(in, value_type);
        } else if (!strcmp(key, "llama.context_length")) {
            cfg.max_seq_len = read_metadata_u32(in, value_type);
        } else if (!strcmp(key, "llama.rope.freq_base")) {
            cfg.rope_theta = read_metadata_f32(in, value_type);
        } else if (!strcmp(key, "llama.attention.layer_norm_rms_epsilon")) {
            cfg.rms_norm_eps = read_metadata_f32(in, value_type);
        } else {
            skip_metadata_value(in, value_type);
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

    TensorInfo *tensors = malloc(tensor_count * sizeof(TensorInfo));
    if (!tensors) die("Out of memory");

    for (uint64_t i = 0; i < tensor_count; i++) {
        tensors[i].name = rd_string(in);
        tensors[i].n_dims = rd_u32(in);
        tensors[i].dims = malloc(tensors[i].n_dims * sizeof(uint64_t));
        if (!tensors[i].dims) die("Out of memory");

        for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
            tensors[i].dims[d] = rd_u64(in);
        }
        tensors[i].type = rd_u32(in);
        tensors[i].offset = rd_u64(in);

        /* Calculate size */
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
            fclose(in);
            return 1;
        }
    }

    /* Get vocab size from embedding tensor */
    for (uint64_t i = 0; i < tensor_count; i++) {
        if (strstr(tensors[i].name, "token_embd.weight") && tensors[i].n_dims >= 2) {
            cfg.vocab_size = tensors[i].dims[1];
            break;
        }
    }

    printf("Config: hidden=%d, intermediate=%d, layers=%d, heads=%d, kv_heads=%d, vocab=%d\n",
           cfg.hidden_size, cfg.intermediate_size, cfg.num_layers,
           cfg.num_heads, cfg.num_kv_heads, cfg.vocab_size);
    printf("        max_seq=%d, rope_theta=%.1f, rms_eps=%e\n",
           cfg.max_seq_len, cfg.rope_theta, cfg.rms_norm_eps);

    /* Align to tensor data start */
    long metadata_end = ftell(in);
    size_t padding = (GGUF_DEFAULT_ALIGNMENT - (metadata_end % GGUF_DEFAULT_ALIGNMENT)) % GGUF_DEFAULT_ALIGNMENT;
    fseek(in, padding, SEEK_CUR);
    long tensor_data_start = ftell(in);

    /* Open output file and write SMOL header */
    FILE *out = fopen(argv[2], "wb");
    if (!out) die("Can't create output file");

    /* SMOL format header */
    fwrite("SMOL", 1, 4, out);
    uint32_t version_out = 2;  /* Version 2 with quant info */
    fwrite(&version_out, 4, 1, out);
    uint32_t quant_type = 0;  /* Q8 */
    fwrite(&quant_type, 4, 1, out);
    uint32_t reserved = 0;
    fwrite(&reserved, 4, 1, out);

    /* Write config */
    fwrite(&cfg.hidden_size, 4, 1, out);
    fwrite(&cfg.intermediate_size, 4, 1, out);
    fwrite(&cfg.num_layers, 4, 1, out);
    fwrite(&cfg.num_heads, 4, 1, out);
    fwrite(&cfg.num_kv_heads, 4, 1, out);
    fwrite(&cfg.vocab_size, 4, 1, out);
    fwrite(&cfg.max_seq_len, 4, 1, out);
    fwrite(&cfg.rope_theta, 4, 1, out);
    fwrite(&cfg.rms_norm_eps, 4, 1, out);

    /* Placeholder tokenizer (will need separate handling) */
    printf("\nWarning: Tokenizer not converted. You'll need to extract it separately.\n");
    uint32_t vocab_count = 0, merge_count = 0;
    fwrite(&vocab_count, 4, 1, out);
    fwrite(&merge_count, 4, 1, out);

    /* Convert and write tensors in SMOL order */
    printf("\nConverting tensors...\n");

    /* Helper to find and convert tensor */
    #define CONVERT_TENSOR(pattern, rows, cols) do { \
        int found = 0; \
        for (uint64_t i = 0; i < tensor_count && !found; i++) { \
            if (strstr(tensors[i].name, pattern)) { \
                fseek(in, tensor_data_start + tensors[i].offset, SEEK_SET); \
                if (tensors[i].type == GGML_TYPE_Q8_0) { \
                    convert_q8_0_to_q8(in, out, rows, cols); \
                    printf("  ✓ %s (Q8_0 → Q8)\n", pattern); \
                } else if (tensors[i].type == GGML_TYPE_F32) { \
                    copy_fp32_tensor(in, out, tensors[i].dims[0]); \
                    printf("  ✓ %s (F32)\n", pattern); \
                } \
                found = 1; \
            } \
        } \
        if (!found) fprintf(stderr, "  ✗ Warning: %s not found\n", pattern); \
    } while(0)

    /* Embedding */
    CONVERT_TENSOR("token_embd.weight", cfg.vocab_size, cfg.hidden_size);

    /* Per-layer tensors */
    for (int l = 0; l < cfg.num_layers; l++) {
        printf("Layer %d:\n", l);
        char pattern[128];
        int hd = cfg.hidden_size / cfg.num_heads;

        snprintf(pattern, sizeof(pattern), "blk.%d.attn_norm.weight", l);
        CONVERT_TENSOR(pattern, 1, cfg.hidden_size);

        snprintf(pattern, sizeof(pattern), "blk.%d.attn_q.weight", l);
        CONVERT_TENSOR(pattern, cfg.num_heads * hd, cfg.hidden_size);

        snprintf(pattern, sizeof(pattern), "blk.%d.attn_k.weight", l);
        CONVERT_TENSOR(pattern, cfg.num_kv_heads * hd, cfg.hidden_size);

        snprintf(pattern, sizeof(pattern), "blk.%d.attn_v.weight", l);
        CONVERT_TENSOR(pattern, cfg.num_kv_heads * hd, cfg.hidden_size);

        snprintf(pattern, sizeof(pattern), "blk.%d.attn_output.weight", l);
        CONVERT_TENSOR(pattern, cfg.hidden_size, cfg.num_heads * hd);

        snprintf(pattern, sizeof(pattern), "blk.%d.ffn_norm.weight", l);
        CONVERT_TENSOR(pattern, 1, cfg.hidden_size);

        snprintf(pattern, sizeof(pattern), "blk.%d.ffn_gate.weight", l);
        CONVERT_TENSOR(pattern, cfg.intermediate_size, cfg.hidden_size);

        snprintf(pattern, sizeof(pattern), "blk.%d.ffn_up.weight", l);
        CONVERT_TENSOR(pattern, cfg.intermediate_size, cfg.hidden_size);

        snprintf(pattern, sizeof(pattern), "blk.%d.ffn_down.weight", l);
        CONVERT_TENSOR(pattern, cfg.hidden_size, cfg.intermediate_size);
    }

    /* Final norm */
    CONVERT_TENSOR("output_norm.weight", 1, cfg.hidden_size);

    /* Cleanup */
    for (uint64_t i = 0; i < tensor_count; i++) {
        free(tensors[i].name);
        free(tensors[i].dims);
    }
    free(tensors);

    fclose(in);
    fclose(out);

    printf("\n✓ Conversion complete: %s\n", argv[2]);
    printf("Note: Use smolc or smolc_full to run inference with this model.\n");

    return 0;
}
