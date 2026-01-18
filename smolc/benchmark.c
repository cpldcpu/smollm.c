/**
 * Simple benchmark tool for SmolLM2 C inference
 * Measures tokens/second for different prompt lengths
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "smolc.h"

#define NUM_RUNS 5

typedef struct {
    double total_time;
    int total_tokens;
    double tokens_per_sec;
} BenchResult;

static double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static BenchResult benchmark_generation(SmolLM2 *m, const char *prompt, int num_tokens) {
    BenchResult result = {0};
    int toks[512];

    printf("Benchmarking: prompt='%s', tokens=%d, runs=%d\n", prompt, num_tokens, NUM_RUNS);

    for (int run = 0; run < NUM_RUNS; run++) {
        // Tokenize
        int n = smolc_tokenize(m, prompt, toks, 512);
        if (!n) {
            fprintf(stderr, "Tokenization failed\n");
            return result;
        }

        smolc_reset_cache(m);

        double start = get_time();

        // Process prompt
        float *logits = NULL;
        for (int i = 0; i < n; i++) {
            logits = smolc_forward(m, toks[i], i);
        }

        // Generate tokens
        for (int i = 0; i < num_tokens; i++) {
            int next = smolc_sample(logits, m->config.vocab_size, 0.0f);
            if (next == 2) break;  // EOS
            logits = smolc_forward(m, next, n + i);
        }

        double elapsed = get_time() - start;

        result.total_time += elapsed;
        result.total_tokens += num_tokens;

        printf("  Run %d: %.3f sec, %.1f tokens/sec\n",
               run + 1, elapsed, num_tokens / elapsed);
    }

    result.tokens_per_sec = result.total_tokens / result.total_time;
    return result;
}

static void print_system_info() {
    printf("=== System Information ===\n");

#ifdef __AVX512F__
    printf("Built with: AVX-512 support\n");
#elif defined(__AVX2__)
    printf("Built with: AVX2 support\n");
#else
    printf("Built with: Scalar code only\n");
#endif

#ifdef _OPENMP
    printf("OpenMP: Enabled\n");
#else
    printf("OpenMP: Disabled\n");
#endif

    printf("Compiler: ");
#ifdef __clang__
    printf("Clang %d.%d.%d\n", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    printf("GCC %d.%d.%d\n", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    printf("Unknown\n");
#endif

    printf("\n");
}

int main(int argc, char **argv) {
    const char *model_path = "../models/smollm2-135m-q8.bin";

    if (argc > 1) {
        if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
            printf("Usage: %s [model_path]\n", argv[0]);
            printf("Default model: %s\n", model_path);
            return 0;
        }
        model_path = argv[1];
    }

    print_system_info();

    printf("=== Loading Model ===\n");
    SmolLM2 model;
    if (smolc_load(&model, model_path)) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    printf("\n");

    printf("=== Benchmark Results ===\n\n");

    // Benchmark different scenarios
    struct {
        const char *name;
        const char *prompt;
        int num_tokens;
    } benchmarks[] = {
        {"Short prompt, few tokens", "Hello", 10},
        {"Short prompt, many tokens", "The capital of France is", 30},
        {"Medium prompt", "Write a function to compute fibonacci numbers", 20},
        {"Longer prompt", "Explain the difference between machine learning and deep learning in simple terms", 50},
    };

    for (size_t i = 0; i < sizeof(benchmarks) / sizeof(benchmarks[0]); i++) {
        printf("--- %s ---\n", benchmarks[i].name);
        BenchResult result = benchmark_generation(&model, benchmarks[i].prompt, benchmarks[i].num_tokens);
        printf("Average: %.1f tokens/sec (%.3f sec total)\n\n",
               result.tokens_per_sec, result.total_time / NUM_RUNS);
    }

    // Compute time breakdown (single forward pass)
    printf("=== Single Token Forward Pass ===\n");
    smolc_reset_cache(&model);
    int tok = 100;  // Arbitrary token

    double start = get_time();
    for (int i = 0; i < 100; i++) {
        smolc_forward(&model, tok, 0);
        smolc_reset_cache(&model);
    }
    double elapsed = get_time() - start;

    printf("Average time per forward pass: %.3f ms\n", elapsed * 10.0);
    printf("Estimated throughput: %.1f tokens/sec\n", 100.0 / elapsed);

    smolc_free(&model);
    return 0;
}
