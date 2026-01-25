# Quantization Verification Report (Q4 vs Q8 vs FP32)

This report summarizes how Q4 and Q8 quantized models compare to the FP32 reference, using the local verification scripts in this repo.

## Method

- Script: `step4_pytorch_q8.py` (quantized model vs FP32)
- Prompts:
  - `Hello`
  - `The capital of France is`
  - `def fibonacci(n):`
- Metrics:
  - Max absolute logit difference
  - Mean absolute logit difference
  - Top-1 token match for the last position
- Generation: greedy, 20 tokens

Commands used:

```bash
TRANSFORMERS_OFFLINE=1 python step4_pytorch_q8.py --model ./models/smollm2-135m-q8.bin
TRANSFORMERS_OFFLINE=1 python step4_pytorch_q8.py --model ./models/smollm2-135m-q4.bin
```

## Results

Per-prompt comparison:

| Prompt | Q8 max / mean | Q8 top-1 | Q4 max / mean | Q4 top-1 |
| --- | --- | --- | --- | --- |
| Hello | 9.4211 / 4.9376 | no | 10.8055 / 5.1738 | no |
| The capital of France is | 11.2621 / 1.5426 | yes | 12.5998 / 2.2032 | no |
| def fibonacci(n): | 11.8937 / 1.7186 | yes | 14.6160 / 2.3894 | yes |

Aggregate:

- Avg max diff: Q8 10.86, Q4 12.67
- Avg mean diff: Q8 2.73, Q4 3.26
- Top-1 match rate: Q8 2/3, Q4 1/3

## Generation (qualitative)

- Q8 (prompt: "The capital of France is")
  - Q8: "The capital of France is Paris, a city known for its historical landmarks, culture, and cultural institutions. Paris is a major"
  - FP32: "The capital of France is Paris. Paris is the largest city in France and the capital of the French department of the Espace"

- Q4 (same prompt)
  - Q4: "The capital of France is the City of Light.\n\nThe city is the heart of the French culture, with the grand"
  - FP32: "The capital of France is Paris. Paris is the largest city in France and the capital of the French department of the Espace"

## Summary

- Q8 is closer to FP32 than Q4 on this prompt set (lower diff metrics and higher top-1 match rate).
- Q4 remains coherent but shows larger deviations from FP32.
- Results are prompt-sensitive; expand the prompt set if you want a more robust comparison.

---

## SMOL-32 Emulator vs C Reference

### Method

- Script: `processor/run_model.c` (assembly forward pass vs pure-C reference)
- Prompt: "The capital of France is" (tokenized, single forward pass at position 0)
- Metrics:
  - Max absolute logit difference across all 49,152 vocab entries
  - Average absolute logit difference
  - Mismatch count (entries differing by >0.1)

Command:
```bash
cd processor && make run_model && ./run_model
```

### Results

| Metric | Value |
|--------|-------|
| Max logit difference | 4.53e-05 |
| Avg logit difference | 2.43e-05 |
| Mismatches (>0.1) | 0 / 49,152 |
| Instructions executed | 19,010,082 |
| Result | **PASS** |

### Text Generation (qualitative)

Emulator-based generation (`processor/generate.c`):
```
Prompt: "The capital of France is"
Output: The capital of France is Paris, a city known for its historical
        landmarks, culture, and cultural institutions. Paris is a major...
```

This matches the C reference (`smolc/smolc`) output exactly under greedy decoding (temperature=0).

### Architecture

The entire forward pass runs in SMOL-32 assembly — a single kernel call (at 0x9000) executes the full 30-layer transformer pipeline. The host only provides the token ID, position, and a model descriptor pointer; all intermediate computation stays in emulator memory.

**9 assembly kernels (2,516 bytes total):**

| Kernel | Bytes | Operation |
|--------|-------|-----------|
| matmul_q8 | 136 | Q8 matrix-vector multiply (Q8MACINC + LOOP) |
| rmsnorm | 212 | RMS normalization (VREDSQS + FRSQRT) |
| rope | 212 | Rotary position embeddings |
| attention | 532 | Multi-head GQA + softmax |
| silu_mul | 44 | SiLU-gate activation |
| residual | 40 | Element-wise addition |
| embed | 100 | Int8→float embedding dequant |
| memcpy | 32 | Float vector copy (KV cache) |
| forward | 1,208 | Full 30-layer forward pass orchestration |

The forward kernel reads model configuration from a descriptor structure (at 0xE0000) and calls the 8 compute kernels via JALR. No host interaction occurs between layers.

---

## Notes

- C vs PyTorch verification for both Q8 and Q4 passed with exact matches once newline parsing was fixed in `step6_verify.py`.
- SMOL-32 emulator uses double-precision accumulation in the Q8 MAC unit, matching the C reference's use of `float` arithmetic to within 4.53e-05.
- The LOOP instruction (`rs--; if (rs > 0) branch`) reduced instruction count by 31% (27.6M → 19.0M) by fusing the common `ADDI R,-1; BNEZ R,label` pattern into a single instruction.
