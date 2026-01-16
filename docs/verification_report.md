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

## Notes

- C vs PyTorch verification for both Q8 and Q4 passed with exact matches once newline parsing was fixed in `step6_verify.py`.
