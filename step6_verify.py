"""
Step 6: Verify C implementation vs PyTorch Q8/Q4
"""

import torch
import subprocess
import argparse
from transformers import AutoTokenizer
from step4_pytorch_q8 import load_model, QUANT_Q4

MODEL_FILE = "./models/smollm2-135m-q8.bin"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
CACHE_DIR = "./models/hf_cache"
DEFAULT_C_Q8 = "./smolc/smolc"
DEFAULT_C_Q4 = "./smolc/smolc_full"


def run_c_inference(c_binary, model_file, prompt, max_tokens=30):
    """Run C inference and return output"""
    result = subprocess.run(
        [c_binary, "-m", model_file, "-p", prompt, "-n", str(max_tokens)],
        capture_output=True,
        text=True
    )
    # Parse output to get generated text
    lines = result.stdout.strip().split('\n')
    # Find "Generating:" line and get text after it
    for i, line in enumerate(lines):
        if line.startswith("Generating:"):
            return '\n'.join(lines[i+1:])
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Verify C vs PyTorch quantized model")
    parser.add_argument("--model", type=str, default=MODEL_FILE, help="Path to quantized model file")
    parser.add_argument("--c-binary", type=str, default=None, help="Path to C binary (optional)")
    parser.add_argument("--max-tokens", type=int, default=30, help="Max tokens to generate")
    args = parser.parse_args()

    print("=== Step 6: Verifying C vs PyTorch ===\n")

    # Load PyTorch quantized model
    print(f"Loading PyTorch model from {args.model}...")
    model, vocab, merges = load_model(args.model)
    quant_label = "Q4" if model.config.quant_type == QUANT_Q4 else "Q8"

    if args.c_binary:
        c_binary = args.c_binary
    else:
        c_binary = DEFAULT_C_Q4 if model.config.quant_type == QUANT_Q4 else DEFAULT_C_Q8

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

    # Test prompts
    test_prompts = [
        "The capital of France is",
        "Hello",
        "def fibonacci(n):",
    ]

    print("\n" + "=" * 60)

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt!r}")
        print("-" * 40)

        # PyTorch generation
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        generated = input_ids.clone()
        kv_cache = None

        for _ in range(args.max_tokens):
            if kv_cache is None:
                logits, kv_cache = model.forward(generated, use_cache=True)
            else:
                logits, kv_cache = model.forward(generated[:, -1:], kv_cache=kv_cache, use_cache=True)

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        pytorch_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"PyTorch {quant_label}: {pytorch_text!r}")

        # C generation
        c_text = run_c_inference(c_binary, args.model, prompt, max_tokens=args.max_tokens)
        print(f"C {quant_label}:       {c_text!r}")

        # Compare
        if pytorch_text.strip() == c_text.strip():
            print("✓ MATCH")
        else:
            # Check if first N tokens match
            pt_words = pytorch_text.split()[:10]
            c_words = c_text.split()[:10]
            if pt_words == c_words:
                print("~ CLOSE (first 10 words match)")
            else:
                print("✗ DIFFER")

    print("\n" + "=" * 60)
    print("=== Verification Complete ===")


if __name__ == "__main__":
    main()
