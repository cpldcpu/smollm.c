"""
Step 2: Verify bare PyTorch implementation against transformers
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import SmolLM2, ModelConfig, load_hf_weights
import json

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
CACHE_DIR = "./models/hf_cache"


def main():
    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    hf_model = hf_model.float().cpu()
    hf_model.eval()

    print("Creating bare PyTorch model...")
    config = ModelConfig()
    model = SmolLM2(config)
    model = model.float().cpu()
    model.eval()

    print("Loading weights from HuggingFace model...")
    load_hf_weights(model, hf_model)

    # Test prompts
    test_prompts = [
        "Hello",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    print("\n=== Verification ===")

    all_passed = True

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt!r}")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # HuggingFace forward
        with torch.no_grad():
            hf_outputs = hf_model(input_ids)
            hf_logits = hf_outputs.logits

        # Our model forward
        with torch.no_grad():
            our_logits, _ = model(input_ids)

        # Compare
        max_diff = (hf_logits - our_logits).abs().max().item()
        mean_diff = (hf_logits - our_logits).abs().mean().item()

        print(f"  HF logits shape: {hf_logits.shape}")
        print(f"  Our logits shape: {our_logits.shape}")
        print(f"  Max absolute difference: {max_diff:.2e}")
        print(f"  Mean absolute difference: {mean_diff:.2e}")

        # Check top predictions match
        hf_top = torch.topk(hf_logits[0, -1, :], k=5)
        our_top = torch.topk(our_logits[0, -1, :], k=5)

        print(f"  HF top 5: {hf_top.indices.tolist()}")
        print(f"  Our top 5: {our_top.indices.tolist()}")

        # Verify
        if max_diff < 1e-4:
            print("  ✓ PASS (max diff < 1e-4)")
        elif max_diff < 1e-3:
            print("  ~ CLOSE (max diff < 1e-3)")
        else:
            print("  ✗ FAIL (max diff >= 1e-3)")
            all_passed = False

        if hf_top.indices.tolist() != our_top.indices.tolist():
            print("  ✗ Top predictions don't match!")
            all_passed = False
        else:
            print("  ✓ Top predictions match")

    # Test generation
    print("\n=== Generation Test ===")
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Simple greedy generation with our model
    generated = input_ids.clone()
    kv_cache = None

    for _ in range(20):
        with torch.no_grad():
            if kv_cache is None:
                logits, kv_cache = model(generated, use_cache=True)
            else:
                logits, kv_cache = model(generated[:, -1:], kv_cache=kv_cache, use_cache=True)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Our model: {generated_text!r}")

    # Compare with HF generation
    hf_generated = hf_model.generate(
        input_ids,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    hf_generated_text = tokenizer.decode(hf_generated[0], skip_special_tokens=True)
    print(f"HF model:  {hf_generated_text!r}")

    if generated_text == hf_generated_text:
        print("✓ Generated text matches!")
    else:
        print("✗ Generated text differs")
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("=== Step 2 Complete: All verifications PASSED ===")
    else:
        print("=== Step 2: Some verifications FAILED ===")
    print("=" * 50)

    return all_passed


if __name__ == "__main__":
    main()
