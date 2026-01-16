"""
Step 1: Transformers Reference Implementation
Downloads SmolLM2-135M-Instruct and generates reference outputs for verification.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
CACHE_DIR = "./models/hf_cache"
OUTPUT_DIR = "./reference_outputs"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
    )
    model = model.float()  # Convert to FP32 for reference
    model = model.cpu()    # CPU for reproducibility
    model.eval()

    # Print model config
    print("\n=== Model Configuration ===")
    config = model.config
    print(f"hidden_size: {config.hidden_size}")
    print(f"intermediate_size: {config.intermediate_size}")
    print(f"num_hidden_layers: {config.num_hidden_layers}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    print(f"num_key_value_heads: {config.num_key_value_heads}")
    print(f"vocab_size: {config.vocab_size}")
    print(f"max_position_embeddings: {config.max_position_embeddings}")
    print(f"rms_norm_eps: {config.rms_norm_eps}")
    print(f"rope_theta: {config.rope_theta}")
    print(f"hidden_act: {config.hidden_act}")
    print(f"tie_word_embeddings: {config.tie_word_embeddings}")

    # Print model structure
    print("\n=== Model Structure ===")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")

    # Test prompts for verification
    test_prompts = [
        "Hello",
        "The capital of France is",
        "def fibonacci(n):",
    ]

    print("\n=== Generating Reference Outputs ===")

    reference_data = {
        "model_id": MODEL_ID,
        "prompts": []
    }

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt!r}")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        print(f"Token IDs: {input_ids.tolist()[0]}")

        # Get logits (no generation, just forward pass)
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Save last token logits for verification
        last_logits = logits[0, -1, :].tolist()
        top_tokens = torch.topk(logits[0, -1, :], k=10)

        print(f"Logits shape: {logits.shape}")
        print(f"Top 10 predictions:")
        for i, (idx, val) in enumerate(zip(top_tokens.indices.tolist(), top_tokens.values.tolist())):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. {idx:5d} ({val:8.4f}): {token!r}")

        # Generate some text
        generated = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated: {generated_text!r}")

        reference_data["prompts"].append({
            "text": prompt,
            "input_ids": input_ids.tolist()[0],
            "last_logits_top10_indices": top_tokens.indices.tolist(),
            "last_logits_top10_values": top_tokens.values.tolist(),
            "generated_text": generated_text
        })

    # Save reference data
    ref_file = os.path.join(OUTPUT_DIR, "reference.json")
    with open(ref_file, "w") as f:
        json.dump(reference_data, f, indent=2)
    print(f"\nReference data saved to: {ref_file}")

    # Save full logits for first prompt (for detailed verification)
    prompt = test_prompts[0]
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs["input_ids"])

    logits_file = os.path.join(OUTPUT_DIR, "logits_hello.pt")
    torch.save({
        "input_ids": inputs["input_ids"],
        "logits": outputs.logits
    }, logits_file)
    print(f"Full logits saved to: {logits_file}")

    print("\n=== Step 1 Complete ===")

if __name__ == "__main__":
    main()
