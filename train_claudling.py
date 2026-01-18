#!/usr/bin/env python3
"""
Train a Claudling: Fine-tune SmolLM2 into a helpful assistant

This script implements supervised fine-tuning (SFT) on instruction-following data
to create a Claude-like assistant from the SmolLM2-135M base model.

Usage:
    python train_claudling.py --phase sft
    python train_claudling.py --phase dpo --sft-model ./claudling-sft
"""

import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import load_dataset
from trl import SFTTrainer, DPOTrainer, DPOConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
import os


def format_oasst_dataset(examples):
    """
    Format OpenAssistant dataset into instruction-following format.

    OASST2 has a tree structure. We extract linear conversation threads.
    """
    conversations = []

    # This is a simplified version - full implementation would trace conversation trees
    for message in examples:
        if message["role"] == "assistant":
            # Build prompt from conversation history
            text = f"<|user|>\n{message['text']}\n<|assistant|>\n"
            conversations.append(text)

    return {"text": conversations}


def format_hh_rlhf_for_sft(example):
    """Format Anthropic HH-RLHF dataset for supervised fine-tuning."""
    # HH-RLHF format: "\n\nHuman: ...\n\nAssistant: ..."
    conversation = example["chosen"]  # Use the preferred response

    # Convert to a cleaner format
    conversation = conversation.replace("\n\nHuman:", "<|user|>\n")
    conversation = conversation.replace("\n\nAssistant:", "<|assistant|>\n")

    return {"text": conversation}


def format_hh_rlhf_for_dpo(example):
    """Format HH-RLHF for DPO training."""
    # Extract the last turn for preference learning
    chosen = example["chosen"]
    rejected = example["rejected"]

    # Find the last human prompt (they should be the same)
    parts = chosen.split("\n\nAssistant:")
    if len(parts) < 2:
        return None

    prompt = parts[-2]  # Everything up to last assistant response
    chosen_response = parts[-1].strip()

    parts_rej = rejected.split("\n\nAssistant:")
    rejected_response = parts_rej[-1].strip()

    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response
    }


def train_sft(args):
    """Phase 1: Supervised Fine-Tuning"""
    print("=" * 60)
    print("Phase 1: Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with optional quantization
    model_kwargs = {}
    if args.use_qlora:
        print("Using QLoRA (4-bit quantization)...")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        **model_kwargs
    )

    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    if args.dataset == "oasst2":
        dataset = load_dataset("OpenAssistant/oasst2")
        # Filter to English, assistant messages only
        dataset = dataset.filter(lambda x: x["lang"] == "en")
        # TODO: Implement proper conversation tree extraction
        train_dataset = dataset["train"]

    elif args.dataset == "hh-rlhf":
        dataset = load_dataset("Anthropic/hh-rlhf")
        train_dataset = dataset["train"].map(format_hh_rlhf_for_sft)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Limit dataset size for faster iteration if requested
    if args.max_samples:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))

    print(f"Training on {len(train_dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to=["tensorboard"] if args.use_tensorboard else [],
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text" if args.dataset == "oasst2" else "text",
        packing=False,  # Disable packing for simplicity
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model()

    # Merge LoRA weights if requested
    if args.merge_lora:
        print("Merging LoRA weights into base model...")
        merged_dir = args.output_dir + "_merged"

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, args.output_dir)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        print(f"Merged model saved to {merged_dir}")

    print("\nSFT training complete!")


def train_dpo(args):
    """Phase 2: Direct Preference Optimization"""
    print("=" * 60)
    print("Phase 2: Direct Preference Optimization (DPO)")
    print("=" * 60)

    if not args.sft_model:
        raise ValueError("--sft-model required for DPO training")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT model
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Reference model (frozen copy of SFT model)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Load preference dataset
    print("Loading preference dataset...")
    dataset = load_dataset("Anthropic/hh-rlhf")
    train_dataset = dataset["train"].map(format_hh_rlhf_for_dpo, remove_columns=dataset["train"].column_names)
    train_dataset = train_dataset.filter(lambda x: x is not None)

    if args.max_samples:
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))

    print(f"Training on {len(train_dataset)} preference pairs")

    # DPO configuration
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=10,
        save_steps=args.save_steps,
        beta=args.dpo_beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_seq_length // 2,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"] if args.use_tensorboard else [],
    )

    # DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("\nStarting DPO training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model()

    print("\nDPO training complete!")


def main():
    parser = argparse.ArgumentParser(description="Train a Claudling assistant")

    # Training phase
    parser.add_argument("--phase", choices=["sft", "dpo"], default="sft",
                      help="Training phase: sft (supervised fine-tuning) or dpo (preference optimization)")

    # Model arguments
    parser.add_argument("--base-model", default="HuggingFaceTB/SmolLM2-135M-Instruct",
                      help="Base model to fine-tune")
    parser.add_argument("--sft-model", help="SFT model path (required for DPO)")
    parser.add_argument("--output-dir", default="./claudling-sft",
                      help="Output directory for checkpoints")

    # Dataset arguments
    parser.add_argument("--dataset", choices=["oasst2", "hh-rlhf"], default="hh-rlhf",
                      help="Dataset to use for SFT")
    parser.add_argument("--max-samples", type=int, help="Limit training samples (for testing)")

    # Training hyperparameters
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                      help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")

    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA weights after training")

    # DPO arguments
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="DPO KL penalty coefficient")

    # Optimization
    parser.add_argument("--use-qlora", action="store_true", help="Use QLoRA (4-bit quantization)")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--use-tensorboard", action="store_true", help="Log to tensorboard")

    args = parser.parse_args()

    # Update output dir based on phase
    if args.phase == "dpo" and args.output_dir == "./claudling-sft":
        args.output_dir = "./claudling-dpo"

    # Run training
    if args.phase == "sft":
        train_sft(args)
    elif args.phase == "dpo":
        train_dpo(args)


if __name__ == "__main__":
    main()
