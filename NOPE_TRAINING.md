# NoPE (No Positional Embeddings) Training Guide

This guide explains how to perform continued pretraining on SmolLM2-135M to generate a NoPE checkpoint, which removes RoPE (Rotary Positional Embeddings) from the model to simplify inference.

## Overview

The NoPE training pipeline allows you to:
1. Load the pretrained SmolLM2-135M model from HuggingFace
2. Remove RoPE from the attention mechanism
3. Perform continued pretraining to adapt the model to work without positional embeddings
4. Export the trained model to a quantized binary format for C/Rust inference

## Quick Start

The easiest way to run NoPE training is using the provided shell script:

```bash
./run_nope_training.sh [data_file] [output_dir] [max_steps]
```

Example:
```bash
chmod +x run_nope_training.sh
./run_nope_training.sh training_data.txt nope_checkpoint 100
```

This will:
1. Generate synthetic training data (if not provided)
2. Train the NoPE model for 100 steps
3. Export to `smollm2_nope_q8.bin`

## Step-by-Step Instructions

### 1. Prepare Training Data

You have three options for preparing training data:

#### Option A: Synthetic Data (for testing)
```bash
python prepare_training_data.py --mode synthetic --num_chars 200000 --output training_data.txt
```

#### Option B: Download Public Domain Texts
```bash
python prepare_training_data.py --mode wikipedia --output training_data.txt
```

This downloads classic texts from Project Gutenberg (Pride and Prejudice, Alice in Wonderland, etc.)

#### Option C: Use Your Own Data
```bash
python prepare_training_data.py --mode custom --input_files file1.txt file2.txt file3.txt --output training_data.txt
```

### 2. Run Continued Pretraining

Basic usage:
```bash
python train_nope.py --data_file training_data.txt --output_dir nope_checkpoint
```

With custom parameters:
```bash
python train_nope.py \
    --data_file training_data.txt \
    --output_dir nope_checkpoint \
    --batch_size 4 \
    --seq_length 512 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
    --max_steps 500 \
    --log_interval 10
```

**Training Parameters:**

- `--model_name`: HuggingFace model to load (default: `HuggingFaceTB/SmolLM2-135M-Instruct`)
- `--data_file`: Path to training text file (required)
- `--seq_length`: Sequence length for training (default: 512)
- `--batch_size`: Training batch size (default: 4)
- `--num_epochs`: Number of epochs (default: 1)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--weight_decay`: Weight decay for AdamW (default: 0.01)
- `--grad_clip`: Gradient clipping threshold (default: 1.0)
- `--max_steps`: Stop after N steps, useful for quick testing (default: 0 = no limit)
- `--log_interval`: Log every N steps (default: 10)
- `--output_dir`: Where to save the checkpoint (default: `./nope_checkpoint`)

**Expected Output:**

The training script will save:
- `model_nope.pt`: PyTorch model weights
- `config.json`: Model configuration (with `use_rope: false`)
- Tokenizer files (vocab.json, merges.txt, etc.)

### 3. Export to Binary Format

Convert the NoPE checkpoint to quantized binary format for C inference:

```bash
# Q8 quantization (recommended)
python step3_quantize_nope.py \
    --checkpoint_dir nope_checkpoint \
    --quant q8 \
    --output smollm2_nope_q8.bin

# Q4 quantization (smaller, slightly lower quality)
python step3_quantize_nope.py \
    --checkpoint_dir nope_checkpoint \
    --quant q4 \
    --group_size 32 \
    --output smollm2_nope_q4.bin
```

The binary format (version 3) includes a `use_rope` flag that the C inference code uses to skip RoPE application.

### 4. Run Inference

Once you have the quantized binary, run inference with the C implementation:

```bash
cd smolc
make
./smolc ../smollm2_nope_q8.bin
```

The C inference code automatically detects NoPE models (version 3 format) and skips RoPE computation, resulting in simpler and potentially faster inference.

## Architecture Changes

### Original Model (with RoPE)
- Attention uses Rotary Positional Embeddings
- Q and K are rotated based on their position in the sequence
- Position information is encoded through rotation

### NoPE Model
- Attention operates without positional embeddings
- Q and K are used directly without rotation
- Model relies on learned positional information from pretraining

## Implementation Details

### Python (train_nope.py)

The NoPE model is implemented in `train_nope.py` with these key classes:

- `AttentionNoPE`: Multi-head attention without RoPE application
- `TransformerBlockNoPE`: Transformer block using NoPE attention
- `SmolLM2NoPE`: Full model without positional embeddings

The training script:
1. Loads the HuggingFace model
2. Copies weights to NoPE model (Q, K, V, O projections are identical)
3. Trains with standard language modeling loss
4. Saves checkpoint with `use_rope: false` flag

### C (smolc/smolc.c)

The C implementation supports both RoPE and NoPE models:

- Binary format version 3 includes `use_rope` flag (uint32)
- `precompute_rope()` is skipped for NoPE models
- `apply_rope()` is conditionally called based on `config.use_rope`
- RoPE buffers are only allocated if needed

## Performance Considerations

### Training
- GPU recommended for faster training (automatically used if available)
- Batch size of 4-8 works well on 16GB GPU
- Typical training: 100-500 steps for basic adaptation

### Inference
- NoPE models skip RoPE computation entirely
- Slightly simpler forward pass
- Same memory requirements for KV cache
- Quality may vary depending on training data and steps

## Troubleshooting

### Out of Memory During Training
- Reduce `--batch_size` (try 2 or 1)
- Reduce `--seq_length` (try 256 or 128)
- Train on CPU if necessary (slower but works)

### Poor Generation Quality
- Increase `--max_steps` (try 500-1000)
- Use more/better training data
- Adjust `--learning_rate` (try 5e-6 or 2e-5)

### Import Errors
```bash
pip install torch transformers
```

## Example Workflow

Complete example from start to finish:

```bash
# 1. Prepare training data
python prepare_training_data.py --mode synthetic --num_chars 500000 --output data.txt

# 2. Train NoPE model
python train_nope.py \
    --data_file data.txt \
    --output_dir my_nope_model \
    --max_steps 300 \
    --batch_size 4 \
    --learning_rate 1e-5

# 3. Quantize to binary
python step3_quantize_nope.py \
    --checkpoint_dir my_nope_model \
    --quant q8 \
    --output my_nope_q8.bin

# 4. Run inference
cd smolc && make && ./smolc ../my_nope_q8.bin
```

## Files

- `train_nope.py`: Main training script with NoPE model implementation
- `prepare_training_data.py`: Data preparation utilities
- `step3_quantize_nope.py`: Export NoPE checkpoint to binary format
- `run_nope_training.sh`: Automated training pipeline
- `smolc/smolc.c`: C inference with NoPE support
- `smolc/smolc.h`: Updated config structure with `use_rope` flag

## Binary Format Version 3

The NoPE binary format extends version 2 with an additional field:

```
Header (52 bytes):
  - magic: "SMOL" (4 bytes)
  - version: 3 (uint32)
  - quant_type: 0=Q8, 1=Q4 (uint32)
  - group_size: for Q4 (uint32)
  - hidden_size, intermediate_size, num_layers, etc. (uint32 each)
  - rope_theta: unused for NoPE (float32)
  - rms_norm_eps (float32)
  - use_rope: 0 for NoPE, 1 for RoPE (uint32)  <-- NEW in v3

[... rest of format same as v2 ...]
```

Version 1 and 2 models are still supported and automatically use RoPE.

## References

- Original RoPE paper: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- SmolLM2: [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- NoPE approach: Simplified positional encoding through continued pretraining
