# NoPE Training for SmolLM2

This directory contains tools for continued pretraining of SmolLM2-135M with NoPE (No Positional Embeddings) to simplify inference by removing RoPE.

## What is NoPE?

NoPE removes Rotary Positional Embeddings (RoPE) from the transformer attention mechanism. Instead of encoding position through rotation, the model learns positional information implicitly during continued pretraining on the same data distribution used for the original pretraining.

**Benefits:**
- ✅ Simpler inference code (no RoPE computation)
- ✅ Slightly faster forward pass
- ✅ Same memory requirements
- ✅ Compatible with existing quantization

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets

# Run the automated pipeline (uses FineWeb-Edu)
chmod +x run_nope_training.sh
./run_nope_training.sh
```

This will:
1. Download 2000 samples from FineWeb-Edu (same as SmolLM2 pretraining)
2. Train for 500 steps with optimized continued pretraining settings
3. Export to quantized binary format (Q8)
4. Ready for inference with `./smolc/smolc`

## Files

| File | Description |
|------|-------------|
| `train_nope.py` | Main training script with NoPE model implementation |
| `prepare_training_data.py` | Download and prepare FineWeb-Edu or other datasets |
| `step3_quantize_nope.py` | Export NoPE checkpoint to binary format (Q8/Q4) |
| `run_nope_training.sh` | Automated end-to-end training pipeline |
| `NOPE_TRAINING.md` | Comprehensive training guide |
| `smolc/smolc.c` | C inference engine with NoPE support |
| `smolc/smolc.h` | Updated config with `use_rope` flag |

## SmolLM2 Pretraining Data

SmolLM2-135M was pretrained on **2 trillion tokens**:

- **FineWeb-Edu (60%)**: Educational web content, 1.3T tokens
  - High-quality educational text classified by Llama3-70B
  - Dataset: [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

- **DCLM-Edu (40%)**: Quality Q&A-style content, 3.8T tokens
  - Filtered using fastText classifier on instruction-following data
  - Dataset: [`HuggingFaceTB/dclm-edu`](https://huggingface.co/datasets/HuggingFaceTB/dclm-edu)

- **The Stack**: Code data across 80+ programming languages
- **Mathematics**: InfiMM-WebMath, FineMath, Cosmopedia

**For NoPE training, we use FineWeb-Edu** to match the original pretraining distribution.

## Usage Examples

### Option 1: Automated Pipeline (Recommended)

```bash
# Uses FineWeb-Edu, 500 training steps, exports to Q8
./run_nope_training.sh
```

### Option 2: Manual Step-by-Step

```bash
# 1. Download FineWeb-Edu samples
python prepare_training_data.py \
    --mode fineweb-edu \
    --num_samples 2000 \
    --output training_data.txt

# 2. Train NoPE model
python train_nope.py \
    --data_file training_data.txt \
    --output_dir nope_checkpoint \
    --max_steps 500 \
    --learning_rate 5e-6

# 3. Export to binary
python step3_quantize_nope.py \
    --checkpoint_dir nope_checkpoint \
    --quant q8 \
    --output smollm2_nope_q8.bin

# 4. Run inference
cd smolc && make && ./smolc ../smollm2_nope_q8.bin
```

### Option 3: Quick Test with Synthetic Data

```bash
# For testing without downloading FineWeb-Edu
python prepare_training_data.py --mode synthetic --output test_data.txt
python train_nope.py --data_file test_data.txt --max_steps 100
```

## Training Parameters

**Recommended settings for continued pretraining:**

- **Learning rate**: `5e-6` (lower than from-scratch to avoid catastrophic forgetting)
- **Training steps**: `500-1000` (sufficient for NoPE adaptation)
- **Batch size**: `4` (works on 16GB GPU)
- **Sequence length**: `512` (matches FineWeb-Edu documents)
- **Dataset**: FineWeb-Edu (same distribution as SmolLM2 pretraining)

## Binary Format

The NoPE model uses binary format **version 3**:

```
Header (52 bytes):
  - magic: "SMOL"
  - version: 3
  - use_rope: 0 (NoPE) or 1 (RoPE)
  [... other config fields ...]
```

The C inference code automatically detects version 3 and skips RoPE computation when `use_rope = 0`.

## GPU Requirements

- **Training**: 16GB+ GPU recommended (CUDA support automatic)
- **CPU Training**: Works but slower (add `--device cpu`)
- **Inference**: CPU-only (C implementation)

**Memory tips:**
- Reduce batch size if OOM: `--batch_size 2` or `--batch_size 1`
- Reduce sequence length: `--seq_length 256`

## Documentation

See [`NOPE_TRAINING.md`](NOPE_TRAINING.md) for:
- Detailed training guide
- All command-line options
- Troubleshooting tips
- Architecture details
- Performance considerations

## References

- **SmolLM2 Paper**: [arXiv:2502.02737](https://arxiv.org/html/2502.02737v1)
- **Model**: [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- **Dataset**: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- **Original Repo**: [huggingface/smollm](https://github.com/huggingface/smollm)

## Support

For issues or questions:
- Check [`NOPE_TRAINING.md`](NOPE_TRAINING.md) troubleshooting section
- Review the SmolLM2 paper for pretraining details
- Ensure `datasets` library is installed: `pip install datasets`
