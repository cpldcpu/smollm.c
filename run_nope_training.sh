#!/bin/bash
# Quick start script for DroPE continued pretraining
# DroPE = Dropping Positional Embeddings (Sakana AI, Dec 2025)
# Uses FineWeb-Edu dataset (same as SmolLM2 pretraining)

set -e

echo "=== SmolLM2 DroPE Continued Pretraining ==="
echo "=== DroPE: Dropping Positional Embeddings (Sakana AI) ==="
echo "=== Using FineWeb-Edu (SmolLM2 pretraining data) ==="
echo ""

# Configuration (DroPE defaults)
DATA_FILE="${1:-training_data.txt}"
OUTPUT_DIR="${2:-drope_checkpoint}"
MAX_STEPS="${3:-10000}"  # ~327M tokens at batch=4, seq=8192

echo "Configuration (DroPE-optimized):"
echo "  Data file: $DATA_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Max training steps: $MAX_STEPS"
echo "  Sequence length: 8192 (SmolLM2 pretraining length)"
echo "  Learning rate: 3e-4 with 490-step warmup"
echo "  QKNorm: Enabled for stability"
echo ""
echo "NOTE: For full DroPE (40B tokens), use --max_steps 1220703"
echo "      This script uses $MAX_STEPS for faster testing"
echo ""

# Step 1: Prepare training data if it doesn't exist
if [ ! -f "$DATA_FILE" ]; then
    echo "Step 1: Preparing training data from FineWeb-Edu..."
    echo "This is the same dataset used to pretrain SmolLM2-135M"
    python prepare_training_data.py \
        --mode fineweb-edu \
        --output "$DATA_FILE" \
        --num_chars 2000000 \
        --num_samples 2000
    echo ""
else
    echo "Step 1: Using existing data file: $DATA_FILE"
    echo ""
fi

# Step 2: Run DroPE continued pretraining
echo "Step 2: Running DroPE continued pretraining..."
echo "Using DroPE-optimized settings (Sakana AI recommendations)"
python train_nope.py \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --seq_length 8192 \
    --num_epochs 1 \
    --learning_rate 3e-4 \
    --warmup_steps 490 \
    --weight_decay 0.01 \
    --use_qk_norm \
    --max_steps "$MAX_STEPS" \
    --log_interval 100

echo ""
echo "Step 3: Exporting to binary format..."
python step3_quantize_nope.py \
    --checkpoint_dir "$OUTPUT_DIR" \
    --output smollm2_drope_q8.bin \
    --quant q8

echo ""
echo "=== Training Complete ==="
echo "DroPE checkpoint saved to: $OUTPUT_DIR"
echo "Binary model saved to: smollm2_drope_q8.bin"
echo ""
echo "You can now use the DroPE model for inference with:"
echo "  cd smolc && make && ./smolc ../smollm2_drope_q8.bin"
echo ""
echo "Note: This was a quick test run ($MAX_STEPS steps)"
echo "For full DroPE quality, train for 40B-100B tokens:"
echo "  ./run_nope_training.sh $DATA_FILE $OUTPUT_DIR 1220703"
