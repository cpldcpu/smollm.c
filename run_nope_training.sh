#!/bin/bash
# Quick start script for NoPE continued pretraining
# Uses FineWeb-Edu dataset (same as SmolLM2 pretraining)

set -e

echo "=== SmolLM2 NoPE Continued Pretraining ==="
echo "=== Using FineWeb-Edu (SmolLM2 pretraining data) ==="
echo ""

# Configuration
DATA_FILE="${1:-training_data.txt}"
OUTPUT_DIR="${2:-nope_checkpoint}"
MAX_STEPS="${3:-500}"

echo "Configuration:"
echo "  Data file: $DATA_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Max training steps: $MAX_STEPS"
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

# Step 2: Run continued pretraining
echo "Step 2: Running NoPE continued pretraining..."
echo "Using settings optimized for SmolLM2 continued pretraining"
python train_nope.py \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --seq_length 512 \
    --num_epochs 1 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --max_steps "$MAX_STEPS" \
    --log_interval 10

echo ""
echo "Step 3: Exporting to binary format..."
python step3_quantize_nope.py \
    --checkpoint_dir "$OUTPUT_DIR" \
    --output smollm2_nope_q8.bin \
    --quant q8

echo ""
echo "=== Training Complete ==="
echo "NoPE checkpoint saved to: $OUTPUT_DIR"
echo "Binary model saved to: smollm2_nope_q8.bin"
echo ""
echo "You can now use the NoPE model for inference with:"
echo "  ./smolc/smolc smollm2_nope_q8.bin"
