#!/bin/bash
# Quick start script for NoPE continued pretraining

set -e

echo "=== SmolLM2 NoPE Continued Pretraining ==="
echo ""

# Configuration
DATA_FILE="${1:-training_data.txt}"
OUTPUT_DIR="${2:-nope_checkpoint}"
MAX_STEPS="${3:-100}"

echo "Configuration:"
echo "  Data file: $DATA_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Max training steps: $MAX_STEPS"
echo ""

# Step 1: Prepare training data if it doesn't exist
if [ ! -f "$DATA_FILE" ]; then
    echo "Step 1: Preparing training data..."
    python prepare_training_data.py --mode synthetic --output "$DATA_FILE" --num_chars 200000
    echo ""
else
    echo "Step 1: Using existing data file: $DATA_FILE"
    echo ""
fi

# Step 2: Run continued pretraining
echo "Step 2: Running NoPE continued pretraining..."
python train_nope.py \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --seq_length 256 \
    --num_epochs 1 \
    --learning_rate 1e-5 \
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
