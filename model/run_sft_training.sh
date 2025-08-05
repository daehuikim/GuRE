#!/bin/bash

# SFT Training Script with Saul-7B-Base
# Usage: ./run_sft_training.sh

set -e  # Exit on error

echo "=== SFT Training Started ==="
echo "Using Saul-7B-Base as baseline model"

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set default parameters for SFT training
MODEL_NAME="Equall/Saul-7B-Base"
TRAIN_DATASET="../data/sft_data_test_10000.jsonl"
VALID_DATASET="../data/sft_data_test_10000.jsonl"
OUTPUT_DIR="./sft_output"
NEW_MODEL="./sft_lora_model"

# Check if training data exists
echo "Checking training data..."
if [ ! -f "$TRAIN_DATASET" ]; then
    echo "Error: Training data not found: $TRAIN_DATASET"
    echo "Please run ./run_make_sft_data.sh first to generate SFT data"
    exit 1
fi

echo "✓ Training data found: $TRAIN_DATASET"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Starting SFT training with Saul-7B-Base..."
echo "Training data: $TRAIN_DATASET"
echo "Output directory: $OUTPUT_DIR"
echo "LoRA model will be saved to: $NEW_MODEL"

# Run SFT training
python sft_trainer.py \
    --model_name "$MODEL_NAME" \
    --dataset_train "$TRAIN_DATASET" \
    --dataset_valid "$VALID_DATASET" \
    --new_model "$NEW_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_seq_length 1024 \
    --save_steps 1000 \
    --logging_steps 100

echo ""
echo "=== SFT Training Completed ==="
echo "LoRA model saved to: $NEW_MODEL"
echo "Training logs saved to: $OUTPUT_DIR" 