#!/bin/bash

# LoRA Model Merge Script
# Usage: ./run_merge_lora.sh [lora_model_path] [output_path]

set -e  # Exit on error

# Set default values
LORA_MODEL_PATH=${1:-"./sft_lora_model"}
OUTPUT_PATH=${2:-"./merged_saul_model"}

echo "=== LoRA Model Merge Started ==="
echo "Base model: Equall/Saul-7B-Base"
echo "LoRA model: $LORA_MODEL_PATH"
echo "Output path: $OUTPUT_PATH"

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if LoRA model exists
echo "Checking LoRA model..."
if [ ! -d "$LORA_MODEL_PATH" ]; then
    echo "Error: LoRA model not found: $LORA_MODEL_PATH"
    echo "Please run ./run_sft_training.sh first to train the LoRA model"
    exit 1
fi

echo "✓ LoRA model found: $LORA_MODEL_PATH"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo ""
echo "Starting LoRA model merge..."

# Run merge
python merge_lora_model.py \
    --base_model "Equall/Saul-7B-Base" \
    --lora_model "$LORA_MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --device_map "auto"

echo ""
echo "=== LoRA Model Merge Completed ==="
echo "Merged model saved to: $OUTPUT_PATH" 