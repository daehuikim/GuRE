#!/bin/bash

# GuRE Inference Script
# Usage: ./run_gure_inference.sh [model_path] [input_csv] [output_csv]

set -e  # Exit on error

# Set default values
MODEL_PATH=${1:-"./merged_saul_model"}
INPUT_CSV=${2:-"../data/testset_top_10000.csv.gz"}
OUTPUT_CSV=${3:-"../data/testset_gure.csv"}

echo "=== GuRE Inference Started ==="
echo "Model path: $MODEL_PATH"
echo "Input CSV: $INPUT_CSV"
echo "Output CSV: $OUTPUT_CSV"

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if model exists
echo "Checking model..."
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found: $MODEL_PATH"
    echo "Please run ./run_merge_lora.sh first to create the merged model"
    exit 1
fi

# Check if input file exists
echo "Checking input file..."
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input file not found: $INPUT_CSV"
    exit 1
fi

echo "✓ Model found: $MODEL_PATH"
echo "✓ Input file found: $INPUT_CSV"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_CSV")"

echo ""
echo "Starting GuRE inference..."

# Run inference
python gure_inference.py \
    --model_path "$MODEL_PATH" \
    --input_csv "$INPUT_CSV" \
    --output_csv "$OUTPUT_CSV" \
    --tensor_parallel_size 1 \
    --temperature 0.0 \
    --top_p 0.9 \
    --max_tokens 1024

echo ""
echo "=== GuRE Inference Completed ==="
echo "Results saved to: $OUTPUT_CSV" 