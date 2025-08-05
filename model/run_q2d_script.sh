#!/bin/bash

# Q2D Script Runner
# Usage: ./run_q2d_script.sh [config_type] [openai_key_path]
# config_type: testset_q2d, testset_q2dcot, etc.

set -e  # Exit on error

# Set default values
CONFIG_TYPE=${1:-"testset_q2d"}
OPENAI_KEY=${2:-"key.txt"}

echo "=== Q2D Script Started ==="
echo "Configuration type: $CONFIG_TYPE"
echo "OpenAI key file: $OPENAI_KEY"

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if OpenAI key file exists
echo "Checking OpenAI key file..."
if [ ! -f "$OPENAI_KEY" ]; then
    echo "Error: OpenAI key file not found: $OPENAI_KEY"
    echo "Please create a file with your OpenAI API key"
    exit 1
fi

echo "✓ OpenAI key file found"

# Set file paths based on configuration type
case $CONFIG_TYPE in
    "testset_q2d")
        INPUT_CSV="../data/testset_top_10000.csv.gz"
        OUTPUT_CSV="../data/testset_q2d.csv"
        OUTPUT_JSON="../data/testset_q2d.json"
        PASSAGE2LABELID="../data/passage2labelid_top_10000.json"
        LABELID2PASSAGE="../data/labelid2passage_top_10000.json"
        TRAIN_DATA="../data/trainset_top_10000.csv.gz"
        DOCS_JSON="../bm25-files-10000/docs00.json"
        ;;
    "testset_q2dcot")
        INPUT_CSV="../data/testset_top_10000.csv.gz"
        OUTPUT_CSV="../data/testset_q2dcot.csv"
        OUTPUT_JSON="../data/testset_q2dcot.json"
        PASSAGE2LABELID="../data/passage2labelid_top_10000.json"
        LABELID2PASSAGE="../data/labelid2passage_top_10000.json"
        TRAIN_DATA="../data/trainset_top_10000.csv.gz"
        DOCS_JSON="../bm25-files-10000/docs00.json"
        ;;
    *)
        echo "Unsupported configuration type: $CONFIG_TYPE"
        echo "Supported types: testset_q2d, testset_q2dcot"
        exit 1
        ;;
esac

# Check if required files exist
echo "Checking required files..."
for file in "$INPUT_CSV" "$PASSAGE2LABELID" "$LABELID2PASSAGE" "$TRAIN_DATA" "$DOCS_JSON"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file not found: $file"
        exit 1
    fi
    echo "✓ $file"
done

# Create output directories
mkdir -p "$(dirname "$OUTPUT_CSV")"
mkdir -p "$(dirname "$OUTPUT_JSON")"

echo ""
echo "Starting Q2D processing..."

# Run Q2D script
python q2d_script.py \
    --key "$OPENAI_KEY" \
    --model_name "gpt-4o-mini" \
    --input_csv "$INPUT_CSV" \
    --output_csv "$OUTPUT_CSV" \
    --output_json "$OUTPUT_JSON" \
    --passage2labelid "$PASSAGE2LABELID" \
    --labelid2passage "$LABELID2PASSAGE" \
    --train_data "$TRAIN_DATA" \
    --docs_json "$DOCS_JSON" \
    --max_parallel_calls 20

echo ""
echo "=== Q2D Script Completed ==="
echo "Results saved to:"
echo "  CSV: $OUTPUT_CSV"
echo "  JSON: $OUTPUT_JSON" 