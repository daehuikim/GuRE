#!/bin/bash

# SFT Data Generation Script
# Usage: ./run_make_sft_data.sh [data_type]
# data_type: test, train, validation, etc.

set -e  # Exit on error

# Set default value
DATA_TYPE=${1:-"test"}

echo "=== SFT Data Generation Started ==="
echo "Data type: $DATA_TYPE"

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set file paths for test data
DATA_CSV="../data/testset_top_10000.csv.gz"
PASSAGE2LABELID="../data/passage2labelid_top_10000.json"
DOCS_JSON="../bm25-files-10000/docs00.json"
OUTPUT="../data/sft_data_test_10000.jsonl"

# Check if required files exist
echo "Checking required files..."
for file in "$DATA_CSV" "$PASSAGE2LABELID" "$DOCS_JSON"; do
    if [ ! -f "$file" ]; then
        echo "Error: File not found: $file"
        exit 1
    fi
    echo "✓ $file"
done

# Execute Python script
echo ""
echo "Generating SFT data..."
python make_sft_data.py \
    --data_csv "$DATA_CSV" \
    --passage2labelid "$PASSAGE2LABELID" \
    --docs_json "$DOCS_JSON" \
    --output "$OUTPUT"

echo ""
echo "=== SFT Data Generation Completed ==="
echo "Output file: $OUTPUT"

# Check file size
if [ -f "$OUTPUT" ]; then
    FILE_SIZE=$(du -h "$OUTPUT" | cut -f1)
    LINE_COUNT=$(wc -l < "$OUTPUT")
    echo "File size: $FILE_SIZE"
    echo "Total lines: $LINE_COUNT"
else
    echo "Error: Output file was not created."
    exit 1
fi 