import pandas as pd
import json
import argparse
import os
from pathlib import Path

def create_sft_data(
    data_csv_path: str,
    passage2labelid_path: str,
    docs_json_path: str,
    output_path: str,
    instruction_template: str = None
):
    """
    Function to create SFT (Supervised Fine-Tuning) data
    
    Args:
        data_csv_path: Input CSV file path
        passage2labelid_path: passage2labelid JSON file path
        docs_json_path: Documents JSON file path
        output_path: Output JSONL file path
        instruction_template: Custom instruction template (optional)
    """
    
    # Default instruction template
    if instruction_template is None:
        instruction_template = """<|begin_of_text|>You are a helpful assistant specializing in generating legal passages that naturally align with the preceding context. 
		Based on the given preceding context, please generate a legal passage that is coherent, relevant, and contextually appropriate.

		### Preceding Context: {context}

		###Legal Passage: {passage} <|end_of_text|>"""
    
    # Check if files exist
    for file_path in [data_csv_path, passage2labelid_path, docs_json_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load data
    print(f"Loading data: {data_csv_path}")
    data = pd.read_csv(data_csv_path)
    
    print(f"Loading passage2labelid: {passage2labelid_path}")
    with open(passage2labelid_path) as f:
        passage2labelid_20 = json.load(f)
    
    print(f"Loading document data: {docs_json_path}")
    df_targets = pd.read_json(docs_json_path, lines=True)
    
    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate SFT data
    print("Generating SFT data...")
    write_data = []
    total_samples = len(data)
    
    for idx, (passage_id, context, quote) in enumerate(zip(data.passage_id, data.destination_context, data.quote)):
        if idx % 1000 == 0:
            print(f"Progress: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
        
        origin_passage = df_targets.iloc[passage2labelid_20[str(passage_id)]]["contents"]
        sample = instruction_template.format(context=context, passage=origin_passage)
        sample = sample.strip()
        write_data.append({"ift_sample": sample, "quote": quote})
    
    # Save results
    print(f"Saving results: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in write_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"SFT data generation completed! Total {len(write_data)} samples saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='SFT Data Generation Tool')
    parser.add_argument('--data_csv', type=str, required=True,
                       help='Input CSV file path (e.g., data/50000_unseen.csv)')
    parser.add_argument('--passage2labelid', type=str, required=True,
                       help='passage2labelid JSON file path (e.g., data/passage2labelid_top_50000.json)')
    parser.add_argument('--docs_json', type=str, required=True,
                       help='Documents JSON file path (e.g., bm25-files-50000/docs00.json)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSONL file path (e.g., data/50000_unseen.jsonl)')
    parser.add_argument('--instruction_template', type=str, default=None,
                       help='Custom instruction template (optional)')
    
    args = parser.parse_args()
    
    try:
        create_sft_data(
            data_csv_path=args.data_csv,
            passage2labelid_path=args.passage2labelid,
            docs_json_path=args.docs_json,
            output_path=args.output,
            instruction_template=args.instruction_template
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())