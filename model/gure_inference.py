from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import json
import os
import pandas as pd
import argparse
instruction = """<|begin_of_text|>You are a helpful assistant specializing in generating legal passages that naturally align with the preceding context. 
Based on the given preceding context, please generate a legal passage that is coherent, relevant, and contextually appropriate.

### Preceding Context: {context}

###Legal Passage: """


# Set the environment variable
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

class LlmGenerator:
    def __init__(self, model_name, dtype, trust_remote_code, tensor_parallel_size, temperature, top_p, max_tokens):
        self.model_name = model_name
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop="<end_of_text>",
            seed=1
        )
        self.llm = LLM(
            model=model_name, 
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "right"
        

    def _generate(self, text_data):
        outputs = self.llm.generate(text_data, self.sampling_params)
        generated = [output.outputs[0].text for output in outputs]
        return generated


def main():
    parser = argparse.ArgumentParser(description="GuRE Inference for Legal Passage Generation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the trained model")
    parser.add_argument("--input_csv", type=str, required=True,
                       help="Input CSV file path")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Output CSV file path")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter (default: 0.9)")
    parser.add_argument("--max_tokens", type=int, default=1024,
                       help="Maximum tokens to generate (default: 1024)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    print(f"Loading model from: {args.model_path}")
    llm_generator = LlmGenerator(
        model_name=args.model_path,
        dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
    
    print(f"Processing input file: {args.input_csv}")
    data = pd.read_csv(args.input_csv)
    
    docs = []
    for destination_context in data.destination_context:
        docs.append(instruction.format(context=destination_context))
    
    print("Generating legal passages...")
    generated_texts = llm_generator._generate(docs)
    data['rewritten_query'] = generated_texts
    
    print(f"Saving results to: {args.output_csv}")
    data.to_csv(args.output_csv, index=False, encoding='utf-8')
    print(f"Inference completed! Results saved to: {args.output_csv}")

if __name__ == '__main__':
    main()

