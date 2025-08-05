import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_model(base_model_name, lora_model_path, output_path, device_map="auto"):
    """
    Merge LoRA weights with the base model
    
    Args:
        base_model_name: Base model name or path
        lora_model_path: Path to the LoRA model
        output_path: Path to save the merged model
        device_map: Device mapping for model loading
    """
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA model: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("Merge completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA model with base model")
    parser.add_argument("--base_model", type=str, required=True,
                       help="Base model name or path (e.g., Equall/Saul-7B-Base)")
    parser.add_argument("--lora_model", type=str, required=True,
                       help="Path to the LoRA model")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save the merged model")
    parser.add_argument("--device_map", type=str, default="auto",
                       help="Device mapping for model loading (default: auto)")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.lora_model):
        raise FileNotFoundError(f"LoRA model path not found: {args.lora_model}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    try:
        merge_lora_model(
            base_model_name=args.base_model,
            lora_model_path=args.lora_model,
            output_path=args.output_path,
            device_map=args.device_map
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())