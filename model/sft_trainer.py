import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
    AddedToken,
    GenerationConfig
)
from peft import LoraConfig,get_peft_model
from trl import SFTTrainer, SFTConfig
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import numpy as np
from accelerate import Accelerator

os.environ["NCCL_TIMEOUT"] = "3600"
os.environ["ACCELERATE_TIMEOUT"] = "3600"
os.environ["WANDB__SERVICE_WAIT"] = "300"
logging.set_verbosity_info()
logging.disable_progress_bar()

def split_tokens(sentence):
    return sentence.split(' ')

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_accuracy(eval_pred):
    predictions, labels = eval_pred
    # Here, predictions is rewards_chosen and rewards_rejected.
    # We want to see how much of the time rewards_chosen > rewards_rejected.
    predictions = np.argmax(predictions, axis=1)
    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {"accuracy": accuracy}

def compute_metrics_bleu(pred):
    references = pred.label_ids
    generated_texts = pred.predictions
    
    bleu_scores = []
    for reference, generated in zip(references, generated_texts):
        reference = np.where(reference != -100, reference, tokenizer.pad_token_id)
        reference_text = tokenizer.decode(reference, skip_special_tokens=False)
        reference_text = reference_text.split("### Answer: ")[-1]
        refs_tokenized = [split_tokens(reference_text)]

        generated = np.where(generated != -100, generated, tokenizer.pad_token_id)
        generated_text = tokenizer.decode(generated, skip_special_tokens=False)
        generated_text = generated_text.split("### Answer: ")[-1]
        cands_tokenized = split_tokens(generated_text)
        
        bleu_score = sentence_bleu(refs_tokenized, cands_tokenized,smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(bleu_score)

    return {
        'bleu': sum(bleu_scores) / len(bleu_scores)
    }

def define_args():
    parser = argparse.ArgumentParser(description="SFT Trainer")

    #Base model and dataset
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--dataset_train", type=str, default="")
    parser.add_argument("--dataset_valid", type=str, default="")

    #New model and output
    parser.add_argument("--new_model", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")

    #lora config
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--use_4bit", type=bool, default=True)
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4")
    parser.add_argument("--use_nested_quant", type=bool, default=False)

    #training arguments
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--optim", type=str, default="paged_adamw_32bit")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--group_by_length", type=bool, default=False)
    parser.add_argument("--local_rank",type=int)

    #save and logging steps
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--logging_steps", type=int, default=2000)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--packing", type=bool, default=False)
    parser.add_argument("--device_map", type=str, default="balanced")
    
    # Merge LoRA arguments
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights with base model after training")
    parser.add_argument("--merged_model_path", type=str, default="", help="Path to save the merged model")
    parser.add_argument("--merge_device_map", type=str, default="auto", help="Device map for merging")

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":

    attn_implementation = "flash_attention_2"
    
    args = define_args()
    model_name = args.model_name
    dataset_train = args.dataset_train
    dataset_valid = args.dataset_valid  
    new_model = args.new_model
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    use_4bit = args.use_4bit
    bnb_4bit_compute_dtype = args.bnb_4bit_compute_dtype
    bnb_4bit_quant_type = args.bnb_4bit_quant_type
    use_nested_quant = args.use_nested_quant
    output_dir = args.output_dir
    num_train_epochs = args.num_train_epochs
    fp16 = args.fp16
    bf16 = args.bf16
    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    gradient_checkpointing = args.gradient_checkpointing
    max_grad_norm = args.max_grad_norm
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optim = args.optim
    lr_scheduler_type = args.lr_scheduler_type
    max_steps = args.max_steps
    warmup_ratio = args.warmup_ratio
    group_by_length = args.group_by_length
    save_steps = args.save_steps
    logging_steps = args.logging_steps
    max_seq_length = args.max_seq_length
    packing = args.packing
    device_map = args.device_map
    local_rank = args.local_rank
    
    # Load datasets
    train_dataset = load_dataset('json', data_files=dataset_train, split="train")
    valid_dataset = load_dataset('json', data_files=dataset_valid, split="train")

    # Load model and tokenizer
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        attn_implementation="flash_attention_2"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        TOKENIZERS_PARALLELISM=False,
        use_fast=False
        )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model.config.eos_token_id = tokenizer.eos_token_id
 
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        ],
        modules_to_save=[
            "embed_tokens",
            "lm_head"
        ]
    )
    
    model = get_peft_model(model,peft_config)

    # Set training parameters
    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="steps",
        eval_steps= save_steps,
        local_rank=local_rank,
        dataset_text_field="ift_sample",
        max_seq_length=max_seq_length,
        dataset_num_proc=10
    )
    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,  
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_arguments,
        compute_metrics=compute_metrics_bleu,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    trainer.train()
    trainer.model.save_pretrained(new_model)
    print(f"Training completed! LoRA model saved to: {new_model}")
