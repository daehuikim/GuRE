# GuRE:Generative Query REwriter for Legal Passage Retrieval

This repo contains the code for the paper LePaRD paper [(kim et al., 2025)](https://arxiv.org/abs/2505.12950).

## dataset

You can download the dataset from [the original LePaRD repo.](https://github.com/rmahari/LePaRD)

# Reference

Please cite the following paper if you use LePaRD:

```bibtex
@misc{kim2025gure,
      title={GuRE:Generative Query REwriter for Legal Passage Retrieval}, 
      author={Daehee Kim and Deokhyung Kang and Jonghwi Kim and Sangwon Ryu and Gary Geunbae Lee},
      year={2025},
      eprint={2505.12950},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.12950}, 
}
```
## Acknowledgement
This codebase is built upon the original LePaRD repository. We extend our gratitude to the [(Mahari et al., 2024)](https://github.com/rmahari/LePaRD) for providing the foundational dataset and baseline implementations.

## Additional Features
This repository includes several enhancements and additional tools for working with the LePaRD dataset:

### SFT Data Generation
Generate Supervised Fine-Tuning (SFT) data from the LePaRD dataset for training language models.

**Usage:**
```bash
cd src/model
./run_make_sft_data.sh
```

This script processes the `testset_top_10000.csv.gz` file and generates SFT training data in JSONL format, saved as `sft_data_test_10000.jsonl`.

**Features:**
- Flexible data processing with configurable file paths
- Progress tracking during data generation
- Automatic output directory creation
- UTF-8 encoding support for proper text handling
- Comprehensive error handling and file validation

**Output Format:**
Each line in the generated JSONL file contains:
```json
{
  "ift_sample": "<instruction>### Preceding Context: {context}\n\n###Legal Passage: {passage}",
  "quote": "original quote text"
}
```

### SFT Training with LoRA
Train a language model using Supervised Fine-Tuning (SFT) with LoRA (Low-Rank Adaptation) on the LePaRD dataset.

**Usage:**
```bash
cd src/model
./run_sft_training.sh
```

This script uses [Saul-7B-Base](https://huggingface.co/Equall/Saul-7B-Base) as the baseline model and trains it on the generated SFT data.

**Features:**
- LoRA training for efficient fine-tuning
- 4-bit quantization for memory efficiency
- Flash Attention 2 for faster training
- Automatic evaluation during training
- BLEU score computation for text generation quality

### LoRA Model Merging
Merge the trained LoRA weights with the base model to create a standalone model.

**Usage:**
```bash
cd src/model
./run_merge_lora.sh [lora_model_path] [output_path]
```

**Examples:**
```bash
# Use default paths
./run_merge_lora.sh

# Specify custom paths
./run_merge_lora.sh ./my_lora_model ./my_merged_model
```

**Features:**
- Automatic device mapping for optimal memory usage
- Tokenizer preservation
- Comprehensive error handling

### GuRE Inference
Generate legal passages using the trained model for inference on test data.

**Usage:**
```bash
cd src/model
./run_gure_inference.sh [model_path] [input_csv] [output_csv]
```

**Examples:**
```bash
# Use default paths
./run_gure_inference.sh

# Specify custom paths
./run_gure_inference.sh ./my_model ../data/my_input.csv ../data/my_output.csv
```

**Features:**
- VLLM-based efficient inference
- Tensor parallelism support
- Configurable sampling parameters
- Automatic output directory creation

### Q2D (Query-to-Document) Processing
Generate legal passages using OpenAI models with few-shot learning approach.

**Usage:**
```bash
cd src/model
./run_q2d_script.sh [config_type] [openai_key_path]
```

**Examples:**
```bash
# Basic Q2D processing
./run_q2d_script.sh testset_q2d key.txt

# Q2D with Chain-of-Thought
./run_q2d_script.sh testset_q2dcot key.txt
```

**Features:**
- OpenAI API integration with retry logic
- Parallel processing for efficiency
- Few-shot learning with BM25 retrieval
- Support for different configuration types
- Automatic output directory creation

# Description

LePaRD is a massive collection of U.S. federal judicial citations to precedent in context. LePaRD builds on millions of expert decisions by extracting quotations to precedents from judicial opinions along with the preceding context. Each row of the dataset corresponds to a quotation to prior case law used in a certain context.

- passage_id: A unique identifier for each passage
- destination_context: The preceding context before the quotation
- quote: The text of the passage that was quoted
- court: The court from which the passage originated
- date: The date when the opinion from which the passage originated was published

Contact [Robert Mahari](https://robertmahari.com/) in case of any questions.


## Data
The original data can be downloaded here: https://www.dropbox.com/scl/fo/0pgqxcz0h2l4wta8yyvb3/ABjE8bNAnq3Vm2bBJziclPE?rlkey=zipkfcso0h9je1xne737ims02&st=6mgtpwa0&dl=0
To run the replication package, make sure to store all files in a folder called *data*.

## Installation

Requires
* For bm25: [anserini](https://github.com/castorini/anserini)
* For dense retrieval: [SBERT](https://github.com/UKPLab/sentence-transformers/) and [Faiss](https://github.com/facebookresearch/faiss)
* For classification experiments: [transformers](https://huggingface.co/docs/transformers/installation)

For example, the following should work:
```
conda create --name lepard python=3.10
conda activate lepard
pip install -r requirements.txt
```

## Experiments

First, split the data into train, dev and test. The output of this process can also be downloaded [here](https://www.dropbox.com/scl/fi/m4z379fyjgi33ppu8q0fs/data_postprocessed.zip?rlkey=nrhton2dkku9gdv8alcj0g7f1&st=mpay7kqc&dl=0).

```shell
python src/model/prepare_data.py
```

### bm25 experiments

```
# reformat input files
python src/model/bm25_pipeline.py

# run anserini and bm25 retrieval
path_anserini="/path/to/anserini"
num_labels="10000" # change this to 20000 / 50000 for other experiments

# build index
sh $path_anserini/target/appassembler/bin/IndexCollection -threads 1 -collection JsonCollection \
 -generator DefaultLuceneDocumentGenerator -input bm25-files-$num_labels \
 -index indexes/index-lepard-passages-$num_labels -storePositions -storeDocvectors -storeRaw 

# retrieve passages devset
sh $path_anserini/target/appassembler/bin/SearchMsmarco -hits 10 -threads 1 \
 -index indexes/index-lepard-passages-$num_labels \
 -queries bm25-files-$num_labels/bm25_input_dev_$num_labels".tsv" \
 -output bm25-files-$num_labels/bm25_output_dev.tsv/

# retrieve passages testset
sh $path_anserini/target/appassembler/bin/SearchMsmarco -hits 10 -threads 1 \
 -index indexes/index-lepard-passages-$num_labels \
 -queries bm25-files-$num_labels/bm25_input_test_$num_labels".tsv" \
 -output bm25-files-$num_labels/bm25_output_test.tsv/

# evaluate
python src/model/evaluate_run.py --dev_predictions bm25-files-$num_labels/bm25_output_dev.tsv --test_predictions bm25-files-$num_labels/bm25_output_test.tsv --experiment bm25
```

### classification experiments

```
num_labels="10000" # change this to 20000 / 50000 for other experiments
model_name="distilbert-base-uncased"
python src/model/train_classification_models.py --n_labels 10000 --model_name $model_name # trains default distilbert models and saves model and predictions in folder "finetuned-$model_name-$num_labels"
# evaluate
python src/model/evaluate_run.py --dev_predictions finetuned-distilbert-base-uncased-10000/predictions_devset_10000.json --test_predictions finetuned-distilbert-base-uncased-10000/predictions_testset_10000.json 
# for all experiments, change num_labels to 20000 and 20000, and also run with legalbert 
```

### SBERT experiments
```
# zero-shot
num_labels="10000" # change this to 20000 / 50000 for other experiments
model_name="sentence-transformers/all-mpnet-base-v2"
python src/model/run_inference_sbert.py --model_name $model_name n_labels $num_labels # creates folder "predictions-sbert" and saves output there (os.path.basename(model_name) + predictions dev/test)
# evaluate
python src/model/evaluate_run.py --dev_predictions predictions-sbert/predictions_devset_all-mpnet-base-v2_$num_labels.json --test_predictions predictions-sbert/predictions_testset_all-mpnet-base-v2_$num_labels.json
```

```
# fine-tune
num_labels="10000" # change this to 20000 / 50000 for other experiments
python src/model/finetune_sbert.py --n_labels $num_labels # saves model in "sbert-finetuned-MultipleNegativesRankingLoss" + os.path.basename(args.model_name) + "-" + args.n_labels
model_name="sbert-finetuned-MultipleNegativesRankingLossall-distilroberta-v1-10000"
# run inference
python src/model/run_inference_sbert.py --model_name $model_name n_labels $num_labels # creates folder "predictions-sbert" and saves output there (os.path.basename(model_name) + predictions dev/test)
# evaluate
python src/model/evaluate_run.py --dev_predictions finetuned-distilbert-base-uncased-10000/predictions_devset_10000.json --test_predictions finetuned-distilbert-base-uncased-10000/predictions_testset_10000.json 
```

### Data replication
The training data may be generated using the Case Law Access project dataset using the precedent_data_extraction.py script

The results of the paper "GuRE:Generative Query REwriter for Legal Passage Retrieval" is in ```results/```

