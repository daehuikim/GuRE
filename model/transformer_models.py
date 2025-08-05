import json
import random
import argparse
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
#from data_helpers import get_dataset_splits, round_float, get_cv_splits
from sklearn import metrics

def evaluate(gold, predictions):
	# acc, pr, rc, f1
	pr = round_float(metrics.precision_score(gold, predictions))
	rc = round_float(metrics.recall_score(gold, predictions))
	f1 = round_float(metrics.f1_score(gold, predictions))
	acc = round_float(metrics.accuracy_score(gold, predictions))
	return " & ".join((pr, rc, f1, acc))

class SequenceClassificationDatasetNoLabels(Dataset):
	def __init__(self, x, tokenizer):
		self.examples = x
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, idx):
		return self.examples[idx]
	def collate_fn(self, batch):
		model_inputs = self.tokenizer([i for i in batch], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
		return {"model_inputs": model_inputs}



class SequenceClassificationDataset(Dataset):
	def __init__(self, x, y, tokenizer):
		self.examples = list(zip(x,y))
		self.tokenizer = tokenizer
		self.device = "cuda" if torch.cuda.is_available() else "cpu"

	def __len__(self):
		return len(self.examples)
	def __getitem__(self, idx):
		return self.examples[idx]
	def collate_fn(self, batch):
		#print (batch)
		model_inputs = self.tokenizer([i[0] for i in batch], return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
		labels = torch.tensor([i[1] for i in batch]).to(self.device)
		return {"model_inputs": model_inputs, "label": labels}

def evaluate_epoch(model, dataset, args):
	model.eval()
	targets = []
	outputs = []
	probs = []
	with torch.no_grad():
		for batch in DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn):
			output = model(**batch["model_inputs"])
			logits = output.logits
			targets.extend(batch['label'].float().tolist())
			outputs.extend(logits.argmax(dim=1).tolist())
			probs.extend(logits.softmax(dim=1)[:,1].tolist())
	return targets, outputs, probs

def train_model(trainset, model_name, args):

	device = "cuda" if torch.cuda.is_available() else "cpu"
	config = AutoConfig.from_pretrained(model_name)
	config.num_labels = int(args.n_labels)
	#config.gradient_checkpointing = True

	model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)
	warmup_steps = 0
	train_dataloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=trainset.collate_fn)
	t_total = int(len(train_dataloader) * args.num_epochs / args.gradient_accumulation_steps)

	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
	{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 8e-6},
	{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	    ]

	optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

	model.zero_grad()
	optimizer.zero_grad()

	use_amp = True
	#scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

	for epoch in range(args.num_epochs):
		model.train()
		t = tqdm(train_dataloader)
		# for i, batch in enumerate(train_dataloader):
		for i, batch in enumerate(t):
			# if not bf16, use_amp should be False
			with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
				output = model(**batch["model_inputs"], labels=batch['label'])
			loss = output.loss
			loss.backward()

			#print (loss)
			scheduler.step()  # Update learning rate schedule
			optimizer.step()
			optimizer.zero_grad()

	return model
