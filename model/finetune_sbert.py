from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
from prepare_data import *
import argparse
import os
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from tqdm import tqdm


def get_training_set(n_labels, train, passage2labelid):
	train, dev, test, passage2labelid, labelid2passage = load_datasets(n_labels=n_labels)
	df_targets = pd.DataFrame([(c,i) for c,i in labelid2passage.items()], columns=["id", "contents"])

	train_samples = []
	for passage_id, destination_context in tqdm(zip(train.passage_id, train.destination_context), total=len(train)):
		origin_passage = passage_dict[str(passage_id)]
		destination_context = destination_context
		train_samples.append(InputExample(texts=[destination_context, origin_passage], label=1))
	return train_samples



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str, default='sentence-transformers/all-distilroberta-v1')
	parser.add_argument('--n_labels', type=str, default="10000")

	args = parser.parse_args()
	train, dev, test, passage2labelid, labelid2passage = load_datasets(n_labels=args.n_labels)
	del dev
	del test
	del labelid2passage

	# hyper-params
	num_epochs = 3
	train_batch_size = 32
	model_save_path = "sbert-finetuned-MultipleNegativesRankingLoss" + os.path.basename(args.model_name) + "-" + args.n_labels

	# set up things
	model = SentenceTransformer(args.model_name)
	model.max_seq_length = 256
	train_samples = get_training_set(args.n_labels, train, passage2labelid)
	print ("processed train samples")
	train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
	train_loss = losses.MultipleNegativesRankingLoss(model)
	
	print (model_save_path)


	# train
	model.old_fit(train_objectives=[(train_dataloader, train_loss)],
		  epochs=num_epochs,
		  warmup_steps=0.1,
		  output_path=model_save_path,
		  )
	print ("training finished")

