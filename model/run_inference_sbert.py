from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
from prepare_data import *
import argparse
import faiss
import os
from tqdm import tqdm

def embedd_sentences(sentences, sbert_model="sentence-transformers/all-mpnet-base-v2", max_seq_length=384):
	embedder = SentenceTransformer(sbert_model)
	embedder.max_seq_length = max_seq_length
	print (len(sentences))
	corpus_embeddings = embedder.encode(sentences, batch_size=32, show_progress_bar=True)
	return corpus_embeddings


def embedd_all_relevant_files(dev, test, n_labels, model_name, df_targets, labelid2passage, query_reformulation="destination_context"):	
	with open("data/passage_dict.json") as f:
		passage_dict = json.load(f)["data"]

	targets = []
	for i in range(int(n_labels)):
		passage_id = labelid2passage[str(i)]
		targets.append(passage_dict[passage_id])

	embeddings_passages = embedd_sentences(targets, sbert_model=model_name)
	
	# Choose which column to embed based on query_reformulation argument
	if query_reformulation == "rewritten_query":
		dev_texts = dev.rewritten_query.tolist()
		test_texts = test.rewritten_query.tolist()
		print("Using rewritten_query column for embedding")
	else:
		dev_texts = dev.destination_context.tolist()
		test_texts = test.destination_context.tolist()
		print("Using destination_context column for embedding")
	
	embeddings_dev = embedd_sentences(dev_texts, sbert_model=model_name)
	embeddings_test = embedd_sentences(test_texts, sbert_model=model_name)

	return embeddings_passages, embeddings_dev, embeddings_test
	
def retrieve_neighbours_gpu(X, queries, batchsize=8192, num_neighbors=10):
	#res = faiss.StandardGpuResources()  # use a single GPU
	n, dim = X.shape[0], X.shape[1]
	index = faiss.IndexFlatIP(dim) # create CPU index
	#gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index) # create GPU index
	#gpu_index_flat.add(X)         # add vectors to the index

    index.add(X) # add vectors to the index
	all_indices = []
	for i in tqdm(range(0, len(queries), batchsize)):
		features = queries[i:i + batchsize]
		distances, indices = index.search(features, num_neighbors)
		all_indices.extend(indices)
	return all_indices

	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str, default='sentence-transformers/all-mpnet-base-v2')
	parser.add_argument('--n_labels', type=str, default="10000")
	parser.add_argument('--query_reformulation', type=str, default="destination_context", 
					   choices=["destination_context", "rewritten_query"],
					   help="Choose which column to use for embedding: destination_context or rewritten_query")

	args = parser.parse_args()
	outpath = "predictions-sbert"
	os.makedirs(outpath, exist_ok=True)
	train, dev, test, passage2labelid, labelid2passage = load_datasets(n_labels=args.n_labels)

	embeddings_passages, embeddings_dev, embeddings_test = embedd_all_relevant_files(
		dev, test, args.n_labels, args.model_name, labelid2passage, 
		query_reformulation=args.query_reformulation
	)
	
	hits = retrieve_neighbours_gpu(embeddings_passages, embeddings_dev)

	# save files somewhere
	true = dev.passage_id.tolist()

	with open(os.path.join(outpath, "predictions_devset_" + os.path.basename(args.model_name) + "_" + args.n_labels + ".json"), "w") as f:
		out = {"true":true, "hits": [[labelid2passage[str(j)] for j in i[:10]] for i in hits]}
		json.dump(out, f) 

	hits = retrieve_neighbours_gpu(embeddings_passages, embeddings_test)
	true = test.passage_id.tolist()
	with open(os.path.join(outpath, "predictions_testset_" + os.path.basename(args.model_name) + "_" + args.n_labels + ".json"), "w") as f:
		out = {"true":true, "hits": [[labelid2passage[str(j)] for j in i[:10]] for i in hits]}
		json.dump(out, f) 

