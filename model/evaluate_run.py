import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from sklearn.metrics import ndcg_score
import json
import pytrec_eval
import json
import argparse
from prepare_data import load_datasets, load_testsets


def convert_to_trec_eval(true, hits):
	qrel = {}
	run = {}
	for i, ref in enumerate(true):
		key = "q" + str(i)
		qrel[key] = {"d" + str(ref): 1}
	#print ("pred", hits[0], len(hits))
	for i, pred in enumerate(hits):
		#print ("pred", pred)
		key = "q" + str(i)	
		run[key] = {"d" + str(p): 10 - j for j,p in enumerate(pred)}
	return qrel, run

def evaluate_recall_at(predicted, true, n_labels):
	print (len(true), len(predicted))
	for recall_at in [1,5,10]:
		is_correct = []
		for i, t in enumerate(true):
			predictions = predicted[i][:recall_at]
			try:
				#predictions = [int(i) for i in predictions if not np.isnan(i)]
				predictions = [i for i in predictions]
			except Exception as e:
				print (str(e))
				print (predictions)
				input("")
			if t in predictions:
				is_correct.append(1)
			else:
				is_correct.append(0)
		print ("n labels dev", n_labels, "-- recall @ ", recall_at, np.round(np.mean(is_correct) * 100, 2))


def load_bm25_predictions(dev, test, labelid2passage):
	df = pd.read_csv(dev, delimiter="\t", header=None)
	df.columns = ["idx", "hit", "rank"]

	predicted_dev = defaultdict(list)
	for i,j in tqdm(zip(df["idx"].tolist(), df["hit"].tolist()), total=len(df)):
		predicted_dev[i].append(labelid2passage[str(j)])
	

	df = pd.read_csv(test, delimiter="\t", header=None)
	df.columns = ["idx", "hit", "rank"]

	predicted_test = defaultdict(list)
	for i,j in tqdm(zip(df["idx"].tolist(), df["hit"].tolist()), total=len(df)):
		predicted_test[i].append(labelid2passage[str(j)])

	predicted_dev = [predicted_dev[i] for i in range(len(predicted_dev))]
	predicted_test = [predicted_test[i] for i in range(len(predicted_test))]
	print (len(predicted_dev), len(predicted_test))
	return predicted_dev, predicted_test


def load_np_predictions(dev, test):
	with open(dev) as f:
		predicted_dev = json.load(f)["hits"]
	with open(test) as f:
		predicted_test = json.load(f)["hits"]		
	return predicted_dev, predicted_test

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--experiment', type=str, default='bert-classification')
	parser.add_argument('--n_labels', type=str, default="10000")
	parser.add_argument('--dev_predictions', type=str, default='...')
	parser.add_argument('--test_predictions', type=str, default='...')	
	args = parser.parse_args()
	dev, test, passage2labelid, labelid2passage = load_testsets(n_labels=args.n_labels)
	if args.experiment == "bm25":
		dev_preds, test_preds = load_bm25_predictions(args.dev_predictions, args.test_predictions, labelid2passage)
	else:
		dev_preds, test_preds = load_np_predictions(args.dev_predictions, args.test_predictions)
	# devset
	evaluate_recall_at(dev_preds, dev.passage_id.tolist(), args.n_labels)
	qrel, run = convert_to_trec_eval(dev.passage_id.tolist(), dev_preds)
	evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
	results = evaluator.evaluate(run)
	print ("devset ndcg", np.round(np.mean([i["ndcg"] for i in results.values()]), 4)  * 100) 
	print ("devset map", np.round(np.mean([i["map"] for i in results.values()]), 4)  * 100)

	# testset
	evaluate_recall_at(test_preds, test.passage_id.tolist(), args.n_labels)
	qrel, run = convert_to_trec_eval(test.passage_id.tolist(), test_preds)
	evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'ndcg'})
	results = evaluator.evaluate(run)
	print ("testset ndcg", np.round(np.mean([i["ndcg"] for i in results.values()]), 4)  * 100) 
	print ("testset map", np.round(np.mean([i["map"] for i in results.values()]), 4)  * 100)

	# python src/evaluate_run.py --dev_predictions bm25-files-10000/bm25_output_dev.tsv --test_predictions bm25-files-10000/bm25_output_test.tsv --experiment bm25
