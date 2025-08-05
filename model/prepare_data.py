import pandas as pd
import random
import json
from collections import Counter

def load_data(dataframe="data/top_10000_training_data_NAACL.csv.gz",  min_length=10, max_length=500):
	df = pd.read_csv(dataframe, compression="gzip")
	df = df.sample(frac=1.0, random_state=42)
	df["seq_lengths"] = [len(i.split()) for i in df.destination_context]
	df = df[df.seq_lengths >= min_length]
	df = df[df.seq_lengths <= max_length]
	
	passage2labelid = {i:j for j,i in enumerate(df.passage_id.unique())}
	labelid2passage = {j:i for i,j in passage2labelid.items()}

	print (len(df))
	print ("unique labels", df.passage_id.nunique())

	train_split = int(len(df) * 0.9)
	dev_split = int(len(df) * 0.95)
	train = df.iloc[:train_split]
	dev = df.iloc[train_split:dev_split]
	test = df.iloc[dev_split:]
	print (len(train), len(dev), len(test))
	return train, dev, test, passage2labelid, labelid2passage
	
	
def load_datasets(n_labels="10000"):
	train = pd.read_csv("data/trainset_top_" + n_labels + ".csv.gz", compression='gzip')
	dev = pd.read_csv("data/devset_top_" + n_labels + ".csv.gz", compression='gzip')
	test = pd.read_csv("data/testset_top_" + n_labels + ".csv.gz", compression='gzip')
	with open("data/passage2labelid_top_" + n_labels + ".json") as f:
		passage2labelid = json.load(f)
	with open("data/labelid2passage_top_" + n_labels + ".json") as f:
		labelid2passage = json.load(f)
	return train, dev, test, passage2labelid, labelid2passage

def load_testsets(n_labels="10000"):
	#train = pd.read_csv("data/trainset_top_" + n_labels + ".csv.gz", compression='gzip')
	dev = pd.read_csv("data/devset_top_" + n_labels + ".csv.gz", compression='gzip')
	test = pd.read_csv("data/testset_top_" + n_labels + ".csv.gz", compression='gzip')
	with open("data/passage2labelid_top_" + n_labels + ".json") as f:
		passage2labelid = json.load(f)
	with open("data/labelid2passage_top_" + n_labels + ".json") as f:
		labelid2passage = json.load(f)
	return dev, test, passage2labelid, labelid2passage


if __name__ == "__main__":
	for n_labels in ["10000", "20000", "50000"]:
		train, dev, test, passage2labelid, labelid2passage = load_data(dataframe="data/top_" + n_labels + "_data.csv.gz")
		
		train.to_csv("data/trainset_top_" + n_labels + ".csv.gz", index=False, compression='gzip')
		dev.to_csv("data/devset_top_" + n_labels + ".csv.gz", index=False, compression='gzip')
		test.to_csv("data/testset_top_" + n_labels + ".csv.gz", index=False, compression='gzip')

		with open("data/passage2labelid_top_" + n_labels + ".json", "w") as outfile:
			json.dump(passage2labelid, outfile)
		with open("data/labelid2passage_top_" + n_labels + ".json", "w") as outfile:
			json.dump(labelid2passage, outfile)

