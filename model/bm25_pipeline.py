import pandas as pd
import random
import json
from collections import Counter
from prepare_data import load_datasets
import os

if __name__ == "__main__":
	
	with open("data/passage_dict.json") as f:
		passage_dict = json.load(f)["data"]
	# yea...
	for n_labels in ["10000", "20000", "50000"]:
		outpath = "bm25-files-" + n_labels
		os.makedirs(outpath, exist_ok=True)

		train, dev, test, passage2labelid, labelid2passage = load_datasets(n_labels=n_labels)

		with open(os.path.join(outpath, "docs00.json"), "w") as outfile:
			for c, i in labelid2passage.items():
				target = passage_dict[str(i)]
				out = {}
				out["contents"] = target
				out["id"] = c
				json.dump(out, outfile)
				outfile.write("\n")

		with open(os.path.join(outpath, "bm25_input_dev_" + n_labels + ".tsv"), "w") as outfile:
			counter = 0
			for i, row in dev.iterrows():
				outfile.write(str(counter) + "\t" + " ".join(row["destination_context"].split()[:500]) + "\n")
				counter += 1

		with open(os.path.join(outpath, "bm25_input_test_" + n_labels + ".tsv"), "w") as outfile:
			counter = 0
			for i, row in test.iterrows():
				outfile.write(str(counter) + "\t" + " ".join(row["destination_context"].split()[:500]) + "\n")
				counter += 1
	# run a few anserini commands afterwards		
