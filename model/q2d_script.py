import pathlib, os
import logging
import argparse 
import random 
import asyncio
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)
from openai import OpenAI

import json 
import time 
import requests 
import tiktoken
import pandas as pd
from tqdm.asyncio import tqdm
import bm25s
import Stemmer 
import multiprocessing
from functools import partial

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

instruction="""Write a following legal passage that is coherent, relevant, and contextually appropriate based on preceding context.

Examples:
### Preceding Context: {context1}
### Legal Passage:{passage1}

### Preceding Context: {context2}
### Legal Passage: {passage2}

### Preceding Context: {context3}
### Legal Passage: {passage3}

Query:
### Preceding Context: {preceding_context}
### Legal Passage: """

CoT_prompt="""You are tasked with writing a coherent, relevant, and contextually appropriate legal passage by following the chain of thought process.
### Note: Examples provided below do not include intermediate steps due to sampling constraints.
### Step 1: Understand the preceding context.
### Step 2: Identify the key legal elements and principles required for coherence.
### Step 3: Generate a legal passage that logically follows and aligns with the context.
### Note: You can generate any intermediate step but, please mark final output with '<output>' tag.

Examples:
### Preceding Context: 3\n.Rule 10b-5, entitled “Employment of manipulative or deceptive device,” makes it unlawful “
### Step 1: We first identify Rule 10b-5’s focus: it prohibits fraud and deceit in securities transactions. 
### Step 2: The phrase “manipulative or deceptive device” signals a prohibition on misleading statements or omissions. Given this context, the key legal language typically includes the requirement that no material facts be misstated or omitted in a manner that would mislead reasonable investors. Because 10b-5 is commonly summarized by referencing the duty not to make materially false or misleading statements, the final quotation should capture this essence.
### Step 3: <output> [t]o make any untrue statement of a material fact or to omit to state a material fact necessary in order to make the statements made, in the light of the circumstances under which they were made, not misleading.

### Preceding Context: As previously discussed, it will also submit for in camera review an affidavit stating whether or not information withheld under the work product prong of Exemption 5 was incorporated or adopted into ICE policy-\n5. CIS\nCIS withholds information under Exemptions 2, 5, 6, 7(C), and 7(E). El Ba- drawi does not contest CIS’s withholdings under Exemptions 2 (“low 2”), 6, and 7(C). See Plaintiffs Mem. in Opp. to CIS at 1 n. 1. Thus, Exemptions “high 2”, 5, and 7(E) are at issue.\nBecause the Vaughn indices and supporting affidavit submitted by CIS in justification of Exemptions 2 and 7(E) are essentially equivalent in form and level of detail to those submitted by the other agency defendants, the analysis of their sufficiency follows a similar course. The vast majority of CIS’s Vaughn indices consist of categorical descriptions of withheld documents, restatements of the statutory language of the FOIA exemptions, and categorical indications of the consequences of disclosing the documents. In brief, CIS has not applied, in detail, the terms of the exemptions to the specific facts of the records at hand. Consequently, CIS’s Motion for Summary Judgment is denied as to Exemptions 2 and 7(E) because the court is unable to engage in the required de novo review of the propriety of the with-holdings.\nFinally, with respect to CIS’s invocation Exemption 5, CIS relies on the same erroneous understanding of the deliberative process privilege as ICE. See supra, Part 111(C)(4) at 49-51. Thus, CIS’s Motion for Summary Judgment is denied as to Exemption 5. For the reasons noted above, CIS will submit for the court’s in camera review all documents withheld in full or in part. Reasonableness of Agency Segrega-bility Determinations\nFOIA requires that “
### Step 1: Here, the context is a discussion of various FOIA exemptions (2, 5, and 7(E)) and how the agency must justify its withholdings.
### Step 2: The court requires a “de novo review” to ensure that exemptions are properly invoked. Because the agency’s justifications are too broad or conclusory, the motion for summary judgment is denied for certain exemptions. Additionally, FOIA mandates that any non-exempt information within a record be disclosed, a principle referred to as “segregability.” That statutory requirement is typically articulated as providing “any reasonably segregable” portions of a record after the exempt content is removed.
### Step 3: <output> [a]ny reasonably segregable portion of a record shall be provided to any person requesting such record after deletion of the portions which are exempt under this subsection.

### Preceding Context: Third, Valdez’ criminal history indicates a relative lack of variety in his criminal activities. Valdez has only been convicted of a small amount of property crime, and no violent crime; virtually all of his criminal activity has been drug related. This factor follows § 3553(a)(l)’s requirement that the Court consider “
### Step 1: We observe this context focuses on sentencing considerations. 
### Step 2: Section 3553(a)(1) instructs the court to examine “the nature and circumstances of the offense and the history and characteristics of the defendant.” Given the mention of Valdez’s prior convictions and the relatively narrow scope of his criminal history, the next logical part is the direct citation to § 3553(a)(1), which references these factors explicitly. Thus, we quote the statutory language that the court uses in making its sentencing determinations.
### Step 3: <output> the history and characteristics of the defendant.

Query:
### Preceding Context: {preceding_context}
### Step 1: """


logger = logging.getLogger(__name__)

def load_api_key(file_path):
    with open(file_path, "r") as f:
        api_key = f.read().strip()
    return api_key


class ChatGPTParallelClient:
    def __init__(self, api_key_path, model_name, max_parallel_calls=20):
        self.api_key = load_api_key(api_key_path)
        self.model_name = model_name
        self.max_parallel_calls = max_parallel_calls
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), before_sleep=print, retry_error_callback=lambda _: None)
    async def get_completion(self, prompt, model_name, session, semaphore):
        async with semaphore:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=self.headers, json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 800,
                "top_p": 0.9
            }) as resp:

                response_json = await resp.json()
                pred = response_json["choices"][0]['message']["content"]
                # Post-processing
                pred = pred.strip()
                return pred

    async def get_completion_list(self, prompts_list):
        semaphore = asyncio.Semaphore(value=self.max_parallel_calls)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(30)) as session:
            return await tqdm.gather(*[self.get_completion(prompt, self.model_name, session, semaphore) for prompt in prompts_list])

    def get_completion_results(self, prompts_list):
        results = asyncio.run(self.get_completion_list(prompts_list))
        return results




def main(args):
    logger.info(args)
    assert os.path.exists(args.key), "Please put your OpenAI API key in the file: {}".format(args.key)
    OPENAI_API_KEY = load_api_key(args.key)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    # Load data files
    print(f"Loading input data: {args.input_csv}")
    data = pd.read_csv(args.input_csv)
    
    print(f"Loading passage mappings: {args.passage2labelid}")
    with open(args.passage2labelid) as f:
        passage2labelid = json.load(f)
    
    print(f"Loading label mappings: {args.labelid2passage}")
    with open(args.labelid2passage) as f:
        labelid2passage = json.load(f)
    
    print(f"Loading training data: {args.train_data}")
    train_data = pd.read_csv(args.train_data, compression='gzip')




    q2d_prompts = []
    #q2d_cot_prompts = []

    print(f"Loading document corpus: {args.docs_json}")
    df_targets = pd.read_json(args.docs_json, lines=True)
    corpus = df_targets.contents.tolist()
    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens,show_progress=False)

    dc = data.destination_context.tolist()
    passage_id = data.passage_id.tolist()
    train_passage_set = set(train_data.passage_id.tolist())
    for i in range(len(dc)):
        origin_passage = df_targets.iloc[passage2labelid[str(passage_id[i])]]
        dc_tokens = bm25s.tokenize([dc[i]], stemmer=stemmer)
        results_dc, scores = retriever.retrieve(dc_tokens, k=10)
        temp_exm=[]
        for j in range(10):
            if (results_dc[0][j] != passage2labelid[str(passage_id[i])]) and len(temp_exm)<3:
                temp_exm.append(results_dc[0][j])
        # find samples in the training set
        temp_texts=[]
        for j in temp_exm:
            passage_id_j = labelid2passage[str(j)]
            if passage_id_j in train_passage_set:
                # Find the index first
                idx = train_data[train_data.passage_id == passage_id_j].index[0]
                temp_texts.append((train_data.at[idx, "destination_context"],
                                   df_targets.at[passage2labelid[str(train_data.at[idx, "passage_id"])], "contents"])) 
                
        q2d_prompts.append(pp_prompt.format(
                                            context1 = temp_texts[0][0],
                                            passage1 =  temp_texts[0][1],
                                            context2 = temp_texts[1][0],
                                            passage2 = temp_texts[1][1],
                                            context3 = temp_texts[2][0],
                                            passage3 = temp_texts[2][1],
                                            preceding_context=dc[i]))
        logger.info(f"Total prompts: {len(q2d_prompts)}")
        

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(20), before_sleep=print, retry_error_callback=lambda _: None)
    async def get_completion(datapoint, model_name, session, semaphore):
        async with semaphore:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
                "model": model_name,
                "messages": [{"role": "user", "content": datapoint}],
                "temperature": 0,
                "max_tokens": 800,
                "top_p": 0.9
            }) as resp:

                response_json = await resp.json()

                pred = response_json["choices"][0]['message']["content"]
                # Post-processing
                pred = pred.strip()
                return pred

    async def get_completion_list(datapoints, max_parallel_calls):
        semaphore = asyncio.Semaphore(value=max_parallel_calls)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(30)) as session:
            return await tqdm.gather(*[get_completion(datapoint, args.model_name, session, semaphore) for datapoint in datapoints])

    logger.info(f"Start inference..")
    results = asyncio.run(get_completion_list(q2d_prompts, args.max_parallel_calls))
    final = []
    for item in results:
        text = item.split("### Key Legal Elements:")[-1]
        final.append(text)


    # Create output directories
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    print(f"Saving JSON results to: {args.output_json}")
    with open(args.output_json, "w") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)
    
    print(f"Saving CSV results to: {args.output_csv}")
    data['rewritten_query'] = final
    data.to_csv(args.output_csv, index=False)
    print("Q2D processing completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q2D (Query-to-Document) Script for Legal Passage Generation")

    # Data generation arguments 
    parser.add_argument("--key", type=str, required=True, help="Path to the OpenAI API key file")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="OpenAI model to be used for generation (default: gpt-4o-mini)")
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Input CSV file path")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output CSV file path")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output JSON file path")
    parser.add_argument("--passage2labelid", type=str, required=True,
                        help="Path to passage2labelid JSON file")
    parser.add_argument("--labelid2passage", type=str, required=True,
                        help="Path to labelid2passage JSON file")
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data CSV file")
    parser.add_argument("--docs_json", type=str, required=True,
                        help="Path to documents JSON file")
    
    # Parallelism arguments
    parser.add_argument("--max_parallel_calls", type=int, default=20, 
                        help="Maximum parallel calls for the API (default: 20)")

    args = parser.parse_args()
    main(args)
