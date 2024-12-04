import json
import os
import sys

from evaluate import load
from openai import OpenAI

from utils.file_utils import oai_get_batch_res, parse_filename
from utils.argument_parser import parse_args

_, dataset, _, _ = parse_args()

preds_dir = os.path.join("files", "preds")
out_dir = os.path.join("evaluation", "files", "indv")
os.makedirs(out_dir, exist_ok=True)
file_out_name = os.path.join(out_dir, f"eval_{dataset.tag}.json")

client = OpenAI()
oai_get_batch_res(client)

out_gts = dataset.get_gts()
all_rouge_scores = {}

rouge = load("rouge")

if os.path.exists(file_out_name):
    with open(file_out_name, "r") as f:
        all_rouge_scores = json.load(f)
else:
    all_rouge_scores = dict()

for file in os.listdir(preds_dir):

    if file.startswith(dataset.tag) and file.endswith(".json"):

        params = parse_filename(file, dataset.tag)
        print(f"Model: {params['model']}, Retriever: {params['retriever']}, Features: {params['features']}, RS: {params['RS']}, K: {params['k']}")

        if file[:-5] in all_rouge_scores.keys():
            print("Individual eval for this already concluded!")
            continue

        with open(os.path.join(preds_dir, file), "r") as f:
            preds = json.load(f)["golds"]
            preds = [p["output"] for p in preds]

        if len(preds) != len(out_gts):
            continue

        sys.stdout.flush()
        rouge_res = []
        for pred, gt in zip(preds, out_gts):
            
            score = rouge.compute(predictions=[pred], references=[gt])
            rouge_res.append(score)

        all_rouge_scores[file[:-5]] = {
            "params": params,
            "rouge1": [r["rouge1"] for r in rouge_res],
            "rouge2": [r["rouge2"] for r in rouge_res],
            "rougeL": [r["rougeL"] for r in rouge_res],
            "rougeLsum": [r["rougeLsum"] for r in rouge_res],
        }
        
        with open(file_out_name, "w") as f:
            json.dump(all_rouge_scores, f)