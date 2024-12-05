import json
import os
import sys

from evaluate import load
from openai import OpenAI

from utils.file_utils import parse_filename, oai_get_batch_res
from utils.argument_parser import parse_args
from utils.output_parser import extract_bfi_scores

_, dataset, _, _ = parse_args()
bfi_model = "GEMMA-2-27B"

bfi_dir = os.path.join("personality_analysis", "files", "inferred_bfi")
out_dir = os.path.join("personality_analysis", "files", "bfi_results")
os.makedirs(out_dir, exist_ok=True)

client = OpenAI()
oai_get_batch_res(client, pred_path=bfi_dir)

file_out_name = os.path.join(out_dir, f"{bfi_model}_{dataset.tag}.json")
gt_len = len(dataset.get_gts())

all_bfi_results = {}

if os.path.exists(file_out_name):
    with open(file_out_name, "r") as f:
        all_bfi_results = json.load(f)
else:
    all_bfi_results = dict()

for file in os.listdir(bfi_dir):

    if file.startswith(dataset.tag) and file.endswith(".json"):

        processed_name = file[:-5]
        processed_name = "_".join(processed_name.split("_")[:-2]) 
        bfi_model = processed_name.split("_")[-1]

        if processed_name == f"{dataset.tag}_UP":

            params = {}

        else:

            params = parse_filename(processed_name, dataset.tag)
            print(f"Model: {params['model']}, Retriever: {params['retriever']}, Features: {params['features']}, RS: {params['RS']}, K: {params['k']}")

        if file[:-5] in all_bfi_results.keys():
            print("Individual eval for this already concluded!")
            continue

        preds = []
        with open(os.path.join(bfi_dir, file), "r") as f:
            bfi_scores = json.load(f)
            preds = [extract_bfi_scores(score) for score in bfi_scores]

        if len(preds) != gt_len:
            continue

        sys.stdout.flush()

        all_bfi_results[processed_name] = {
            "params": params,
            "bfi": preds
        }
        
        with open(file_out_name, "w") as f:
            json.dump(all_bfi_results, f)