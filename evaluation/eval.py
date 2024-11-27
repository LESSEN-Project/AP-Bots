import json
import os
import re

from evaluate import load
from openai import OpenAI
import pandas as pd

from utils.file_utils import oai_get_batch_res
from utils.argument_parser import parse_args

_, dataset, _, _ = parse_args()

preds_dir = os.path.join("files", "preds")
out_dir = os.path.join("evaluation", "files")
client = OpenAI()
oai_get_batch_res(client)

out_gts = dataset.get_gts()
all_res = []
models = []
cols = ["model", "features", "retriever", "k", "RS"]

rouge = load("rouge")
bleu = load("bleu")
cols.extend(["rouge1", "rouge2", "rougeL", "rougeLsum", "bleu"])

for file in os.listdir(preds_dir):

    if file.startswith(dataset.tag) and file.endswith(".json"):

        params = file[len(dataset.tag)+1:-5].split("_")

        k = re.findall(r'\((.*?)\)', params[-1])[0]
        retriever = params[2]
        features = params[1]
        if features != "None":
            features = ":".join(eval(features))
        model = params[0]
        rs = re.findall(r'\((.*?)\)', params[-2])[0]

        with open(os.path.join(preds_dir, file), "r") as f:
            preds = json.load(f)["golds"]
            preds = [p["output"] for p in preds]

        if len(preds) != len(out_gts):
            continue

        print(f"Model: {model}, Retriever: {retriever}, Features: {features}, RS: {rs}, K: {k}")

        res_dict = {    
            "model": model,
            "features": features,
            "retriever": retriever,
            "k": k,
            "RS": rs
        }

        rouge_results = rouge.compute(predictions=preds, references=out_gts)
        bleu_results = bleu.compute(predictions=preds, references=[[gt] for gt in out_gts])
        res_dict = res_dict | rouge_results
        res_dict["bleu"] = bleu_results["bleu"]
        
        all_res.append(res_dict)

df = pd.DataFrame(all_res)
df = df[cols]
df = df.round(dict([(c, 4) for c in df.columns if df[c].dtype == "float64"]))
df.to_csv(os.path.join(out_dir, f"eval_{dataset.tag}.csv"), index=False, columns=cols)