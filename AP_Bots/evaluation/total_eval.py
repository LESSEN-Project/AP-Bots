import json
import os

from evaluate import load
from openai import OpenAI
import pandas as pd

from AP_Bots.utils.file_utils import oai_get_batch_res, parse_filename
from AP_Bots.utils.argument_parser import parse_args
from AP_Bots.utils.output_parser import parse_react_output, parse_r1_output

_, dataset, _, _ = parse_args()

preds_dir = os.path.join("files", "preds")
out_dir = os.path.join("evaluation", "files", "total")
os.makedirs(out_dir, exist_ok=True)

client = OpenAI()
oai_get_batch_res(client)

out_gts = dataset.get_gts()
all_res = []
models = []
cols = ["model", "features", "retriever", "RS", "k", "PS"]

rouge = load("rouge")
bleu = load("bleu")
meteor = load("meteor")
cols.extend(["rouge1", "rouge2", "rougeL", "rougeLsum", "bleu", "meteor"])

for file in os.listdir(preds_dir):

    if file.startswith(dataset.tag) and file.endswith(".json"):

        with open(os.path.join(preds_dir, file), "r") as f:
            preds = json.load(f)["golds"]
            preds = [p["output"] for p in preds]

        if len(preds) != len(out_gts):
            continue

        params = parse_filename(file, dataset.tag)
        print(f"Model: {params['model']}, Retriever: {params['retriever']}, Features: {params['features']}, RS: {params['RS']}, K: {params['k']}, PS: {params['PS']}")

        if params["PS"] == "react":
            preds = [parse_react_output(p) for p in preds]

        if params["model"].startswith("R1"):
            preds = [parse_r1_output(p)[1] for p in preds]

        rouge_results = rouge.compute(predictions=preds, references=out_gts)
        bleu_results = bleu.compute(predictions=preds, references=[[gt] for gt in out_gts])
        meteor_results = meteor.compute(predictions=preds, references=[[gt] for gt in out_gts])
        res_dict = params | rouge_results
        res_dict["bleu"] = bleu_results["bleu"]
        res_dict["meteor"] = meteor_results["meteor"]
        
        all_res.append(res_dict)

df = pd.DataFrame(all_res)
df = df[cols]
df = df.round(dict([(c, 4) for c in df.columns if df[c].dtype == "float64"]))
df.to_csv(os.path.join(out_dir, f"eval_{dataset.tag}.csv"), index=False, columns=cols)