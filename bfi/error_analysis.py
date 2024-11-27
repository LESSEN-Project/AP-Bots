import json
import os

from evaluate import load

from utils.argument_parser import parse_args
from utils.misc import get_model_list

args, dataset, final_feature_list, k = parse_args()
pred_path = os.path.join("files", "preds")
out_path = os.path.join("bfi", "files", "indv_scores")
os.makedirs(out_path, exist_ok=True)

all_models = get_model_list()
out_gts = dataset.get_gts()
rouge = load("rouge")

all_rouge_scores = {}
for model_name in all_models:

    exp_name = f"{dataset.tag}_{model_name}_{final_feature_list}_{args.retriever}_RS({args.repetition_step})_K({k}))"
    print(exp_name)
    pred_out_path = os.path.join(pred_path, f"{exp_name}.json")

    with open(pred_out_path, "r") as f:
        preds = json.load(f)["golds"]
        preds = [p["output"] for p in preds]

    if len(preds) != len(out_gts):
        print("Predictions for this experiment is not concluded yet!")
        continue

    rouge_res = []
    for pred, gt in zip(preds, out_gts):
        
        score = rouge.compute(predictions=[pred], references=[gt])
        rouge_res.append(score)

    all_rouge_scores[model_name] = rouge_res

    with open(os.path.join(out_path, f"{dataset.tag}_{final_feature_list}_{args.retriever}_RS({args.repetition_step})_K({k})).json"), "w") as f:
        json.dump(all_rouge_scores, f)