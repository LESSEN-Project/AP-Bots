import json
import os
import numpy as np

from utils import get_args, parse_dataset, parse_cot_output

args = get_args()
dataset = parse_dataset(args.dataset)

rand_k = 20
preds_dir = "preds"

out_gts = dataset.get_gts()
rand_samples = np.random.choice(range(len(out_gts)), rand_k, replace=False)

model_samples = {}
for file in os.listdir(preds_dir):
    if file.startswith(args.dataset) and file.endswith(".json") and "K(0)" in file:
        with open(os.path.join(preds_dir, file), "r") as f:
            preds = json.load(f)["golds"]
            preds = [p["output"] for p in preds]
            if "CoT" in file:
                preds = [parse_cot_output(p) for p in preds]
        if len(preds) != len(out_gts):
            continue

        model_samples[file[len(args.dataset)+1:-5]] = [preds[idx] for idx in rand_samples]

for i, sample in enumerate(rand_samples):
    print()
    print(f"GT: {out_gts[sample]}")
    for model in model_samples.keys():
        print()
        print(model)
        print(model_samples[model][i])