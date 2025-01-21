import json
import os
import numpy as np

from AP_Bots.utils.argument_parser import parse_args

_, dataset, _, _ = parse_args()

rand_k = 20
preds_dir = "files/preds"

out_gts = dataset.get_gts()
rand_samples = np.random.choice(range(len(out_gts)), rand_k, replace=False)

model_samples = {}
for file in os.listdir(preds_dir):
    if file.startswith(dataset.tag) and file.endswith(".json") and "WF" in file:
        with open(os.path.join(preds_dir, file), "r") as f:
            preds = json.load(f)["golds"]
            preds = [p["output"] for p in preds]
        if len(preds) != len(out_gts):
            continue

        model_samples[file[len(dataset.tag)+1:-5]] = [preds[idx] for idx in rand_samples]

for i, sample in enumerate(rand_samples):
    print()
    print(f"GT: {out_gts[sample]}")
    for model in model_samples.keys():
        print()
        print(model)
        print(model_samples[model][i])