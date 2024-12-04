import os
import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from utils.argument_parser import get_args, parse_dataset


def get_model_and_k(exp_key: str) -> Tuple[str, int]:
    """Extract model name and k value from experiment key."""
    parts = exp_key.split("_")
    model_name = parts[-5] 
    k = int(exp_key.split("K(")[-1].split(")")[0])
    return model_name, k


def load_eval_results(eval_file_path: str) -> Dict[str, Any]:
    """Load and filter evaluation results based on specific parameters."""
    with open(eval_file_path, 'r') as f:
        eval_data = json.load(f)

    # Filter experiments based on criteria
    filtered_results = {}
    for key, value in eval_data.items():
        params = value.get('params', {})
        if (params.get('RS') == '1' and
            params.get('features') == "" and
            params.get('retriever') == "contriever" and
            params.get('k') in ['0', '10'] and
            params.get('model_name') in ['GEMMA-2-9B', 'GEMMA-2-27B', 'LLAMA-3.1-8B', 'LLAMA-3.1-70B']):
            filtered_results[key] = value

    return filtered_results


def load_predictions(pred_dir: str, experiment_keys: List[str]) -> Dict[str, Dict[int, List[str]]]:
    """Load predictions and organize them by model and k value."""
    predictions = defaultdict(dict)  # model -> k -> predictions

    for exp_key in experiment_keys:
        pred_file = os.path.join(pred_dir, f"{exp_key}.json")
        if os.path.exists(pred_file):
            model_name, k = get_model_and_k(exp_key)
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
                # Extract predictions from the golds list
                preds = []
                for item in pred_data.get('golds', []):
                    if isinstance(item, dict) and 'output' in item:
                        preds.append(item['output'])
                predictions[model_name][k] = preds

    # Keep only models that have both k=0 and k=10
    return {model: k_preds for model, k_preds in predictions.items()
            if '0' in k_preds and '10' in k_preds}

def load_bfi(bfi_dir, dataset, bfi_model="GEMMA-2-27B"):

    pass    

args = get_args()
dataset = parse_dataset(args.dataset)

# Load evaluation results
eval_file = os.path.join("evaluation", "files", "indv", f"eval_{args.dataset}.json")
eval_results = load_eval_results(eval_file)

# Load predictions
pred_dir = os.path.join("files", "preds")
predictions = load_predictions(pred_dir, list(eval_results.keys()))

# Load ground truth
ground_truth = dataset.get_gts()
print(f"Loaded {len(ground_truth)} ground truth samples")

bfi_dir = os.path.join("personality_analysis", "files", "inferred_bfi")