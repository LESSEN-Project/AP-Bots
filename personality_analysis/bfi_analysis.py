import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from utils.argument_parser import get_args, parse_dataset


def get_model_and_k(exp_key: str) -> Tuple[str, int]:
    """Extract model name and k value from experiment key."""
    parts = exp_key.split("_")
    model_name = parts[-5] 
    k = exp_key.split("K(")[-1].split(")")[0]
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
            params.get('model') in ['GEMMA-2-9B', 'GEMMA-2-27B', 'LLAMA-3.1-8B', 'LLAMA-3.1-70B']):
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

def load_bfi(bfi_file, experiment_keys):

    bfi_predictions = defaultdict(dict)

    with open(bfi_file, "r") as f:
        bfi_res = json.load(f) 

    up_exp_name = f"{dataset.tag}_UP"
    up_exp = pd.DataFrame(bfi_res[up_exp_name]["bfi"])

    bfi_res = {key: bfi_res[key] for key in experiment_keys if key in bfi_res.keys()}

    for key in bfi_res.keys():
        if key in experiment_keys:
            model_name, k = get_model_and_k(key)
            bfi_predictions[model_name][k] = pd.DataFrame(bfi_res[key]["bfi"])

    return up_exp, bfi_predictions

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

bfi_model="GEMMA-2-27B"
bfi_file = os.path.join("personality_analysis", "files", "bfi_results", f"{bfi_model}_{dataset.tag}.json")
up_bfi_results, exp_bfi_results = load_bfi(bfi_file, list(eval_results.keys()))

for key in up_bfi_results.columns:
    print(up_bfi_results[key].value_counts())

# Create directory for saving plots
visuals_dir = os.path.join("personality_analysis", "files", "visuals", "bfi_analysis")
os.makedirs(visuals_dir, exist_ok=True)

print("BFI summary for User profiles:")

for model_key in exp_bfi_results:
    for k_key in exp_bfi_results[model_key]:

        df = exp_bfi_results[model_key][k_key]
        print(f"BFI analysis for {model_key, k_key}:")
        k_exp_key = [k for k in eval_results.keys() if get_model_and_k(k)[0] == model_key and get_model_and_k(k)[1] == k_key][0]
        rougeL = eval_results[k_exp_key]['rougeL']

        # Adding Visualizations
        for trait in df.columns:
            
            f_value, p_value = stats.f_oneway(rougeL, up_bfi_results[trait])
            print(f"ANOVA between ROUGE-L and user {trait}: F({f_value:.4f}), p({p_value:.4e})")

            trait_diff = int(df[trait].mean()) - up_bfi_results[trait]
            f_value, p_value = stats.f_oneway(rougeL, trait_diff)
            print(f"ANOVA between ROUGE-L and {trait} diff: F({f_value:.4f}), p({p_value:.4e})")

            # Plotting Box Plot to visualize distribution
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=up_bfi_results[trait], y=rougeL)
            plt.title(f'Box Plot of ROUGE-L vs {trait}')
            plt.xlabel(f'{trait} Scores')
            plt.ylabel('ROUGE-L')
            plt.savefig(os.path.join(visuals_dir, f'boxplot_rougeL_vs_{trait}_{model_key}_{k_key}.png'))
            plt.close()

            # Plotting Violin Plot for detailed distribution and density
            plt.figure(figsize=(10, 6))
            sns.violinplot(x=up_bfi_results[trait], y=rougeL)
            plt.title(f'Violin Plot of ROUGE-L vs {trait}')
            plt.xlabel(f'{trait} Scores')
            plt.ylabel('ROUGE-L')
            plt.savefig(os.path.join(visuals_dir, f'violinplot_rougeL_vs_{trait}_{model_key}_{k_key}.png'))
            plt.close()

            # Scatter Plot with jitter to observe potential patterns
            plt.figure(figsize=(10, 6))
            sns.stripplot(x=up_bfi_results[trait], y=rougeL, jitter=True, alpha=0.6)
            plt.title(f'Scatter Plot of ROUGE-L vs {trait} (with jitter)')
            plt.xlabel(f'{trait} Scores')
            plt.ylabel('ROUGE-L')
            plt.savefig(os.path.join(visuals_dir, f'scatterplot_rougeL_vs_{trait}_{model_key}_{k_key}.png'))
            plt.close()
