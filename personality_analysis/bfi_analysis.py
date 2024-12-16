import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats
from collections import defaultdict

from personality_analysis.analysis_utils import load_eval_results, load_predictions, get_model_and_k, get_exp_eval_results
from utils.argument_parser import get_args, parse_dataset


def load_bfi(bfi_file, experiment_keys):

    print(experiment_keys)
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

bfi_model="LLAMA-3.3-70B"
bfi_file = os.path.join("personality_analysis", "files", "bfi_results", f"{bfi_model}_{dataset.tag}.json")
up_bfi_results, exp_bfi_results = load_bfi(bfi_file, list(eval_results.keys()))
# up_bfi_results = up_bfi_results.astype(int)

for key in up_bfi_results.columns:
    print(f"Number of infinite values: {np.sum(np.isinf(up_bfi_results[key]))}")
    print(f"Number of NA values: {up_bfi_results[key].isna().sum()}")
    
    # Fill NA values with mean
    if up_bfi_results[key].isna().any():
        up_bfi_results[key] = up_bfi_results[key].fillna(up_bfi_results[key].mean())
        
    up_bfi_results[key] = up_bfi_results[key].astype(int)

# Create directory for saving plots
visuals_dir = os.path.join("personality_analysis", "files", "visuals", "bfi_analysis")
os.makedirs(visuals_dir, exist_ok=True)

print("BFI summary for User profiles:")
print(exp_bfi_results)
for model_key in exp_bfi_results:
    for k_key in exp_bfi_results[model_key]:

        df = exp_bfi_results[model_key][k_key]
        print(f"BFI analysis for {model_key, k_key}:")
        rougeL = get_exp_eval_results(eval_results, model_key, k_key)
        rougeL = [round(r*100, 1) for r in rougeL]
        # print(pd.Series(rougeL).describe())

        # Adding Visualizations
        for trait in df.columns:
            
            # f_value, p_value = stats.f_oneway(rougeL, up_bfi_results[trait])
            # print(f"ANOVA between ROUGE-L and {trait}: F({f_value:.4f}), p({p_value:.4e})")

            trait_diff = abs(df[trait] - up_bfi_results[trait])
            trait_diff = trait_diff.astype('category')
            # ANOVA analysis
            f_value, p_value = stats.f_oneway(rougeL, trait_diff)
            print(f"ANOVA between ROUGE-L and {trait} diff: F({f_value:.4f}), p({p_value:.4e})")
            
            # Correlation analysis
            corr_coef, corr_p = stats.pearsonr(rougeL, trait_diff.astype(float))
            print(f"Correlation between ROUGE-L and {trait} diff: r({corr_coef:.4f}), p({corr_p:.4e})")

            # Plotting Box Plot to visualize distribution
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=trait_diff, y=rougeL)
            plt.title(f'Box Plot of ROUGE-L vs {trait}')
            plt.xlabel(f'{trait} Scores')
            plt.ylabel('ROUGE-L')
            plt.savefig(os.path.join(visuals_dir, f'boxplot_rougeL_vs_{trait}_diff_{model_key}_{k_key}.png'))
            plt.close()

            # Plotting Violin Plot for detailed distribution and density
            plt.figure(figsize=(10, 6))
            sns.violinplot(x=trait_diff, y=rougeL)
            plt.title(f'Violin Plot of ROUGE-L vs {trait}')
            plt.xlabel(f'{trait} Scores')
            plt.ylabel('ROUGE-L')
            plt.savefig(os.path.join(visuals_dir, f'violinplot_rougeL_vs_{trait}_diff_{model_key}_{k_key}.png'))
            plt.close()

            # Scatter Plot with jitter to observe potential patterns
            plt.figure(figsize=(10, 6))
            sns.stripplot(x=trait_diff, y=rougeL, jitter=True, alpha=0.6)
            plt.title(f'Scatter Plot of ROUGE-L vs {trait} (with jitter)')
            plt.xlabel(f'{trait} Scores')
            plt.ylabel('ROUGE-L')
            plt.savefig(os.path.join(visuals_dir, f'scatterplot_rougeL_vs_{trait}_diff_{model_key}_{k_key}.png'))
            plt.close()
