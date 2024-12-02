import json
import os
import argparse
import numpy as np
from exp_datasets import AmazonDataset
from typing import Dict, List, Any, Tuple
from retriever import Retriever
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="amazon_Grocery_and_Gourmet_Food_2018", type=str)
    return parser.parse_args()

def parse_dataset(dataset):
    if dataset.startswith("amazon"):
        year = int(dataset.split("_")[-1])
        category = "_".join(dataset.split("_")[1:-1])
        return AmazonDataset(category, year)
    else:
        raise Exception("Dataset not known!")

def get_model_and_k(exp_key: str) -> Tuple[str, int]:
    """Extract model name and k value from experiment key."""
    parts = exp_key.split("_")
    model_name = parts[-5]  # Model name is typically the 5th part from the end
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
        if (params.get('RS') == 1 and 
            params.get('features') == "" and
            params.get('retriever') == "contriever" and
            params.get('k') in [0, 10]):
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
            if 0 in k_preds and 10 in k_preds}

def analyze_distances(distances: List[float], model: str, k: int):
    """Analyze and plot the distribution of distances."""
    distances = np.array(distances)
    
    # Create output directory if it doesn't exist
    plot_dir = os.path.join("bfi", "files", "distance_distribution")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Calculate statistics
    print(f"\nAnalysis for {model} (k={k}):")
    print(f"Mean distance: {np.mean(distances):.4f}")
    print(f"Median distance: {np.median(distances):.4f}")
    print(f"Std deviation: {np.std(distances):.4f}")
    
    # Calculate percentiles
    percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        value = np.percentile(distances, p)
        print(f"{p}th percentile: {value:.4f}")
    
    return {
        'mean': np.mean(distances),
        'median': np.median(distances),
        'std': np.std(distances),
        'percentiles': {p: np.percentile(distances, p) for p in percentiles}
    }

def analyze_sample_changes(k0_distances: List[float], k10_distances: List[float], model: str):
    """Analyze how individual samples change when k is increased from 0 to 10."""
    
    # Calculate per-sample changes (k10 - k0)
    # Negative change means improvement (distance decreased)
    changes = np.array(k10_distances) - np.array(k0_distances)
    
    # Categorize changes
    improved = changes < 0  # Distance decreased (improved)
    worsened = changes > 0  # Distance increased (worsened)
    unchanged = np.isclose(changes, 0, atol=1e-6)  # Distance stayed same
    
    # Calculate statistics
    num_samples = len(changes)
    num_improved = np.sum(improved)
    num_worsened = np.sum(worsened)
    num_unchanged = np.sum(unchanged)
    
    # Calculate average improvement for different categories
    avg_improvement = -np.mean(changes[improved]) if any(improved) else 0  # Make positive for reporting
    avg_deterioration = np.mean(changes[worsened]) if any(worsened) else 0
    
    # Print analysis
    print(f"\nPer-sample analysis for {model}:")
    print(f"Total samples: {num_samples}")
    print(f"Improved samples: {num_improved} ({(num_improved/num_samples)*100:.1f}%)")
    print(f"Worsened samples: {num_worsened} ({(num_worsened/num_samples)*100:.1f}%)")
    print(f"Unchanged samples: {num_unchanged} ({(num_unchanged/num_samples)*100:.1f}%)")
    print(f"Average distance reduction when improved: {avg_improvement:.4f}")
    print(f"Average distance increase when worsened: {avg_deterioration:.4f}")
    
    # Create scatter plot of k=0 vs k=10 distances
    plt.figure(figsize=(10, 10))
    plt.scatter(k0_distances, k10_distances, alpha=0.5)
    
    # Add diagonal line (y=x)
    min_val = min(min(k0_distances), min(k10_distances))
    max_val = max(max(k0_distances), max(k10_distances))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='No change line')
    
    plt.title(f'K=0 vs K=10 Distances\n{model}')
    plt.xlabel('Distance with K=0')
    plt.ylabel('Distance with K=10')
    plt.legend()
    
    # Points above the line got worse (k10 > k0)
    # Points below the line improved (k10 < k0)
    
    plot_path = os.path.join("bfi", "files", "distance_distribution", f'scatter_{model}.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Create histogram of changes
    plt.figure(figsize=(10, 6))
    plt.hist(changes, bins=50, density=True)
    plt.axvline(0, color='r', linestyle='--', label='No change')
    plt.title(f'Distribution of Distance Changes\n{model}')
    plt.xlabel('Change in Distance (K10 - K0)\nNegative = Improved (distance decreased)')
    plt.ylabel('Density')
    plt.legend()
    
    plot_path = os.path.join("bfi", "files", "distance_distribution", f'changes_{model}.png')
    plt.savefig(plot_path)
    plt.close()
    
    return {
        'total_samples': num_samples,
        'improved': num_improved,
        'worsened': num_worsened,
        'unchanged': num_unchanged,
        'avg_improvement': avg_improvement,
        'avg_deterioration': avg_deterioration,
        'changes': changes.tolist()
    }

def compare_k_settings(model: str, k0_distances: List[float], k10_distances: List[float]):
    """Compare distance distributions between k=0 and k=10 settings."""
    k0_stats = analyze_distances(k0_distances, model, 0)
    k10_stats = analyze_distances(k10_distances, model, 10)
    
    # Calculate improvement metrics
    mean_improvement = k0_stats['mean'] - k10_stats['mean']
    median_improvement = k0_stats['median'] - k10_stats['median']
    
    print(f"\nImprovements for {model}:")
    print(f"Mean distance reduction: {mean_improvement:.4f}")
    print(f"Median distance reduction: {median_improvement:.4f}")
    
    # Perform statistical test
    t_stat, p_value = stats.ttest_ind(k0_distances, k10_distances)
    print(f"T-test p-value: {p_value:.4e}")
    
    # Analyze per-sample changes
    sample_analysis = analyze_sample_changes(k0_distances, k10_distances, model)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(k0_distances, bins=50, alpha=0.5, label=f'k=0 (mean={k0_stats["mean"]:.4f})', density=True)
    plt.hist(k10_distances, bins=50, alpha=0.5, label=f'k=10 (mean={k10_stats["mean"]:.4f})', density=True)
    
    # Add vertical lines for means
    plt.axvline(k0_stats['mean'], color='blue', linestyle='dashed', alpha=0.5)
    plt.axvline(k10_stats['mean'], color='orange', linestyle='dashed', alpha=0.5)
    
    plt.title(f'Distance Distribution Comparison\n{model}')
    plt.xlabel('Distance (1 - cosine similarity)')
    plt.ylabel('Density')
    plt.legend()
    
    plot_path = os.path.join("bfi", "files", "distance_distribution", f'comparison_{model}.png')
    plt.savefig(plot_path)
    plt.close()
    
    return {
        'k0_stats': k0_stats,
        'k10_stats': k10_stats,
        'mean_improvement': mean_improvement,
        'median_improvement': median_improvement,
        'p_value': p_value,
        'sample_analysis': sample_analysis
    }

def main():
    # Parse arguments
    args = get_args()
    dataset = parse_dataset(args.dataset)
    
    # File paths
    eval_file = f"/home/myazan1/AP-Bots/evaluation/files/indv/eval_{dataset.tag}.json"
    pred_dir = "/home/myazan1/AP-Bots/files/preds"
    
    # Load and filter evaluation results
    filtered_results = load_eval_results(eval_file)
    print(f"Found {len(filtered_results)} matching experiments")
    
    # Load predictions
    predictions = load_predictions(pred_dir, list(filtered_results.keys()))
    print(f"Loaded predictions for {len(predictions)} models with both k=0 and k=10")
    
    # Load ground truth
    ground_truth = dataset.get_gts()
    print(f"Loaded {len(ground_truth)} ground truth samples")
    
    # Initialize retriever
    retriever = Retriever(dataset)
    
    # Compare k settings for each model
    comparison_results = {}
    for model, k_preds in predictions.items():
        print(f"\nAnalyzing model: {model}")
        k0_distances = retriever.calculate_one_to_one_distances(k_preds[0], ground_truth)
        k10_distances = retriever.calculate_one_to_one_distances(k_preds[10], ground_truth)
        comparison_results[model] = compare_k_settings(model, k0_distances, k10_distances)
    
    # Print summary of improvements
    print("\nSummary of Improvements:")
    for model, results in comparison_results.items():
        print(f"\n{model}:")
        sample_analysis = results['sample_analysis']
        print(f"Mean distance reduction: {results['mean_improvement']:.4f}")
        print(f"Samples improved: {sample_analysis['improved']} ({(sample_analysis['improved']/sample_analysis['total_samples'])*100:.1f}%)")
        print(f"Samples worsened: {sample_analysis['worsened']} ({(sample_analysis['worsened']/sample_analysis['total_samples'])*100:.1f}%)")
        print(f"Average distance reduction when improved: {sample_analysis['avg_improvement']:.4f}")
        print(f"Average distance increase when worsened: {sample_analysis['avg_deterioration']:.4f}")
        print(f"Statistically significant: {results['p_value'] < 0.05}")

if __name__ == "__main__":
    main()
