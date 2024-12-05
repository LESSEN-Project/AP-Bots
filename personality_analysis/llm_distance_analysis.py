import json
import os
import numpy as np
import pandas as pd

from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

from retriever import Retriever
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


def analyze_distances(distances: List[float], model: str, k: int):
    """Analyze and plot the distribution of distances."""
    distances = np.array(distances)

    # Create output directory if it doesn't exist
    plot_dir = os.path.join("personality_analysis", "files", "visuals", "distance_distribution")
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

    plot_path = os.path.join("personality_analysis", "files", "visuals", "distance_distribution", f'scatter_{model}.png')
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

    plot_path = os.path.join("personality_analysis", "files", "visuals", "distance_distribution", f'changes_{model}.png')
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
    k0_stats = analyze_distances(k0_distances, model, '0')
    k10_stats = analyze_distances(k10_distances, model, '10')

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

    plot_path = os.path.join("personality_analysis", "files", "visuals", "distance_distribution", f'comparison_{model}.png')
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


def analyze_rouge_correlation(distances: List[float], rouge_scores: List[float], model: str, k: int):
    """Analyze correlation between distances and ROUGE-L scores."""

    # Calculate correlation
    correlation, p_value = stats.pearsonr(distances, rouge_scores)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, rouge_scores, alpha=0.5)

    # Add trend line
    z = np.polyfit(distances, rouge_scores, 1)
    p = np.poly1d(z)
    plt.plot(distances, p(distances), "r--", alpha=0.8)

    plt.title(f'Distance vs ROUGE-L Score\n{model} (k={k})\nCorrelation: {correlation:.4f} (p={p_value:.4e})')
    plt.xlabel('Distance')
    plt.ylabel('ROUGE-L Score')

    plot_path = os.path.join("personality_analysis", "files", "visuals", "rouge_correlation", f'correlation_{model}_k{k}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    return {
        'correlation': correlation,
        'p_value': p_value
    }


def analyze_rouge_improvement(k0_distances: List[float], k10_distances: List[float],
                              k0_rouge: List[float], k10_rouge: List[float], model: str):
    """Analyze how distance improvements correlate with ROUGE score improvements."""

    # Calculate changes
    distance_changes = np.array(k10_distances) - np.array(k0_distances)
    rouge_changes = np.array(k10_rouge) - np.array(k0_rouge)

    # Calculate correlation between changes
    correlation, p_value = stats.pearsonr(distance_changes, rouge_changes)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(distance_changes, rouge_changes, alpha=0.5)

    # Add trend line
    z = np.polyfit(distance_changes, rouge_changes, 1)
    p = np.poly1d(z)
    plt.plot(distance_changes, p(distance_changes), "r--", alpha=0.8)

    plt.title(f'Distance Change vs ROUGE-L Change\n{model}\nCorrelation: {correlation:.4f} (p={p_value:.4e})')
    plt.xlabel('Change in Distance (K10 - K0)\nNegative = Distance Decreased')
    plt.ylabel('Change in ROUGE-L Score (K10 - K0)\nPositive = Score Improved')

    # Add quadrant labels
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)

    plot_path = os.path.join("personality_analysis", "files", "visuals", "rouge_correlation", f'improvement_correlation_{model}.png')
    plt.savefig(plot_path)
    plt.close()

    # Calculate statistics for different quadrants
    better_distance_better_rouge = np.sum((distance_changes < 0) & (rouge_changes > 0))
    better_distance_worse_rouge = np.sum((distance_changes < 0) & (rouge_changes < 0))
    worse_distance_better_rouge = np.sum((distance_changes > 0) & (rouge_changes > 0))
    worse_distance_worse_rouge = np.sum((distance_changes > 0) & (rouge_changes < 0))

    total_samples = len(distance_changes)

    print(f"\nQuadrant Analysis for {model}:")
    print(f"Better Distance & Better ROUGE: {better_distance_better_rouge} ({better_distance_better_rouge/total_samples*100:.1f}%)")
    print(f"Better Distance & Worse ROUGE: {better_distance_worse_rouge} ({better_distance_worse_rouge/total_samples*100:.1f}%)")
    print(f"Worse Distance & Better ROUGE: {worse_distance_better_rouge} ({worse_distance_better_rouge/total_samples*100:.1f}%)")
    print(f"Worse Distance & Worse ROUGE: {worse_distance_worse_rouge} ({worse_distance_worse_rouge/total_samples*100:.1f}%)")

    return {
        'correlation': correlation,
        'p_value': p_value,
        'quadrants': {
            'better_distance_better_rouge': better_distance_better_rouge,
            'better_distance_worse_rouge': better_distance_worse_rouge,
            'worse_distance_better_rouge': worse_distance_better_rouge,
            'worse_distance_worse_rouge': worse_distance_worse_rouge
        }
    }


def analyze_initial_distance_impact(k0_distances: List[float], k10_distances: List[float], 
                                 k0_rouge: List[float], k10_rouge: List[float], model: str):
    """Analyze how initial distances affect improvements when k increases."""
    
    # Convert to numpy arrays
    k0_distances = np.array(k0_distances)
    k10_distances = np.array(k10_distances)
    k0_rouge = np.array(k0_rouge)
    k10_rouge = np.array(k10_rouge)
    
    # Calculate changes
    distance_changes = k10_distances - k0_distances
    rouge_changes = k10_rouge - k0_rouge
    
    # Define high distance threshold as 75th percentile of initial distances
    high_dist_threshold = np.percentile(k0_distances, 75)
    
    # Split samples into high and low initial distance groups
    high_dist_mask = k0_distances >= high_dist_threshold
    low_dist_mask = k0_distances < high_dist_threshold
    
    # Prepare data for both CSV and printing
    high_dist_stats = {
        'model': model,
        'group': 'high_distance',
        'threshold': high_dist_threshold,
        'num_samples': np.sum(high_dist_mask),
        'mean_initial_distance': np.mean(k0_distances[high_dist_mask]),
        'mean_distance_change': np.mean(distance_changes[high_dist_mask]),
        'mean_rouge_change': np.mean(rouge_changes[high_dist_mask]),
        'num_rouge_improved': np.sum(rouge_changes[high_dist_mask] > 0),
        'num_rouge_worsened': np.sum(rouge_changes[high_dist_mask] < 0),
        'pct_rouge_improved': np.mean(rouge_changes[high_dist_mask] > 0) * 100,
        'pct_rouge_worsened': np.mean(rouge_changes[high_dist_mask] < 0) * 100
    }
    
    low_dist_stats = {
        'model': model,
        'group': 'low_distance',
        'threshold': high_dist_threshold,
        'num_samples': np.sum(low_dist_mask),
        'mean_initial_distance': np.mean(k0_distances[low_dist_mask]),
        'mean_distance_change': np.mean(distance_changes[low_dist_mask]),
        'mean_rouge_change': np.mean(rouge_changes[low_dist_mask]),
        'num_rouge_improved': np.sum(rouge_changes[low_dist_mask] > 0),
        'num_rouge_worsened': np.sum(rouge_changes[low_dist_mask] < 0),
        'pct_rouge_improved': np.mean(rouge_changes[low_dist_mask] > 0) * 100,
        'pct_rouge_worsened': np.mean(rouge_changes[low_dist_mask] < 0) * 100
    }
    
    # Print analysis
    print(f"\nInitial Distance Impact Analysis for {model}")
    print(f"High distance threshold (75th percentile): {high_dist_threshold:.4f}")
    
    print("\nHigh Initial Distance Samples:")
    print(f"Number of samples: {high_dist_stats['num_samples']}")
    print(f"Mean initial distance: {high_dist_stats['mean_initial_distance']:.4f}")
    print(f"Mean distance change: {high_dist_stats['mean_distance_change']:.4f}")
    print(f"Mean ROUGE change: {high_dist_stats['mean_rouge_change']:.4f}")
    print(f"Samples with improved ROUGE: {high_dist_stats['num_rouge_improved']} ({high_dist_stats['pct_rouge_improved']:.1f}%)")
    print(f"Samples with worse ROUGE: {high_dist_stats['num_rouge_worsened']} ({high_dist_stats['pct_rouge_worsened']:.1f}%)")
    
    print("\nLow Initial Distance Samples:")
    print(f"Number of samples: {low_dist_stats['num_samples']}")
    print(f"Mean initial distance: {low_dist_stats['mean_initial_distance']:.4f}")
    print(f"Mean distance change: {low_dist_stats['mean_distance_change']:.4f}")
    print(f"Mean ROUGE change: {low_dist_stats['mean_rouge_change']:.4f}")
    print(f"Samples with improved ROUGE: {low_dist_stats['num_rouge_improved']} ({low_dist_stats['pct_rouge_improved']:.1f}%)")
    print(f"Samples with worse ROUGE: {low_dist_stats['num_rouge_worsened']} ({low_dist_stats['pct_rouge_worsened']:.1f}%)")
    
    # Create scatter plot of initial distance vs changes
    plt.figure(figsize=(12, 5))
    
    # Distance changes subplot
    plt.subplot(1, 2, 1)
    plt.scatter(k0_distances, distance_changes, alpha=0.5)
    plt.axvline(high_dist_threshold, color='r', linestyle='--', label='75th percentile')
    plt.axhline(0, color='k', linestyle='-', alpha=0.2)
    plt.xlabel('Initial Distance (k=0)')
    plt.ylabel('Change in Distance (k10 - k0)')
    plt.title('Initial Distance vs Distance Change')
    plt.legend()
    
    # ROUGE changes subplot
    plt.subplot(1, 2, 2)
    plt.scatter(k0_distances, rouge_changes, alpha=0.5)
    plt.axvline(high_dist_threshold, color='r', linestyle='--', label='75th percentile')
    plt.axhline(0, color='k', linestyle='-', alpha=0.2)
    plt.xlabel('Initial Distance (k=0)')
    plt.ylabel('Change in ROUGE-L (k10 - k0)')
    plt.title('Initial Distance vs ROUGE Change')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join("personality_analysis", "files", "visuals", "initial_distance_impact", f'initial_distance_impact_{model}.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    
    return {
        'high_dist_stats': high_dist_stats,
        'low_dist_stats': low_dist_stats
    }

def main():
    # Parse arguments
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
    
    # Initialize retriever
    retriever = Retriever(dataset)
    
    # Initialize results dictionary and list for CSV data
    comparison_results = {}
    csv_data = []
    
    for model, k_preds in predictions.items():
        print(f"\nAnalyzing model: {model}")
        
        # Get predictions for k=0 and k=10
        k0_preds = k_preds['0']
        k10_preds = k_preds['10']
        
        # Get ROUGE-L scores for k=0 and k=10
        k0_exp_key = [k for k in eval_results.keys() if get_model_and_k(k)[0] == model and get_model_and_k(k)[1] == '0'][0]
        k10_exp_key = [k for k in eval_results.keys() if get_model_and_k(k)[0] == model and get_model_and_k(k)[1] == '10'][0]
        
        k0_rouge = eval_results[k0_exp_key]['rougeL']
        k10_rouge = eval_results[k10_exp_key]['rougeL']
        
        # Calculate distances
        k0_distances = retriever.calculate_one_to_one_distances(k0_preds, ground_truth)
        k10_distances = retriever.calculate_one_to_one_distances(k10_preds, ground_truth)
        
        # Basic distance analysis
        results = compare_k_settings(model, k0_distances, k10_distances)
        
        # Analyze impact of initial distances
        impact_results = analyze_initial_distance_impact(
            k0_distances, k10_distances, k0_rouge, k10_rouge, model
        )
        
        # Add results to CSV data
        csv_data.append(impact_results['high_dist_stats'])
        csv_data.append(impact_results['low_dist_stats'])
        
        results['initial_distance_impact'] = impact_results
        comparison_results[model] = results
    
    # Save results to CSV
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join("personality_analysis", "files", "csv", "initial_distance_impact.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, float_format='%.4f')
    
    # Print summary of improvements and initial distance impact
    print("\nSummary of Improvements and Initial Distance Impact:")
    for model, results in comparison_results.items():
        print(f"\n{model}:")
        impact = results['initial_distance_impact']
        
        print("\nHigh Initial Distance Samples:")
        high_dist = impact['high_dist_stats']
        print(f"Count: {high_dist['num_samples']}")
        print(f"Mean initial distance: {high_dist['mean_initial_distance']:.4f}")
        print(f"Mean distance change: {high_dist['mean_distance_change']:.4f}")
        print(f"Mean ROUGE change: {high_dist['mean_rouge_change']:.4f}")
        print(f"ROUGE improved/worsened: {high_dist['num_rouge_improved']}/{high_dist['num_rouge_worsened']}")
        
        print("\nLow Initial Distance Samples:")
        low_dist = impact['low_dist_stats']
        print(f"Count: {low_dist['num_samples']}")
        print(f"Mean initial distance: {low_dist['mean_initial_distance']:.4f}")
        print(f"Mean distance change: {low_dist['mean_distance_change']:.4f}")
        print(f"Mean ROUGE change: {low_dist['mean_rouge_change']:.4f}")
        print(f"ROUGE improved/worsened: {low_dist['num_rouge_improved']}/{low_dist['num_rouge_worsened']}")

if __name__ == "__main__":
    main()
