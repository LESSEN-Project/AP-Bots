import json
import os
import numpy as np
import pandas as pd

from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

from AP_Bots.retriever import Retriever
from AP_Bots.utils.argument_parser import get_args, parse_dataset
from AP_Bots.personality_analysis.analysis_utils import load_eval_results, load_predictions, get_exp_eval_results


def analyze_distances(distances: List[float], model: str, k: int):
    """Analyze and plot the distribution of distances."""
    distances = np.array(distances)

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


def analyze_sample_changes(k0_distances: List[float], kmax_distances: List[float], k_max, model: str, visuals_dir: str):

    changes = np.array(kmax_distances) - np.array(k0_distances)

    improved = changes < 0 
    worsened = changes > 0 
    unchanged = np.isclose(changes, 0, atol=1e-6) 

    num_samples = len(changes)
    num_improved = np.sum(improved)
    num_worsened = np.sum(worsened)
    num_unchanged = np.sum(unchanged)

    avg_improvement = -np.mean(changes[improved]) if any(improved) else 0  
    avg_deterioration = np.mean(changes[worsened]) if any(worsened) else 0

    print(f"\nPer-sample analysis for {model}:")
    print(f"Total samples: {num_samples}")
    print(f"Improved samples: {num_improved} ({(num_improved/num_samples)*100:.1f}%)")
    print(f"Worsened samples: {num_worsened} ({(num_worsened/num_samples)*100:.1f}%)")
    print(f"Unchanged samples: {num_unchanged} ({(num_unchanged/num_samples)*100:.1f}%)")
    print(f"Average distance reduction when improved: {avg_improvement:.4f}")
    print(f"Average distance increase when worsened: {avg_deterioration:.4f}")

    plt.figure(figsize=(10, 10))
    plt.scatter(k0_distances, kmax_distances, alpha=0.5)

    min_val = min(min(k0_distances), min(kmax_distances))
    max_val = max(max(k0_distances), max(kmax_distances))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='No change line')

    plt.title(f'K=0 vs K={k_max} Distances\n{model}')
    plt.xlabel('Distance with K=0')
    plt.ylabel(f'Distance with K={k_max}')
    plt.legend()

    plot_path = os.path.join(visuals_dir, "distance_distribution")
    os.makedirs(plot_path, exist_ok=True)
    
    plt.savefig(os.path.join(plot_path, f'scatter_{model}.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(changes, bins=50, density=True)
    plt.axvline(0, color='r', linestyle='--', label='No change')
    plt.title(f'Distribution of Distance Changes\n{model}')
    plt.xlabel('Change in Distance (K10 - K0)\nNegative = Improved (distance decreased)')
    plt.ylabel('Density')
    plt.legend()

    plt.savefig(os.path.join(plot_path, f'changes_{model}.png'))
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


def compare_k_settings(model: str, k0_distances: List[float], kmax_distances: List[float], k_max, visuals_dir: str):
    """Compare distance distributions between k=0 and k=10 settings."""
    k0_stats = analyze_distances(k0_distances, model, '0')
    kmax_stats = analyze_distances(kmax_distances, model, '10')

    # Calculate improvement metrics
    mean_improvement = k0_stats['mean'] - kmax_stats['mean']
    median_improvement = k0_stats['median'] - kmax_stats['median']

    print(f"\nImprovements for {model}:")
    print(f"Mean distance reduction: {mean_improvement:.4f}")
    print(f"Median distance reduction: {median_improvement:.4f}")

    # Perform statistical test
    t_stat, p_value = stats.ttest_ind(k0_distances, kmax_distances)
    print(f"T-test p-value: {p_value:.4e}")

    # Analyze per-sample changes
    sample_analysis = analyze_sample_changes(k0_distances, kmax_distances, k_max, model, visuals_dir)

    # Plot comparison
    plt.figure(figsize=(12, 6))

    # Plot histograms
    plt.hist(k0_distances, bins=50, alpha=0.5, label=f'k=0 (mean={k0_stats["mean"]:.4f})', density=True)
    plt.hist(kmax_distances, bins=50, alpha=0.5, label=f'k={k_max} (mean={kmax_stats["mean"]:.4f})', density=True)

    # Add vertical lines for means
    plt.axvline(k0_stats['mean'], color='blue', linestyle='dashed', alpha=0.5)
    plt.axvline(kmax_stats['mean'], color='orange', linestyle='dashed', alpha=0.5)

    plt.title(f'Distance Distribution Comparison\n{model}')
    plt.xlabel('Distance (1 - cosine similarity)')
    plt.ylabel('Density')
    plt.legend()

    plot_path = os.path.join(visuals_dir, "distance_distribution")
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f'comparison_{model}.png'))
    plt.close()

    return {
        'k0_stats': k0_stats,
        f'k{k_max}_stats': kmax_stats,
        'mean_improvement': mean_improvement,
        'median_improvement': median_improvement,
        'p_value': p_value,
        'sample_analysis': sample_analysis
    }


def analyze_initial_distance_impact(k0_distances: List[float], kmax_distances: List[float], 
                                 k0_rouge: List[float], kmax_rouge: List[float], k_max, model: str, visuals_dir: str):
    """Analyze how initial distances affect improvements when k increases."""
    
    # Convert to numpy arrays
    k0_distances = np.array(k0_distances)
    k10_distances = np.array(kmax_distances)
    k0_rouge = np.array(k0_rouge)
    k10_rouge = np.array(kmax_rouge)
    
    # Calculate changes
    distance_changes = kmax_distances - k0_distances
    rouge_changes = kmax_rouge - k0_rouge
    
    high_dist_threshold = np.percentile(k0_distances, 75)
    
    high_dist_mask = k0_distances >= high_dist_threshold
    low_dist_mask = k0_distances < high_dist_threshold
    
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
    plt.ylabel(f'Change in Distance (k{k_max} - k0)')
    plt.title('Initial Distance vs Distance Change')
    plt.legend()
    
    # ROUGE changes subplot
    plt.subplot(1, 2, 2)
    plt.scatter(k0_distances, rouge_changes, alpha=0.5)
    plt.axvline(high_dist_threshold, color='r', linestyle='--', label='75th percentile')
    plt.axhline(0, color='k', linestyle='-', alpha=0.2)
    plt.xlabel('Initial Distance (k=0)')
    plt.ylabel(f'Change in ROUGE-L (k{k_max} - k0)')
    plt.title('Initial Distance vs ROUGE Change')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(visuals_dir, "initial_distance_impact")
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(os.path.join(plot_path, f'{model}.png'))
    plt.close()
    
    return {
        'high_dist_stats': high_dist_stats,
        'low_dist_stats': low_dist_stats
    }

def main():

    k_range =  ["0", "10"]
    k_max = max(k_range)
    args = get_args()
    dataset = parse_dataset(args.dataset)

    visuals_dir = os.path.join("personality_analysis", "files", "visuals", dataset.tag)
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Load evaluation results
    eval_file = os.path.join("evaluation", "files", "indv", f"eval_{args.dataset}.json")
    eval_results = load_eval_results(eval_file, k_range)
    
    # Load predictions
    pred_dir = os.path.join("files", "preds")
    predictions = load_predictions(pred_dir, list(eval_results.keys()), k_max)
    
    # Load ground truth
    ground_truth = dataset.get_gts()
    print(f"Loaded {len(ground_truth)} ground truth samples")
    
    # Initialize retriever
    retriever = Retriever(dataset)
    
    comparison_results = {}
    csv_data = []
    
    for model, k_preds in predictions.items():
        print(f"\nAnalyzing model: {model}")

        k_max = [k for k in k_preds.keys() if k != "0"][0]
        
        k0_preds = k_preds['0']
        kmax_preds = k_preds[k_max]
        
        k0_rouge = get_exp_eval_results(eval_results, model, "0")
        kmax_rouge = get_exp_eval_results(eval_results, model, k_max)
        
        # Calculate distances
        k0_distances = retriever.calculate_one_to_one_distances(k0_preds, ground_truth)
        kmax_distances = retriever.calculate_one_to_one_distances(kmax_preds, ground_truth)
        
        results = compare_k_settings(model, k0_distances, kmax_distances, k_max, visuals_dir)
        
        impact_results = analyze_initial_distance_impact(
            k0_distances, kmax_distances, k0_rouge, kmax_rouge, k_max, model, visuals_dir
        )
        
        csv_data.append(impact_results['high_dist_stats'])
        csv_data.append(impact_results['low_dist_stats'])
        
        results['initial_distance_impact'] = impact_results
        comparison_results[model] = results
    
    # Save results to CSV
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join("personality_analysis", "files", "csv", dataset.tag, "initial_distance_impact.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, float_format='%.4f')
    
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
