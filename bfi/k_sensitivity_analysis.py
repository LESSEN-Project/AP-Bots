import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_individual_scores(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_model_family(model_name):
    return model_name.split('-')[0]

def filter_experiments(scores_data):
    """Filter experiments based on parameters"""
    filtered_exps = {}
    for exp_name, exp_data in scores_data.items():
        params = exp_data['params']
        if (params['features'] == "" and 
            params['RS'] == "1" and 
            params['retriever'] == "contriever"):
            filtered_exps[exp_name] = exp_data
    
    # Print summary of filtered experiments
    print("\nFiltered Experiments Summary:")
    model_counts = defaultdict(int)
    k_values = defaultdict(int)
    for exp_data in filtered_exps.values():
        model = exp_data['params']['model']
        k = exp_data['params']['k']
        model_counts[model] += 1
        k_values[k] += 1
    
    print("\nModels found:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} experiments")
    
    print("\nK values found:")
    for k, count in sorted(k_values.items(), key=lambda x: int(x[0])):
        print(f"  k={k}: {count} experiments")
    
    return filtered_exps

def group_by_k_and_family(filtered_exps):
    """Group experiments by k value and model family"""
    k_family_scores = defaultdict(lambda: defaultdict(list))
    for exp_name, exp_data in filtered_exps.items():
        k = int(exp_data['params']['k'])
        model = exp_data['params']['model']
        family = get_model_family(model)
        scores = exp_data['rougeL']
        k_family_scores[k][family].append(scores)
    return k_family_scores

def analyze_k_sensitivity(k_family_scores):
    """Analyze how samples respond to increasing k"""
    k_values = sorted(k_family_scores.keys())
    if len(k_values) < 2:
        print("Not enough k values to analyze sensitivity")
        return None
    
    families = list(k_family_scores[k_values[0]].keys())
    n_samples = len(k_family_scores[k_values[0]][families[0]][0])
    
    # Calculate improvement for each sample as k increases
    improvements = defaultdict(list)
    for family in families:
        for sample_idx in range(n_samples):
            sample_scores = []
            for k in k_values:
                if family in k_family_scores[k]:
                    # Average score across all models in family for this k
                    scores = [exp[sample_idx] for exp in k_family_scores[k][family]]
                    sample_scores.append(np.mean(scores))
            
            if len(sample_scores) >= 2:
                # Calculate absolute and relative improvement
                abs_improvement = sample_scores[-1] - sample_scores[0]
                # Handle zero initial scores
                if sample_scores[0] == 0:
                    rel_improvement = float('inf') if abs_improvement > 0 else 0
                else:
                    rel_improvement = abs_improvement / sample_scores[0]
                
                # Calculate improvement rate (slope)
                k_array = np.array(k_values, dtype=float)
                score_array = np.array(sample_scores)
                slope = np.polyfit(k_array, score_array, 1)[0]
                
                improvements[family].append({
                    'sample_idx': sample_idx,
                    'rel_improvement': rel_improvement,
                    'abs_improvement': abs_improvement,
                    'improvement_rate': slope,
                    'initial_score': sample_scores[0],
                    'final_score': sample_scores[-1],
                    'scores_by_k': {k: score for k, score in zip(k_values, sample_scores)}
                })
    
    return improvements, k_values

def identify_underperforming_samples(improvements, threshold_percentile=10):
    """Identify samples that show least improvement with increasing k"""
    underperforming = {}
    for family, samples in improvements.items():
        # Calculate multiple metrics for underperformance
        abs_improvements = np.array([s['abs_improvement'] for s in samples])
        rates = np.array([s['improvement_rate'] for s in samples])
        
        # Use combined metric (normalize both and take average)
        abs_improvements_norm = (abs_improvements - np.min(abs_improvements)) / (np.max(abs_improvements) - np.min(abs_improvements))
        rates_norm = (rates - np.min(rates)) / (np.max(rates) - np.min(rates))
        combined_metric = (abs_improvements_norm + rates_norm) / 2
        
        threshold = np.percentile(combined_metric, threshold_percentile)
        
        # Find samples below threshold
        underperforming[family] = [
            sample for sample, metric in zip(samples, combined_metric)
            if metric < threshold
        ]
        
        # Sort by absolute improvement
        underperforming[family].sort(key=lambda x: x['abs_improvement'])
    
    return underperforming

def plot_k_sensitivity(k_family_scores, underperforming, k_values, output_file):
    """Plot how underperforming samples change with k"""
    families = list(underperforming.keys())
    n_families = len(families)
    
    fig, axes = plt.subplots(n_families, 1, figsize=(12, 6*n_families))
    if n_families == 1:
        axes = [axes]
    
    for idx, (family, samples) in enumerate(underperforming.items()):
        ax = axes[idx]
        
        # Plot worst 5 samples
        for sample in samples[:5]:
            scores = [sample['scores_by_k'][k] for k in k_values]
            ax.plot(k_values, scores, marker='o', label=f'Sample {sample["sample_idx"]}')
        
        # Plot average performance for reference
        avg_scores = []
        for k in k_values:
            if family in k_family_scores[k]:
                scores = np.mean([np.mean(exp) for exp in k_family_scores[k][family]])
                avg_scores.append(scores)
        ax.plot(k_values, avg_scores, 'k--', linewidth=2, label='Average')
        
        ax.set_title(f'{family} Family - Worst Performing Samples vs K')
        ax.set_xlabel('K value')
        ax.set_ylabel('RougeL Score')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main():
    # Load and filter data
    file_path = "evaluation/files/indv/eval_amazon_Grocery_and_Gourmet_Food_2018.json"
    scores_data = load_individual_scores(file_path)
    filtered_exps = filter_experiments(scores_data)
    
    # Group by k and family
    k_family_scores = group_by_k_and_family(filtered_exps)
    
    # Analyze k sensitivity
    improvements, k_values = analyze_k_sensitivity(k_family_scores)
    if improvements is None:
        return
    
    # Identify underperforming samples
    underperforming = identify_underperforming_samples(improvements)
    
    # Print analysis
    print("\n=== K-Sensitivity Analysis ===")
    print(f"Analyzed k values: {k_values}")
    
    for family, samples in underperforming.items():
        print(f"\n{family} Family:")
        print("Top 5 worst performing samples (least improvement with increasing k):")
        for sample in samples[:5]:
            print(f"\nSample {sample['sample_idx']}:")
            print(f"  Absolute improvement: {sample['abs_improvement']:.4f}")
            print(f"  Improvement rate (slope): {sample['improvement_rate']:.4f}")
            if sample['initial_score'] > 0:
                print(f"  Relative improvement: {sample['rel_improvement']:.2%}")
            else:
                print("  Relative improvement: N/A (initial score was 0)")
            print(f"  Initial score (k={k_values[0]}): {sample['initial_score']:.4f}")
            print(f"  Final score (k={k_values[-1]}): {sample['final_score']:.4f}")
            print("  Scores by k:")
            for k, score in sorted(sample['scores_by_k'].items(), key=lambda x: int(x[0])):
                print(f"    k={k}: {score:.4f}")
    
    # Create visualization
    plot_k_sensitivity(k_family_scores, underperforming, k_values, 'k_sensitivity.png')
    print("\nVisualization saved as 'k_sensitivity.png'")

if __name__ == "__main__":
    main()
