import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_individual_scores(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_model_family(model_name):
    return model_name.split('-')[0]

def get_experiment_info(exp_name):
    # Remove .json extension if present
    exp_name = exp_name.replace('.json', '')
    parts = exp_name.split('_')
    # Find the part that contains the model name (it will have dashes)
    model_parts = [p for p in parts if '-' in p]
    if model_parts:
        model_name = model_parts[0]
    else:
        model_name = "unknown"
    # Features are in brackets
    features = [p for p in parts if p.startswith('[') and p.endswith(']')]
    features = features[0] if features else ""
    return model_name, features

def analyze_sample_patterns(scores_data):
    # Convert scores to a matrix where each row is a sample and each column is an experiment
    experiments = list(scores_data.keys())
    n_samples = len(scores_data[experiments[0]]['rougeL'])
    
    # Create a matrix of rougeL scores
    score_matrix = np.zeros((n_samples, len(experiments)))
    experiment_info = []
    
    for i, exp in enumerate(experiments):
        score_matrix[:, i] = scores_data[exp]['rougeL']
        model_name, features = get_experiment_info(exp)
        experiment_info.append({
            'model': model_name,
            'features': features
        })
    
    # Calculate mean score for each sample across all configurations
    sample_means = np.mean(score_matrix, axis=1)
    
    # Find consistently low/high performing samples
    low_threshold = np.percentile(sample_means, 25)
    high_threshold = np.percentile(sample_means, 75)
    
    consistently_low = np.where(sample_means < low_threshold)[0]
    consistently_high = np.where(sample_means > high_threshold)[0]
    
    # Calculate variance for each sample
    sample_vars = np.var(score_matrix, axis=1)
    
    return {
        'consistently_low_samples': consistently_low.tolist(),
        'consistently_high_samples': consistently_high.tolist(),
        'sample_means': sample_means.tolist(),
        'sample_vars': sample_vars.tolist(),
        'score_matrix': score_matrix,
        'experiment_info': experiment_info
    }

def analyze_model_families(scores_data):
    # Group models by family
    family_scores = defaultdict(list)
    for exp_name, exp_data in scores_data.items():
        model_name, _ = get_experiment_info(exp_name)
        family = get_model_family(model_name)
        family_scores[family].extend(exp_data['rougeL'])
    
    # Calculate statistics for each family
    family_stats = {}
    for family, scores in family_scores.items():
        scores_array = np.array(scores)
        family_stats[family] = {
            'mean': np.mean(scores_array),
            'std': np.std(scores_array),
            'median': np.median(scores_array),
            'size': len(scores)
        }
    
    # Perform one-way ANOVA to test if families are significantly different
    if len(family_scores) >= 2:
        family_groups = [np.array(scores) for scores in family_scores.values()]
        f_stat, p_value = stats.f_oneway(*family_groups)
        anova_results = {'f_stat': f_stat, 'p_value': p_value}
    else:
        anova_results = {'f_stat': None, 'p_value': None}
    
    return {
        'family_stats': family_stats,
        'anova_results': anova_results,
        'family_scores': family_scores
    }

def plot_family_distributions(family_scores):
    plt.figure(figsize=(12, 6))
    
    # Create violin plot
    plt.subplot(1, 2, 1)
    data = [scores for scores in family_scores.values()]
    sns.violinplot(data=data)
    plt.xticks(range(len(family_scores)), family_scores.keys(), rotation=45)
    plt.title('RougeL Score Distribution by Model Family')
    plt.ylabel('RougeL Score')
    
    # Create box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(data=data)
    plt.xticks(range(len(family_scores)), family_scores.keys(), rotation=45)
    plt.title('RougeL Score Box Plot by Model Family')
    plt.ylabel('RougeL Score')
    
    plt.tight_layout()
    plt.savefig('model_family_distributions.png')
    plt.close()

def main():
    # Load the data
    file_path = "evaluation/files/indv/eval_amazon_Grocery_and_Gourmet_Food_2018.json"
    scores_data = load_individual_scores(file_path)
    
    # Analyze sample patterns
    sample_analysis = analyze_sample_patterns(scores_data)
    print("\n=== Sample Analysis ===")
    print(f"Number of consistently low performing samples: {len(sample_analysis['consistently_low_samples'])}")
    print(f"Number of consistently high performing samples: {len(sample_analysis['consistently_high_samples'])}")
    print(f"Average sample variance: {np.mean(sample_analysis['sample_vars']):.4f}")
    
    # Find samples with highest variance
    sample_vars = np.array(sample_analysis['sample_vars'])
    high_var_indices = np.argsort(sample_vars)[-5:]  # Top 5 highest variance samples
    print("\nSamples with highest variance (indicating inconsistent performance):")
    for idx in high_var_indices:
        print(f"Sample {idx}: Variance = {sample_vars[idx]:.4f}")
    
    # Analyze model families
    family_analysis = analyze_model_families(scores_data)
    print("\n=== Model Family Analysis ===")
    print("Family Statistics:")
    for family, stats in family_analysis['family_stats'].items():
        print(f"\n{family}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Size: {stats['size']}")
    
    print("\nANOVA Test Results:")
    if family_analysis['anova_results']['f_stat'] is not None:
        print(f"F-statistic: {family_analysis['anova_results']['f_stat']:.4f}")
        print(f"p-value: {family_analysis['anova_results']['p_value']:.4f}")
    else:
        print("Not enough families to perform ANOVA test.")
    
    # Create visualization
    plot_family_distributions(family_analysis['family_scores'])
    print("\nVisualization saved as 'model_family_distributions.png'")

if __name__ == "__main__":
    main()
