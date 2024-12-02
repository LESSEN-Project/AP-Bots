import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def load_and_filter_data(file_path):
    """Load data and filter for specified conditions"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Define models to include
    included_models = {
        'LLAMA': ['LLAMA-3.2-3B', 'LLAMA-3.1-8B'],
        'GEMMA': ['GEMMA-2-9B', 'GEMMA-2-27B']
    }
    
    filtered_data = {}
    for exp_name, exp_data in data.items():
        params = exp_data['params']
        model = params['model']
        model_family = model.split('-')[0]
        
        if (params['features'] == "" and 
            params['RS'] == 1 and 
            params['k'] in [0, 10, 50] and
            model_family in included_models and
            model in included_models[model_family]):
            filtered_data[exp_name] = exp_data
    
    return filtered_data

def analyze_scores(filtered_data):
    """Analyze rouge scores for different k values and models"""
    results = {k: {} for k in [0, 10, 50]}
    
    for exp_name, exp_data in filtered_data.items():
        k = exp_data['params']['k']
        model = exp_data['params']['model']
        scores = np.array(exp_data['rougeL'])
        
        if k in results:
            if model not in results[k]:
                results[k][model] = []
            results[k][model].extend(scores)
    
    return results

def create_model_display_name(model):
    """Create a display name for the model"""
    family = model.split('-')[0]
    size = model.split('-')[-1]
    return f"{family}\n{size}"

def analyze_score_transitions(results, output_dir):
    """Analyze how scores change between k=0 and k=50"""
    # First, create comprehensive statistics DataFrame
    stats_data = []
    bins = [-np.inf, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, np.inf]
    bin_labels = ['0', '0-0.05', '0.05-0.1', '0.1-0.15', '0.15-0.2', '0.2-0.25', '0.25-0.3', '>0.3']
    
    for k in results.keys():
        for model in results[k].keys():
            scores = np.array(results[k][model])
            binned_scores = pd.cut(scores, bins=bins, labels=bin_labels)
            bin_counts = binned_scores.value_counts()
            bin_percentages = (bin_counts / len(scores) * 100).round(2)
            
            # Calculate statistics
            stats = {
                'Model': model,
                'k': k,
                'Mean': np.mean(scores).round(4),
                'Median': np.median(scores).round(4),
                'Std': np.std(scores).round(4),
                'Min': np.min(scores).round(4),
                'Max': np.max(scores).round(4),
                'Zero_Count': np.sum(scores == 0),
                'Zero_Percentage': (np.sum(scores == 0) / len(scores) * 100).round(2)
            }
            
            # Add bin percentages
            for bin_label in bin_labels:
                stats[f'Bin_{bin_label}_Count'] = bin_counts.get(bin_label, 0)
                stats[f'Bin_{bin_label}_Percentage'] = bin_percentages.get(bin_label, 0)
            
            # Add quartile information
            q1, q3 = np.percentile(scores, [25, 75])
            stats.update({
                'Q1': q1.round(4),
                'Q3': q3.round(4),
                'IQR': (q3 - q1).round(4)
            })
            
            stats_data.append(stats)
    
    # Create and save comprehensive statistics DataFrame
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values(['Model', 'k'])
    stats_csv_path = f'{output_dir}/comprehensive_statistics.csv'
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"\nSaved comprehensive statistics to: {stats_csv_path}")
    
    # Continue with transition analysis
    bins = [-np.inf, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, np.inf]
    bin_labels = ['0', '0-0.05', '0.05-0.1', '0.1-0.15', '0.15-0.2', '0.2-0.25', '0.25-0.3', '>0.3']
    
    # Prepare data for each model
    transition_stats = {}
    detailed_transitions = {}
    
    for model in results[0].keys():
        scores_k0 = np.array(results[0][model])
        scores_k50 = np.array(results[50][model])
        
        # Basic statistics
        total_samples = len(scores_k0)
        improved = np.sum(scores_k50 > scores_k0)
        worsened = np.sum(scores_k50 < scores_k0)
        unchanged = np.sum(scores_k50 == scores_k0)
        
        # Calculate how many non-zero scores became zero and vice versa
        became_zero = np.sum((scores_k0 > 0) & (scores_k50 == 0))
        zero_to_nonzero = np.sum((scores_k0 == 0) & (scores_k50 > 0))
        
        # Create bins for k=0 and k=50
        bins_k0 = pd.cut(scores_k0, bins=bins, labels=bin_labels)
        bins_k50 = pd.cut(scores_k50, bins=bins, labels=bin_labels)
        
        # Create transition DataFrame
        transitions_df = pd.DataFrame({
            'k0_bin': bins_k0,
            'k50_bin': bins_k50,
            'k0_score': scores_k0,
            'k50_score': scores_k50
        })
        
        # Calculate mean score change for each starting bin
        transition_summary = []
        for start_bin in bin_labels:
            bin_data = transitions_df[transitions_df['k0_bin'] == start_bin]
            if len(bin_data) > 0:
                # Calculate where scores moved to
                dest_counts = bin_data['k50_bin'].value_counts()
                total_in_bin = len(bin_data)
                
                # Calculate mean score change
                mean_score_change = (bin_data['k50_score'] - bin_data['k0_score']).mean()
                
                # Get top 3 destinations
                top_destinations = dest_counts.nlargest(3)
                dest_str = ' | '.join([f"{idx}: {val/total_in_bin*100:.1f}%" 
                                     for idx, val in top_destinations.items()])
                
                transition_summary.append({
                    'Start Bin': start_bin,
                    'Count': total_in_bin,
                    'Mean Score Change': mean_score_change,
                    'Top Destinations': dest_str
                })
        
        detailed_transitions[model] = pd.DataFrame(transition_summary)
        
        # Store basic statistics
        transition_stats[model] = {
            'improved': improved / total_samples * 100,
            'worsened': worsened / total_samples * 100,
            'unchanged': unchanged / total_samples * 100,
            'became_zero': became_zero,
            'became_zero_pct': became_zero / total_samples * 100,
            'zero_to_nonzero': zero_to_nonzero,
            'zero_to_nonzero_pct': zero_to_nonzero / total_samples * 100,
            'total_samples': total_samples
        }
    
    # Print transition statistics and detailed transitions
    print("\nScore Transition Analysis (k=0 → k=50):")
    for model in transition_stats.keys():
        stats = transition_stats[model]
        print(f"\n{model}:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Improved scores: {stats['improved']:.1f}%")
        print(f"  Worsened scores: {stats['worsened']:.1f}%")
        print(f"  Unchanged scores: {stats['unchanged']:.1f}%")
        print(f"  Non-zero → Zero: {stats['became_zero']} samples ({stats['became_zero_pct']:.1f}%)")
        print(f"  Zero → Non-zero: {stats['zero_to_nonzero']} samples ({stats['zero_to_nonzero_pct']:.1f}%)")
        
        print(f"\nDetailed Transitions for {model}:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(detailed_transitions[model].to_string(index=False))
        print("\n" + "="*80)
        
        # Save the transition DataFrame to CSV
        csv_filename = f'{output_dir}/transitions_{model.replace("-", "_").lower()}.csv'
        detailed_transitions[model].to_csv(csv_filename, index=False)
        print(f"Saved transition details to: {csv_filename}")

def plot_comparisons(results, output_dir):
    """Create plots comparing all models"""    

    plot_data = []
    for k in sorted(results.keys()):
        for model in sorted(results[k].keys()):
            scores = results[k][model]
            display_name = create_model_display_name(model)
            plot_data.extend([(score, k, display_name) for score in scores])
    
    df = pd.DataFrame(plot_data, columns=['score', 'k', 'model'])
    df['k'] = pd.Categorical(df['k'], categories=[0, 10, 50], ordered=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Box plot for all models
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='model', y='score', hue='k', 
                hue_order=[0, 10, 50],
                palette='Set2')
    plt.title('Rouge-L Score Distribution by Model and k')
    plt.xticks(rotation=0)
    plt.ylabel('Rouge-L Score')
    plt.xlabel('Model')
    plt.legend(title='k')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_distributions_boxplot.png')
    plt.close()
    
    # Violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='model', y='score', hue='k',
                  hue_order=[0, 10, 50],
                  palette='Set2')
    plt.title('Rouge-L Score Distribution (Violin Plot)')
    plt.xticks(rotation=0)
    plt.ylabel('Rouge-L Score')
    plt.xlabel('Model')
    plt.legend(title='k')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/score_distributions_violin.png')
    plt.close()
    
    # Zero scores analysis
    plt.figure(figsize=(12, 6))
    zero_scores = df.groupby(['k', 'model'])['score'].apply(
        lambda x: (x == 0).mean() * 100
    ).reset_index()
    
    sns.barplot(data=zero_scores, x='model', y='score', hue='k',
                hue_order=[0, 10, 50],
                palette='Set2')
    plt.title('Percentage of Zero Scores by Model and k')
    plt.xticks(rotation=0)
    plt.ylabel('Percentage of Zeros')
    plt.xlabel('Model')
    plt.legend(title='k')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/zero_scores_analysis.png')
    plt.close()
    
    # Score distribution analysis
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 1.0]
    bin_labels = ['0', '0-0.05', '0.05-0.1', '0.1-0.15', '0.15-0.2', '0.2-0.25', '0.25-0.3', '>0.3']
    
    # Create a function to bin scores with special handling for zeros
    def custom_bin(x):
        if x == 0:
            return '0'
        for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
            if left <= x < right:
                return bin_labels[i+1]
        return bin_labels[-1]
    
    df['score_bin'] = df['score'].apply(custom_bin)
    df['score_bin'] = pd.Categorical(df['score_bin'], categories=bin_labels, ordered=True)
    
    dist_data = df.groupby(['model', 'k', 'score_bin']).size().reset_index(name='count')
    dist_data['percentage'] = dist_data.groupby(['model', 'k'])['count'].transform(lambda x: x / x.sum() * 100)
    
    # Create distribution plot for each k value
    for k_val in [0, 10, 50]:
        plt.figure(figsize=(14, 7))
        k_data = dist_data[dist_data['k'] == k_val]
        
        sns.barplot(data=k_data, x='score_bin', y='percentage', hue='model',
                   palette='Set2')
        plt.title(f'Score Distribution (k={k_val})', pad=20)
        plt.xticks(rotation=45)
        plt.ylabel('Percentage of Scores')
        plt.xlabel('Score Range')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add a vertical line after the zero category
        plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.savefig(f'{output_dir}/score_distribution_k{k_val}.png', bbox_inches='tight', dpi=300)
        plt.close()

def main():
    """Main function to run the analysis"""
    # Create output directories if they don't exist
    visuals_dir = os.path.join('bfi', 'files', 'visuals')
    csvs_dir = os.path.join('bfi', 'files', 'csv')

    os.makedirs(visuals_dir, exist_ok=True)
    os.makedirs(csvs_dir, exist_ok=True)
    
    # Load and filter data
    input_file = 'evaluation/files/indv/eval_amazon_Grocery_and_Gourmet_Food_2018.json'
    filtered_data = load_and_filter_data(input_file)
    
    # Analyze scores
    results = analyze_scores(filtered_data)
    
    # Create visualizations
    analyze_score_transitions(results, csvs_dir)
    plot_comparisons(results, visuals_dir)

if __name__ == "__main__":
    main()
