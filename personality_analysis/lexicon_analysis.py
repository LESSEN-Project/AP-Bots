import spacy
from collections import Counter
import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import ttest_ind, mannwhitneyu, kruskal
import seaborn as sns
import matplotlib.pyplot as plt
import os

from utils.argument_parser import get_args, parse_dataset
from personality_analysis.analysis_utils import load_eval_results, load_predictions, get_model_and_k, get_exp_eval_results

nlp = spacy.load("en_core_web_sm")

ext_lexicons = {
    "good", "well", "new", "love",
    "we", "our", "us", "help",
    "really", "actually", "real",
    "said", "care", "friend"
}

agr_lexicons = {
    "wrong", "honor", "judge",
    "fight", "attack",
    "we", "our", "us", "help",
    "bad", "hate",
    "care", "thank"
}

con_lexicons = {
    "work", "better", "best",
    "work", "price", "market",
    "wrong", "honor", "judge",
    "fight", "attack",
    "when", "now", "then"
}

neu_lexicons = {
    "worry", "fear", "afraid",
    "bad", "wrong", "hate",
    "trauma", "depressed",
    "sad", "disappoint", "cry",
    "mad", "angry",
    "feel", "hard", "cool"
}

opn_lexicons = {
    "research", "wonder",
    "know", "how", "think",
    "we", "our", "us", "help",
    "see", "look", "eye",
    "will", "going", "to"
}

personality_lexicons = {
    "Extraversion": ext_lexicons,
    "Agreeableness": agr_lexicons,
    "Conscientiousness": con_lexicons,
    "Neuroticism": neu_lexicons,
    "Openness": opn_lexicons
}


def get_features(text):

    doc = nlp(text)
    tokens = [t.text.lower() for t in doc if not t.is_space and not t.is_punct]
    num_tokens = len(tokens)
    num_sents = len(list(doc.sents))

    # Basic lexical features
    avg_sentence_length = round((np.mean([len([t for t in sent if not t.is_space and not t.is_punct]) 
                                    for sent in doc.sents]) 
                           if num_sents > 0 else 0))
    type_token_ratio = round(len(set(tokens)) / (num_tokens + 1e-9) if num_tokens > 0 else 0, 4)

    # POS distribution
    pos_counts = Counter([token.pos_ for token in doc])
    noun_ratio = round(pos_counts.get("NOUN", 0) / (num_tokens + 1e-9), 4)
    verb_ratio = round(pos_counts.get("VERB", 0) / (num_tokens + 1e-9), 4)
    adj_ratio = round(pos_counts.get("ADJ", 0) / (num_tokens + 1e-9), 4)
    adv_ratio = round(pos_counts.get("ADV", 0) / (num_tokens + 1e-9), 4)

    # Sentiment
    sentiment = round(TextBlob(text).sentiment.polarity, 4)

    # Personality-based lexical features
    personality_features = {}
    for trait, lex_set in personality_lexicons.items():
        count = sum(1 for w in tokens if w in lex_set)
        ratio = round(count / (num_tokens + 1e-9), 4)
        personality_features[f"{trait.lower()}_ratio"] = ratio

    features = {
        "num_tokens": num_tokens,
        "num_sents": num_sents,
        "avg_sentence_length": avg_sentence_length,
        "type_token_ratio": type_token_ratio,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "adj_ratio": adj_ratio,
        "adv_ratio": adv_ratio,
        "sentiment_polarity": sentiment
    }
    features.update(personality_features)
    return features


def aggregate_features(text_list):
    all_features = [get_features(txt) for txt in text_list]
    df = pd.DataFrame(all_features)

    df = df.mean().round(4).to_dict()
    df["num_tokens"] = int(df["num_tokens"])
    df["num_sents"] = int(df["num_sents"])
    df["avg_sentence_length"] = int(df["avg_sentence_length"])

    return df


def calculate_feature_differences(user_features, llm_features):
    """Calculate absolute differences between user and LLM lexicon features."""
    differences = {}
    for feature in user_features:
        if feature in llm_features:
            differences[feature] = round(abs(user_features[feature] - llm_features[feature]), 4)
    return differences


def analyze_feature_impact(feature_differences, rouge_k0, rouge_k10, n_bins=5):
    """Analyze how ROUGE score changes across different feature ranges using binning."""
    rouge_change = [round(k10 - k0, 4) for k10, k0 in zip(rouge_k10, rouge_k0)]
    
    # Create DataFrame with all features and ROUGE change
    df = pd.DataFrame(feature_differences)
    df['rouge_change'] = rouge_change
    
    feature_impacts = {}
    for feature in df.columns:
        if feature != 'rouge_change':
            try:
                # Create bins with labels
                df['bin'], bin_edges = pd.qcut(df[feature], n_bins, retbins=True, labels=False)
                
                # Calculate mean ROUGE change for each bin
                bin_stats = df.groupby('bin')['rouge_change'].agg(['mean', 'std', 'count']).round(4)
                
                # Add bin boundaries
                bin_stats['start'] = bin_edges[:-1]
                bin_stats['end'] = bin_edges[1:]
                
                feature_impacts[feature] = bin_stats
                
                # Perform Kruskal-Wallis H-test to check if differences between bins are significant
                groups = [group['rouge_change'].values for _, group in df.groupby('bin')]
                if len(groups) > 1:  # Only perform test if we have at least 2 groups
                    h_stat, p_val = kruskal(*groups)
                    feature_impacts[feature].attrs['h_stat'] = round(h_stat, 4)
                    feature_impacts[feature].attrs['p_value'] = round(p_val, 4)
                else:
                    print(f"Skipping statistical test for {feature}: not enough distinct groups")
                    feature_impacts[feature].attrs['h_stat'] = 0
                    feature_impacts[feature].attrs['p_value'] = 1.0
                    
            except ValueError as e:
                # Skip features that can't be binned (e.g., constant values)
                print(f"Skipping {feature}: {str(e)}")
                continue
    
    return feature_impacts


def plot_feature_impacts(feature_impacts, model_key, output_dir):
    """Create visualization showing ROUGE score changes across feature bins."""
    features = list(feature_impacts.keys())
    if not features:
        print(f"No features to plot for {model_key}")
        return
        
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        stats = feature_impacts[feature]
        
        # Plot mean ROUGE change for each bin
        x = range(len(stats))
        ax.bar(x, stats['mean'], yerr=stats['std'], alpha=0.6)
        
        # Add bin ranges as x-tick labels
        labels = [f"[{start:.3f},\n{end:.3f}]" for start, end in zip(stats['start'], stats['end'])]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        ax.set_xlabel(f"{feature} bins")
        ax.set_ylabel('Mean ROUGE-L Change')
        
        # Add statistical test results
        h_stat = feature_impacts[feature].attrs['h_stat']
        p_val = feature_impacts[feature].attrs['p_value']
        ax.set_title(f"{feature}\nH={h_stat:.2f}, p={p_val:.4f}")
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Hide empty subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_key}_feature_impacts.png'), dpi=150)
    plt.close()


args = get_args()
dataset = parse_dataset(args.dataset)

_, _, retr_gts = dataset.get_retr_data()

eval_file = os.path.join("evaluation", "files", "indv", f"eval_{args.dataset}.json")
eval_results = load_eval_results(eval_file)

pred_dir = os.path.join("files", "preds")
predictions = load_predictions(pred_dir, list(eval_results.keys()))

base_dir = os.path.join("personality_analysis", "files")
visuals_dir = os.path.join("personality_analysis", "files", "visuals", "lexicon_analysis")
csv_dir = os.path.join("personality_analysis", "files", "csv")

os.makedirs(visuals_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

user_df_path = os.path.join(csv_dir, "lexicon_user_features.csv")

if os.path.exists(user_df_path):
    user_df = pd.read_csv(user_df_path)
else:
    user_features = []
    for i, u_texts in enumerate(retr_gts):
        user_agg = aggregate_features(u_texts)  
        user_features.append(user_agg)

        if (i+1) % 100 == 0:
            print(i)

    user_df = pd.DataFrame(user_features)
    user_df.to_csv(user_df_path, index=False)

# Store k=0 data for later comparison
k0_data = {}
for model_key in predictions:
    if '0' in predictions[model_key]:
        print(f"\nAnalyzing initial differences for {model_key}...")
        
        # Calculate initial differences
        differences_list = []
        for i, (llm_text, user_feats) in enumerate(zip(predictions[model_key]['0'], 
                                                      user_df.to_dict('records'))):
            llm_feats = get_features(llm_text)
            differences = calculate_feature_differences(user_feats, llm_feats)
            differences_list.append(differences)
        
        k0_data[model_key] = {
            'differences': differences_list,
            'rouge': get_exp_eval_results(eval_results, model_key, '0')
        }

# Compare with k=10 data
for model_key in k0_data:
    if '10' in predictions[model_key]:
        print(f"\nAnalyzing ROUGE score changes for {model_key}...")
        
        # Get k=10 ROUGE scores
        rouge_k10 = get_exp_eval_results(eval_results, model_key, '10')
        
        # Analyze feature impacts on ROUGE changes
        feature_impacts = analyze_feature_impact(
            k0_data[model_key]['differences'],
            k0_data[model_key]['rouge'],
            rouge_k10
        )
        
        # Print results for features with significant differences
        print("\nFeature impacts on ROUGE-L changes:")
        for feature, stats in feature_impacts.items():
            if stats.attrs['p_value'] < 0.05:  # Show only significant results
                print(f"\n{feature} (H={stats.attrs['h_stat']:.2f}, p={stats.attrs['p_value']:.4f}):")
                print(stats[['mean', 'std', 'count']].round(4))
        
        # Create visualizations
        plot_feature_impacts(feature_impacts, model_key, visuals_dir)

for model_key in predictions:
    for k_key in predictions[model_key]:

        rougeL = get_exp_eval_results(eval_results, model_key, k_key)

        print(f"Lexicon analysis for {model_key, k_key}:")

        llm_texts = predictions[model_key][k_key]
        
        # Calculate LLM features and differences
        differences_list = []
        for i, (llm_text, user_feats) in enumerate(zip(llm_texts, user_df.to_dict('records'))):
            llm_feats = get_features(llm_text)
            differences = calculate_feature_differences(user_feats, llm_feats)
            differences_list.append(differences)
            
            if (i+1) % 100 == 0:
                print(f"Processed {i+1} samples")
        
        llm_df_path = os.path.join(csv_dir, f"lexicon_{model_key}_{k_key}_features.csv")

        if os.path.exists(llm_df_path):
            llm_df = pd.read_csv(llm_df_path)
        else:
            llm_df = pd.DataFrame([get_features(text) for text in llm_texts])
            llm_df.to_csv(llm_df_path, index=False)

        print("Aggregated user features and single LLM text features saved.")

        features = [c for c in user_df.columns]

        user_plot_df = user_df.assign(Source="User")
        llm_plot_df = llm_df.assign(Source="LLM")
        combined_df = pd.concat([user_plot_df, llm_plot_df], ignore_index=True)

        print("\n=== Statistical Tests at User Level ===")
        for feature in features:
            user_values = user_df[feature].dropna()
            llm_values = llm_df[feature].dropna()
            if len(user_values) > 1 and len(llm_values) > 1:
                t_stat, t_pval = ttest_ind(user_values, llm_values, equal_var=False)
                u_stat, u_pval = mannwhitneyu(user_values, llm_values, alternative='two-sided')

                print(f"Feature: {feature}")
                print(f"  Mean (User): {user_values.mean():.4f}, Mean (LLM): {llm_values.mean():.4f}")
                print(f"  t-test p-value: {t_pval:.4f}")
                print(f"  Mann-Whitney U p-value: {u_pval:.4f}\n")
            else:
                print(f"Feature: {feature}")
                print("  Not enough data for statistical tests.\n")

        """
        for feature in features:
            plt.figure(figsize=(8,6))
            sns.boxplot(x="Source", y=feature, data=combined_df)
            plt.title(f"{feature} Distribution across Users")
            plt.savefig(os.path.join(visuals_dir, f"{model_key}_{k_key}_{feature}_user_boxplot.png"), dpi=150)
            plt.close()
        """

        mean_values = combined_df.groupby("Source")[features].mean().reset_index()
        melted_means = mean_values.melt(id_vars="Source", value_vars=features, var_name="Feature", value_name="MeanValue")

        # Add ROUGE-L scores to the feature dataframes
        user_df_with_rouge = user_df[features].copy()
        user_df_with_rouge['rouge_L'] = rougeL
        
        llm_df_with_rouge = llm_df[features].copy()
        llm_df_with_rouge['rouge_L'] = rougeL

        plt.figure(figsize=(10,8))
        sns.heatmap(user_df_with_rouge.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt='.2f')
        plt.title("User Feature Correlations with ROUGE-L (User texts)")
        plt.tight_layout()
        plt.savefig(os.path.join(visuals_dir, f"{model_key}_{k_key}_user_feature_correlations_users.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(10,8))
        sns.heatmap(llm_df_with_rouge.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt='.2f')
        plt.title("LLM Feature Correlations with ROUGE-L (LLM texts)")
        plt.tight_layout()
        plt.savefig(os.path.join(visuals_dir, f"{model_key}_{k_key}_llm_feature_correlations_users.png"), dpi=150)
        plt.close()

        print("Analysis completed. Check CSV files and 'plots' directory for results.")