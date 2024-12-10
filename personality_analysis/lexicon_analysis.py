import spacy
from collections import Counter
import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy.stats import ttest_ind, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
import os

from utils.argument_parser import get_args, parse_dataset
from personality_analysis.analysis_utils import load_eval_results, load_predictions, get_model_and_k

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


def plot_feature_differences(user_df, llm_df, model_key, k_key, output_dir):
    """Create plots showing feature differences between user and LLM texts"""
    features = [c for c in user_df.columns]
    
    # Create difference plot
    plt.figure(figsize=(15, 8))
    differences = []
    for feature in features:
        user_mean = user_df[feature].mean()
        llm_mean = llm_df[feature].mean()
        # Calculate percentage difference
        if user_mean != 0:
            pct_diff = ((llm_mean - user_mean) / user_mean) * 100
        else:
            pct_diff = 0 if llm_mean == 0 else 100
        differences.append(pct_diff)
    
    colors = ['red' if d < 0 else 'blue' for d in differences]
    plt.bar(features, differences, color=colors)
    plt.xticks(rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title(f'Feature Differences (% change from User) for {model_key} k={k_key}')
    plt.ylabel('Percentage Difference (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_key}_{k_key}_feature_differences.png'), dpi=150)
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

for model_key in predictions:
    for k_key in ['0', '10']:
        if k_key in predictions[model_key]:
            print(f"\nAnalyzing features for {model_key} (k={k_key})...")

            llm_texts = predictions[model_key][k_key]
            llm_df_path = os.path.join(csv_dir, f"lexicon_{model_key}_{k_key}_features.csv")

            if os.path.exists(llm_df_path):
                llm_df = pd.read_csv(llm_df_path)
            else:
                # Calculate LLM features
                llm_features = []
                for i, llm_text in enumerate(llm_texts):
                    llm_feats = get_features(llm_text)
                    llm_features.append(llm_feats)
                    
                    if (i+1) % 100 == 0:
                        print(f"Processed {i+1} samples")
                
                llm_df = pd.DataFrame(llm_features)
                llm_df.to_csv(llm_df_path, index=False)
            
            # Calculate differences
            differences_list = []
            for user_feats, llm_feats in zip(user_df.to_dict('records'), llm_df.to_dict('records')):
                differences = calculate_feature_differences(user_feats, llm_feats)
                differences_list.append(differences)

            features = [c for c in user_df.columns]

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

            # Create visualization of feature differences
            plot_feature_differences(user_df, llm_df, model_key, k_key, visuals_dir)

            print(f"Analysis completed for {model_key} k={k_key}. Check CSV files and 'plots' directory for results.")