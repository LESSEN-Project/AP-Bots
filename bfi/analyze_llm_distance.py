import json
import os
import argparse
import numpy as np
from exp_datasets import AmazonDataset
from typing import Dict, List, Any
from retriever import Retriever
import matplotlib.pyplot as plt
from scipy import stats

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
            params.get('k') == 0):  # Only k=0 for now
            filtered_results[key] = value
    
    return filtered_results

def load_predictions(pred_dir: str, experiment_keys: List[str]) -> Dict[str, List[str]]:
    """Load predictions from files corresponding to filtered experiments."""
    predictions = {}
    for exp_key in experiment_keys:
        pred_file = os.path.join(pred_dir, f"{exp_key}.json")
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
                # Extract predictions from the golds list
                preds = []
                for item in pred_data.get('golds', []):
                    if isinstance(item, dict) and 'output' in item:
                        preds.append(item['output'])
                predictions[exp_key] = preds
    
    return predictions

def calculate_distances(retriever: Retriever, predictions: List[str], ground_truth: List[str]) -> np.ndarray:
    """Calculate semantic distances between predictions and ground truth."""
    # Get similarities using the retriever
    similarities, _ = retriever._neural_retrieval(predictions, ground_truth)
    
    # Convert similarities to distances (1 - similarity)
    if isinstance(similarities, float):
        distances = [1 - similarities]
    else:
        distances = 1 - np.array(similarities)
    return distances

def analyze_distances(distances: np.ndarray, exp_key: str):
    """Analyze and plot the distribution of distances."""
    print(f"\nAnalysis for {exp_key}:")
    print(f"Mean distance: {np.mean(distances):.4f}")
    print(f"Median distance: {np.median(distances):.4f}")
    print(f"Std deviation: {np.std(distances):.4f}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, density=True, alpha=0.7)
    plt.axvline(np.mean(distances), color='r', linestyle='dashed', linewidth=2, label=f'Mean ({np.mean(distances):.4f})')
    plt.axvline(np.median(distances), color='g', linestyle='dashed', linewidth=2, label=f'Median ({np.median(distances):.4f})')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(distances)
    x = np.linspace(min(distances), max(distances), 100)
    p = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2, label='Normal fit')
    
    plt.title(f'Distribution of Semantic Distances\n{exp_key}')
    plt.xlabel('Distance (1 - cosine similarity)')
    plt.ylabel('Density')
    plt.legend()
    
    # Save plot
    os.makedirs("files/plots", exist_ok=True)
    plt.savefig(os.path.join("files/plots", f'distance_distribution_{exp_key}.png'))
    plt.close()

def main():
    # Parse arguments
    args = get_args()
    dataset = parse_dataset(args.dataset)
    
    # File paths
    eval_file = os.path.join("evaluation", "files", "indv", f"eval_{dataset.tag}.json")
    pred_dir = os.path.join("files", "preds")
    
    # Load and filter evaluation results
    filtered_results = load_eval_results(eval_file)
    print(f"Found {len(filtered_results)} matching experiments")
    
    # Load predictions
    predictions = load_predictions(pred_dir, list(filtered_results.keys()))
    print(f"Loaded predictions for {len(predictions)} experiments")
    
    # Load ground truth
    ground_truth = dataset.get_gts()
    print(f"Loaded {len(ground_truth)} ground truth samples")
    
    # Initialize retriever with specified model
    retriever = Retriever(dataset, model="sentence-transformers/all-MiniLM-L6-v2", device="cuda:0")
    
    # Calculate and analyze distances for each experiment
    for exp_key, preds in predictions.items():
        if len(preds) != len(ground_truth):
            print(f"Warning: Number of predictions ({len(preds)}) doesn't match ground truth ({len(ground_truth)}) for {exp_key}")
            continue
            
        print(f"\nCalculating distances for {exp_key}...")
        distances = calculate_distances(retriever, preds, ground_truth)
        analyze_distances(distances, exp_key)

if __name__ == "__main__":
    main()
