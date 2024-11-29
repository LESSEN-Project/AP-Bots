import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple

def load_individual_scores(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def get_model_scores_by_k(scores_data: Dict) -> Dict[str, Dict[int, List[float]]]:
    """Get scores for each model at k=0 and k=50"""
    model_scores = {}
    
    for exp_name, exp_data in scores_data.items():
        params = exp_data['params']
        
        # Filter for our conditions
        if (params['features'] == "" and 
            params['RS'] == "1" and 
            params['retriever'] == "contriever" and 
            params['k'] in ['0', '50']):
            
            model = params['model']
            k = int(params['k'])
            
            if model not in model_scores:
                model_scores[model] = {0: None, 50: None}
            
            model_scores[model][k] = exp_data['rougeL']
    
    # Only keep models that have both k=0 and k=50
    return {model: scores for model, scores in model_scores.items() 
            if scores[0] is not None and scores[50] is not None}

def analyze_k_impact(model_scores: Dict[str, Dict[int, List[float]]]) -> Dict[str, Dict]:
    """Analyze how scores change from k=0 to k=50 for each model"""
    results = {}
    
    for model, k_scores in model_scores.items():
        scores_k0 = np.array(k_scores[0])
        scores_k50 = np.array(k_scores[50])
        
        # Calculate differences
        differences = scores_k50 - scores_k0
        
        # Get indices of improved and degraded samples
        improved_idx = np.where(differences > 0)[0]
        degraded_idx = np.where(differences < 0)[0]
        unchanged_idx = np.where(differences == 0)[0]
        
        # Calculate statistics
        results[model] = {
            'total_samples': len(scores_k0),
            'improved': {
                'count': len(improved_idx),
                'percentage': len(improved_idx) / len(scores_k0) * 100,
                'indices': improved_idx.tolist(),
                'avg_improvement': np.mean(differences[improved_idx]) if len(improved_idx) > 0 else 0
            },
            'degraded': {
                'count': len(degraded_idx),
                'percentage': len(degraded_idx) / len(scores_k0) * 100,
                'indices': degraded_idx.tolist(),
                'avg_degradation': np.mean(differences[degraded_idx]) if len(degraded_idx) > 0 else 0
            },
            'unchanged': {
                'count': len(unchanged_idx),
                'percentage': len(unchanged_idx) / len(scores_k0) * 100,
                'indices': unchanged_idx.tolist()
            }
        }
    
    return results

def find_overlapping_samples(impact_results: Dict[str, Dict]) -> Dict[str, Set[int]]:
    """Find samples that consistently improve or degrade across models"""
    models = list(impact_results.keys())
    if not models:
        return {}
    
    # Initialize with first model's indices
    degraded_samples = set(impact_results[models[0]]['degraded']['indices'])
    improved_samples = set(impact_results[models[0]]['improved']['indices'])
    
    # Intersect with other models
    for model in models[1:]:
        degraded_samples &= set(impact_results[model]['degraded']['indices'])
        improved_samples &= set(impact_results[model]['improved']['indices'])
    
    return {
        'consistently_degraded': degraded_samples,
        'consistently_improved': improved_samples
    }

def main():
    # Load data
    file_path = "evaluation/files/indv/eval_amazon_Grocery_and_Gourmet_Food_2018.json"
    scores_data = load_individual_scores(file_path)
    print(scores_data.keys())
    
    print("\n=== Available Experiments ===")
    for exp_name, exp_data in scores_data.items():
        params = exp_data['params']
        if (params['features'] == "" and 
            params['RS'] == "1" and 
            params['retriever'] == "contriever"):
            print(f"- {exp_name}")
            print(f"  Model: {params['model']}, k={params['k']}")
    
    # Get scores by model and k
    model_scores = get_model_scores_by_k(scores_data)
    
    print("\n=== Models with both k=0 and k=50 experiments ===")
    for model in model_scores.keys():
        print(f"- {model}")
    
    # Analyze impact of increasing k
    impact_results = analyze_k_impact(model_scores)
    
    print("\n=== Impact Analysis of Increasing k from 0 to 50 ===")
    for model, results in impact_results.items():
        print(f"\n{model}:")
        print(f"Total samples: {results['total_samples']}")
        
        print("\nImproved samples:")
        print(f"- Count: {results['improved']['count']} ({results['improved']['percentage']:.2f}%)")
        print(f"- Average improvement: {results['improved']['avg_improvement']:.4f}")
        
        print("\nDegraded samples:")
        print(f"- Count: {results['degraded']['count']} ({results['degraded']['percentage']:.2f}%)")
        print(f"- Average degradation: {results['degraded']['avg_degradation']:.4f}")
        
        print("\nUnchanged samples:")
        print(f"- Count: {results['unchanged']['count']} ({results['unchanged']['percentage']:.2f}%)")
    
    # Find overlapping samples
    overlapping = find_overlapping_samples(impact_results)
    
    print("\n=== Overlapping Samples Across All Models ===")
    print(f"Samples that consistently degrade: {len(overlapping['consistently_degraded'])}")
    if overlapping['consistently_degraded']:
        print("Sample indices:", sorted(list(overlapping['consistently_degraded'])))
    
    print(f"\nSamples that consistently improve: {len(overlapping['consistently_improved'])}")
    if overlapping['consistently_improved']:
        print("Sample indices:", sorted(list(overlapping['consistently_improved'])))

if __name__ == "__main__":
    main()
