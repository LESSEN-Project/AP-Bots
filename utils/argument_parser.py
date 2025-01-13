import argparse
import numpy as np
import copy

from exp_datasets import LampDataset, AmazonDataset

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--dataset", default="amazon_Grocery_and_Gourmet_Food_2018", type=str)
    parser.add_argument("-k", "--top_k", default=-1, type=int)
    parser.add_argument('-f', '--features', nargs='+', type=str, default=None)
    parser.add_argument("-r", "--retriever", default="contriever", type=str)
    parser.add_argument("-ce", "--counter_examples", default=None, type=int)
    parser.add_argument("-rs", "--repetition_step", default=1, type=int)
    parser.add_argument("-ob", "--openai_batch", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("-ps", "--prompt_style", default="regular", type=str)

    return parser.parse_args()

def parse_dataset(dataset):

    if dataset.startswith("lamp"):
        num = int(dataset.split("_")[1])
        data_split = dataset.split("_")[2]
        split = dataset.split("_")[-1]
        return LampDataset(num, data_split, split)
    
    elif dataset.startswith("amazon"):
        year = int(dataset.split("_")[-1])
        category = "_".join(dataset.split("_")[1:-1])
        return AmazonDataset(category, year)
    
    else:
        raise Exception("Dataset not known!")

def parse_args():

    final_feature_list = []

    args = get_args()
    dataset = parse_dataset(args.dataset)

    if args.features:
        final_feature_list = copy.copy(args.features)

    if args.counter_examples:
        final_feature_list.append(f"CE({args.counter_examples})")

    _, retr_texts, retr_gts = dataset.get_retr_data()
    if args.top_k == -1:
        k = get_k(retr_texts if dataset.name == "lamp" else retr_gts)
    else:
        k = args.top_k

    return args, dataset, final_feature_list, k

def get_k(retr_texts):

    mean = []
    for retr_text in retr_texts:
        mean.append(np.mean([len(text.split(" ")) for text in retr_text]))
    mean = np.mean(mean)
    if mean < 50:
        return 50
    else:
        return 10