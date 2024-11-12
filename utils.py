import argparse
import os
import json
import random
import re
import numpy as np

from exp_datasets import LampDataset, AmazonDataset

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--dataset", default="lamp_5_dev_user", type=str)
    parser.add_argument("-k", "--top_k", default=None, type=int)
    parser.add_argument('-f', '--features', nargs='+', type=str, default=None)
    parser.add_argument("-r", "--retriever", default="contriever", type=str)
    parser.add_argument("-ce", "--counter_examples", default=None, type=int)
    parser.add_argument("-rs", "--repetition_step", default=1, type=int)

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

def list_files_in_directory(root_dir):

    file_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def shuffle_lists(list1, list2):

   zipped_list = list(zip(list1, list2))
   random.shuffle(zipped_list)
   list1_shuffled, list2_shuffled = zip(*zipped_list)
   list1_shuffled = list(list1_shuffled)
   list2_shuffled = list(list2_shuffled)
   return list1_shuffled, list2_shuffled

def softmax(x):

    e_x = np.exp(x - np.max(x))
    return (e_x/e_x.sum()).tolist()

def get_k(retr_texts):

    mean = []
    for retr_text in retr_texts:
        mean.append(np.mean([len(text.split(" ")) for text in retr_text]))
    mean = np.mean(mean)
    if mean < 50:
        return 50
    else:
        return 7

def oai_get_or_create_file(client, filename):

    files = client.files.list()
    existing_file = next((file for file in files if file.filename == filename), None)

    if existing_file:
        print(f"File '{filename}' already exists. File ID: {existing_file.id}")
        return existing_file.id
    else:
        with open(os.path.join("preds", filename), "rb") as file_data:
            new_file = client.files.create(
                file=file_data,
                purpose="batch"
            )
        print(f"File '{filename}' created. File ID: {new_file.id}")
        return new_file.id
    
def parse_json(output):

    try:
        idx = output.find("{")
        if idx != 0:
            output = output[idx:]
            if output.endswith("```"):
                output = output[:-3]
        output = json.loads(output, strict=False)["Title"]
    except Exception as e:
        try:
            match = re.search(r'"Title":\s*(.+)$', output, re.MULTILINE)
            if match:
                return match.group(1).strip().rstrip(',').strip()
            else:
                match = re.search(r'"title":\s*(.+)$', output, re.MULTILINE)
                if match:
                    return match.group(1).strip().rstrip(',').strip()
        except Exception as e:
            print(output)
            print(e)
    return output

def parse_cot_output(output: str) -> str:

    last_line = output.strip().splitlines()[-1]
    last_line = last_line.replace('"', '')
    last_line = last_line.replace('*', '')
    return last_line