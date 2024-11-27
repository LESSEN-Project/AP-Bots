## First look at the error analysis. Use the best performing method only.
## Use this code to get the BFI analysis of different methods. Include k=0 and user profile analysis.

import os
import json
import sys

import numpy as np
from openai import OpenAI

from models import LLM
from prompts import BFI_analysis
from utils import parse_dataset, get_args, oai_get_or_create_file, get_k

out_path = "bfi"
os.makedirs(out_path, exist_ok=True)

bfi_model = "GEMMA-2-27B"
# bfi_model = "GPT-4o-mini"
llm = LLM(model_name=bfi_model)

all_models = ["UP", "MINISTRAL-8B-INSTRUCT", "LLAMA-3.2-3B", "GEMMA-2-2B", "LLAMA-3.1-8B", "GEMMA-2-9B", "GEMMA-2-27B"]
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.01

args = get_args()
dataset = parse_dataset(args.dataset)

_, retr_texts, retr_gts = dataset.get_retr_data() 

print(f"Number of users: {len(retr_texts)}")

final_feature_list = []
if args.features:
    final_feature_list = args.features
else:
    features = None

if args.counter_examples:
    final_feature_list.append(f"CE({args.counter_examples})")

if args.top_k == -1:
    k = get_k(retr_texts if dataset.name == "lamp" else retr_gts)
else:
    k = args.top_k

for model_name in all_models:

    if model_name == "UP":
        exp_name = f"{dataset.tag}_{model_name}"
    else:
        exp_name = f"{dataset.tag}_{model_name}_{final_feature_list}_{args.retriever}_RS({args.repetition_step})_K({k}))"

    print(exp_name)
    pred_out_path = os.path.join("preds", f"{exp_name}.json")
    bfi_out_path = os.path.join(out_path, f"{exp_name}_BFI.json")

    if os.path.exists(bfi_out_path):
        print("BFI analysis results already exist for the experiment!")
        continue

    if model_name != "UP":
        if not os.path.exists(pred_out_path):
            print("Predictions for this experiment doesn't exist!")
            continue

        with open(pred_out_path, "r") as f:
            preds = json.load(f)["golds"]
            preds = [p["output"] for p in preds]

        if len(preds) != len(retr_texts):
            print("Predictions for this experiment is not concluded yet!")
            continue
    
    else:
        preds = retr_texts if dataset.name == "lamp" else retr_gts

    all_prompts = []
    for i in range(len(preds)):

        reviews = preds[i]

        if isinstance(reviews, list):
            if dataset.name == "amazon":

                _, ratings = dataset.get_ratings(i)
                reviews = [f"\nReview: {review}\nRating: {rating}\n" for review, rating in zip(reviews, ratings)]

            max_k = 10 if len(reviews) > 10 else len(reviews)
            reviews = np.random.choice(reviews, size=max_k, replace=False)
            context = llm.prepare_context(BFI_analysis(text=""), reviews)

        else:
            if dataset.name == "amazon":

                rating, _ = dataset.get_ratings(i)       
                context = f"\nReview: {reviews}\nRating: {rating}\n"  

            else:
                context = reviews   
        
        all_prompts.append(BFI_analysis(context))

    if llm.family == "GPT" and args.openai_batch:

        batch_file_path = os.path.join(out_path, f"{args.dataset}_{model_name}.jsonl")

        with open(batch_file_path, "w") as file:
            for i, prompt in enumerate(all_prompts):

                json_line = json.dumps({"custom_id": str(i), "method": "POST", "url": "/v1/chat/completions", 
                                        "body": {"model": llm.repo_id, 
                                                "messages": prompt, "max_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE}})
                file.write(json_line + '\n')

        client = OpenAI()
        batch_input_file_id = oai_get_or_create_file(client, f"{args.dataset}_batch.jsonl")

        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

    else:

        if os.path.exists(bfi_out_path):
            with open(bfi_out_path, "w") as f:
                bfi_results = json.load(f)
                print(f"Continuing from index {len(bfi_results)}!")
        else:
            bfi_results = []
        
        start_index = len(bfi_results)

        for _ in range(len(all_prompts[start_index:])):

            response = llm.prompt_chatbot(all_prompts[start_index], gen_params={"max_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE})
            bfi_results.append(response)
            
            if start_index + 1 % 500 == 0:
                print(f"Step: {start_index}")  
                with open(bfi_out_path, "w") as f:
                    json.dump(bfi_results, f)

            start_index = start_index + 1
            sys.stdout.flush()

        print("Finished experiment!")

        with open(bfi_out_path, "w") as f:
            json.dump(bfi_results, f)