import os
import json
import sys
import copy

import torch 
import numpy as np
from openai import OpenAI

from AP_Bots.models import LLM
from AP_Bots.prompts import get_BFI_prompts
from AP_Bots.utils.argument_parser import parse_args
from AP_Bots.utils.file_utils import oai_get_or_create_file
from AP_Bots.utils.misc import get_model_list

pred_path = os.path.join("files", "preds")
bfi_path = os.path.join("personality_analysis", "files", "inferred_bfi")
os.makedirs(bfi_path, exist_ok=True)
args, dataset, final_feature_list, k = parse_args()
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.01

bfi_model = "LLAMA-3.3-70B"
print(f"BFI Model: {bfi_model}")

model_params = None
if bfi_model.endswith("70B"):
    model_params = {
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True
        }
    }
llm = LLM(model_name=bfi_model, model_params=model_params)

all_models = get_model_list() + ["UP"]

_, retr_texts, retr_gts = dataset.get_retr_data() 
print(f"Number of users: {len(retr_texts)}")

for model_name in all_models:

    if model_name == "UP":
        exp_name = f"{dataset.tag}_{model_name}"
    else:
        exp_name = f"{dataset.tag}_{model_name}_{final_feature_list}_{args.retriever}_RS({args.repetition_step})_K({k})"

    print(exp_name)
    pred_out_path = os.path.join(pred_path, f"{exp_name}.json")
    bfi_out_path = os.path.join(bfi_path, f"{exp_name}_BFI_{bfi_model}.json")

    if os.path.exists(bfi_out_path):
        with open(bfi_out_path, "r") as f:
            bfi_results = json.load(f)
        
        if len(bfi_results) == len(retr_texts):
            print("BFI results already exists for the experiment!")
            continue
        else:
            print(f"Continuing from index {len(bfi_results)}!")

    else:
        bfi_results = []


    if os.path.exists(pred_out_path):

        with open(pred_out_path, "r") as f:
            preds = json.load(f)["golds"]
            preds = [p["output"] for p in preds]

        if len(preds) != len(retr_texts):
            print("Predictions for this experiment is not finished yet!")
            continue

    elif model_name == "UP":

        preds = retr_texts if dataset.name == "lamp" else retr_gts
    
    else:
        
        print("Predictions for this experiment doesn't exist!")
        continue
        
    all_prompts = []
    for i in range(len(preds)):

        reviews = preds[i]

        if isinstance(reviews, list):
            if dataset.name == "amazon":

                _, ratings = dataset.get_ratings(i)
                reviews = [f"\nReview: {review}\nRating: {rating}\n" for review, rating in zip(reviews, ratings)]

            max_k = 10 if len(reviews) > 10 else len(reviews)
            reviews = np.random.choice(reviews, size=max_k, replace=False)
            context = llm.prepare_context(get_BFI_prompts(dataset, text=""), reviews)

        else:
            if dataset.name == "amazon":

                rating, _ = dataset.get_ratings(i)       
                context = f"\nReview: {reviews}\nRating: {rating}\n"  

            else:
                context = reviews   
        
        all_prompts.append(get_BFI_prompts(dataset, context))

    if llm.family == "GPT" and args.openai_batch:

        batch_file_path = os.path.join(bfi_path, f"{exp_name}_BFI_{bfi_model}.jsonl")

        with open(batch_file_path, "w") as file:
            for i, prompt in enumerate(all_prompts):

                json_line = json.dumps({"custom_id": str(i), "method": "POST", "url": "/v1/chat/completions", 
                                        "body": {"model": llm.repo_id, 
                                                "messages": prompt, "max_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE}})
                file.write(json_line + '\n')

        client = OpenAI()
        batch_input_file_id = oai_get_or_create_file(client, batch_file_path)

        client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

    else:
        
        start_index = copy.copy(len(bfi_results))

        for _ in range(len(all_prompts[start_index:])):

            response = llm.prompt_chatbot(all_prompts[start_index], gen_params={"max_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE})
            bfi_results.append(response)
            
            if (start_index+1) % 500 == 0:
                print(f"Step: {start_index+1}")  
                with open(bfi_out_path, "w") as f:
                    json.dump(bfi_results, f)

            start_index = start_index + 1
            sys.stdout.flush()

        print("Finished experiment!")

        with open(bfi_out_path, "w") as f:
            json.dump(bfi_results, f)