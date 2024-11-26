import os
import json
import sys

import numpy as np
from openai import OpenAI

from models import LLM
from prompts import BFI_analysis
from utils import parse_dataset, get_bfi_args, oai_get_or_create_file

out_path = "preds"
os.makedirs(out_path, exist_ok=True)

model_name = "LLAMA-3.1-70B"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.01

args = get_bfi_args()
dataset = parse_dataset(args.dataset)

_, _, retr_gts = dataset.get_retr_data() 
print(f"Number of users: {len(retr_gts)}")

bfi_file_name = os.path.join(out_path, f"{args.dataset}_{model_name}_user_BFI_scores.json")

if os.path.exists(bfi_file_name):
    print("Ground truth already exists for the model!")
    exit

llm = LLM(model_name=model_name)

all_prompts = []
for i in range(len(retr_gts)):

    reviews = retr_gts[i]
    _, ratings = dataset.get_ratings(i)

    combined_reviews = [f"\nReview: {review}\nRating: {rating}\n" for review, rating in zip(reviews, ratings)]
    max_k = 20 if len(combined_reviews) > 20 else len(combined_reviews)
    combined_reviews = np.random.choice(combined_reviews, size=max_k, replace=False)

    context = llm.prepare_context(BFI_analysis(text=""), combined_reviews)
    all_prompts.append(BFI_analysis(context))

if llm.family == "GPT":

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
    if os.path.exists(bfi_file_name):
        with open(bfi_file_name, "w") as f:
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
            with open(bfi_file_name, "w") as f:
                json.dump(bfi_results, f)

        print(start_index)
        start_index = start_index + 1
        sys.stdout.flush()

    print("Finished experiment!")

    with open(bfi_file_name, "w") as f:
        json.dump(bfi_results, f)