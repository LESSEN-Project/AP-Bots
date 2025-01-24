import os
import time
import json
import sys
import copy

import torch 

from models import LLM
from prompts import prepare_res_prompt
from feature_processor import FeatureProcessor
from retriever import Retriever

from AP_Bots.utils.argument_parser import parse_args
from AP_Bots.utils.file_utils import oai_get_or_create_file
from AP_Bots.utils.misc import get_model_list

args, dataset, final_feature_list, k = parse_args()
MAX_NEW_TOKENS = 64 if dataset.name == "lamp" else 128
MAX_NEW_TOKENS = MAX_NEW_TOKENS if args.prompt_style == "regular" else MAX_NEW_TOKENS * 10
pred_path = os.path.join("files", "preds")
os.makedirs(pred_path, exist_ok=True)

if dataset.name == "lamp":
    ids = dataset.get_ids()    

LLMs = get_model_list()
# LLMs = ["GPT-4o-mini"]

retriever = Retriever(dataset, args.retriever)
all_context = retriever.get_context(k) 

if args.features:
    feature_processor = FeatureProcessor(dataset)
    prepared_features = feature_processor.prepare_features(args.features)
else:
    features = None

if args.counter_examples:
    ce_k = 3 if k == 50 else 1
    all_ce_examples = retriever.contrastive_retrieval(args.counter_examples, ce_k)

queries, _, _ = dataset.get_retr_data() 

print(f"Running experiments for {dataset.tag} with Features: {final_feature_list}, Retriever: {args.retriever}, Repetition Step: {args.repetition_step}, Prompt Style: {args.prompt_style} and K: {k}")
sys.stdout.flush()

for model_name in LLMs:

    exp_name = f"{dataset.tag}_{model_name}_{final_feature_list}_{args.retriever}_RS({args.repetition_step})_K({k})"

    if args.prompt_style == "react":
        exp_name = f"{exp_name}_PS({args.prompt_style})"

    out_path = os.path.join(pred_path, f"{exp_name}.json")

    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
             all_res = json.load(f)["golds"]
    else:
        all_res = []

    print(model_name) 
    if len(all_res) == len(queries):
        print("Experiment for this LLM is already concluded!")
        continue

    elif len(all_res) != 0 and args.openai_batch:
        print("Batch openai jobs can only be done on the whole dataset!")
        continue

    MAX_NEW_TOKENS = MAX_NEW_TOKENS * 20 if model_name.startswith("R1") else MAX_NEW_TOKENS
    model_params = None
    if model_name.endswith("70B"):
        print("70B model, using quantization!")
        model_params = {
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True
            }
        }
    
    llm = LLM(model_name=model_name, model_params=model_params)

    print(f"Starting from sample no. {len(all_res)}")

    start_time = time.time()
    sys.stdout.flush() 

    cont_idx = copy.copy(len(all_res))

    for _ in range(len(queries) - len(all_res)):
        
        query = queries[cont_idx]       
        if dataset.name == "amazon":
            query_rating, _ = dataset.get_ratings(cont_idx) 
            query = f"{query}\nRating:\n{query_rating}"
            
        context = all_context[cont_idx]    

        if args.features:
            features = prepared_features[cont_idx]
        
        if args.counter_examples:
            ce_examples = all_ce_examples[cont_idx]
        else:
            ce_examples = None

        start_bot_time = time.time() 

        prompt = prepare_res_prompt(dataset, query, llm, examples=context, features=features, counter_examples=ce_examples, repetition_step=args.repetition_step, prompt_style=args.prompt_style)
        prompt = [{"role": "user", "content": prompt}]
        id = ids[cont_idx] if dataset.name == "lamp" else cont_idx

        if llm.family == "GPT" and args.openai_batch:

            with open(os.path.join(pred_path, f"{exp_name}.jsonl"), "a+") as file:
                        json_line = json.dumps({"custom_id": str(id), "method": "POST", "url": "/v1/chat/completions", 
                                                "body": {"model": llm.repo_id, 
                                                "messages": prompt, "max_tokens": MAX_NEW_TOKENS}})
                        file.write(json_line + '\n')

        else:

            res = llm.prompt_chatbot(prompt, gen_params={"max_new_tokens": MAX_NEW_TOKENS})
            end_bot_time = time.time()
            all_res.append({
                    "id": id,
                    "output": res,
                    "prompt": prompt,
                    "model_inf_time": round(end_bot_time - start_bot_time, 2), 
            })

            if (cont_idx+1)%500==0 or (cont_idx+1)==len(queries):
                print(cont_idx+1)
                with open(out_path, "w") as f:
                    task = f"LaMP_{dataset.num}" if dataset.name == "lamp" else dataset.tag          
                    json.dump({
                        "task": task,
                        "golds": all_res
                    }, f)

        sys.stdout.flush()
        cont_idx += 1 

    if llm.family == "GPT" and args.openai_batch:

        print("Created batch job for the experiment!")
        batch_input_file_id = oai_get_or_create_file(llm.model, os.path.join(pred_path, f"{exp_name}.jsonl"))

        llm.model.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )       

    else:

        end_time = time.time()
        print(f"Took {(end_time-start_time)/3600} hours!")
        del llm
        llm = []
        torch.cuda.empty_cache()