import json
import argparse
from datasets import load_dataset

from AP_Bots.models import LLM
from AP_Bots.sentiment_analysis.helpers import *


parser = argparse.ArgumentParser()
parser.add_argument("-k", "--few_k", default=1, type=int)
parser.add_argument("-l", "--language", default="nl", type=str)
parser.add_argument("-ob", "--openai_batch", default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

llm_list = ["GPT-4o-mini", "LLAMA-3.1-8B", "LLAMA-3.2-3B", "GEMMA-2-2B"]

translator = LLM("GPT-4o-mini", default_prompt=get_translate_prompt())

daily_dialog_train = load_dataset("li2017dailydialog/daily_dialog")["train"]
daily_dialog_test = load_dataset("li2017dailydialog/daily_dialog")["test"]

few_shot_examples = get_few_shot_samples(daily_dialog_train, translator, 1)
emotion_labels_inverse = get_emotion_labels_inverse()

for llm in llm_list:

    if args.language == "nl":
        classifier = LLM(llm, default_prompt=get_classifier_prompt_nl())

    all_res = []
    exp_name = f"preds_{llm}_{args.language}_k({args.few_k})"

    for i, sample in enumerate(daily_dialog_test):

        try:
            params = {"text": str(daily_dialog_test["dialog"][i])}
            translated_text = translator.generate(prompt_params=params)
            pred = classifier.generate(prompt_params={"text": translated_text, "few_shot_examples": few_shot_examples})
            all_res.append([emotion_labels_inverse[p] for p in eval(pred)])
        except Exception as e:
            print(e)
            all_res.append([])
            
        with open(f"{exp_name}.json", "w") as f:
            json.dump(all_res, f)