import json
import re

def parse_cot_output(output: str) -> str:

    last_line = output.strip().splitlines()[-1]
    last_line = last_line.replace('"', '')
    last_line = last_line.replace('*', '')

    return last_line

def extract_bfi_scores(input_str):

    input_str = input_str.strip()
    if input_str.startswith("```json"):
        input_str = input_str[7:].strip()

    if input_str.startswith("```"):
        input_str = input_str[3:].strip()

    if input_str.endswith("```"):
        input_str = input_str[:-3].strip()
        
    try:
        json_object = json.loads(input_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e} in string: {input_str}")

    return json_object

def parse_react_output(text: str) -> str:

    try:
        start_tag = "<review>"
        end_tag = "</review>"
        start_idx = text.find(start_tag) + len(start_tag)
        end_idx = text.find(end_tag)
        if start_idx != -1 and end_idx != -1:
            return text[start_idx:end_idx].strip()
        return ""
    except Exception as e:
        print(f"Error parsing react output: {e}")
        return ""

def parse_r1_output(text):

    match = re.match(r'<think>(.*?)</think>(.*)', text, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        content = match.group(2).strip()
        return thought, content
    return None, text.strip()