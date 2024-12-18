import json
import re
import yaml

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