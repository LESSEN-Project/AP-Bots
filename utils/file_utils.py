import os
import random
import json

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
    
def oai_get_batch_res(client):

    batches = client.batches.list()
    files = client.files.list()

    for batch in batches:
        
        filename = [file.filename for file in files if file.id == batch.input_file_id]

        if filename and batch.output_file_id:

            merged_res = []
            path_to_file = os.path.join("preds", filename[0])

            data = []
            with open(path_to_file, "r") as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            batch_res = client.files.content(batch.output_file_id).text
            batch_res = [json.loads(line) for line in batch_res.splitlines()]

            for sample in data:
                res = [res["response"]["body"]["choices"][0]["message"]["content"] for res in batch_res if res["custom_id"] == sample["custom_id"]]
                merged_res.append({
                    "id": sample["custom_id"],
                    "prompt": sample["body"]["messages"][0]["content"],
                    "output": res[0].strip(),
                    "model_inf_time": "n/a", 
            })
            with open(os.path.join("preds", f"{filename[0].split('.')[0]}.json"), "w") as f:
                json.dump({
                    "golds": merged_res
                }, f)