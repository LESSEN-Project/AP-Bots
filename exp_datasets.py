import os
import re
import json
import requests
import gzip
import datetime
import urllib
from abc import ABC, abstractmethod

import itertools
import pandas as pd


class Dataset(ABC):
    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def get_gts(self):
        pass

    @abstractmethod
    def get_var_names(self):
        pass

    @abstractmethod
    def get_retr_data(self):
        pass


class LampDataset(Dataset):

    def __init__(self, num, data_split="dev", split="user", dataset_dir="datasets"):
        self.name = "lamp"
        self.num = num
        self.split = split
        self.data_split = data_split
        self.tag = f"lamp_{self.num}_{self.data_split}_{self.split}"
        self.dataset_dir = dataset_dir
        self.dataset = None

    def get_dataset(self):

        os.makedirs(self.dataset_dir, exist_ok=True)
        data_path = os.path.join(self.dataset_dir, f"{self.tag}_data.json")

        base_url = "https://ciir.cs.umass.edu/downloads/LaMP/"
        if self.split == "time":
            base_url = f"{base_url}/time"
        if self.num == 2:
            base_url = f"{base_url}/LaMP_{self.num}/new/{self.data_split}/{self.data_split}"
        else:
            base_url = f"{base_url}/LaMP_{self.num}/{self.data_split}/{self.data_split}"

        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                data = json.load(f)
        else:
            with urllib.request.urlopen(f"{base_url}_questions.json") as url:
                data = json.load(url)
            with open(data_path, "w") as f:
                json.dump(data, f)
        return data
    
    def get_gts(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        base_url = "https://ciir.cs.umass.edu/downloads/LaMP/"
        if self.split == "time":
            base_url = f"{base_url}/time"
        if self.num == 2:
            base_url = f"{base_url}/LaMP_{self.num}/new/{self.data_split}/{self.data_split}"
        else:
            base_url = f"{base_url}/LaMP_{self.num}/{self.data_split}/{self.data_split}"

        if self.split != "test":
            gts_path = os.path.join(self.dataset_dir, f"{self.tag}_gts.json")
            if os.path.exists(gts_path):
                with open(gts_path, "r") as f:
                    gts = json.load(f)
            else:
                with urllib.request.urlopen(f"{base_url}_outputs.json") as url:
                    gts = json.load(url)["golds"]
                with open(gts_path, "w") as f:
                    json.dump(gts, f)
            return [p["output"] for p in gts]
        else:
            print("Ground truth for test set not available!")
            return None

    def get_var_names(self):
        if self.num == 1:
            retr_gt_name = "title"
            retr_text_name = "abstract"
            retr_prompt_name = "abstract"
        elif self.num == 2:
            retr_gt_name = "tag"
            retr_text_name = "description"
            retr_prompt_name = "description"        
        elif self.num == 3:
            retr_gt_name = "score"
            retr_text_name = "text"
            retr_prompt_name = "review"
        elif self.num == 4:
            retr_gt_name = "title"
            retr_text_name = "text"
            retr_prompt_name = "article"
        elif self.num == 5:
            retr_gt_name = "title"
            retr_text_name = "abstract"
            retr_prompt_name = "abstract"
        elif self.num == 7:
            retr_gt_name = None
            retr_text_name = "text"
            retr_prompt_name = "tweet"        
        return retr_text_name, retr_gt_name, retr_prompt_name

    def get_retr_data(self):
        if not self.dataset:
            self.dataset = self.get_dataset()
        queries = []
        retr_text = []
        retr_gts = []
        prof_text_name, prof_gt_name, _ = self.get_var_names()
        for sample in self.dataset:
            if self.num in [3, 4, 5, 7]:
                text_idx = sample["input"].find(":") + 1
                queries.append(sample["input"][text_idx:].strip())
            elif self.num == 1:
                queries.append(re.findall(r'"(.*?)"', sample["input"]))
            elif self.num == 2:
                text_idx = sample["input"].find("description:") + 1
                queries.append(sample["input"][text_idx+len("description:"):].strip())
            retr_text.append([p[prof_text_name] for p in sample["profile"]])
            if self.num != 7:
                retr_gts.append([p[prof_gt_name] for p in sample["profile"]])
            else:
                retr_gts = retr_text
        return queries, retr_text, retr_gts

    def get_ids(self):
        data = self.get_dataset()
        return [i["id"] for i in data]


class AmazonDataset(Dataset):

    def __init__(self, category, year, dataset_dir="datasets"):
        self.name = "amazon"
        self.category = category
        self.year = year
        self.dataset = None
        self.tag = f"amazon_{self.category}_{self.year}"
        self.dataset_dir = dataset_dir
        self.min_user_samples = 20

    def get_dataset(self):

        dfs = self.get_amazon_dfs()
        df, df_meta = self.process_dfs(dfs)
        data_path = os.path.join(self.dataset_dir, f"amazon_{self.category}_{self.year}_user_data.json")
        user_var = "user_id" if self.year == 2023 else "reviewerID"

        user_counts = df[user_var].value_counts()
        users_with_enough_samples = set(user_counts[user_counts >= self.min_user_samples].index)
        df = df[df[user_var].isin(users_with_enough_samples)]

        all_users = df[user_var].unique()
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                all_user_data = json.load(f)
                if len(all_user_data) == len(all_users):
                    print("User data for this category is already created!")
                    return all_user_data
                start_idx = len(all_user_data)
        else:
            all_user_data = []
            start_idx = 0

        print("Processing user data...")
        print(f"Number of users: {len(all_users)}")

        df_meta = df_meta.set_index('asin' if self.year == 2018 else 'parent_asin')

        for num_user, user_id in enumerate(all_users[start_idx:], start=start_idx):

            sample_user = df[df[user_var] == user_id].sort_values("unixReviewTime" if self.year == 2018 else "timestamp")
            sample_user = sample_user.join(df_meta, on='asin' if self.year == 2018 else 'parent_asin', how='inner')
            
            user_data = {
                "ID": user_id,
                "History": []
            }
            
            for idx, row in sample_user.iterrows():
                date_time = datetime.datetime.fromtimestamp(row["unixReviewTime"] if self.year == 2018 else row["timestamp"] / 1000)
                formatted_date = date_time.strftime("%Y-%m-%d %H:%M:%S")

                prod_info = {
                    "Name": row["title" if self.year == 2018 else "productTitle"],
                    "Descriptions": row["description"],
                    "Review": row["reviewText"],
                    "Score": row["overall" if self.year == 2018 else "rating"],
                    "Review Time": formatted_date
                }
                
                if idx == sample_user.index[-1]:
                    user_data["Product"] = prod_info
                else:
                    user_data["History"].append(prod_info)

            all_user_data.append(user_data)

            if (num_user + 1) % 500 == 0:
                print(f"Step: {num_user + 1}")
                with open(data_path, "w") as f:
                    json.dump(all_user_data, f)

        print("Finished processing user data!")
        with open(data_path, "w") as f:
            json.dump(all_user_data, f)

        self.dataset = all_user_data
        return all_user_data

    def get_gts(self):
        if not self.dataset:
            self.dataset = self.get_dataset()
        return [d["Product"]["Review"] for d in self.dataset]
    
    def get_var_names(self):
        retr_gt_name = "Review", "Rating"
        retr_text_name = "Product"
        retr_prompt_name = "Product"
        return retr_text_name, retr_gt_name, retr_prompt_name

    def get_retr_data(self):
        queries = []
        retr_texts = []
        retr_gts = []
        if not self.dataset:
            self.dataset = self.get_dataset()
        for sample in self.dataset:
            queries.append(sample["Product"]["Name"])
            retr_texts.append([item["Name"] for item in sample["History"]])
            retr_gts.append([str(item["Review"]) if not isinstance(item["Review"], str) else item["Review"] for item in sample["History"]])
        return queries, retr_texts, retr_gts
    
    def get_ratings(self, idx):
        if not self.dataset:
            self.dataset = self.get_dataset()
        return self.dataset[idx]["Product"]["Score"], [item["Score"] for item in self.dataset[idx]["History"]]

    def get_statistics(self):

        _, _, retr_gts = self.get_retr_data()
        all_reviews = list(itertools.chain.from_iterable(retr_gts))
        all_reviews = [str(review) for review in all_reviews]

        word_counts = [len(review.split(" ")) for review in all_reviews]
        df = pd.Series(word_counts)

        print(df.value_counts(ascending=False))
        print(df.describe())  

    def download_amazon_datasets(self, target_dest="datasets"):
        os.makedirs(target_dest, exist_ok=True)
        review_link, review_save_loc = self.get_review_links(target_dest)
        self.download_file(review_link, review_save_loc, "Reviews")
        
        meta_link, meta_save_loc = self.get_meta_links(target_dest)
        self.download_file(meta_link, meta_save_loc, "Metadata")

    def get_review_links(self, target_dest):
        if self.year == 2018:
            review_link = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/{self.category}_5.json.gz"
            review_save_loc = os.path.join(target_dest, f"amazon_{self.category}_{self.year}.json.gz")
        elif self.year == 2023:
            review_link = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/{self.category}.jsonl.gz"
            review_save_loc = os.path.join(target_dest, f"amazon_{self.category}_{self.year}.jsonl.gz")
        return review_link, review_save_loc

    def get_meta_links(self, target_dest):
        if self.year == 2018:
            meta_link = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/meta_{self.category}.json.gz"
            meta_save_loc = os.path.join(target_dest, f"amazon_{self.category}_{self.year}_meta.json.gz")
        elif self.year == 2023:
            meta_link = f"https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_{self.category}.jsonl.gz"
            meta_save_loc = os.path.join(target_dest, f"amazon_{self.category}_{self.year}_meta.jsonl.gz")
        return meta_link, meta_save_loc

    @staticmethod
    def download_file(link, save_loc, file_type):
        if os.path.exists(save_loc):
            print(f"{file_type} for this category already exists!")
        else:
            print(f"Downloading {file_type.lower()} for the category!")
            response = requests.get(link)
            if response.status_code == 200:
                with open(save_loc, 'wb') as f:
                    f.write(response.content)
                print(f"{file_type} downloaded successfully!")
            else:
                print(f"Failed to download {file_type.lower()}. Status code: {response.status_code}")

    @staticmethod
    def parse_amazon(path):
        g = gzip.open(path, 'rb')
        for l in g:
            yield json.loads(l)

    def get_amazon_dfs(self, dataset_dir="datasets"):
        self.download_amazon_datasets(dataset_dir)
        extension = "jsonl.gz" if self.year == 2023 else "json.gz"
        paths = [
            os.path.join(dataset_dir, f"amazon_{self.category}_{self.year}.{extension}"),
            os.path.join(dataset_dir, f"amazon_{self.category}_{self.year}_meta.{extension}")
        ]
        dfs = []
        for path in paths: 
            i = 0
            df = {}
            for d in self.parse_amazon(path):
                df[i] = d
                i += 1
            dfs.append(pd.DataFrame.from_dict(df, orient="index"))
        return dfs[0], dfs[1]

    def process_dfs(self, dfs):
        df, df_meta = dfs
        if self.year == 2018:
            df = df[~df["reviewerName"].str.contains("ustomer", na=False)]
            df = df.drop_duplicates(["reviewerID", "unixReviewTime"])
            df = df[df["reviewerID"].isin(df.value_counts("reviewerID")[(df.value_counts("reviewerID") > 1)].index)]
            df_meta = df_meta.drop_duplicates("asin")
        elif self.year == 2023:
            df = df.drop_duplicates(["user_id", "timestamp"])
            df = df[df["user_id"].isin(df.value_counts("user_id")[(df.value_counts("user_id") > 2)].index)]
            df_meta = df_meta.drop_duplicates("parent_asin")
        return df, df_meta