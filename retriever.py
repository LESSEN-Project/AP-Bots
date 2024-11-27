import os
import json

import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer


class Retriever:

    def __init__(self, dataset, model: str = "contriever", device: str = "cuda:0"):

        self.model = model
        self.device = device
        self.dataset = dataset
        self.save_loc = os.path.join("files", "retrieval_res")
        os.makedirs(self.save_loc, exist_ok=True)
        self._init_model()

    def _init_model(self):

        if self.model == "contriever":
            self.retr_model = SentenceTransformer("nishimoto/contriever-sentencetransformer", device=self.device)
        elif self.model == "dpr":
            self.retr_model = SentenceTransformer("sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base", device=self.device)  
        else:
            self.retr_model = SentenceTransformer(self.model, device=self.device)

    def check_file(self):

        file_path = os.path.join(self.save_loc, f"{self.dataset.tag}_{self.model}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                all_idxs = json.load(f)
        else:
            print("Retrieval results are not cached, starting from 0!")
            all_idxs = []
        return all_idxs
    
    def save_file(self, obj):

        file_path = os.path.join(self.save_loc, f"{self.dataset.tag}_{self.model}.json")
        with open(file_path, "w") as f:
            json.dump(obj, f)

    def get_retrieval_results(self, queries: List[str], retr_texts: List[List[str]]) -> List[List[int]]:

        return self._neural_retrieval(queries, retr_texts)

    def _encode(self, docs):

        if not isinstance(docs, np.ndarray):
            return self.retr_model.encode(docs)
        else:
            return docs

    def _neural_retrieval(self, queries: List[str], docs: List[str]):

        query_embeds = self._encode(queries)
        doc_embeds = self._encode(docs)
        similarities = self.retr_model.similarity(query_embeds, doc_embeds).numpy().squeeze().tolist()
        sorted_idxs = np.argsort(similarities)[::-1].tolist()
        if isinstance(similarities, float):
            similarities = [similarities]
            
        return similarities, sorted_idxs

    def semantic_consensus_weighting(self, outputs):

        embeds = self._encode(outputs)
        similarity_matrix = self.retr_model.similarity(embeds, embeds)   
        aggregated_scores = similarity_matrix.sum(axis=1)
        best_response_index = np.argmax(aggregated_scores)
        
        return outputs[best_response_index]

    def contrastive_retrieval(self, queries, retr_texts, retr_gts, num_ce, ce_k):

        _, retr_gt_name, retr_prompt_name = self.dataset.get_var_names()
        _, ce_retr_res = self.get_retrieval_results(queries, queries)

        all_ce_examples = []
        for ce_retr in ce_retr_res:

            ce_idxs = ce_retr[-num_ce:]
            ce_examples = []
            for ce in ce_idxs:
                ce_example = []
                max_range = len(retr_texts[ce]) if ce_k > len(retr_texts[ce]) else ce_k
                if isinstance(retr_gt_name, tuple):
                    _, doc_ratings = self.dataset.get_ratings(ce)
                for j in range(max_range):
                    if retr_gt_name:
                        if isinstance(retr_gt_name, tuple):
                            ce_example.append(f"{retr_prompt_name.capitalize()}:\n{retr_texts[ce][j]}\n{retr_gt_name[0].capitalize()}:\n{retr_gts[ce][j]}\n{retr_gt_name[1].capitalize()}:\n{doc_ratings[j]}")
                        else:
                            ce_example.append(f"{retr_prompt_name.capitalize()}:\n{retr_texts[ce][j]}\n{retr_gt_name.capitalize()}:\n{retr_gts[ce][j]}")
                    else:
                        ce_example.append(f"{retr_prompt_name.capitalize()}:\n{retr_texts[ce][j]}")
                ce_examples.append(ce_example)
            all_ce_examples.append(ce_examples)
            
        return all_ce_examples

    def get_context(self, queries: List[str], retr_texts: List[List[str]], retr_gts: List[List[str]], k: str) -> List[List[str]]:

        if k == 0:
            return [""] * len(queries)

        all_idxs = self.check_file()
        all_examples = []
        _, retr_gt_name, retr_prompt_name = self.dataset.get_var_names()

        for i, query in enumerate(queries):
            
            retr_text = retr_texts[i]
            retr_gt = retr_gts[i]
            if len(all_idxs) > i:
                sorted_idxs = all_idxs[i]
            else:
                _, sorted_idxs = self.get_retrieval_results(query, retr_text)
                
                all_idxs.append(sorted_idxs)
                if ((i+1)%500 == 0) or (i+1 == len(queries)):
                    print(i)     
                    self.save_file(all_idxs)

            texts = [retr_text[doc_id] for doc_id in sorted_idxs[:k]]                
            gts = [retr_gt[doc_id] for doc_id in sorted_idxs[:k]]

            if isinstance(retr_gt_name, tuple):
                _, doc_ratings = self.dataset.get_ratings(i)
                doc_ratings = [doc_ratings[doc_id] for doc_id in sorted_idxs[:k]]
                
            examples = []

            for i, (text, gt) in enumerate(zip(texts, gts)):
                if text != gt:      
                    if isinstance(retr_gt_name, tuple):
                        example = f"{retr_prompt_name.capitalize()}:\n{text}\n{retr_gt_name[0].capitalize()}:\n{gt}\n{retr_gt_name[1].capitalize()}:\n{doc_ratings[i]}"
                    else:
                        example = f"{retr_prompt_name.capitalize()}:\n{text}\n{retr_gt_name.capitalize()}:\n{gt}\n"
                else:
                    example = f"{retr_prompt_name.capitalize()}:\n{text}"
                examples.append(example)
            all_examples.append(examples)

        return all_examples