import os
import re
import math
from datetime import datetime

import numpy as np

from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, MilvusClient, model, Function, FunctionType


class VectorDB:

    def __init__(self, uri="apbots.db"):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f"{current_dir}/db", exist_ok=True)
        self.embedder = model.DefaultEmbeddingFunction()
        self.client = MilvusClient(uri=f"{current_dir}/db/{uri}")
        self.initialize_db()

    def initialize_db(self):
        
        if not self.client.has_collection("user"):
            fields = [
                FieldSchema(name="user_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="user_name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="password", dtype=DataType.VARCHAR, max_length=100)
            ]
            schema = CollectionSchema(fields, description="User collection")
            self.client.create_collection(collection_name="user", schema=schema)
        
        if not self.client.has_collection("chat_history"):
            fields = [
                FieldSchema(name="conv_id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="user_id", dtype=DataType.INT64),
                FieldSchema(name="conversation", dtype=DataType.JSON),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="start_time", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="end_time", dtype=DataType.VARCHAR, max_length=100),
            ]
            schema = CollectionSchema(fields, description="Chat history collection")
            self.client.create_collection(collection_name="chat_history", schema=schema)

        if not self.client.has_collection("chat_turns"):
            fields = [
                FieldSchema(name="turn_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="user_id", dtype=DataType.INT64),
                FieldSchema(name="conv_id", dtype=DataType.INT64),
                FieldSchema(name="role", dtype=DataType.VARCHAR, max_length=10),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000, enable_analyzer=True),
                FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
                
            ]

            schema = CollectionSchema(fields, enable_dynamic_field=True, description="Chat turns collection")

            index_params = self.client.prepare_index_params()
            
            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )
            
            self.client.create_collection(collection_name="chat_turns", schema=schema, index_params=index_params)

    def authenticate_user(self, username, password):

        res = self.client.query(collection_name="user", filter=f"user_name == '{username}'", output_fields=["user_id", "password"])
        if not res:
            return False, "User does not exist. Please sign up.", -1
        if res[0]['password'] != password:
            return False, "Incorrect password.", -1
        
        return True, "Login successful.", res[0]["user_id"]

    def sign_up_user(self, username, password):

        res = self.client.query(collection_name="user", filter=f"user_name == '{username}'", output_fields=["user_id", "user_name"])
        if res:
            return False, "Username already exists.", -1
        self.client.insert(collection_name="user", data={"user_name": username,
                                                        "password": password})
        
        user_id = self.client.query(collection_name="user", filter=f"user_name == '{username}'", output_fields=["user_id"])[0]["user_id"]
        return True, "Sign-up successful. Please log in.", user_id

    def change_password(self, user_id, new_password):

        cur_user = self.client.get("user", ids=user_id)
        cur_user[0]["password"] = new_password
        self.client.upsert(collection_name="user", data=cur_user[0])

    def delete_user(self, user_id):

        self.client.delete(collection_name="user", ids=user_id)
        user_conv_ids = self.client.query(collection_name="chat_history", filter=f"user_id == {user_id}", output_fields=["conv_id"])
        if user_conv_ids:
            self.delete_conversation([uc_id["conv_id"] for uc_id in user_conv_ids])

    def get_embedding(self, text):
        return self.embedder.encode_documents(text)[0]

    def get_all_user_convs(self, user_id):
        return self.client.query(collection_name="chat_history", filter=f"user_id == {user_id}", consistency_level="Strong")

    @staticmethod
    def gen_conv_id(user_id):

        start_time = datetime.now().isoformat()
        return start_time, hash(f"{start_time}{user_id}") % (2**31)

    def save_conversation(self, conv_id, user_id, start_time, turn, title):

        conv_str = f"User: {turn['user_message']}\nAssistant: {turn['assistant_message']}"
        self.client.insert(collection_name="chat_history", data={
            "conv_id": conv_id,
            "user_id": user_id,
            "conversation": [turn],
            "title": title,
            "start_time": start_time,
            "end_time": datetime.now().isoformat(),
        })
        # Also insert individual turns:
        self.insert_turn(user_id, conv_id, "user", turn["user_message"], start_time)
        self.insert_turn(user_id, conv_id, "assistant", turn["assistant_message"], datetime.now().isoformat())

    def update_conversation(self, conv_id, turn):

        cur_conv = self.client.get("chat_history", ids=conv_id)
        cur_conv[0]["end_time"] = datetime.now().isoformat()
        conv_hist = cur_conv[0]["conversation"]
        conv_hist.append(turn)
        cur_conv[0]["conversation"] = conv_hist
        all_turns = " ".join(f"User: {t['user_message']}\nAssistant: {t['assistant_message']}" for t in conv_hist)

        self.client.upsert(collection_name="chat_history", data=cur_conv[0])
        # Insert the new turn into chat_turns:
        self.insert_turn(cur_conv[0]["user_id"], conv_id, "user", turn["user_message"], datetime.now().isoformat())
        self.insert_turn(cur_conv[0]["user_id"], conv_id, "assistant", turn["assistant_message"], datetime.now().isoformat())

    def delete_conversation(self, conv_id):
        self.client.delete(collection_name="chat_history", ids=conv_id)
        turns = self.client.query(collection_name="chat_turns", filter=f"conv_id == {conv_id}", output_fields=["turn_id"])
        if turns:
            turn_ids = [t["turn_id"] for t in turns]
            self.client.delete(collection_name="chat_turns", ids=turn_ids)

    def insert_turn(self, user_id, conv_id, role, text, timestamp):
        self.client.insert(collection_name="chat_turns", data={
            "user_id": user_id,
            "conv_id": conv_id,
            "role": role,
            "text": text,
            "timestamp": timestamp,
            "dense_vector": self.get_embedding([text])
        })

    def dense_search(self, query_text, search_filter, top_k=3):

        query_embedding = self.get_embedding([query_text])
        results = self.client.search(
            collection_name="chat_turns",
            data=[query_embedding],
            anns_field="dense_vector",
            search_params={
            "metric_type": "IP",
            },
            limit=top_k,
            filter=search_filter,
            output_fields=["turn_id", "conv_id", "role", "text", "timestamp", "sparse_vector"]
        )
        return results

    def bm25_search(self, query_text, search_filter, top_k=3):
        """
        Perform a BM25 search on conversation turns for the specified user.
        Returns the top_k turns ranked by BM25 score.
        """
        # Retrieve all conversation turns for this user.
        docs = self.client.query(
            collection_name="chat_turns",
            filter=search_filter,
            output_fields=["turn_id", "conv_id", "role", "text", "timestamp"]
        )
        if not docs:
            return []

        # Simple tokenizer: lowercase and extract alphanumeric words.
        def tokenize(text):
            return re.findall(r'\w+', text.lower())

        # Build corpus: list of token lists and store doc info.
        corpus = []
        doc_info = []
        for doc in docs:
            tokens = tokenize(doc["text"])
            corpus.append(tokens)
            doc_info.append(doc)

        N = len(corpus)
        # Compute document frequency for each term.
        df = {}
        for tokens in corpus:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        avg_dl = sum(len(tokens) for tokens in corpus) / N

        # BM25 parameters.
        k1 = 1.5
        b = 0.75

        # Tokenize query.
        query_tokens = tokenize(query_text)
        # Pre-compute idf for query tokens.
        idf = {}
        for token in query_tokens:
            n_q = df.get(token, 0)
            idf[token] = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)

        scores = []
        # Compute BM25 score for each document.
        for tokens in corpus:
            doc_len = len(tokens)
            score = 0.0
            # Compute term frequency for the document.
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            # Sum contributions for each query token.
            for token in query_tokens:
                if token in tf:
                    f = tf[token]
                    score += idf[token] * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * (doc_len / avg_dl))))
            scores.append(score)

        # Get indices of the top_k documents.
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            res = doc_info[idx].copy()
            res["bm25_score"] = scores[idx]
            results.append(res)

        return results
