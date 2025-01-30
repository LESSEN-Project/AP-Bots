import os
from datetime import datetime

import numpy as np

from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, MilvusClient, model


class VectorDB:

    def __init__(self, uri="db/apbots.db"):

        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f"{current_dir}/db", exist_ok=True)
        self.embedder = model.DefaultEmbeddingFunction()
        self.client = MilvusClient(uri=uri)
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
                FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=384)
            ]
            schema = CollectionSchema(fields, description="Chat history collection")

            index_params = self.client.prepare_index_params()

            index_params.add_index(
                field_name="dense_vector",
                index_name="dense_vector_index",
                index_type="IVF_FLAT",
                metric_type="IP",
                params={"nlist": 128},
            )

            self.client.create_collection(collection_name="chat_history", schema=schema, index_params=index_params)

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

    def save_conversation(self, user_id, turn, title):

        start_time = datetime.now().isoformat()
        conv_id = hash(f"{start_time}{user_id}") % (2**31)
        conv_str = f"User: {turn['user_message']}\nAssistant: {turn['assistant_message']}"

        self.client.insert(collection_name="chat_history", data={
            "conv_id": hash(f"{start_time}{user_id}") % (2**31) ,
            "user_id": user_id,
            "conversation": [turn],
            "title": title,
            "start_time": start_time,
            "end_time": datetime.now().isoformat(),
            "dense_vector": self.get_embedding([conv_str])
        })
        
        return conv_id

    def update_conversation(self, conv_id, turn):

        cur_conv = self.client.get("chat_history", ids=conv_id)
        cur_conv[0]["end_time"] = datetime.now().isoformat()
        
        conv_hist = cur_conv[0]["conversation"]
        conv_hist.append(turn)
        cur_conv[0]["conversation"] = conv_hist

        all_turns = " ".join(f"User: {turn['user_message']}\nAssistant: {turn['assistant_message']}" for turn in conv_hist)
        cur_conv[0]["dense_vector"] = self.get_embedding([all_turns])

        self.client.upsert(collection_name="chat_history", data=cur_conv[0])

    def delete_conversation(self, conv_id):
        self.client.delete(collection_name="chat_history", ids=conv_id)