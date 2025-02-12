import os
import re
import math
from datetime import datetime
import uuid

import chromadb
from chromadb.utils import embedding_functions


class UserCollection:
    def __init__(self, client):
        self.collection = client.get_or_create_collection("user")

    def authenticate_user(self, username, password):
        res = self.collection.get(where={"user_name": username})
        print(res)
        if not res["metadatas"]:
            return False, "User does not exist. Please sign up.", -1
        user_data = res["metadatas"][0]
        if user_data["password"] != password:
            return False, "Incorrect password.", -1
        return True, "Login successful.", user_data["user_id"]

    def sign_up_user(self, username, password):
        res = self.collection.get(where={"user_name": username})
        if res["metadatas"]:
            return False, "Username already exists.", -1

        user_id = uuid.uuid4().int % (2**31)
        self.collection.add(
            ids=[str(user_id)],
            documents=[""],
            metadatas=[{"user_id": user_id, "user_name": username, "password": password}],
        )
        return True, "Sign-up successful. Please log in.", user_id

    def change_password(self, user_id, new_password):
        res = self.collection.get(ids=[str(user_id)])
        if not res["metadatas"]:
            return
        user_metadata = res["metadatas"][0]
        user_metadata["password"] = new_password
        self.collection.add(
            ids=[str(user_id)],
            documents=[""],
            metadatas=[user_metadata],
        )

    def delete_user(self, user_id):
        self.collection.delete(ids=[str(user_id)])


class ChatCollection:
    def __init__(self, client, embedder):
        self.chat_history = client.get_or_create_collection("chat_history")
        self.chat_turns = client.get_or_create_collection("chat_turns", embedding_function=embedder)
        self.embedder = embedder

    def get_all_user_convs(self, user_id):
        return self.chat_history.get(where={"user_id": user_id})

    def save_conversation(self, conv_id, user_id, start_time, turn, title):
        conv_data = {
            "conv_id": conv_id,
            "user_id": user_id,
            "title": title,
            "start_time": start_time,
            "end_time": datetime.now().isoformat(),
        }
        self.chat_history.add(
            ids=[str(conv_id)],
            documents=[""],
            metadatas=[conv_data],
        )
        self.insert_turn(user_id, conv_id, "user", turn["user_message"], start_time)
        self.insert_turn(user_id, conv_id, "assistant", turn["assistant_message"], datetime.now().isoformat())

    def update_conversation(self, conv_id, turn):
        res = self.chat_history.get(ids=[str(conv_id)])
        if not res["metadatas"]:
            return
        conv_entry = res["metadatas"][0]
        conv_entry["end_time"] = datetime.now().isoformat()
        self.chat_history.add(
            ids=[str(conv_id)],
            documents=[""],
            metadatas=[conv_entry],
        )
        user_id = conv_entry["user_id"]
        self.insert_turn(user_id, conv_id, "user", turn["user_message"], datetime.now().isoformat())
        self.insert_turn(user_id, conv_id, "assistant", turn["assistant_message"], datetime.now().isoformat())

    def delete_conversation(self, conv_id):
        if not isinstance(conv_id, list):
            conv_ids = [conv_id]
        else:
            conv_ids = conv_id

        self.chat_history.delete(ids=[str(cid) for cid in conv_ids])

        for cid in conv_ids:
            turns = self.chat_turns.get(where={"conv_id": cid})
            if turns["ids"]:
                self.chat_turns.delete(ids=turns["ids"])

    def insert_turn(self, user_id, conv_id, role, text, timestamp):
        turn_id = str(abs(hash(f"{conv_id}_{role}_{timestamp}_{text}")))
        turn_data = {
            "turn_id": turn_id,
            "user_id": user_id,
            "conv_id": conv_id,
            "role": role,
            "text": text,
            "timestamp": timestamp,
        }
        # Get embedding; note that self.embedder expects a list.
        embedding = self.embedder([text])[0]
        self.chat_turns.add(
            ids=[turn_id],
            documents=[text],
            metadatas=[turn_data],
            embeddings=[embedding],
        )

    def dense_search(self, query_text, search_filter, top_k=3, distance_threshold=0.5):
        results = self.chat_turns.query(
            query_texts=[query_text],
            n_results=top_k,
            where=search_filter,
            include=["metadatas", "documents", "distances"],
        )
        filtered_results = []
        for idx, turn_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][idx]
            if distance > distance_threshold:
                continue
            meta = results["metadatas"][0][idx]
            filtered_results.append({
                "turn_id": meta.get("turn_id", turn_id),
                "conv_id": meta["conv_id"],
                "role": meta["role"],
                "text": results["documents"][0][idx],
                "timestamp": meta["timestamp"],
                "distance": distance,
            })
        return filtered_results

    def bm25_search(self, query_text, search_filter, top_k=3):
        docs = self.chat_turns.get(where=search_filter, include=["metadatas", "documents"])
        if not docs["documents"]:
            return []

        def tokenize(text):
            return re.findall(r'\w+', text.lower())

        corpus = []
        doc_info = []
        for doc, meta in zip(docs["documents"], docs["metadatas"]):
            tokens = tokenize(doc)
            corpus.append(tokens)
            doc_info.append({
                "turn_id": meta.get("turn_id"),
                "conv_id": meta["conv_id"],
                "role": meta["role"],
                "text": doc,
                "timestamp": meta["timestamp"],
            })

        N = len(corpus)
        df = {}
        for tokens in corpus:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        avg_dl = sum(len(tokens) for tokens in corpus) / N
        k1 = 1.5
        b = 0.75

        query_tokens = tokenize(query_text)
        idf = {}
        for token in query_tokens:
            n_q = df.get(token, 0)
            idf[token] = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1)

        scores = []
        for tokens in corpus:
            doc_len = len(tokens)
            score = 0.0
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token in query_tokens:
                if token in tf:
                    f = tf[token]
                    score += idf[token] * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * (doc_len / avg_dl))))
            scores.append(score)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results_list = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc_info[idx]["bm25_score"] = scores[idx]
                results_list.append(doc_info[idx])
        return results_list

    def hybrid_search(self, query_text, search_filter, top_k=3, distance_threshold=0.5, window_size=1):
        dense_results = self.dense_search(query_text, search_filter, top_k, distance_threshold)
        bm25_results = self.bm25_search(query_text, search_filter, top_k)

        merged_results = {}
        for result in dense_results:
            merged_results[result["turn_id"]] = {
                **result,
                "bm25_score": None,
                "distance": result["distance"],
            }

        for result in bm25_results:
            if result["turn_id"] in merged_results:
                merged_results[result["turn_id"]]["bm25_score"] = result["bm25_score"]
            else:
                merged_results[result["turn_id"]] = {
                    **result,
                    "bm25_score": result["bm25_score"],
                    "distance": None,
                }

        merged_results = list(merged_results.values())

        conversation_groups = {}
        for res in merged_results:
            conv_id = res["conv_id"]
            conversation_groups.setdefault(conv_id, []).append(res)

        conversation_windows = []
        for conv_id, turn_data in conversation_groups.items():
            turn_ids = [entry["turn_id"] for entry in turn_data]
            scored_turns = {entry["turn_id"]: entry for entry in turn_data}
            conversation_windows.append(
                self.get_merged_conversation_window(conv_id, turn_ids, window_size, scored_turns)
            )
        return conversation_windows

    def get_merged_conversation_window(self, conv_id, turn_ids, window_size=1, scored_turns=None):
        conv = self.chat_turns.get(where={"conv_id": conv_id}, include=["metadatas", "documents"])
        combined = []
        for meta, doc, tid in zip(conv["metadatas"], conv["documents"], conv["ids"]):
            combined.append({
                "turn_id": meta.get("turn_id", tid),
                "role": meta["role"],
                "text": doc,
                "timestamp": meta["timestamp"],
            })

        sorted_conv = sorted(combined, key=lambda x: x["timestamp"])
        turn_indices = [i for i, turn in enumerate(sorted_conv) if turn["turn_id"] in turn_ids]
        if not turn_indices:
            return []
        start = max(0, min(turn_indices) - window_size)
        end = min(len(sorted_conv), max(turn_indices) + window_size + 1)
        merged_conversation = sorted_conv[start:end]

        for turn in merged_conversation:
            tid = turn["turn_id"]
            if tid in scored_turns:
                turn["bm25_score"] = scored_turns[tid]["bm25_score"]
                turn["distance"] = scored_turns[tid]["distance"]
        return merged_conversation

    def delete_conversations_by_user(self, user_id):
        convs = self.chat_history.get(where={"user_id": user_id})
        if convs["metadatas"]:
            conv_ids = [m["conv_id"] for m in convs["metadatas"]]
            self.delete_conversation(conv_ids)


class VectorDB:
    def __init__(self, persist_directory="apbots"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(f"{current_dir}/db", exist_ok=True)
        db_dir = f"{current_dir}/db/{persist_directory}"
        os.makedirs(db_dir, exist_ok=True)

        self.embedder = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.client = chromadb.PersistentClient(path=db_dir)
        self.initialize_db()

    def initialize_db(self):
        self.user_collection = UserCollection(self.client)
        self.chat_collection = ChatCollection(self.client, self.embedder)

    def authenticate_user(self, username, password):
        return self.user_collection.authenticate_user(username, password)

    def sign_up_user(self, username, password):
        return self.user_collection.sign_up_user(username, password)

    def change_password(self, user_id, new_password):
        return self.user_collection.change_password(user_id, new_password)

    def delete_user(self, user_id):
        self.user_collection.delete_user(user_id)
        self.chat_collection.delete_conversations_by_user(user_id)

    def get_embedding(self, text):
        return self.embedder(text)[0]

    def get_all_user_convs(self, user_id):
        return self.chat_collection.get_all_user_convs(user_id)

    @staticmethod
    def gen_conv_id(user_id):
        start_time = datetime.now().isoformat()
        return start_time, hash(f"{start_time}{user_id}") % (2**31)

    def save_conversation(self, conv_id, user_id, start_time, turn, title):
        self.chat_collection.save_conversation(conv_id, user_id, start_time, turn, title)

    def update_conversation(self, conv_id, turn):
        self.chat_collection.update_conversation(conv_id, turn)

    def delete_conversation(self, conv_id):
        self.chat_collection.delete_conversation(conv_id)

    def insert_turn(self, user_id, conv_id, role, text, timestamp):
        self.chat_collection.insert_turn(user_id, conv_id, role, text, timestamp)

    def dense_search(self, query_text, search_filter, top_k=3, distance_threshold=0.5):
        return self.chat_collection.dense_search(query_text, search_filter, top_k, distance_threshold)

    def bm25_search(self, query_text, search_filter, top_k=3):
        return self.chat_collection.bm25_search(query_text, search_filter, top_k)

    def hybrid_search(self, query_text, search_filter, top_k=3, distance_threshold=0.5, window_size=1):
        return self.chat_collection.hybrid_search(query_text, search_filter, top_k, distance_threshold, window_size)

    def get_merged_conversation_window(self, conv_id, turn_ids, window_size=1, scored_turns=None):
        return self.chat_collection.get_merged_conversation_window(conv_id, turn_ids, window_size, scored_turns)
