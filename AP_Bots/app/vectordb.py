from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, MilvusClient


class VectorDB:

    def __init__(self, uri="./apbots.db"):
        self.client = MilvusClient(uri=uri)

    def initialize_db(self):
        
        if not self.client.has_collection("user"):
            fields = [
                FieldSchema(name="user_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="user_name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="password", dtype=DataType.VARCHAR, max_length=100)
            ]
            schema = CollectionSchema(fields, description="User collection")
            client.create_collection(collection_name="user", schema=schema)
        
        # Check and create conversation collection
        if not self.client.has_collection("conversation"):
            fields = [
                FieldSchema(name="conv_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="user_id", dtype=DataType.INT64),
                FieldSchema(name="conversation", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="chatbot", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="start_time", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="end_time", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
            ]
            schema = CollectionSchema(fields, description="Conversation collection")
            client.create_collection(collection_name="conversation", schema=schema)

        return client

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
            return False, "Username already exists."
        self.client.insert(collection_name="user", data={"user_name": username,
                                                        "password": password})
        
        user_id = self.client.query(collection_name="user", filter=f"user_name == '{username}'", output_fields=["user_id"])[0]["user_id"]
        return True, "Sign-up successful. Please log in.", user_id

    def change_password(self, user_id, new_password):

        cur_user = self.client.get("user", ids=user_id)
        cur_user[0]["password"] = new_password
        self.client.upsert(collection_name="user", data=cur_user[0])

    def save_conversation(self, user_id, conversation, chatbot_name, embedding):

        start_time = datetime.now().isoformat()
        self.client.insert(collection_name="conversations", data={
            "user_id": user_id,
            "conversation": conversation,
            "chatbot": chatbot_name,
            "start_time": start_time,
            "end_time": datetime.now().isoformat(),
            "embedding": embedding
        })