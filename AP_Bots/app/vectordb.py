from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, MilvusClient


def initialize_db():

    client = MilvusClient()
    # Check and create user collection
    if not client.has_collection("user"):
        fields = [
            FieldSchema(name="user_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="user_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="password", dtype=DataType.VARCHAR, max_length=100)
        ]
        schema = CollectionSchema(fields, description="User collection")
        Collection("user", schema)
    
    # Check and create conversation collection
    if not client.has_collection("conversation"):
        fields = [
            FieldSchema(name="user_id", dtype=DataType.INT64),
            FieldSchema(name="conversation", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="chatbot", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="start_time", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="end_time", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        schema = CollectionSchema(fields, description="Conversation collection")
        Collection("conversation", schema)