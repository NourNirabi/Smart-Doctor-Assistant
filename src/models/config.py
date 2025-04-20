"""Configuration settings for the RAG Chatbot."""

# Paths
DATA_PATH = r".\data\processed\documents_data_after.txt"
CHROMA_PATH = "chroma_db"

# Model settings
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# Vector store settings
COLLECTION_NAME = "example_collection"

# Text splitting settings
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100

# Retrieval settings
NUM_RESULTS = 3