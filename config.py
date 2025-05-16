import os
from dotenv import load_dotenv

# --- Load Environment ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Config ---
DATA_DIR = "data"
CHROMA_PERSIST_DIR = "chroma_store"
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_CHAT_MODEL = "gpt-4o"
LLM_UTILITY_MODEL = "gpt-4o" # Used for compression and summaries
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
TOP_K_RESULTS = 5

# Embedding dimensions (specific to text-embedding-3-large)
EMBEDDING_DIMENSIONS = 1024 