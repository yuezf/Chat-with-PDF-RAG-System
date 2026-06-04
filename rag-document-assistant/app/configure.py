import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY not set. Please add it to your .env file.")

# EMBEDDING_MODEL = "text-embedding-3-small"

CHROMA_DB_PATH = Path("chroma_db")

CHROMA_COLLECTION_NAME = "documents"