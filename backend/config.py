import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
