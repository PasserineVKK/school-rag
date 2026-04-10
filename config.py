import os
from dotenv import load_dotenv

load_dotenv()

# ====================== CẤU HÌNH ======================
DATA_DIR = "data"
CHROMA_PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "school_regulations"

# Embedding - Cache rõ ràng để tránh tải lại
EMBEDDING_MODEL = "BAAI/bge-m3"   # Model tốt + ổn định

# Thư mục cache model (để trong project, không để ổ C)
HF_CACHE_DIR = os.path.join(os.getcwd(), "hf_cache")
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE_DIR

# LLM
GEMINI_MODEL = "gemini-2.5-flash-lite"   # hoặc gemini-2.5-flash nếu quota thấp

MEMORY_K = 4
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200