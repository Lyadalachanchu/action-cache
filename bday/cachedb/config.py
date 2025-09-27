# cachedb/config.py
import os
from pathlib import Path

DB_PATH = os.getenv("CACHE_DB_PATH", str(Path(__file__).resolve().parent / "cache.sqlite3"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # keep it light for local
COSINE_THRESHOLD_PROMPT = float(os.getenv("COSINE_THRESHOLD_PROMPT", "0.97"))
COSINE_THRESHOLD_QA = float(os.getenv("COSINE_THRESHOLD_QA", "0.92"))
COSINE_THRESHOLD_PLAN = float(os.getenv("COSINE_THRESHOLD_PLAN", "0.70"))
