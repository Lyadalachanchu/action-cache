# cachedb/embeddings.py
import os, json, math, hashlib
from typing import List
from .config import EMBEDDING_MODEL, EMBEDDING_DIM

def _hash_embed(text: str, dim: int) -> List[float]:
    """
    Deterministic fallback embedding that works offline.
    Not semantically meaningful but consistent for prototyping.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # expand deterministically
    vals = []
    seed = int.from_bytes(h[:8], "big")
    x = seed
    for i in range(dim):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        vals.append((x / 0x7FFFFFFF) * 2 - 1)  # [-1, 1]
    # L2 normalize
    norm = math.sqrt(sum(v*v for v in vals)) or 1.0
    return [v / norm for v in vals]

def embed(text: str) -> List[float]:
    """
    Switch to OpenAI embeddings if OPENAI_API_KEY is set.
    Otherwise use deterministic offline embedding.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _hash_embed(text, EMBEDDING_DIM)

    # Lazy import to avoid hard dependency
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        vec = resp.data[0].embedding
        # normalize
        norm = math.sqrt(sum(v*v for v in vec)) or 1.0
        return [v / norm for v in vec]
    except Exception:
        # Fallback if API fails
        return _hash_embed(text, EMBEDDING_DIM)

def cosine(a: List[float], b: List[float]) -> float:
    return sum(x*y for x, y in zip(a, b))
