"""
answer_cache.py
---------------
Stable answer cache for questions like Wikipedia birthdates.
"""

import json
import os
import time
from typing import Optional, Dict, Any

DEFAULT_TTL = int(os.getenv("ANSWER_CACHE_TTL_SECONDS", "2592000"))  # 30 days

class AnswerCache:
    def __init__(self, path: Optional[str] = None, ttl_seconds: Optional[int] = None):
        self.path = path or os.getenv("ANSWER_CACHE_FILE", "answer_cache.json")
        self.ttl = ttl_seconds if ttl_seconds is not None else DEFAULT_TTL
        self.kv: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            print(f"[info] No answer cache found at {self.path} (starting fresh).")
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.kv = json.load(f)
            print(f"[info] Loaded {len(self.kv)} answers from {self.path}.")
        except Exception as e:
            print(f"[warn] Failed to load answer cache: {e}")

    def save(self):
        try:
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.kv, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[warn] Failed to save answer cache: {e}")

    def get(self, goal: str) -> Optional[Dict[str, Any]]:
        rec = self.kv.get(goal)
        if not rec:
            return None
        if time.time() - rec.get("ts", 0) > self.ttl:
            return None
        return rec

    def put(self, goal: str, answer: str, meta: Dict[str, Any]):
        self.kv[goal] = {"answer": answer, "meta": meta, "ts": time.time()}
        self.save()
