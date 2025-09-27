"""
plan_store.py
-------------
JSON-backed plan cache. Stores LLM-generated subgoal/action plans so
subsequent runs can skip planning and save tokens.
"""

import json
import os
from typing import Optional, Dict, Any

class PlanStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv("ACTION_PLAN_FILE", "plan_cache.json")
        self.plans: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            print(f"[info] No plan map found at {self.path} (starting fresh).")
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self.plans = raw
            print(f"[info] Loaded {len(self.plans)} plans from {self.path}.")
        except Exception as e:
            print(f"[warn] Failed to load {self.path}: {e}")

    def save(self):
        try:
            tmp = f"{self.path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.plans, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[warn] Failed to save {self.path}: {e}")

    def _key(self, intent: str, topic_key: str) -> str:
        return f"{intent}:{topic_key}"

    def get(self, intent: str, topic_key: str):
        return self.plans.get(self._key(intent, topic_key))

    def put(self, intent: str, topic_key: str, plan: Dict[str, Any]):
        self.plans[self._key(intent, topic_key)] = plan
        self.save()
