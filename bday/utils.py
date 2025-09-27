"""
utils.py
--------
Small utilities: pricing, token accounting, and intent parsing.
"""

import os
import re
from typing import Optional

# --- OpenAI pricing (USD per 1K tokens). Override with env OPENAI_PRICE_INPUT/OUTPUT ---
PRICES_PER_1K = {
    "gpt-4":        {"input": 0.03,  "output": 0.06},
    "gpt-4-turbo":  {"input": 0.01,  "output": 0.03},
    "gpt-4o":       {"input": 0.005, "output": 0.015},
    "gpt-4o-mini":  {"input": 0.0005,"output": 0.0015},
}

def get_prices(model: str):
    """Return {'input': float, 'output': float} or None if unknown."""
    pi = os.getenv("OPENAI_PRICE_INPUT")
    po = os.getenv("OPENAI_PRICE_OUTPUT")
    if pi and po:
        try:
            return {"input": float(pi), "output": float(po)}
        except Exception:
            print("[warn] OPENAI_PRICE_INPUT/OUTPUT invalid; cost accounting disabled.")
            return None
    return PRICES_PER_1K.get(model)


class TokenCost:
    """Accumulates token usage and converts to $ when pricing is known."""
    def __init__(self, model: str):
        self.model = model
        self.prices = get_prices(model)   # may be None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add_usage(self, usage):
        """Add an OpenAI usage object (if present)."""
        if not usage:
            return
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def usd(self) -> Optional[float]:
        if not self.prices:
            return None
        return (self.prompt_tokens / 1000.0) * self.prices["input"] + \
               (self.completion_tokens / 1000.0) * self.prices["output"]


# --- Intent helpers ---

def classify_intent(user_goal: str) -> str:
    g = (user_goal or "").lower()
    if " born" in g or "birth" in g:
        return "wikipedia_birth_date"
    return "wikipedia_research"

def topic_key_from_goal(user_goal: str) -> str:
    """Compact cache key from the goal (keeps up to 6 keywords)."""
    toks = re.findall(r"[a-z0-9']+", (user_goal or "").lower())
    stop = {"the","a","an","of","and","or","to","in","on","for","with","how","what","when","why","who"}
    toks = [t for t in toks if t not in stop]
    return "_".join(toks[:6])

def extract_person_name(user_goal: str) -> str:
    """Loose person-name extractor for 'When was X born?' type goals."""
    if not user_goal:
        return ""
    for pat in [r"when was (.*?) born", r"birth date of (.*?)$", r"(.*?) birth year", r"what year was (.*?) born"]:
        m = re.search(pat, user_goal, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return user_goal.strip()
