# cachedb/resolver.py
from typing import Optional, Dict, Any
# Prefer Weaviate repos when available, but always fall back to SQLite.
try:
    from .weaviate_repos import AnswersRepo, PlansRepo, LLMRepo  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    from .repos import AnswersRepo, PlansRepo, LLMRepo
from .repos import DOMRepo
from .migrate_from_json import canonicalize

answers = AnswersRepo()
plans = PlansRepo()
llm = LLMRepo()
dom = DOMRepo()

def get_answer(question: str) -> Optional[Dict[str, Any]]:
    cq = canonicalize(question)
    hit = answers.approx_get(question, canonical_q=cq)
    return hit  # returns dict or None

def get_plan(goal: str) -> Optional[Dict[str, Any]]:
    hit = plans.approx_get(goal)
    return hit

def robust_click_hint(hint: str, site_domain: Optional[str] = None):
    return dom.approx_by_role(hint, site_domain=site_domain, top_k=5)
