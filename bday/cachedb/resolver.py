# cachedb/resolver.py
from typing import Optional, Dict, Any
from .repos import AnswersRepo, PlansRepo, LLMRepo, DOMRepo
from .migrate_from_json import canonicalize

answers = AnswersRepo()
plans = PlansRepo()
llm = LLMRepo()
dom = DOMRepo()

def get_answer(question: str) -> Optional[Dict[str, Any]]:
    cq = canonicalize(question)
    hit = answers.approx_get(question, canonical_q=cq)
    return hit  # returns dict or None

def get_plan(goal: str, site_domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
    hit = plans.approx_get(goal, site_domain=site_domain)
    return hit

def robust_click_hint(hint: str, site_domain: Optional[str] = None):
    return dom.approx_by_role(hint, site_domain=site_domain, top_k=5)
