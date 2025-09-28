# cachedb_integrations/cache_adapters.py
"""Drop-in adapters to replace your JSON caches with the new hybrid DB.
Usage in your project:
    from cachedb.db import init_db
    from cachedb_integrations.cache_adapters import AnswerCacheAdapter, PlanStoreAdapter, LLMCacheAdapter

    init_db()  # once at startup

    answer_cache = AnswerCacheAdapter()
    plan_store = PlanStoreAdapter()
    llm_cache = LLMCacheAdapter()
"""
from __future__ import annotations
from typing import Any, Dict, Optional, List
import time

# Prefer Weaviate-backed repositories when available
try:
    from cachedb.weaviate_repos import AnswersRepo, PlansRepo, LLMRepo
except ImportError:  # pragma: no cover - fallback for environments without weaviate
    from cachedb.repos import AnswersRepo, PlansRepo, LLMRepo

from cachedb.resolver import get_answer, get_plan
from cachedb.migrate_from_json import canonicalize

# ---------------------------- Answer Cache -----------------------------------

class AnswerCacheAdapter:
    def __init__(self):
        self.repo = AnswersRepo()

    def get(self, question: str) -> Optional[Dict[str, Any]]:
        hit = get_answer(question)
        return hit  # {'answer_text', 'sources', ...} or None

    def put(self, question: str, answer_text: str,
            sources: Optional[List[Dict[str, str]]] = None,
            confidence: float = 1.0,
            freshness_horizon: Optional[int] = None,
            evidence: Optional[Dict[str, Any]] = None) -> int:
        cq = canonicalize(question)
        return self.repo.put(
            canonical_q=cq,
            question_text=question,
            answer_text=answer_text,
            confidence=confidence,
            freshness_horizon=freshness_horizon,
            evidence=evidence or {},
            sources=sources or []
        )

# ---------------------------- Plan Store -------------------------------------

class PlanStoreAdapter:
    def __init__(self):
        self.repo = PlansRepo()

    def approx_get(self, goal_text: str) -> Optional[Dict[str, Any]]:
        return get_plan(goal_text)

    def put(self, intent_key: str, goal_text: str, plan_json: Dict[str, Any],
            site_domain: Optional[str] = None, env_fingerprint: Optional[str] = None,
            success_rate: float = 0.5, version: Optional[str] = None) -> int:
        return self.repo.put(
            intent_key=intent_key,
            goal_text=goal_text,
            plan_json=plan_json,
            site_domain=site_domain,
            env_fingerprint=env_fingerprint,
            success_rate=success_rate,
            version=version
        )

# ---------------------------- LLM Cache --------------------------------------

class LLMCacheAdapter:
    def __init__(self):
        self.repo = LLMRepo()

    def approx_get(self, prompt: str) -> Optional[Dict[str, Any]]:
        return self.repo.approx_get(prompt)

    def put(self, model: str, prompt: str, text: str,
            usage: Optional[Dict[str, Any]] = None,
            tool_state: Optional[str] = None,
            ttl: Optional[int] = None,
            source_tag: Optional[str] = None,
            version: Optional[str] = None) -> int:
        return self.repo.put(
            model=model,
            prompt=prompt,
            output_text=text,
            usage=usage,
            tool_state=tool_state,
            ttl=ttl,
            source_tag=source_tag,
            version=version
        )

# ------------------------- Subgoal Store (per-subgoal) ------------------------

class SubgoalStoreAdapter:
    """Cache actions for a single subgoal description.

    Backed by PlansRepo, using intent_key="SUBGOAL" and goal_text=description,
    with plan_json={"actions": [...]}.
    """
    def __init__(self):
        self.repo = PlansRepo()

    def approx_get(self, description: str) -> Optional[Dict[str, Any]]:
        hit = self.repo.approx_get(description)
        if not hit:
            return None
        pj = hit.get("plan_json", {})
        actions = pj.get("actions")
        if not actions:
            return None
        return {"description": hit.get("goal_text", description), "actions": actions, "similarity": hit.get("similarity", 1.0)}

    def put(self, description: str, actions: List[Dict[str, Any]],
            success_rate: float = 0.8) -> int:
        return self.repo.put(
            intent_key="SUBGOAL",
            goal_text=description,
            plan_json={"actions": actions},
            success_rate=success_rate,
            version="subgoal-v1"
        )
