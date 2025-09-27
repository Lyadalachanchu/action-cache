# utils.py
"""
Utility functions for the agent.
Now using the DB-backed cache adapters instead of JSON-based caches.
"""

from cachedb_integrations.cache_adapters import (
    AnswerCacheAdapter as AnswerCache,
    PlanStoreAdapter as PlanStore,
    LLMCacheAdapter as LLMCache,
)

# Re-export adapters under familiar names so rest of code can stay unchanged
answer_cache = AnswerCache()
plan_store = PlanStore()
llm_cache = LLMCache()

def classify_intent(goal: str) -> str:
    """
    Dummy intent classifier — replace with real logic.
    """
    if "born" in goal or "birth" in goal:
        return "wikipedia_birth_date"
    if "died" in goal or "death" in goal:
        return "wikipedia_death_date"
    return "general_query"

def topic_key_from_goal(goal: str) -> str:
    """
    Produce a key like wikipedia_birth_date:was_marie_curie_born
    """
    intent = classify_intent(goal)
    key = goal.lower().replace(" ", "_")
    return f"{intent}:{key}"
