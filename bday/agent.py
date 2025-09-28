# agent.py
import sys
from cachedb_integrations.cache_adapters import (
    AnswerCacheAdapter as AnswerCache,
    PlanStoreAdapter as PlanStore,
    LLMCacheAdapter as LLMCache,
)

# Use absolute import when running as a script
import dom_extractors  # noqa: F401  (imported for side effects / future use)

class LLMBrowserAgent:
    def __init__(self):
        # Swap in DB-backed caches
        self.answer_cache = AnswerCache()
        self.plan_store = PlanStore()
        self.llm_cache = LLMCache()

    def run(self, goal: str):
        # 1) Check answer cache
        hit = self.answer_cache.get(goal)
        if hit:
            print(f"[CACHE HIT] {goal} → {hit['answer_text']}")
            return hit["answer_text"]

        # 2) Check for reusable plan
        plan = self.plan_store.approx_get(goal)
        if plan:
            print(f"[PLAN] Found plan template: {plan['intent_key']}")
            # You'd execute plan["plan_json"]["actions"] here
            # For now, just print them
            for step in plan["plan_json"].get("actions", []):
                print(" →", step)

        # 3) Fall back to LLM
        maybe = self.llm_cache.approx_get(goal)
        if maybe:
            print(f"[LLM CACHE HIT] {maybe['output_text']}")
            return maybe["output_text"]

        print(f"[NO CACHE] Need to browse/LLM for: {goal}")
        # TODO: implement browsing logic
        return None
