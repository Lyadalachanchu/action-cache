# cachedb/migrate_from_json.py
"""
One-time migration from your existing JSON caches into the new hybrid DB.

Looks for these files (override via env vars):
- ANSWER_JSON=/mnt/data/answer_cache.json
- PLAN_JSON=/mnt/data/plan_cache.json
- LLM_JSON=/mnt/data/llm_cache.json
"""
import os, json, time
from pathlib import Path
from .db import init_db
from .repos import AnswersRepo, PlansRepo, LLMRepo

BASE_DIR = Path(__file__).resolve().parent.parent

def _candidate_paths(env_name: str, filename: str):
    env_path = os.getenv(env_name)
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).with_name(filename))
    candidates.append(BASE_DIR / filename)
    candidates.append(Path(f"/mnt/data/{filename}"))
    # Deduplicate while preserving order
    seen = set()
    filtered = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        filtered.append(p)
    return filtered

ANSWER_JSON = _candidate_paths("ANSWER_JSON", "answer_cache.json")
PLAN_JSON = _candidate_paths("PLAN_JSON", "plan_cache.json")
LLM_JSON = _candidate_paths("LLM_JSON", "llm_cache.json")

def _load(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for candidate in paths:
        p = Path(candidate)
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            continue
    return None

def canonicalize(question: str) -> str:
    # naive canonicalizer for demo: extract "who/what" + "attr" style
    # e.g., "When was Marie Curie born?" -> "PERSON:Marie Curie|ATTR:birth_date"
    q = question.lower()
    attr = "unknown"
    if "born" in q or "birth" in q:
        attr = "birth_date"
    if "died" in q or "death" in q:
        attr = "death_date"
    # very rough entity extraction
    # take capitalized words from original question as entity
    ent = []
    for tok in question.split():
        if tok[:1].isupper():
            ent.append(tok.strip(" ?!.,;:"))
    entity = " ".join(ent) or "UNKNOWN"
    return f"ENTITY:{entity}|ATTR:{attr}"

def run():
    init_db()
    ans_repo = AnswersRepo()
    plan_repo = PlansRepo()
    llm_repo = LLMRepo()

    # answers -----------------------------------------------------------------
    ans = _load(ANSWER_JSON)
    if isinstance(ans, dict):
        # expected format (flexible): {question: {answer:..., url:..., title:..., ts:...}, ...}
        for q, meta in ans.items():
            if not isinstance(meta, dict): 
                continue
            cq = meta.get("canonical_q") or canonicalize(q)
            answer_text = meta.get("answer") or meta.get("text") or ""
            sources = []
            if "url" in meta or "title" in meta:
                sources.append({"url": meta.get("url"), "title": meta.get("title")})
            evidence = {"cached_ts": meta.get("ts"), "note": "migrated from JSON answer_cache"}
            ans_repo.put(cq, q, answer_text, confidence=1.0, freshness_horizon=None, evidence=evidence, sources=sources)

    # plans -------------------------------------------------------------------
    plans = _load(PLAN_JSON)
    if isinstance(plans, dict):
        # expected: {goal_key: {subgoals:[...], actions:[...]}} OR any plan_json
        for goal_key, plan_json in plans.items():
            if not isinstance(plan_json, dict):
                plan_json = {"template": plan_json}
            intent_key = goal_key.split(":")[0] if ":" in goal_key else goal_key
            goal_text = plan_json.get("goal") or goal_key
            site_domain = plan_json.get("site") or None
            plan_repo.put(intent_key, goal_text, plan_json, site_domain=site_domain, success_rate=0.5, version="migrated")

    # llm cache ---------------------------------------------------------------
    llm = _load(LLM_JSON)
    if isinstance(llm, dict):
        # expected: {(model, prompt_hash): {"prompt":..., "text":..., "usage":...}}
        for k, meta in llm.items():
            if not isinstance(meta, dict): 
                continue
            model = meta.get("model") or "unknown"
            prompt = meta.get("prompt") or meta.get("input") or ""
            output = meta.get("text") or meta.get("output") or ""
            usage = meta.get("usage") or {}
            llm_repo.put(model, prompt, output, usage=usage, source_tag="migrated", version="migrated")

if __name__ == "__main__":
    run()
    print("Migration finished.")
