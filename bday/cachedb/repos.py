# cachedb/repos.py
import json, time
from typing import Optional, List, Tuple
from .db import get_connection
from .embeddings import embed, cosine
from .config import (
    COSINE_THRESHOLD_PROMPT, COSINE_THRESHOLD_QA, COSINE_THRESHOLD_PLAN
)

def _now() -> float:
    return time.time()

def _to_json(x) -> str:
    return json.dumps(x, ensure_ascii=False)

def _from_json(s: Optional[str], default=None):
    if not s: return default
    try:
        return json.loads(s)
    except Exception:
        return default

# ------------------------------ LLM Cache -----------------------------------

class LLMRepo:
    def put(self, model: str, prompt: str, output_text: str,
            usage: Optional[dict] = None, tool_state: Optional[str] = None,
            ttl: Optional[int] = None, source_tag: Optional[str] = None, version: Optional[str] = None) -> int:
        v = embed(prompt)
        conn = get_connection()
        with conn:
            cur = conn.execute(
                "INSERT INTO llm_cache (model, prompt_norm_hash, prompt_text, tool_state, output_text, usage_json, ts, ttl, source_tag, version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (model, str(hash(prompt)), prompt, tool_state, output_text, _to_json(usage or {}), _now(), ttl, source_tag, version)
            )
            llm_id = cur.lastrowid
            conn.execute(
                "INSERT INTO llm_cache_vectors (llm_cache_id, prompt_vec_json) VALUES (?, ?)",
                (llm_id, _to_json(v))
            )
        return llm_id

    def approx_get(self, prompt: str, top_k: int = 5) -> Optional[dict]:
        q = embed(prompt)
        conn = get_connection()
        cur = conn.execute("SELECT l.*, lv.prompt_vec_json FROM llm_cache l JOIN llm_cache_vectors lv ON l.id = lv.llm_cache_id")
        best, best_sim, best_row = None, -1.0, None
        rows = cur.fetchall()
        for r in rows:
            v = _from_json(r["prompt_vec_json"], [])
            sim = cosine(q, v)
            if sim > best_sim:
                best_sim, best_row = sim, r
        if best_row and best_sim >= COSINE_THRESHOLD_PROMPT:
            return {
                "id": best_row["id"],
                "model": best_row["model"],
                "prompt_text": best_row["prompt_text"],
                "output_text": best_row["output_text"],
                "usage": _from_json(best_row["usage_json"], {}),
                "ts": best_row["ts"],
                "similarity": best_sim
            }
        return None

# ------------------------------ Answers -------------------------------------

class AnswersRepo:
    def put(self, canonical_q: str, question_text: str, answer_text: str,
            confidence: float = 1.0, freshness_horizon: Optional[int] = None,
            evidence: Optional[dict] = None, sources: Optional[list] = None) -> int:
        qv = embed(canonical_q + " " + question_text)
        conn = get_connection()
        with conn:
            cur = conn.execute(
                "INSERT INTO answers (canonical_q, question_text, answer_text, confidence, freshness_horizon, ts, evidence_json, sources_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (canonical_q, question_text, answer_text, confidence, freshness_horizon, _now(), _to_json(evidence or {}), _to_json(sources or []))
            )
            ans_id = cur.lastrowid
            conn.execute(
                "INSERT INTO answers_vectors (answer_id, q_vec_json) VALUES (?, ?)",
                (ans_id, _to_json(qv))
            )
        return ans_id

    def approx_get(self, question_text: str, canonical_q: Optional[str] = None, top_k: int = 5) -> Optional[dict]:
        key = (canonical_q or "") + " " + question_text
        q = embed(key)
        conn = get_connection()
        cur = conn.execute("SELECT a.*, av.q_vec_json FROM answers a JOIN answers_vectors av ON a.id = av.answer_id")
        best_row, best_sim = None, -1.0
        for r in cur.fetchall():
            v = _from_json(r["q_vec_json"], [])
            sim = cosine(q, v)
            # Freshness: if freshness_horizon set and expired, downweight
            expired = False
            fh = r["freshness_horizon"]
            if fh is not None:
                expired = (_now() - r["ts"]) > fh
                if expired:
                    sim *= 0.8  # light penalty
            if sim > best_sim:
                best_sim, best_row = sim, r
        if best_row and best_sim >= COSINE_THRESHOLD_QA:
            return {
                "id": best_row["id"],
                "canonical_q": best_row["canonical_q"],
                "question_text": best_row["question_text"],
                "answer_text": best_row["answer_text"],
                "confidence": best_row["confidence"],
                "sources": _from_json(best_row["sources_json"], []),
                "evidence": _from_json(best_row["evidence_json"], {}),
                "similarity": best_sim,
                "ts": best_row["ts"]
            }
        return None

# ------------------------------ Plans ---------------------------------------

class PlansRepo:
    def put(self, intent_key: str, goal_text: str, plan_json: dict,
            site_domain: Optional[str] = None, env_fingerprint: Optional[str] = None,
            success_rate: float = 0.5, version: Optional[str] = None) -> int:
        gv = embed(intent_key + " " + goal_text)
        conn = get_connection()
        with conn:
            cur = conn.execute(
                "INSERT INTO plans (intent_key, goal_text, site_domain, env_fingerprint, plan_json, success_rate, version, ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (intent_key, goal_text, site_domain, env_fingerprint, json.dumps(plan_json, ensure_ascii=False), success_rate, version, _now())
            )
            plan_id = cur.lastrowid
            conn.execute(
                "INSERT INTO plans_vectors (plan_id, goal_vec_json) VALUES (?, ?)",
                (plan_id, json.dumps(gv))
            )
        return plan_id

    def approx_get(self, goal_text: str, top_k: int = 5) -> Optional[dict]:
        conn = get_connection()
        cur = conn.execute("SELECT * FROM plans WHERE goal_text = ?", (goal_text,))
        row = cur.fetchone()
        if row:
            return {
                "id": row["id"],
                "intent_key": row["intent_key"],
                "goal_text": row["goal_text"],
                "site_domain": row["site_domain"],
                "plan_json": json.loads(row["plan_json"]),
                "success_rate": row["success_rate"],
                "similarity": 1.0,
            }

        q = embed(goal_text)
        cur = conn.execute("SELECT p.*, pv.goal_vec_json FROM plans p JOIN plans_vectors pv ON p.id = pv.plan_id")
        best, best_sim = None, -1.0
        for r in cur.fetchall():
            v = json.loads(r["goal_vec_json"])
            sim = cosine(q, v) * (0.9 + 0.2 * float(r["success_rate"] or 0.5))  # favor successful plans
            if sim > best_sim:
                best_sim, best = sim, r
        if best and best_sim >= COSINE_THRESHOLD_PLAN:
            return {
                "id": best["id"],
                "intent_key": best["intent_key"],
                "goal_text": best["goal_text"],
                "site_domain": best["site_domain"],
                "plan_json": json.loads(best["plan_json"]),
                "success_rate": best["success_rate"],
                "similarity": best_sim,
            }
        return None

# ------------------------------ DOM -----------------------------------------

class DOMRepo:
    def put(self, site_domain: str, selector: str, role: str, text: str,
            attrs: Optional[dict] = None, quality_score: float = 0.5) -> int:
        dv = embed(role + " " + text + " " + selector)
        conn = get_connection()
        with conn:
            cur = conn.execute(
                "INSERT INTO dom_chunks (site_domain, selector, role, text, attrs_json, ts, quality_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (site_domain, selector, role, text, json.dumps(attrs or {}, ensure_ascii=False), _now(), quality_score)
            )
            dom_id = cur.lastrowid
            conn.execute(
                "INSERT INTO dom_vectors (dom_id, dom_vec_json) VALUES (?, ?)",
                (dom_id, json.dumps(dv))
            )
        return dom_id

    def approx_by_role(self, hint: str, site_domain: Optional[str] = None, top_k: int = 5) -> List[dict]:
        q = embed(hint)
        conn = get_connection()
        cur = conn.execute("SELECT d.*, dv.dom_vec_json FROM dom_chunks d JOIN dom_vectors dv ON d.id = dv.dom_id")
        scored = []
        for r in cur.fetchall():
            if site_domain and r["site_domain"] != site_domain:
                continue
            v = json.loads(r["dom_vec_json"])
            sim = cosine(q, v) * (0.9 + 0.2 * float(r["quality_score"] or 0.5))
            scored.append((sim, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for sim, r in scored[:top_k]:
            out.append({
                "selector": r["selector"],
                "role": r["role"],
                "text": r["text"],
                "similarity": sim
            })
        return out
