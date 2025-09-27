#!/usr/bin/env python3
# t_agent.py — Main automation script with caching and LLM integration
import argparse
import asyncio
import os
import json
import re
import time
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from playwright.async_api import async_playwright

# --- DB + cache
from cachedb.db import init_db, get_connection
from cachedb.repos import AnswersRepo, PlansRepo, LLMRepo
from cachedb_integrations.cache_adapters import (
    SubgoalStoreAdapter as SubgoalStore,
    SubgoalManifestAdapter as SubgoalManifest,
    LLMCacheAdapter as LLMCache,
)
from cachedb.config import DB_PATH

# --- Provider-agnostic LLM client
from llm_client import LLMClient

# --- Import our core agent
from agent_core import LLMBrowserAgent, MAX_PAGE_TEXT_CHARS
from urllib.parse import quote

# ===================== Setup =====================
load_dotenv()
init_db()

answers_repo = AnswersRepo()
plans_repo = PlansRepo()
llm_repo = LLMRepo()
subgoal_store = SubgoalStore()
subgoal_manifest = SubgoalManifest()
llm_cache = LLMCache()
llm_client = LLMClient()

# Prefer OpenAI as LLM (for reliable usage accounting and JSON)
if llm_client.provider.lower() != "openai":
    print(f"⚠️ LLM provider is '{llm_client.provider}'. To force OpenAI, set:")
    print("   export OPENAI_API_KEY=sk-... ; export OPENAI_MODEL=gpt-4o-mini ; export FORCE_PROVIDER=openai")

# Global tracking
RUN_TOKENS = {"prompt": 0, "completion": 0, "total": 0}
EXECUTION_START_TIME = 0
TOTAL_EXECUTION_TIME = 0


def improved_canonicalize(question: str) -> str:
    """Better canonicalization that properly categorizes different question types"""
    q = question.lower()

    # Current time/date questions
    if any(word in q for word in ["current", "today", "now", "what year", "what time", "what date"]):
        if "year" in q:
            return "QUERY:current_year"
        elif "time" in q:
            return "QUERY:current_time"
        elif "date" in q:
            return "QUERY:current_date"
        else:
            return "QUERY:current_temporal"

    # Birth date questions
    if ("when was" in q or "when did" in q) and ("born" in q or "birth" in q):
        # Extract entity name
        entity = "UNKNOWN"
        if "when was" in q and "born" in q:
            start = q.find("when was") + len("when was")
            end = q.find("born")
            if start < end:
                entity = question[start:end].strip().strip("?").strip()

        if entity != "UNKNOWN":
            return f"ENTITY:{entity}|ATTR:birth_date"
        else:
            return "QUERY:birth_date_unknown_person"

    # Death date questions
    if ("when did" in q or "when was") and ("die" in q or "death" in q):
        entity = "UNKNOWN"
        # Similar extraction logic...
        if entity != "UNKNOWN":
            return f"ENTITY:{entity}|ATTR:death_date"
        else:
            return "QUERY:death_date_unknown_person"

    # Age questions
    if "how old" in q:
        start = q.find("how old is") + len("how old is") if "how old is" in q else q.find("how old") + len("how old")
        entity = question[start:].strip().strip("?").strip()
        if entity:
            return f"ENTITY:{entity}|ATTR:age"
        else:
            return "QUERY:age_unknown_person"

    # General information questions
    if any(start in q for start in ["what is", "what are", "what was", "what were"]):
        # Extract the main subject
        subject = q.replace("what is", "").replace("what are", "").replace("what was", "").replace("what were", "").strip("? ")
        return f"QUERY:what_is_{subject[:30].replace(' ', '_')}"

    # How questions
    if q.startswith("how"):
        subject = q.replace("how", "").strip("? ")[:30].replace(" ", "_")
        return f"QUERY:how_{subject}"

    # Where questions
    if q.startswith("where"):
        subject = q.replace("where", "").strip("? ")[:30].replace(" ", "_")
        return f"QUERY:where_{subject}"

    # Default: use a hash of the question to ensure uniqueness
    import hashlib
    question_hash = hashlib.md5(q.encode()).hexdigest()[:8]
    return f"QUERY:unique_{question_hash}"


async def _chat_async(messages, temperature=0.0, model_hint="", cache_mode="approx"):
    """Call LLMClient, store in llm_cache, and print token usage with timing"""

    start_time = time.time()

    # Check LLM cache first (optional)
    prompt_text = "\n".join(m["content"] for m in messages)
    cached_response = None
    if cache_mode != "off":
        cached_response = llm_cache.approx_get(prompt_text)
        if cached_response:
            end_time = time.time()
            duration = end_time - start_time
            similarity = cached_response.get("similarity", 1.0)
            print(f"[LLM CACHE HIT] {llm_client.provider.upper()} {model_hint or 'call'} (sim={similarity:.3f}) - cached response in {duration:.2f}s")

            # Still count tokens for tracking (but they're "free")
            usage = cached_response.get("usage", {})
            pt = int(usage.get("prompt_tokens", 0))
            ct = int(usage.get("completion_tokens", 0))
            tt = int(usage.get("total_tokens", pt + ct))
            RUN_TOKENS["prompt"] += 0  # Cached = 0 tokens used
            RUN_TOKENS["completion"] += 0
            RUN_TOKENS["total"] += 0
            print(f"    Cached tokens: prompt={pt} completion={ct} total={tt} (0 tokens charged)")
            return cached_response["output_text"]

    # Make fresh LLM call
    loop = asyncio.get_event_loop()

    def _call():
        return llm_client.chat(messages, temperature=temperature)

    text, usage = await loop.run_in_executor(None, _call)

    end_time = time.time()
    duration = end_time - start_time

    # persist in LLM cache
    llm_cache.put(
        model=f"{llm_client.provider}:{os.getenv('OPENAI_MODEL') or 'default'}",
        prompt=prompt_text,
        text=text,
        usage=usage or {"hint": model_hint},
    )

    # tokens
    pt = int(usage.get("prompt_tokens", 0))
    ct = int(usage.get("completion_tokens", 0))
    tt = int(usage.get("total_tokens", pt + ct))
    RUN_TOKENS["prompt"] += pt
    RUN_TOKENS["completion"] += ct
    RUN_TOKENS["total"] += tt

    print(f"[LLM {llm_client.provider.upper()}] {model_hint or 'call'} tokens: prompt={pt} completion={ct} total={tt} in {duration:.2f}s")
    return text


# ---------------- Helpers ----------------
def _loose_json_parse(text: str):
    """Extract a JSON object/array from arbitrary LLM output."""
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    for start_pos in range(len(raw)):
        if raw[start_pos] in "{[":
            for end_pos in range(len(raw) - 1, start_pos - 1, -1):
                if raw[end_pos] in "}]":
                    try:
                        return json.loads(raw[start_pos:end_pos + 1])
                    except:
                        continue
    raise ValueError(f"No valid JSON found in: {text[:200]}...")


def print_db_stats():
    """Print cache database statistics."""
    with get_connection() as conn:
        answer_count = conn.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
        plan_count = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
        llm_count = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        print(f"📊 DB stats: answers={answer_count}, plans={plan_count}, llm_cache={llm_count}")


def show_counts(when: str):
    print(f"\n📊 Cache counts ({when}):")
    print_db_stats()


def purge_answer_for(goal: str):
    """Remove cached answer for a specific goal."""
    key = improved_canonicalize(goal)
    print(f"🗑️ Purging cache for: '{goal}' (canonical: '{key}')")

    with get_connection() as conn:
        # First show what we're about to delete
        rows = conn.execute("SELECT id, question_text, answer_text FROM answers WHERE canonical_q=?", (key,)).fetchall()
        for r in rows:
            print(f"   Deleting: Q='{r[1]}' A='{r[2][:50]}...'")

        # Now delete
        for r in rows:
            conn.execute("DELETE FROM answers_vectors WHERE answer_id=?", (r['id'],))
            conn.execute("DELETE FROM answers WHERE id=?", (r['id'],))

    print(f"✅ Purged {len(rows)} cached answer(s)")


def strict_purge_all_wrong_answers():
    """Emergency purge of obviously wrong cached answers"""
    print("🚨 EMERGENCY PURGE: Cleaning up wrong cached answers...")

    with get_connection() as conn:
        # Find answers where the question and answer don't match
        cursor = conn.execute("""
            SELECT id, question_text, answer_text, canonical_q
            FROM answers
        """)

        wrong_answers = []
        for row in cursor.fetchall():
            question = row[1].lower()
            answer = row[2].lower()

            # Check for obvious mismatches
            if ("current year" in question or "what year" in question) and "born" in answer:
                wrong_answers.append(row)
            elif "born" in question and "current" in answer:
                wrong_answers.append(row)
            # Add more mismatch patterns as needed

        print(f"Found {len(wrong_answers)} obviously wrong cached answers:")
        for row in wrong_answers:
            print(f"   Q: '{row[1]}' -> A: '{row[2][:50]}...'")

        if wrong_answers:
            confirm = input("Delete these wrong answers? (y/N): ")
            if confirm.lower() == 'y':
                for row in wrong_answers:
                    conn.execute("DELETE FROM answers_vectors WHERE answer_id=?", (row[0],))
                    conn.execute("DELETE FROM answers WHERE id=?", (row[0],))
                print(f"✅ Deleted {len(wrong_answers)} wrong answers")
            else:
                print("❌ Cancelled - no answers deleted")
        else:
            print("✅ No obviously wrong answers found")


def _print_actions(title: str, subgoals: List[Dict]):
    """Pretty-print subgoals and their actions."""
    print(f"\n🧭 {title}:")
    for i, sg in enumerate(subgoals, 1):
        print(f"  • Subgoal {i}: {sg.get('description', 'No description')}")
        for j, act in enumerate(sg.get("actions", []), 1):
            atype = act.get("action", "unknown")
            # Format action details nicely
            if atype == "goto":
                url = act.get("url", "")
                print(f"      {j}. {atype} -> {url}")
            elif atype == "scroll":
                direction = act.get("direction", "down")
                print(f"      {j}. {atype} {direction}")
            elif atype == "read_page":
                print(f"      {j}. {atype}")
            else:
                # Fallback to show raw action
                known = {k: v for k, v in act.items() if k != "action"}
                print(f"      {j}. {atype} {known}")


def preview_stored_plan(goal: str):
    hit = plan_store.approx_get(goal, site_domain="wikipedia.org")
    if not hit:
        print("\n[PLAN PREVIEW] No stored plan found for this goal (or similar).")
        print("  Tip: run once with --force-plan to create & store a plan.")
        return
    subgoals = (hit["plan_json"] or {}).get("subgoals", [])
    print(f"\n[PLAN PREVIEW] intent_key={hit['intent_key']} similarity={hit.get('similarity',1):.3f}")
    _print_actions("Stored plan actions", subgoals)


class ExtendedLLMBrowserAgent(LLMBrowserAgent):
    """Extended agent with planning and execution capabilities"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If all actions come from subgoal cache, skip LLM extraction
        self.no_llm_when_cached = True

    async def _extract_answer(self, goal: str, all_page_texts: List[str]) -> str:
        """Extract answer from collected page texts using LLM"""
        if not all_page_texts:
            return "No information could be retrieved from Wikipedia."

        # Ensure each page contributes within the overall budget so later pages aren't dropped
        pages = list(all_page_texts)
        total_len = sum(len(p) for p in pages)
        if total_len > MAX_PAGE_TEXT_CHARS and len(pages) > 0:
            per_page_budget = max(1000, MAX_PAGE_TEXT_CHARS // len(pages))
            trimmed_pages = [p[:per_page_budget] for p in pages]
            combined_text = "\n\n".join(trimmed_pages) + "... [truncated]"
        else:
            combined_text = "\n\n".join(pages)

        # Debug: Print first part of each page content
        print(f"\n🔍 DEBUG: Extracted content preview:")
        for i, text in enumerate(all_page_texts, 1):
            preview = text[:300].replace('\n', ' ')
            print(f"   Page {i}: {preview}...")
            # Look for birth dates specifically
            if "born" in text.lower() or "birth" in text.lower():
                import re
                birth_matches = re.findall(r'born[^\.]*\d{4}[^\.]*', text.lower())
                if birth_matches:
                    print(f"   -> Found birth info: {birth_matches[0]}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based ONLY on the provided Wikipedia text. "
                    "Be concise but thorough. If the information isn't in the provided text, say so clearly. "
                    "Provide specific dates, facts, and details when available. "
                    "For comparison questions, look for birth dates in both pieces of content to make the comparison."
                )
            },
            {
                "role": "user",
                "content": f"Question: {goal}\n\nWikipedia content:\n{combined_text}\n\nAnswer:"
            }
        ]

        try:
            answer = await _chat_async(messages, model_hint="extract_answer")
            return answer.strip()
        except Exception as e:
            print(f"❌ Answer extraction failed: {e}")
            return f"Error extracting answer: {e}"

    async def _create_plan(self, goal: str) -> List[Dict]:
        """Create a new execution plan for the given goal"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Wikipedia research planner. Create a plan to answer the user's question using only Wikipedia. "
                    "Return a JSON array of subgoals, each with a 'description' and 'actions' array. "
                    "Each action must have this exact format:\n"
                    "- goto: {\"action\": \"goto\", \"url\": \"https://en.wikipedia.org/wiki/PageName\"}\n"
                    "- read_page: {\"action\": \"read_page\"}\n"
                    "- scroll: {\"action\": \"scroll\", \"direction\": \"up\" or \"down\"}\n"
                    "Only use wikipedia.org URLs. Use the actual person/entity names in URLs, not placeholders like PERSON."
                )
            },
            {
                "role": "user",
                "content": f"Create a research plan to answer: {goal}\n\nExample format:\n[\n  {{\n    \"description\": \"Find information about James Blunt\",\n    \"actions\": [\n      {{\"action\": \"goto\", \"url\": \"https://en.wikipedia.org/wiki/James_Blunt\"}},\n      {{\"action\": \"read_page\"}}\n    ]\n  }}\n]"
            }
        ]

        try:
            response = await _chat_async(messages, model_hint="create_plan", cache_mode="off")
            subgoals = _loose_json_parse(response)

            if not isinstance(subgoals, list):
                raise ValueError("Plan must be a list of subgoals")

            # Fix action format if needed
            for subgoal in subgoals:
                if "actions" in subgoal:
                    fixed_actions = []
                    for action in subgoal["actions"]:
                        fixed_action = self._normalize_action(action)
                        if fixed_action:
                            fixed_actions.append(fixed_action)
                    subgoal["actions"] = fixed_actions

            return subgoals

        except Exception as e:
            print(f"❌ Plan creation failed: {e}")
            return []

    def _heuristic_entities(self, goal: str) -> List[str]:
        """Very simple entity extraction: sequences of capitalized words or quoted phrases."""
        g = goal.strip()
        # Quoted phrases first
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", g)
        names = [q[0] or q[1] for q in quoted if (q[0] or q[1])]
        # Capitalized sequences (skip leading How/What/When/Is/Are)
        parts = re.findall(r"(?:\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", g)
        stop_first = {"How","What","When","Where","Why","Who","Is","Are","Do","Does","Did","Will","Can"}
        for p in parts:
            if p.split()[0] in stop_first:
                continue
            if p not in names:
                names.append(p)
        # Deduplicate while preserving order
        seen, out = set(), []
        for n in names:
            k = n.strip()
            if k and k not in seen:
                seen.add(k)
                out.append(k)
        return out[:4]

    def _build_heuristic_subgoals(self, goal: str) -> List[Dict[str, Any]]:
        """Construct minimal subgoals without using LLM."""
        ents = self._heuristic_entities(goal)
        subgoals: List[Dict[str, Any]] = []
        for e in ents:
            slug = quote(e.replace(" ", "_"))
            subgoals.append({
                "description": f"Find information about {e}",
                "actions": [
                    {"action": "goto", "url": f"https://en.wikipedia.org/wiki/{slug}"},
                    {"action": "read_page"}
                ]
            })
        # If comparative/difference question, add a compute subgoal placeholder (no LLM)
        low = goal.lower()
        if any(k in low for k in ["how many more", "difference", "older than", "younger than", "compare", "versus", "vs"]):
            subgoals.append({
                "description": "Compute comparison based on collected pages",
                "actions": [
                    {"action": "scroll", "direction": "up"},
                    {"action": "scroll", "direction": "down"}
                ]
            })
        return subgoals or []

    def _extract_year(self, text: str) -> Optional[int]:
        # Prefer YYYY-MM-DD
        m = re.search(r"(19|20)\d{2}-\d{2}-\d{2}", text)
        if m:
            return int(m.group(0)[:4])
        # Month Day, Year
        m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+((19|20)\d{2})", text, re.IGNORECASE)
        if m:
            return int(m.group(2))
        # Born ... 1977 etc.
        m = re.search(r"\b(19|20)\d{2}\b", text)
        if m:
            return int(m.group(0))
        return None

    def _extract_population(self, text: str) -> Optional[int]:
        # Look for a population number (largest number after 'population')
        m_iter = re.finditer(r"population[^\d]*(\d[\d,\.\s]{3,})", text, re.IGNORECASE)
        best = None
        for m in m_iter:
            raw = m.group(1)
            num = re.sub(r"[^0-9]", "", raw)
            try:
                val = int(num)
                best = max(best or 0, val)
            except Exception:
                continue
        return best

    async def _heuristic_answer(self, goal: str, pages: List[str]) -> Optional[str]:
        if not pages:
            return None
        g = goal.lower()
        # Age difference between two people
        if any(k in g for k in ["older than", "younger than", "how much older", "age difference"]):
            if len(pages) >= 2:
                y1 = self._extract_year(pages[0])
                y2 = self._extract_year(pages[1])
                if y1 and y2:
                    diff = abs(y1 - y2)
                    rel = "older" if (y2 and y1 and y2 < y1) else "younger"
                    return f"Approximate age difference: {diff} years ({rel})."
        # Population difference between two cities
        if any(k in g for k in ["how many more people", "population difference", "more people than"]):
            if len(pages) >= 2:
                p1 = self._extract_population(pages[0])
                p2 = self._extract_population(pages[1])
                if p1 and p2:
                    return f"Population difference (page1 - page2): {p1 - p2:,}."
        # How old is X
        if g.startswith("how old") and pages:
            y = self._extract_year(pages[0])
            if y:
                from datetime import datetime
                age = datetime.utcnow().year - y
                return f"Approximate age based on birth year {y}: {age} years."
        return None

    async def run(self, goal: str, no_cache=False, plan_preview=False, force_plan=False):
        """Main execution method with robust error handling"""
        global EXECUTION_START_TIME, TOTAL_EXECUTION_TIME
        EXECUTION_START_TIME = time.time()

        print(f"\n🚀 STARTING AUTOMATION: {goal}")
        print("=" * 60)
        print(f"provider={llm_client.provider.upper()}  db={DB_PATH}")

        # No answer-level cache: always compute using subgoal cache only

        # Plan preview
        if plan_preview:
            preview_stored_plan(goal)

        # Planning phase (no plan-level caching; we’ll cache per subgoal instead)
        print(f"\n📋 PLANNING PHASE")
        print("=" * 60)

        # Zero-LLM path: if we have a subgoal manifest and all subgoals have cached actions, skip planning
        canonical_goal = improved_canonicalize(goal)
        manifest = subgoal_manifest.get(canonical_goal, site_domain="wikipedia.org")
        subgoals = []
        used_manifest = False
        if manifest:
            all_cached = True
            for desc in manifest:
                hit = subgoal_store.approx_get(desc, site_domain="wikipedia.org")
                if not hit:
                    all_cached = False
                    break
            if all_cached:
                used_manifest = True
                subgoals = [{"description": d, "actions": subgoal_store.approx_get(d, site_domain="wikipedia.org")["actions"]} for d in manifest]
                print("📝 [NO-LLM PLAN] Using cached subgoal manifest and actions.")

        if not subgoals:
            # Fallback: derive subgoal descriptions from goal (cheap heuristic), but DO NOT call LLM
            # Only accept if all are cached; else, call LLM planner
            heuristic_descs = [f"Find information about {e}" for e in self._heuristic_entities(goal)]
            if heuristic_descs:
                all_cached = True
                for desc in heuristic_descs:
                    if not subgoal_store.approx_get(desc, site_domain="wikipedia.org"):
                        all_cached = False
                        break
                if all_cached:
                    subgoals = [{"description": d, "actions": subgoal_store.approx_get(d, site_domain="wikipedia.org")["actions"]} for d in heuristic_descs]
                    used_manifest = True
                    print("📝 [NO-LLM PLAN] Using cached actions for heuristic descriptions.")

        # Always generate subgoals independent of cache
        print("📝 [CREATING NEW PLAN]")
        plan_creation_start = time.time()
        # Disable LLM cache for planning to avoid unrelated plans
        subgoals = await self._create_plan(goal)
        plan_creation_time = time.time() - plan_creation_start
        if not subgoals:
            print("❌ Failed to create plan")
            return
        _print_actions("New plan actions", subgoals)
        print(f"📝 [PLAN CREATED] in {plan_creation_time:.3f}s")
        # Store manifest after plan generation (cache only after generating subgoals)
        try:
            descs = [sg.get("description", "") for sg in subgoals if sg.get("description")]
            if descs:
                subgoal_manifest.put(canonical_goal, descs, site_domain="wikipedia.org")
        except Exception as e:
            print(f"⚠️ Failed to store manifest: {e}")

        # Execute the plan
        print(f"\n🎬 EXECUTION PHASE")
        print("=" * 60)

        execution_start = time.time()
        all_page_texts = []

        all_actions_from_cache = True
        for i, subgoal in enumerate(subgoals, 1):
            subgoal_start = time.time()
            print(f"\n▶ Subgoal {i}: {subgoal.get('description', 'No description')}")

            # Check browser health before each subgoal
            if not await self.check_browser_health():
                print(f"❌ Browser connection lost before subgoal {i}")
                break

            # Subgoal-level cache: try to reuse actions by description
            desc = subgoal.get("description", "")
            cached_sg = None if force_plan else subgoal_store.approx_get(desc, site_domain="wikipedia.org")
            if cached_sg:
                actions = cached_sg.get("actions", [])
                print(f"   📦 Using cached actions for subgoal (sim={cached_sg.get('similarity',1):.3f})")
            else:
                actions = subgoal.get("actions", [])
                all_actions_from_cache = False
            for j, action in enumerate(actions, 1):
                action_start = time.time()
                print(f"   [{j}] EXECUTE {action.get('action', 'unknown')} …")

                try:
                    result = await self._exec_action(action)
                    action_time = time.time() - action_start

                    if result is None:
                        print(f"   ❌ Action failed in {action_time:.3f}s, continuing...")
                        continue
                    elif isinstance(result, str) and len(result) > 100:
                        # This is page text content
                        all_page_texts.append(result)
                        print(f"   ✅ Collected page content ({len(result)} chars) in {action_time:.3f}s")
                    elif result:
                        print(f"   ✅ Action completed successfully in {action_time:.3f}s")
                    else:
                        print(f"   ⚠️ Action completed with warnings in {action_time:.3f}s")

                except Exception as e:
                    action_time = time.time() - action_start
                    print(f"   ❌ Action execution error in {action_time:.3f}s: {e}")
                    continue

            # If no cached actions existed but we executed some, store them now
            if not cached_sg and actions:
                try:
                    subgoal_store.put(desc, actions, site_domain="wikipedia.org", success_rate=0.8)
                    print("   💾 Stored subgoal actions in cache")
                except Exception as e:
                    print(f"   ⚠️ Failed to store subgoal actions: {e}")

            subgoal_time = time.time() - subgoal_start
            print(f"   ⏱️ Subgoal {i} completed in {subgoal_time:.3f}s")

        execution_time = time.time() - execution_start
        print(f"🎬 [EXECUTION COMPLETED] in {execution_time:.3f}s")

        # Extract final answer
        print(f"\n🧠 ANSWER EXTRACTION")
        print("=" * 60)

        if all_page_texts:
            print(f"📖 Extracting answer from {len(all_page_texts)} page(s) of content...")
            # If we used only cached actions and configured to avoid LLM, try heuristic answer
            if all_actions_from_cache and self.no_llm_when_cached:
                final_answer = await self._heuristic_answer(goal, all_page_texts)
                if not final_answer:
                    print("   ⚠️ Heuristic extractor could not answer without LLM. Skipping LLM per configuration.")
                    final_answer = "No LLM run (cached-only mode). Sufficient structured extractors not available for this question."
            else:
                extraction_start = time.time()
                final_answer = await self._extract_answer(goal, all_page_texts)
                extraction_time = time.time() - extraction_start
                print(f"🧠 [EXTRACTION COMPLETED] in {extraction_time:.3f}s")

            print(f"\n✅ FINAL ANSWER:")
            print("=" * 60)
            print(final_answer)
        else:
            error_msg = "❌ No content was successfully retrieved from Wikipedia."
            print(error_msg)

        # Print comprehensive timing and token summary
        TOTAL_EXECUTION_TIME = time.time() - EXECUTION_START_TIME
        print(f"\n📊 EXECUTION SUMMARY:")
        print("=" * 60)
        print(f"⏱️  TOTAL TIME: {TOTAL_EXECUTION_TIME:.2f}s")
        print(f"🎯 TOKENS USED: prompt={RUN_TOKENS['prompt']} completion={RUN_TOKENS['completion']} total={RUN_TOKENS['total']}")
        if RUN_TOKENS['total'] > 0:
            cost_estimate = (RUN_TOKENS['prompt'] * 0.0000015) + (RUN_TOKENS['completion'] * 0.000006)  # GPT-4o-mini pricing
            print(f"💰 ESTIMATED COST: ${cost_estimate:.4f}")
        else:
            print(f"💰 ESTIMATED COST: $0.0000 (cached)")
        print("=" * 60)


async def main():
    ap = argparse.ArgumentParser(description="LLM-powered Wikipedia research agent")
    ap.add_argument("goal", nargs="*", help='Research goal (e.g., "When was Marie Curie born?")')
    ap.add_argument("--no-cache", action="store_true", help="Bypass answer cache")
    ap.add_argument("--force-plan", action="store_true", help="Skip answer cache and force planning/execution")
    ap.add_argument("--plan-preview", action="store_true", help="Print stored plan actions even on cache hit")
    ap.add_argument("--purge", action="store_true", help="Purge cached answer for the given goal")
    ap.add_argument("--emergency-purge", action="store_true", help="Emergency purge of obviously wrong cached answers")
    ap.add_argument("--show-counts", action="store_true", help="Print DB table counts before/after")
    ap.add_argument("--headless", action="store_true", help="Use local headless Chromium if Lightpanda fails")
    args = ap.parse_args()

    goal = " ".join(args.goal).strip() if args.goal else "When was Marie Curie born?"

    if args.purge:
        purge_answer_for(goal)
        return

    if args.emergency_purge:
        strict_purge_all_wrong_answers()
        return

    if args.show_counts:
        show_counts("before")

    token = os.getenv("LIGHTPANDA_TOKEN")
    agent = None

    try:
        # Initialize playwright context
        async with async_playwright() as p:
            browser = None

            # Try connecting to Lightpanda first
            if token:
                try:
                    print("Connecting to Lightpanda browser…")
                    browser = await p.chromium.connect_over_cdp(f"wss://cloud.lightpanda.io/ws?token={token}")
                    print("✅ Connected to Lightpanda.")
                except Exception as e:
                    print("❌ Lightpanda connection failed:", e)
                    browser = None

            # Fallback to local browser
            if not browser:
                print("Launching local Chromium instead…")
                browser = await p.chromium.launch(headless=args.headless)

            page = await browser.new_page()

            # Create agent with browser references
            agent = ExtendedLLMBrowserAgent(page=page, browser=browser)
            agent.use_headless = args.headless
            agent.token = token
            agent.playwright_context = p
            agent.playwright_instance = p

            # Run the automation
            await agent.run(goal, no_cache=args.no_cache, plan_preview=args.plan_preview, force_plan=args.force_plan)

            # Cleanup
            await browser.close()

    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        if agent:
            await agent.cleanup()

    if args.show_counts:
        show_counts("after")


if __name__ == "__main__":
    asyncio.run(main())
