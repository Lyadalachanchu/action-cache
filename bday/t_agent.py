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
    AnswerCacheAdapter as AnswerCache,
    PlanStoreAdapter as PlanStore,
    LLMCacheAdapter as LLMCache,
)
from cachedb.config import DB_PATH

# --- Provider-agnostic LLM client
from llm_client import LLMClient

# --- Import our core agent
from agent_core import LLMBrowserAgent, MAX_PAGE_TEXT_CHARS

# ===================== Setup =====================
load_dotenv()
init_db()

answers_repo = AnswersRepo()
plans_repo = PlansRepo()
llm_repo = LLMRepo()
answer_cache = AnswerCache()
plan_store = PlanStore()
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


async def _chat_async(messages, temperature=0.0, model_hint=""):
    """Call LLMClient, store in llm_cache, and print token usage with timing"""

    start_time = time.time()

    # Check LLM cache first
    prompt_text = "\n".join(m["content"] for m in messages)
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

    async def _extract_answer(self, goal: str, all_page_texts: List[str]) -> str:
        """Extract answer from collected page texts using LLM"""
        if not all_page_texts:
            return "No information could be retrieved from Wikipedia."

        combined_text = "\n\n".join(all_page_texts)
        if len(combined_text) > MAX_PAGE_TEXT_CHARS:
            combined_text = combined_text[:MAX_PAGE_TEXT_CHARS] + "... [truncated]"

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
            response = await _chat_async(messages, model_hint="create_plan")
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

    async def run(self, goal: str, no_cache=False, plan_preview=False, force_plan=False):
        """Main execution method with robust error handling"""
        global EXECUTION_START_TIME, TOTAL_EXECUTION_TIME
        EXECUTION_START_TIME = time.time()

        print(f"\n🚀 STARTING AUTOMATION: {goal}")
        print("=" * 60)
        print(f"provider={llm_client.provider.upper()}  db={DB_PATH}")

        # Check for cached answer first
        if not force_plan and not no_cache:
            cache_start = time.time()
            cached = answer_cache.get(goal)
            cache_time = time.time() - cache_start

            if cached:
                TOTAL_EXECUTION_TIME = time.time() - EXECUTION_START_TIME
                # Fix the canonicalization display
                canon_key = improved_canonicalize(goal)
                cached_copy = cached.copy()
                cached_copy['canonical_q'] = canon_key
                print(f"✅ [ANSWER CACHE HIT] {cached_copy} (retrieved in {cache_time:.3f}s)")
                print(f"📊 TOTAL EXECUTION TIME: {TOTAL_EXECUTION_TIME:.2f}s | TOKENS: prompt=0 completion=0 total=0 (cached)")
                return

        # Plan preview
        if plan_preview:
            preview_stored_plan(goal)

        # Check for cached plan
        print(f"\n📋 PLANNING PHASE")
        print("=" * 60)

        plan_start = time.time()
        plan_hit = None
        if not force_plan:
            # Use canonicalized goal for plan lookup to ensure proper matching
            canonical_goal = improved_canonicalize(goal)
            print(f"📝 [PLAN LOOKUP] Using canonical goal: '{canonical_goal}'")
            plan_hit = plan_store.approx_get(canonical_goal, site_domain="wikipedia.org")
            if plan_hit:
                similarity = plan_hit.get("similarity", 1.0)
                print(f"📝 [PLAN CACHE DEBUG] Found plan similarity={similarity:.3f}")
                if similarity >= 0.95:  # Much higher threshold to prevent wrong matches
                    plan_time = time.time() - plan_start
                    print(f"📝 [PLAN LIBRARY HIT] {len(plan_hit['plan_json'].get('subgoals', []))} subgoals (sim={similarity:.3f}) in {plan_time:.3f}s")
                else:
                    print(f"📝 [PLAN CACHE MISS] Similarity {similarity:.3f} below threshold 0.95")
                    plan_hit = None
            else:
                plan_time = time.time() - plan_start
                print(f"📝 [PLAN CACHE MISS] No similar plans found (search took {plan_time:.3f}s)")

        if plan_hit:
            subgoals = plan_hit["plan_json"].get("subgoals", [])
            _print_actions("Reused plan actions", subgoals)
        else:
            # Create new plan
            print("📝 [CREATING NEW PLAN]")
            plan_creation_start = time.time()
            subgoals = await self._create_plan(goal)
            plan_creation_time = time.time() - plan_creation_start

            if not subgoals:
                print("❌ Failed to create plan")
                return

            _print_actions("New plan actions", subgoals)
            print(f"📝 [PLAN CREATED] in {plan_creation_time:.3f}s")

            # Store the plan with canonicalized intent key
            canonical_intent = improved_canonicalize(goal)
            plan_store.put(
                intent_key=canonical_intent,  # Use canonicalized key to prevent wrong matches
                goal_text=goal,
                plan_json={"subgoals": subgoals},
                site_domain="wikipedia.org"
            )
            print(f"📝 [PLAN STORED] with intent_key='{canonical_intent}'")

        # Execute the plan
        print(f"\n🎬 EXECUTION PHASE")
        print("=" * 60)

        execution_start = time.time()
        all_page_texts = []

        for i, subgoal in enumerate(subgoals, 1):
            subgoal_start = time.time()
            print(f"\n▶ Subgoal {i}: {subgoal.get('description', 'No description')}")

            # Check browser health before each subgoal
            if not await self.check_browser_health():
                print(f"❌ Browser connection lost before subgoal {i}")
                break

            actions = subgoal.get("actions", [])
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

            subgoal_time = time.time() - subgoal_start
            print(f"   ⏱️ Subgoal {i} completed in {subgoal_time:.3f}s")

        execution_time = time.time() - execution_start
        print(f"🎬 [EXECUTION COMPLETED] in {execution_time:.3f}s")

        # Extract final answer
        print(f"\n🧠 ANSWER EXTRACTION")
        print("=" * 60)

        if all_page_texts:
            extraction_start = time.time()
            print(f"📖 Extracting answer from {len(all_page_texts)} page(s) of content...")
            final_answer = await self._extract_answer(goal, all_page_texts)
            extraction_time = time.time() - extraction_start
            print(f"🧠 [EXTRACTION COMPLETED] in {extraction_time:.3f}s")

            # Cache the answer
            if not no_cache:
                answer_cache.put(goal, final_answer)

            print(f"\n✅ FINAL ANSWER:")
            print("=" * 60)
            print(final_answer)
        else:
            error_msg = "❌ No content was successfully retrieved from Wikipedia."
            print(error_msg)
            if not no_cache:
                answer_cache.put(goal, error_msg)

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
