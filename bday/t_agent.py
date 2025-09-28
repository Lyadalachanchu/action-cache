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




async def _chat_async(messages, temperature=0.0, model_hint="", bypass_cache=False):
    """Call LLMClient, store in llm_cache, and print token usage with timing"""

    start_time = time.time()

    # Check LLM cache first (unless bypassed)
    cached_response = None
    if not bypass_cache:
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

    # persist in LLM cache (unless bypassed)
    if not bypass_cache:
        if 'prompt_text' not in locals():
            prompt_text = "\n".join(m["content"] for m in messages)
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
    import hashlib
    key = f"QUERY:unique_{hashlib.md5(goal.lower().encode()).hexdigest()[:8]}"
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
    hit = plans_repo.approx_get(goal)
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

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based ONLY on the provided Wikipedia content. "
                    "Be concise but thorough. If the information isn't in the provided text, say so clearly. "
                    "Provide specific dates, facts, and details when available."
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

    async def _create_subgoals(self, goal: str) -> List[Dict]:
        """Create subgoal descriptions without specific actions"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Wikipedia research planner. Break down the user's question into very granular, atomic subgoals. "
                    "Each subgoal should be a single, specific task that can be accomplished with 1-3 simple actions. "
                    "Return a JSON array of subgoals with only descriptions. Do NOT include actions yet. "
                    "Make subgoals as specific and reusable as possible. "
                    "Examples of good granular subgoals: 'Navigate to Taylor Swift Wikipedia page', 'Extract birth date from current page', 'Navigate to Grammy Awards page', 'Count Taylor Swift Grammy wins'."
                )
            },
            {
                "role": "user",
                "content": f"Break down this research goal into granular subgoals: {goal}\n\nExample format:\n[\n  {{\"description\": \"Navigate to Taylor Swift Wikipedia page\"}},\n  {{\"description\": \"Extract birth date from Taylor Swift page\"}},\n  {{\"description\": \"Navigate to Beyoncé Wikipedia page\"}},\n  {{\"description\": \"Extract Grammy count from Beyoncé page\"}},\n  {{\"description\": \"Compare the extracted information\"}}\n]"
            }
        ]

        try:
            # Bypass LLM cache for subgoal planning to ensure fresh, specific plans
            response = await _chat_async(messages, model_hint="create_subgoals", bypass_cache=True)
            subgoals = _loose_json_parse(response)

            if not isinstance(subgoals, list):
                raise ValueError("Subgoals must be a list")

            # Ensure each subgoal has a description
            for subgoal in subgoals:
                if not isinstance(subgoal, dict) or "description" not in subgoal:
                    raise ValueError("Each subgoal must have a description")

            return subgoals

        except Exception as e:
            print(f"❌ Subgoal creation failed: {e}")
            return []

    async def _create_actions_for_subgoal(self, subgoal_description: str) -> List[Dict]:
        """Create specific actions for a single subgoal"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Wikipedia action planner. For the given subgoal, create specific actions to achieve it. "
                    "Return a JSON array of actions. Each action must have this exact format:\n"
                    "- goto: {\"action\": \"goto\", \"url\": \"https://en.wikipedia.org/wiki/PageName\"}\n"
                    "- read_page: {\"action\": \"read_page\"}\n"
                    "- scroll: {\"action\": \"scroll\", \"direction\": \"up\" or \"down\"}\n"
                    "Only use Wikipedia URLs (en.wikipedia.org). Use actual names, not placeholders.\n"
                    "IMPORTANT: For comparison tasks, do NOT navigate to new pages - the data should already be available from previous subgoals. Use minimal actions like just reading the current page or scrolling."
                )
            },
            {
                "role": "user",
                "content": f"Create actions for this subgoal: {subgoal_description}\n\nExample format:\n[\n  {{\"action\": \"goto\", \"url\": \"https://en.wikipedia.org/wiki/James_Blunt\"}},\n  {{\"action\": \"read_page\"}}\n]"
            }
        ]

        try:
            # Don't bypass cache for action generation - these can be reused
            response = await _chat_async(messages, model_hint="create_actions")
            actions = _loose_json_parse(response)

            if not isinstance(actions, list):
                raise ValueError("Actions must be a list")

            # Fix action format if needed
            fixed_actions = []
            for action in actions:
                fixed_action = self._normalize_action(action)
                if fixed_action:
                    fixed_actions.append(fixed_action)

            return fixed_actions

        except Exception as e:
            print(f"❌ Action creation failed for subgoal '{subgoal_description}': {e}")
            return []




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

        # Planning phase: Two-phase approach for better caching
        print(f"\n📋 PLANNING PHASE")
        print("=" * 60)

        # Phase 1: Create subgoal descriptions
        print("📝 [CREATING SUBGOALS]")
        subgoal_creation_start = time.time()
        subgoals = await self._create_subgoals(goal)
        subgoal_creation_time = time.time() - subgoal_creation_start
        if not subgoals:
            print("❌ Failed to create subgoals")
            return
        
        print(f"📝 Generated {len(subgoals)} subgoals:")
        for i, sg in enumerate(subgoals, 1):
            print(f"  • Subgoal {i}: {sg.get('description', 'No description')}")
        print(f"📝 [SUBGOALS CREATED] in {subgoal_creation_time:.3f}s")

        # Phase 2: Get or create actions for each subgoal
        print(f"\n🔧 ACTION PLANNING PHASE")
        print("=" * 60)
        
        action_planning_start = time.time()
        for i, subgoal in enumerate(subgoals, 1):
            desc = subgoal.get("description", "")
            print(f"\n▶ Planning actions for subgoal {i}: {desc}")
            
            # Check if we have cached actions for this subgoal
            cached_sg = subgoal_store.approx_get(desc)
            similarity_threshold = 0.75  # Balanced threshold for action reuse
            
            if cached_sg and not force_plan and cached_sg.get('similarity', 0) >= similarity_threshold:
                actions = cached_sg.get("actions", [])
                print(f"   📦 Using cached actions (sim={cached_sg.get('similarity',1):.3f})")
                subgoal["actions"] = actions
            else:
                if cached_sg:
                    print(f"   🔍 Found similar subgoal but similarity too low ({cached_sg.get('similarity',0):.3f} < {similarity_threshold})")
                # Generate new actions for this subgoal
                print(f"   🔨 Generating new actions...")
                actions = await self._create_actions_for_subgoal(desc)
                if actions:
                    subgoal["actions"] = actions
                    # Store the new actions in cache, but avoid caching overly complex sequences
                    if len(actions) <= 5:  # Only cache simple action sequences
                        try:
                            subgoal_store.put(desc, actions, success_rate=0.8)
                            print(f"   💾 Stored new actions in cache")
                        except Exception as e:
                            print(f"   ⚠️ Failed to store actions: {e}")
                    else:
                        print(f"   ⚠️ Skipping cache storage - action sequence too complex ({len(actions)} actions)")
                else:
                    print(f"   ❌ Failed to generate actions for subgoal")
                    subgoal["actions"] = []
        
        action_planning_time = time.time() - action_planning_start
        print(f"🔧 [ACTION PLANNING COMPLETED] in {action_planning_time:.3f}s")
        
        # Print final plan summary
        _print_actions("Final execution plan", subgoals)

        # Execute the plan
        print(f"\n🎬 EXECUTION PHASE")
        print("=" * 60)

        execution_start = time.time()
        all_page_texts = []
        for i, subgoal in enumerate(subgoals, 1):
            subgoal_start = time.time()
            print(f"\n▶ Executing subgoal {i}: {subgoal.get('description', 'No description')}")

            # Check browser health before each subgoal
            if not await self.check_browser_health():
                print(f"❌ Browser connection lost before subgoal {i}")
                break

            # Execute the actions for this subgoal (already planned)
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
            print(f"📖 Extracting answer from {len(all_page_texts)} page(s) of content...")
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
                    browser = await p.chromium.connect_over_cdp(
                        f"wss://cloud.lightpanda.io/ws?token={token}"
                    )

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
