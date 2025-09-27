#!/usr/bin/env python3
# t_agent.py — Strict Wikipedia-only browsing + closed-book extraction
import argparse
import asyncio
import os
import json
import re
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv
from playwright.async_api import async_playwright

# --- DB + cache
from cachedb.db import init_db, get_connection
from cachedb.migrate_from_json import canonicalize
from cachedb.repos import AnswersRepo, PlansRepo, LLMRepo
from cachedb_integrations.cache_adapters import (
    AnswerCacheAdapter as AnswerCache,
    PlanStoreAdapter as PlanStore,
    LLMCacheAdapter as LLMCache,
)
from cachedb.config import DB_PATH

# --- Provider-agnostic LLM client
from llm_client import LLMClient

# ===================== Globals =====================
STRICT_WIKI_ONLY = True  # hard guard: only act on *.wikipedia.org and only extract from page text
MAX_PAGE_TEXT_CHARS = 18000
WIKI_HOME = "https://en.wikipedia.org"
WIKI_SEARCH = WIKI_HOME + "/w/index.php?search="

# Common misnamings → correct canonical titles
WIKI_TITLE_FIXES = {
    "Crisis_of_the_Roman_Empire": "Crisis_of_the_Third_Century",
    "Crisis_of_Roman_Empire": "Crisis_of_the_Third_Century",
}

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

RUN_TOKENS = {"prompt": 0, "completion": 0, "total": 0}


async def _chat_async(messages, temperature=0.0, model_hint=""):
    """Call LLMClient, store in llm_cache, and print token usage (closed-book => temp=0.0)."""
    loop = asyncio.get_event_loop()

    def _call():
        return llm_client.chat(messages, temperature=temperature)

    text, usage = await loop.run_in_executor(None, _call)

    # persist in LLM cache
    llm_cache.put(
        model=f"{llm_client.provider}:{os.getenv('OPENAI_MODEL') or 'default'}",
        prompt="\n".join(m["content"] for m in messages),
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

    print(f"[LLM {llm_client.provider.upper()}] {model_hint or 'call'} tokens: prompt={pt} completion={ct} total={tt}")
    return text


# ---------------- Helpers ----------------
def _loose_json_parse(text: str):
    """Extract a JSON object/array from arbitrary LLM output."""
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except Exception:
        pass

    def _extract_balanced(s, opener, closer):
        start = s.find(opener)
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(s)):
            if s[i] == opener:
                depth += 1
            elif s[i] == closer:
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
        return None

    candidate = _extract_balanced(raw, "{", "}")
    if candidate is None:
        candidate = _extract_balanced(raw, "[", "]")
    if candidate:
        return json.loads(candidate)

    raise ValueError("No JSON object/array found")


def purge_answer_for(question: str):
    cq = canonicalize(question)
    conn = get_connection()
    with conn:
        rows = conn.execute("SELECT id FROM answers WHERE canonical_q=?", (cq,)).fetchall()
        for r in rows:
            conn.execute("DELETE FROM answers_vectors WHERE answer_id=?", (r['id'],))
            conn.execute("DELETE FROM answers WHERE id=?", (r['id'],))
    print(f"[PURGE] removed cached answer for canonical_q={cq}")


def show_counts(stage: str):
    conn = get_connection()
    cur = conn.cursor()
    tables = ["answers", "plans", "llm_cache"]
    print(f"\n[DB COUNTS] {stage}")
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"  {t:10s}: {cur.fetchone()[0]}")
    print(f"  db_path   : {DB_PATH}")
    conn.close()


def _flatten_actions(subgoals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat = []
    for sg in subgoals:
        for act in sg.get("actions", []) or []:
            flat.append(act)
    return flat


def _print_actions(title: str, subgoals: List[Dict[str, Any]]):
    print(f"\n🧭 {title}:")
    for sg in subgoals:
        print(f"  • Subgoal {sg.get('id', '?')}: {sg.get('title', '').strip()}")
        actions = sg.get("actions") or []
        if not actions:
            print("      (no actions)")
        else:
            for i, a in enumerate(actions, 1):
                atype = a.get("action") or a.get("type")
                # print known fields, stay compact
                known = {k: v for k, v in a.items() if k in ("action", "type", "url", "selector", "text", "direction")}
                print(f"      {i}. {atype} {known}")


def preview_stored_plan(goal: str):
    hit = plan_store.approx_get(goal, site_domain="wikipedia.org")
    if not hit:
        print("\n[PLAN PREVIEW] No stored plan found for this goal (or similar).")
        print("  Tip: run once with --force-plan to create & store a plan.")
        return
    subgoals = (hit["plan_json"] or {}).get("subgoals", [])
    print(f"\n[PLAN PREVIEW] intent_key={hit['intent_key']} similarity={hit.get('similarity',1):.3f}")
    _print_actions("Stored plan actions", subgoals)


def _ensure_wikipedia_url(url: str) -> bool:
    return "wikipedia.org" in (url or "").lower()


def _normalize_wiki_url(url: str) -> str:
    if not url:
        return url
    if "wikipedia.org" not in url:
        return url
    for bad, good in WIKI_TITLE_FIXES.items():
        if bad in url:
            return url.replace(bad, good)
    return url


async def _goto_with_retry(agent, url: str, retries: int = 2) -> bool:
    """
    Navigate with retries. If the page was closed, re-open a new one.
    Returns True/False for success.
    """
    url = _normalize_wiki_url(url)
    for attempt in range(retries + 1):
        try:
            await agent.page.goto(url, wait_until="domcontentloaded")
            await agent.page.wait_for_load_state("networkidle")
            return True
        except Exception as e:
            err = str(e)
            print(f"     nav attempt {attempt+1} failed: {err}")
            # Reopen page if closed
            if "has been closed" in err or "Target" in err:
                try:
                    if agent.browser:
                        new_page = await agent.browser.new_page()
                    else:
                        new_page = await agent.page.context.new_page()
                    agent.page = new_page
                    print("     ↻ Reopened a fresh page, retrying…")
                except Exception as open_err:
                    print(f"     failed to reopen page: {open_err}")
            await asyncio.sleep(1.0)
    return False


async def _wiki_search(agent, query: str) -> bool:
    """Go to Wikipedia search results for query (still on wikipedia)."""
    q = quote_plus(query.strip())
    search_url = WIKI_SEARCH + q
    print(f"   → WIKI SEARCH {query!r} ({search_url})")
    return await _goto_with_retry(agent, search_url)


async def _extract_page_text(page) -> str:
    # strict: get visible main content only
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    text = await page.evaluate(
        """
        () => {
          const kill = (sel) => document.querySelectorAll(sel).forEach(n => n.remove());
          kill('script,style,nav,header,footer,aside');
          const main = document.querySelector('#mw-content-text') || document.body;
          return main.innerText || '';
        }
        """
    )
    text = (text or "").strip()
    if len(text) > MAX_PAGE_TEXT_CHARS:
        text = text[:MAX_PAGE_TEXT_CHARS]
    return text


async def _closed_book_extract(question: str, page_title: str, page_url: str, page_text: str) -> Dict[str, Any]:
    """
    Use the LLM ONLY to extract from the provided page_text.
    If info is not present verbatim in page_text, return found=false.
    """
    prompt = f"""
You are an information extractor. You MUST answer ONLY from the provided PAGE TEXT.
If the answer is not present verbatim in PAGE TEXT, respond with found=false and answer="".

Return strict JSON with keys: found (boolean), answer (string), evidence (short quote).

QUESTION:
{question}

PAGE TITLE:
{page_title}

PAGE URL:
{page_url}

PAGE TEXT (truncated):
{page_text}
"""
    text = await _chat_async([{"role": "user", "content": prompt}], temperature=0.0, model_hint="closed-book-extract")
    try:
        obj = _loose_json_parse(text)
        if not isinstance(obj, dict):
            raise ValueError("parsed non-dict")
        # sanitize
        obj.setdefault("found", False)
        obj.setdefault("answer", "")
        obj.setdefault("evidence", "")
        return obj
    except Exception:
        # very defensive fallback: never fabricate
        return {"found": False, "answer": "", "evidence": ""}


# ---------------- Agent ----------------
class LLMBrowserAgent:
    def __init__(self, page, browser=None, verbose: bool = True):
        self.page = page
        self.browser = browser
        self.verbose = verbose
        self.subgoals: List[dict] = []
        self.collected_information: List[dict] = []

    async def create_subgoals_and_actions(self, user_goal: str) -> bool:
        print("\n📋 PLANNING PHASE")
        print("=" * 60)

        # Try reuse
        plan_hit = plan_store.approx_get(user_goal, site_domain="wikipedia.org")
        if plan_hit:
            subgoals = plan_hit["plan_json"].get("subgoals", [])
            if subgoals:
                self.subgoals = subgoals
                sim = plan_hit.get("similarity", 1.0)
                print(f"📝 [PLAN LIBRARY HIT] {len(self.subgoals)} subgoals (sim={sim:.3f})")
                _print_actions("Reused plan actions", self.subgoals)
                return True

        # Plan with LLM (JSON only)
        planning_prompt = f"""
You are a browser-automation planner that must ONLY use Wikipedia.
Break the goal into 3–5 subgoals; each subgoal MUST include 'actions'.
Allowed actions:
- goto(url)  [only wikipedia URLs]
- type(selector,text)
- press_enter()
- click(selector)
- read_page()
- scroll(direction)

Prefer search-first strategy:
1) goto("{WIKI_HOME}/wiki/Main_Page")
2) type("#searchInput","<query>")
3) press_enter()
4) click() the most relevant /wiki/ link
5) read_page()

Return STRICT JSON ONLY:
{{
  "subgoals":[
    {{
      "id": 1,
      "title": "Search for TOPIC on Wikipedia",
      "actions": [
        {{ "action": "goto", "url": "{WIKI_HOME}/wiki/Main_Page" }},
        {{ "action": "type", "selector": "#searchInput", "text": "TOPIC" }},
        {{ "action": "press_enter" }},
        {{ "action": "click", "selector": "a[href^='/wiki/']:not([href*=':'])" }},
        {{ "action": "read_page" }}
      ]
    }}
  ]
}}
Goal: "{user_goal}"
"""
        text = await _chat_async(
            [{"role": "user", "content": planning_prompt}],
            temperature=0.0,
            model_hint="planning",
        )

        try:
            plan_data = _loose_json_parse(text)
            subgoals = plan_data.get("subgoals", []) if isinstance(plan_data, dict) else []
            if not subgoals:
                raise ValueError("No 'subgoals' key")
            self.subgoals = subgoals
            print(f"📝 Created {len(self.subgoals)} subgoals.")
            _print_actions("Created plan actions", self.subgoals)

            plan_json = {"params": {}, "subgoals": self.subgoals, "actions": _flatten_actions(self.subgoals)}
            plans_repo.put(
                intent_key="wikipedia_research",
                goal_text=user_goal,
                plan_json=plan_json,
                site_domain="wikipedia.org",
                success_rate=0.6,
                version="v1",
            )
            print("🗂 Plan stored → plans table (site=wikipedia.org)")
            return True

        except Exception as e:
            print("❌ Planning parse failed:", e)
            print("🔎 RAW LLM planning output (first 600 chars):")
            print(text[:600])

            # fallback minimal plan (still wiki-only)
            self.subgoals = [
                {
                    "id": 1,
                    "title": "Open Wikipedia main page",
                    "actions": [
                        {"action": "goto", "url": f"{WIKI_HOME}/wiki/Main_Page"},
                        {"action": "read_page"},
                    ],
                }
            ]
            print("🛟 Using fallback plan.")
            _print_actions("Fallback plan actions", self.subgoals)

            plans_repo.put(
                intent_key="wikipedia_research",
                goal_text=user_goal,
                plan_json={"params": {}, "subgoals": self.subgoals, "actions": _flatten_actions(self.subgoals)},
                site_domain="wikipedia.org",
                success_rate=0.5,
                version="v1",
            )
            print("🗂 Fallback plan stored → plans table (site=wikipedia.org)")
            return True

    async def _exec_action(self, action: Dict[str, Any]) -> Optional[str]:
        """Execute a single action with strict Wikipedia guard. Returns 'read' text or None."""
        atype = action.get("action") or action.get("type")

        if atype == "goto":
            url = action.get("url", "")
            if url and not url.startswith("http"):
                # treat as title
                candidate = f"{WIKI_HOME}/wiki/{url.replace(' ', '_')}"
                print(f"   → normalizing title to URL: {candidate}")
                url = candidate
            if STRICT_WIKI_ONLY and not _ensure_wikipedia_url(url):
                print(f"   ⛔ BLOCKED non-wikipedia goto: {url}")
                return None
            url = _normalize_wiki_url(url)
            print(f"   → GOTO {url}")
            ok = await _goto_with_retry(self, url)
            if not ok:
                # Fallback: try a Wikipedia search using the last path segment as a query
                topic = (url.rsplit("/", 1)[-1] or "").replace("_", " ")
                if topic:
                    print("   → GOTO failed; attempting Wikipedia search fallback…")
                    await _wiki_search(self, topic)
            return None

        if atype == "type":
            sel = action.get("selector", "#searchInput")
            text = action.get("text", "")
            print(f"   → TYPE {text!r} into {sel}")
            try:
                await self.page.fill(sel, text, timeout=5000)
            except Exception:
                # fallback JS fill
                await self.page.evaluate(
                    """(s,t)=>{const el=document.querySelector(s); if(el){el.value=''; el.value=t; el.dispatchEvent(new Event('input',{bubbles:true}));}}""",
                    sel, text
                )
            return None

        if atype == "press_enter":
            print("   → PRESS ENTER")
            try:
                await self.page.keyboard.press("Enter")
                await self.page.wait_for_load_state("networkidle")
            except Exception:
                pass
            return None

        if atype == "click":
            sel = action.get("selector", "a")
            print(f"   → CLICK {sel}")
            try:
                # Prefer actual article links (avoid namespaces with ':')
                link = await self.page.query_selector("a[href^='/wiki/']:not([href*=':'])")
                if link:
                    await link.click()
                    await self.page.wait_for_load_state("networkidle")
                    return None
                # Fallback to provided selector
                await self.page.click(sel, timeout=4000)
                await self.page.wait_for_load_state("networkidle")
            except Exception as e:
                print(f"     click failed: {e}")
            return None

        if atype == "scroll":
            direction = action.get("direction", "down").lower()
            print(f"   → SCROLL {direction}")
            if direction == "down":
                await self.page.evaluate("window.scrollBy(0, 800)")
            else:
                await self.page.evaluate("window.scrollBy(0, -800)")
            return None

        if atype == "read_page":
            # strict read
            title = await self.page.title()
            url = self.page.url
            if STRICT_WIKI_ONLY and not _ensure_wikipedia_url(url):
                print(f"   ⛔ BLOCKED non-wikipedia read_page: {url}")
                return None
            print(f"   → READ_PAGE ({title[:80]} …)")
            text = await _extract_page_text(self.page)
            return json.dumps({"title": title, "url": url, "text": text})  # return packed

        print(f"   → SKIP unknown action: {atype}")
        return None

    async def provide_final_answer(self, user_goal: str):
        print(f"\n🧩 SYNTHESIZING (no outside facts; only from extracted notes)…")
        print("=" * 60)

        # Only use collected notes that were found=true
        usable = [info for info in self.collected_information if info.get("found")]
        if not usable:
            final_answer = "Information not found on the visited Wikipedia pages."
        else:
            # Minimal synthesis without adding new facts
            lines = []
            for info in usable:
                lines.append(f"- {info['answer']}  (source: {info['page_title']})")
            final_answer = " ".join(lines)

        print("\n🎯 FINAL ANSWER")
        print("=" * 80)
        print(final_answer)
        print("=" * 80)

        cq = canonicalize(user_goal)
        answers_repo.put(
            canonical_q=cq,
            question_text=user_goal,
            answer_text=final_answer,
            confidence=0.8 if usable else 0.2,
            evidence={"note": "strict wiki-only; closed-book extraction"},
            sources=[{"title": info["page_title"], "url": info["page_url"]} for info in self.collected_information],
        )
        print(f"💾 Stored answer → answers table (canonical={cq})")

    async def run(self, goal: str, no_cache: bool = False, plan_preview: bool = False, force_plan: bool = False):
        print(f"\n🚀 STARTING AUTOMATION: {goal}")
        print("=" * 60)
        print(f"provider={llm_client.provider.upper()}  db={DB_PATH}")

        # cache
        if not no_cache and not force_plan:
            hit = answer_cache.get(goal)
            if hit:
                sim = round(hit.get("similarity", 1.0), 4)
                print("\n[CACHE HIT] (answers)")
                print("  canonical_q:", hit.get("canonical_q"))
                print("  similarity :", sim)
                print("  answer     :", hit["answer_text"])
                print("  sources    :", hit.get("sources"))
                if plan_preview:
                    preview_stored_plan(goal)
                print(f"\n[RUN TOKENS] prompt={RUN_TOKENS['prompt']} completion={RUN_TOKENS['completion']} total={RUN_TOKENS['total']}")
                RUN_TOKENS.update({"prompt": 0, "completion": 0, "total": 0})
                return

        # go to Wikipedia home
        await self.page.goto(f"{WIKI_HOME}/wiki/Main_Page", wait_until="domcontentloaded")
        await self.page.wait_for_load_state("networkidle")

        if not await self.create_subgoals_and_actions(goal):
            print("❌ Planning failed.")
            return

        # Execute plan strictly; collect notes via closed-book extraction
        for sg in self.subgoals:
            print(f"\n▶ Subgoal {sg.get('id','?')}: {sg.get('title','').strip()}")
            actions = sg.get("actions") or []
            if not actions:
                print("   (no actions)")
                continue

            last_read_pack = None
            for i, a in enumerate(actions, 1):
                atype = a.get("action") or a.get("type")
                print(f"   [{i}] EXECUTE {atype} …")
                result = await self._exec_action(a)
                if atype == "read_page" and result:
                    last_read_pack = json.loads(result)

            # If we read a page in this subgoal, attempt closed-book extraction from it
            if last_read_pack:
                page_title = last_read_pack.get("title", "")
                page_url = last_read_pack.get("url", "")
                page_text = last_read_pack.get("text", "")

                # closed-book extraction on the current page
                extracted = await _closed_book_extract(goal, page_title, page_url, page_text)
                self.collected_information.append({
                    "subgoal_id": sg.get("id", len(self.collected_information) + 1),
                    "subgoal_title": sg.get("title", "step"),
                    "found": bool(extracted.get("found")),
                    "answer": extracted.get("answer", ""),
                    "evidence": extracted.get("evidence", ""),
                    "page_title": page_title,
                    "page_url": page_url,
                })
                status = "FOUND" if extracted.get("found") else "NOT FOUND"
                print(f"   → EXTRACT [{status}] {extracted.get('answer','')!r}  evidence: {extracted.get('evidence','')!r}")

        await self.provide_final_answer(goal)

        print(f"\n[RUN TOKENS] prompt={RUN_TOKENS['prompt']} completion={RUN_TOKENS['completion']} total={RUN_TOKENS['total']}")
        RUN_TOKENS.update({"prompt": 0, "completion": 0, "total": 0})


# ---------------- CLI ----------------
async def main():
    ap = argparse.ArgumentParser(description="CLI Browser Agent (OpenAI LLM + Lightpanda browser + STRICT WIKIPEDIA ONLY)")
    ap.add_argument("goal", nargs="*", help='Goal, e.g. "When was Marie Curie born?"')
    ap.add_argument("--no-cache", action="store_true", help="Bypass answer cache")
    ap.add_argument("--force-plan", action="store_true", help="Skip answer cache and force planning/execution")
    ap.add_argument("--plan-preview", action="store_true", help="Print stored plan actions even on cache hit")
    ap.add_argument("--purge", action="store_true", help="Purge cached answer for the given goal")
    ap.add_argument("--show-counts", action="store_true", help="Print DB table counts before/after")
    ap.add_argument("--headless", action="store_true", help="Use local headless Chromium if Lightpanda fails")
    args = ap.parse_args()

    goal = " ".join(args.goal).strip() if args.goal else "When was Marie Curie born?"

    if args.purge:
        purge_answer_for(goal)

    if args.show_counts:
        show_counts("before")

    token = os.getenv("LIGHTPANDA_TOKEN")
    async with async_playwright() as p:
        try:
            if token:
                print("Connecting to Lightpanda browser…")
                browser = await p.chromium.connect_over_cdp(f"wss://cloud.lightpanda.io/ws?token={token}")
                print("✅ Connected to Lightpanda.")
            else:
                raise RuntimeError("No LIGHTPANDA_TOKEN in env")
        except Exception as e:
            print("❌ Lightpanda connection failed:", e)
            print("Launching local Chromium instead…")
            browser = await p.chromium.launch(headless=args.headless)

        page = await browser.new_page()
        agent = LLMBrowserAgent(page=page, browser=browser)
        await agent.run(goal, no_cache=args.no_cache, plan_preview=args.plan_preview, force_plan=args.force_plan)
        await browser.close()

    if args.show_counts:
        show_counts("after")


if __name__ == "__main__":
    asyncio.run(main())
