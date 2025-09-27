"""
agent.py
--------
LLM-driven planner agent (Wikipedia-only) with:
- PlanStore (persisted plans) for cache wins
- TokenCost metrics (tokens & $)
- LLM policy: always / verify / fallback
- Optional DOM-first extraction for 'wikipedia_birth_date'
- LLM memoization (prompt cache) to avoid re-paying for identical prompts
- Answer cache for stable facts (birthdates)
"""

import asyncio
import json
import os
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from plan_store import PlanStore
from answer_cache import AnswerCache
from utils import TokenCost, classify_intent, topic_key_from_goal, extract_person_name, LLMCache
from dom_extractors import extract_birthdate_dom, format_birthdate

load_dotenv()

# ---- Budgets / caps to keep costs predictable ----
INTERACTIVE_LIMIT = 10          # show at most 10 elements to the LLM
PAGE_BUDGET_BIRTH = 1200        # chars for birth-date extraction/verify
PAGE_BUDGET_RESEARCH = 1800     # chars for general research extraction
MAX_TOK_PLANNING = 300
MAX_TOK_ACTION   = 150
MAX_TOK_EXTRACT  = 120
MAX_TOK_SYNTH    = 400

# ---- Policy ----
class LLMPolicy(str, Enum):
    ALWAYS = "always"      # always use LLM to extract
    VERIFY = "verify"      # DOM first, then small LLM to verify/format
    FALLBACK = "fallback"  # DOM first, LLM only if DOM fails


class LLMBrowserAgent:
    def __init__(self, llm_policy: str = "fallback"):
        self.model = os.getenv("OPENAI_MODEL") or "gpt-4"
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.page = None

        self.action_history: List[str] = []
        self.subgoals: List[Dict[str, Any]] = []
        self.current_subgoal_index = 0
        self.collected_information: List[Dict[str, Any]] = []

        # Caches/metrics
        self.plan_store = PlanStore()
        self.answer_cache = AnswerCache()
        self.llm_cache = LLMCache()
        self.tokens = TokenCost(self.model)
        self.policy = LLMPolicy(llm_policy)

        print(f"LLM Browser Agent ready (model={self.model}, policy={self.policy.value})")

    # ---------- tiny wrapper with memoization + token accounting ----------
    async def _chat(self, prompt: str, *, max_tokens: int, temperature: float = 0.1) -> str:
        cached = self.llm_cache.get(self.model, prompt)
        if cached:
            # cached completions count as zero tokens/cost
            return cached["text"]
        resp = await self.openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content
        self.tokens.add_usage(getattr(resp, "usage", None))
        usage = None
        if getattr(resp, "usage", None):
            usage = dict(prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens)
        self.llm_cache.put(self.model, prompt, text, usage)
        return text

    # ---------- Browser helpers ----------

    async def check_browser_health(self) -> bool:
        try:
            await self.page.title()
            return True
        except Exception as e:
            print(f"Browser health check failed: {e}")
            return False

    async def get_page_info(self) -> Dict[str, Any]:
        title = await self.page.title()
        url = self.page.url
        try:
            await self.page.wait_for_load_state('networkidle', timeout=15000)
            await asyncio.sleep(2)
        except Exception:
            await asyncio.sleep(2)

        html = await self.page.content()
        interactive_elements = await self.page.evaluate("""
            () => {
                const elements = document.querySelectorAll('a, button, input, select, [onclick], [role="button"]');
                return Array.from(elements)
                    .filter(el => {
                        const rect = el.getBoundingClientRect();
                        return rect.width > 0 && rect.height > 0 &&
                               getComputedStyle(el).visibility !== 'hidden' &&
                               getComputedStyle(el).display !== 'none';
                    })
                    .map(el => ({
                        tag: el.tagName.toLowerCase(),
                        text: el.textContent?.trim().slice(0, 100) || '',
                        id: el.id || '',
                        className: el.className || '',
                        href: el.href || '',
                        selector: el.id ? `#${el.id}` :
                                 el.className ? `.${el.className.split(' ')[0]}` :
                                 el.tagName.toLowerCase()
                    }))
                    .slice(0, 20);
            }
        """)
        interactive_elements = interactive_elements[:INTERACTIVE_LIMIT]
        return {
            "title": title,
            "url": url,
            "html": html[:8000] + "..." if len(html) > 8000 else html,
            "interactive_elements": interactive_elements
        }

    # ---------- Planning (with PlanStore cache) ----------

    async def create_subgoals_and_actions(self, user_goal: str) -> bool:
        print("\n📋 PLANNING PHASE: Creating subgoals and action plan...")
        print("=" * 60)

        intent = classify_intent(user_goal)
        topic_key = topic_key_from_goal(user_goal)

        cached = self.plan_store.get(intent, topic_key)
        if cached:
            self.subgoals = cached.get("subgoals", [])
            print(f"📝 Using cached plan for [{intent}:{topic_key}] — {len(self.subgoals)} subgoals.")
            return True

        planning_prompt = f"""
        You are a browser automation planner for Wikipedia research. Break down the following user goal into specific subgoals and concrete actions.

        User Goal: {user_goal}

        CONTEXT: You will start on Wikipedia (https://en.wikipedia.org) and must stay on Wikipedia throughout the entire process.

        Available actions:
        - click(selector) - Click an element
        - type(selector, text) - Type text into an input field
        - press_enter() - Press Enter key
        - read_page() - Read and analyze page content
        - goto(url) - Navigate to a URL (ONLY use Wikipedia URLs like https://en.wikipedia.org/wiki/...)
        - scroll(direction) - Scroll up or down
        - wait(seconds) - Wait for specified time
        - done() - Task completed

        IMPORTANT RULES:
        - NEVER navigate to non-Wikipedia sites
        - Use Wikipedia search (#searchInput) and result links
        - After typing in a search box, use press_enter()

        Respond ONLY with a JSON object in this format:
        {{
            "subgoals": [
                {{
                    "id": 1,
                    "title": "Search on Wikipedia",
                    "description": "Use the Wikipedia search to find the topic",
                    "actions": [
                        {{"action": "type", "parameters": {{"selector": "#searchInput", "text": "..."}}}},
                        {{"action": "press_enter", "parameters": {{}}}}
                    ]
                }},
                {{
                    "id": 2,
                    "title": "Open the best article",
                    "description": "Click the most relevant result",
                    "actions": [
                        {{"action": "click", "parameters": {{"selector": "li.mw-search-result a"}}}}
                    ]
                }},
                {{
                    "id": 3,
                    "title": "Extract information",
                    "description": "Read the article and extract information relevant to the question",
                    "actions": [
                        {{"action": "read_page", "parameters": {{}}}}
                    ]
                }}
            ]
        }}
        Make the plan specific to answering: {user_goal}
        """

        text = await self._chat(planning_prompt, max_tokens=MAX_TOK_PLANNING, temperature=0.1)
        try:
            plan = json.loads(text)
            self.subgoals = plan.get("subgoals", [])
            print(f"📝 Created {len(self.subgoals)} subgoals.")
            self.plan_store.put(intent, topic_key, {"goal": user_goal, "subgoals": self.subgoals})
            return True
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse planning response: {e}")
            return False

    # ---------- LLM action selection ----------

    async def ask_llm_for_action(self, page_info: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        recent = self.action_history[-5:] if self.action_history else []
        history_text = "\n".join([f"- {a}" for a in recent]) if recent else "None"

        current_subgoal = None
        planned_actions = []
        if self.current_subgoal_index < len(self.subgoals):
            current_subgoal = self.subgoals[self.current_subgoal_index]
            planned_actions = current_subgoal.get("actions", [])

        repeated_action_warning = ""
        if len(self.action_history) >= 2 and "type(#searchInput)" in self.action_history[-1] \
           and "type(#searchInput)" in self.action_history[-2]:
            repeated_action_warning = "⚠️ You just typed. Use press_enter() next."

        prompt = f"""
        You are following a structured plan to answer the user's goal within Wikipedia.

        Current page:
        - Title: {page_info['title']}
        - URL: {page_info['url']}
        - Interactive elements: {json.dumps(page_info['interactive_elements'], indent=2)}

        Recent actions:
        {history_text}
        {repeated_action_warning}

        CURRENT SUBGOAL:
        {json.dumps(current_subgoal, indent=2) if current_subgoal else "None (finish soon)"}

        User goal: {user_goal}

        Available actions:
        1. click(selector)
        2. type(selector, text)
        3. press_enter()
        4. read_page()
        5. goto(url)
        6. scroll(direction)
        7. wait(seconds)
        8. done()

        RULES:
        - Only use selectors from interactive_elements.
        - After typing in a search box, press_enter() to submit.
        - Do not repeat the same action if it was just performed.

        Respond ONLY with JSON:
        {{
          "action": "action_name",
          "parameters": {{ "selector": "...", "text": "..." }},
          "reasoning": "why this is the next step"
        }}
        """

        text = await self._chat(prompt, max_tokens=MAX_TOK_ACTION, temperature=0.1)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"action": "wait", "parameters": {"seconds": 1}, "reasoning": "parse error"}

    # ---------- Subgoal progression ----------

    def check_subgoal_completion(self, action: str):
        if not self.subgoals or self.current_subgoal_index >= len(self.subgoals):
            return
        current = self.subgoals[self.current_subgoal_index]
        if action == "read_page":
            print(f"\n✅ SUBGOAL COMPLETED: {current['title']}")
            self.current_subgoal_index += 1
            if self.current_subgoal_index < len(self.subgoals):
                nxt = self.subgoals[self.current_subgoal_index]
                print(f"➡️ NEXT SUBGOAL: {nxt['title']}")
            else:
                print("🏁 ALL SUBGOALS COMPLETED - PREPARING FINAL ANSWER!")

    # ---------- LLM helpers for extraction/verification ----------

    async def llm_extract_birthdate(self, page_text: str, person_name: str):
        prompt = (
            f"Extract the birth date of {person_name}.\n\n"
            f"Page:\n{page_text}\n\n"
            f'Respond exactly as:\n"{person_name} was born on [Month Day, Year]"'
        )
        return await self._chat(prompt, max_tokens=MAX_TOK_EXTRACT, temperature=0.0)

    async def llm_verify_birthdate(self, page_text: str, person_name: str, dom_raw_date: str):
        formatted = format_birthdate(dom_raw_date) or dom_raw_date
        prompt = (
            f"We extracted this birth date for {person_name}: '{formatted}'.\n"
            f"Check against the page snippet below. If correct, respond exactly:\n"
            f"\"{person_name} was born on {formatted}\"\n"
            f"If not, respond exactly with the corrected date in that format.\n\n"
            f"Page snippet:\n{page_text}"
        )
        return await self._chat(prompt, max_tokens=MAX_TOK_EXTRACT, temperature=0.0)

    # ---------- Execute one action ----------

    async def execute_action(self, action_data: Dict[str, Any], user_goal: str) -> bool:
        action = action_data.get("action")
        params = action_data.get("parameters", {})
        action_summary = f"{action}({params.get('selector', '')}) - {action_data.get('reasoning', '')[:50]}..."
        self.action_history.append(action_summary)

        print(f"Executing: {action} with params: {params}")
        print(f"Reasoning: {action_data.get('reasoning', 'No reasoning provided')}")

        # Anti-typing loop safeguard
        if action == "type" and len([a for a in self.action_history[-3:] if a.startswith("type(")]) >= 2:
            print("🚨 BLOCKED: Multiple typing actions detected. Forcing press_enter().")
            action = "press_enter"
            params = {}

        self.check_subgoal_completion(action)

        if action == "click":
            sel = params.get("selector")
            try:
                await self.page.wait_for_selector(sel, timeout=5000, state='visible')
                await self.page.click(sel, timeout=5000)
            except Exception as e:
                print(f"Failed to click {sel}: {e}")
                return False

        elif action == "type":
            sel = params.get("selector")
            text = params.get("text", "")
            try:
                await self.page.wait_for_selector(sel, timeout=5000)
                escaped_text = text.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
                ok = await self.page.evaluate(f"""
                    () => {{
                        const el = document.querySelector('{sel}');
                        if (!el) return false;
                        el.value = '';
                        el.value = '{escaped_text}';
                        el.dispatchEvent(new Event('input', {{bubbles: true}}));
                        el.dispatchEvent(new Event('change', {{bubbles: true}}));
                        return true;
                    }}
                """)
                if not ok:
                    print(f"JavaScript typing failed for {sel}")
                    return False
            except Exception as e:
                print(f"Failed to type in {sel}: {e}")
                return False

        elif action == "press_enter":
            try:
                submitted = await self.page.evaluate("""
                    () => {
                        const input = document.querySelector('#searchInput, input[type="search"]');
                        if (!input) return false;
                        const form = input.closest('form');
                        if (form) { form.submit(); return true; }
                        return false;
                    }
                """)
                if not submitted:
                    await self.page.keyboard.press('Enter')
                try:
                    await self.page.wait_for_load_state('networkidle', timeout=10000)
                except Exception:
                    await asyncio.sleep(2)
            except Exception as e:
                if "Execution context was destroyed" in str(e):
                    try:
                        await self.page.wait_for_load_state('networkidle', timeout=10000)
                    except Exception:
                        await asyncio.sleep(2)
                else:
                    print(f"Failed to press Enter: {e}")
                    return False

        elif action == "goto":
            url = params.get("url")
            if url and "wikipedia.org" not in (url.lower()):
                print(f"❌ BLOCKED: Non-Wikipedia URL requested: {url}")
                return False
            try:
                await self.page.goto(url, wait_until='networkidle', timeout=30000)
            except Exception as e:
                print(f"Failed to navigate to {url}: {e}")
                return False

        elif action == "scroll":
            direction = params.get("direction", "down")
            try:
                await self.page.keyboard.press("PageDown" if direction == "down" else "PageUp")
            except Exception as e:
                print(f"Scroll failed: {e}")

        elif action == "wait":
            await asyncio.sleep(params.get("seconds", 1))

        elif action == "read_page":
            # Intent & page text with budget applied
            intent = classify_intent(user_goal)
            page_text = await self.page.evaluate("""
                () => {
                    const scripts = document.querySelectorAll('script, style, nav, header, footer');
                    scripts.forEach(el => el.remove());
                    const content = document.querySelector('#mw-content-text, main, .content, body');
                    return content ? content.innerText : document.body.innerText;
                }
            """)
            budget = PAGE_BUDGET_BIRTH if intent == "wikipedia_birth_date" else PAGE_BUDGET_RESEARCH
            page_text = page_text[:budget]

            answer_text: Optional[str] = None

            if intent == "wikipedia_birth_date":
                # Answer cache short-circuit (stable fact)
                cached_ans = self.answer_cache.get(user_goal)
                if cached_ans and self.policy != LLMPolicy.ALWAYS:
                    print("⚡ Answer cache hit (birth date).")
                    answer_text = cached_ans["answer"]
                else:
                    person = extract_person_name(user_goal)
                    if self.policy == LLMPolicy.ALWAYS:
                        answer_text = await self.llm_extract_birthdate(page_text, person)
                    elif self.policy == LLMPolicy.VERIFY:
                        dom_raw = await extract_birthdate_dom(self.page)
                        if dom_raw:
                            answer_text = await self.llm_verify_birthdate(page_text, person, dom_raw)
                        else:
                            answer_text = await self.llm_extract_birthdate(page_text, person)
                    else:  # FALLBACK
                        dom_raw = await extract_birthdate_dom(self.page)
                        if dom_raw:
                            answer_text = f"{person} was born on {format_birthdate(dom_raw)}"
                        else:
                            answer_text = await self.llm_extract_birthdate(page_text, person)

                    # Save to answer cache if it looks good
                    if answer_text and any(s in answer_text.lower() for s in [" was born on "]):
                        self.answer_cache.put(user_goal, answer_text, {
                            "policy": self.policy.value,
                            "url": self.page.url,
                            "title": await self.page.title()
                        })

            else:
                # General research extraction with LLM
                current_subgoal = None
                if self.current_subgoal_index < len(self.subgoals):
                    current_subgoal = self.subgoals[self.current_subgoal_index]
                extraction_prompt = f"""
                Extract information relevant to this subgoal: {current_subgoal['title'] if current_subgoal else 'Information extraction'}
                Description: {current_subgoal['description'] if current_subgoal else 'Extract relevant information'}
                Overall goal: {user_goal}

                Page:
                {page_text}

                Extract only what is relevant. If not found, say "Information not found on this page."
                """
                answer_text = await self._chat(extraction_prompt, max_tokens=MAX_TOK_EXTRACT, temperature=0.1)

            # record snippet
            info = {
                "subgoal_id": self.subgoals[self.current_subgoal_index]['id'] if self.current_subgoal_index < len(self.subgoals) else len(self.collected_information) + 1,
                "subgoal_title": self.subgoals[self.current_subgoal_index]['title'] if self.current_subgoal_index < len(self.subgoals) else "Information extraction",
                "information": answer_text,
                "page_title": await self.page.title(),
                "page_url": self.page.url
            }
            self.collected_information.append(info)

            # If this was the last subgoal, synthesize a final answer
            if self.current_subgoal_index >= len(self.subgoals) - 1:
                await self.provide_final_answer(user_goal)
                return True

            return False

        elif action == "done":
            return True

        return False

    # ---------- Final synthesis ----------

    async def provide_final_answer(self, user_goal: str):
        print(f"\n🧩 SYNTHESIZING INFORMATION FROM ALL SUBGOALS...")
        print("=" * 60)

        parts = []
        for info in self.collected_information:
            parts.append(
                f"--- {info['subgoal_title']} ---\n"
                f"Source: {info['page_title']} ({info['page_url']})\n"
                f"Information: {info['information']}\n"
            )
        all_info = "\n\n".join(parts)

        synthesis_prompt = f"""
        You researched: {user_goal}

        Here is the collected information:
        {all_info}

        Provide a concise, well-structured final answer that:
        1) directly answers the question,
        2) synthesizes across steps,
        3) cites page titles inline where relevant,
        4) clearly states missing info if anything is missing.
        """

        text = await self._chat(synthesis_prompt, max_tokens=MAX_TOK_SYNTH, temperature=0.1)
        final_answer = text

        print("\n🎯 COMPREHENSIVE FINAL ANSWER")
        print("=" * 80)
        print(final_answer)
        print("=" * 80)

        # quick summary
        print("\n📊 RESEARCH SUMMARY")
        print(f"  • Steps: {len(self.collected_information)}")
        uniq = len({i['page_url'] for i in self.collected_information})
        print(f"  • Unique pages: {uniq}")

    # ---------- Orchestration ----------

    async def run_automation(self, user_goal: str, max_steps: int = 12):
        print("\n🚀 STARTING AUTOMATION")
        print(f"Goal: {user_goal}")
        print("=" * 60)

        if not await self.create_subgoals_and_actions(user_goal):
            print("❌ Planning failed. Ending.")
            return

        print("\n🎬 EXECUTION PHASE")
        print("=" * 60)

        for step in range(max_steps):
            if not await self.check_browser_health():
                print("\n❌ BROWSER CONNECTION LOST — ending.")
                break

            try:
                page_info = await self.get_page_info()
                action_data = await self.ask_llm_for_action(page_info, user_goal)
                print(f"\n🔄 STEP {step + 1}: {action_data.get('action','(unknown)')}")
                is_done = await self.execute_action(action_data, user_goal)
            except Exception as e:
                print(f"\n❌ ERROR IN STEP {step + 1}: {e}")
                if "has been closed" in str(e) or "Target page" in str(e):
                    print("Browser connection lost during step. Ending automation.")
                    break
                continue

            if is_done:
                print("\n✅ AUTOMATION COMPLETED SUCCESSFULLY!")
                break

            await asyncio.sleep(1)

        # Token/cost summary
        print("\n💰 TOKEN/COST SUMMARY")
        print(f"  • Calls: {self.tokens.calls}")
        print(f"  • Prompt tokens: {self.tokens.prompt_tokens}")
        print(f"  • Completion tokens: {self.tokens.completion_tokens}")
        print(f"  • Total tokens: {self.tokens.total_tokens}")
        usd = self.tokens.usd
        print(f"  • Estimated cost: ${usd:.4f}" if usd is not None else "  • Estimated cost: N/A (pricing unknown)")
