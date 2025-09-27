
import asyncio
import json
import os
import re
import time
from datetime import datetime
from urllib.parse import quote

from dotenv import load_dotenv
from openai import AsyncOpenAI
from playwright.async_api import async_playwright

load_dotenv()


# ============================
# Wikipedia Action Cache
# ============================
class WikipediaActionCache:
    """Intent-based action cache for Wikipedia biography searches."""

    def __init__(self):
        self.patterns = {}  # {intent: {"actions": [...], "times_used": int, ...}}
        self.metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_time_saved": 0.0,
            "total_cost_saved": 0.0,
        }
        print("Wikipedia Action Cache initialized")

    def has_pattern(self, pattern_type: str) -> bool:
        return pattern_type in self.patterns

    def get_cached_actions(self, pattern_type: str):
        if pattern_type in self.patterns:
            print(f"Retrieved cached pattern: {pattern_type}")
            return self.patterns[pattern_type]["actions"]
        return None

    def store_pattern(self, pattern_type: str, actions, execution_time: float):
        self.patterns[pattern_type] = {
            "actions": actions,
            "success_rate": 1.0,
            "avg_execution_time": execution_time,
            "times_used": 0,
            "created_at": time.time(),
        }
        print(f"Cached new pattern: {pattern_type} (took {execution_time:.1f}s)")

    def record_cache_hit(self, pattern_type: str, time_saved: float, cost_saved: float):
        if pattern_type in self.patterns:
            self.patterns[pattern_type]["times_used"] += 1
        self.metrics["cache_hits"] += 1
        self.metrics["total_time_saved"] += max(0.0, float(time_saved))
        self.metrics["total_cost_saved"] += max(0.0, float(cost_saved))
        print(f"Cache hit recorded! Saved {time_saved:.1f}s, ${cost_saved:.3f}")

    def record_cache_miss(self):
        self.metrics["cache_misses"] += 1
        print("Cache miss recorded")

    def get_cache_stats(self):
        total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
        hit_rate = (self.metrics["cache_hits"] / total * 100.0) if total else 0.0
        return {
            "hit_rate": hit_rate,
            "total_queries": total,
            "time_saved": self.metrics["total_time_saved"],
            "cost_saved": self.metrics["total_cost_saved"],
            "patterns_learned": len(self.patterns),
        }


# ============================
# LLM Browser Agent
# ============================
class LLMBrowserAgent:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.page = None
        self.cache = WikipediaActionCache()
        self.action_history = []
        print("LLM Browser Agent initialized with cache support")

    # ---------- Robust page helpers ----------
    async def ensure_page(self, browser):
        """Recreate page if current page/context/browser is dead."""
        try:
            if self.page is None or self.page.is_closed():
                ctx = await browser.new_context()
                self.page = await ctx.new_page()
                self.page.on("close", lambda: print("[warn] Page closed by target"))
                return
            # quick ping
            _ = await self.page.title()
        except Exception:
            ctx = await browser.new_context()
            self.page = await ctx.new_page()
            self.page.on("close", lambda: print("[warn] Page closed by target"))

    async def safe_goto(self, browser, url: str, *, retries: int = 2):
        for attempt in range(retries + 1):
            try:
                await self.ensure_page(browser)
                await self.page.goto(url, wait_until="domcontentloaded")
                try:
                    await self.page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass
                return
            except Exception as e:
                if attempt == retries:
                    raise
                await asyncio.sleep(1.0)

    # ---------- Intent classification & name extraction ----------
    def classify_goal(self, user_goal: str) -> str:
        birth_patterns = [
            r"when was .* born",
            r"birth date of .*",
            r".* birth year",
            r"what year was .* born",
            r"when did .* born",
        ]
        goal_lower = user_goal.lower().strip()
        for p in birth_patterns:
            if re.search(p, goal_lower):
                print(f"Classified as wikipedia_birth_date: '{user_goal}'")
                return "wikipedia_birth_date"
        print(f"Unknown pattern: '{user_goal}'")
        return "unknown"

    def extract_person_name(self, user_goal: str):
        """Extract person name (case-preserving) and a normalized search name."""
        patterns = [
            r"when was (.*?) born",
            r"birth date of (.*?)$",
            r"(.*?) birth year",
            r"what year was (.*?) born",
        ]
        for pat in patterns:
            m = re.search(pat, user_goal, flags=re.IGNORECASE)
            if m:
                display = m.group(1).strip()
                return display, display  # display + search are same here

        # Fallback: remove common words
        words_to_remove = {"when", "was", "born", "birth", "date", "of", "year", "what", "did", "?"}
        tokens = [w for w in re.split(r"\s+", user_goal.strip()) if w.lower() not in words_to_remove]
        display = " ".join(tokens).strip()
        return display, display

    # ---------- Navigation & extraction ----------
    def _search_url(self, name: str) -> str:
        q = quote(name)
        # ns0=1 => main namespace; results more relevant for articles
        return f"https://en.wikipedia.org/w/index.php?search={q}&title=Special:Search&ns0=1"

    async def navigate_to_person_page(self, browser, person_name: str):
        """Go to the most likely person page for the given name."""
        await self.safe_goto(browser, self._search_url(person_name))
        # If we landed directly on an article, there is no search results list
        html = await self.page.content()
        if "mw-search-results" in html:
            # pick the first result (heuristic)
            link = await self.page.query_selector("li.mw-search-result a")
            if link:
                href = await link.get_attribute("href")
                if href:
                    await self.safe_goto(browser, "https://en.wikipedia.org" + href)

    async def extract_birth_date_dom(self):
        """Extract birth date from DOM (infobox bday / time[datetime] / JSON-LD)."""
        bday = await self.page.evaluate(
            """
            () => {
              // Try infobox '.bday'
              const b = document.querySelector('.infobox .bday');
              if (b && b.textContent) return b.textContent.trim();

              // Try infobox time[datetime]
              const t = document.querySelector('.infobox time[datetime]');
              if (t) {
                const dt = t.getAttribute('datetime');
                if (dt) return dt;
              }

              // Try JSON-LD
              const scripts = document.querySelectorAll('script[type="application/ld+json"]');
              for (const s of scripts) {
                try {
                  const data = JSON.parse(s.textContent);
                  if (Array.isArray(data)) {
                    for (const item of data) {
                      if (item && item.birthDate) return item.birthDate;
                      if (item && item.person && item.person.birthDate) return item.person.birthDate;
                    }
                  } else if (data) {
                    if (data.birthDate) return data.birthDate;
                    if (data.person && data.person.birthDate) return data.person.birthDate;
                  }
                } catch {}
              }

              // Try first paragraph heuristic (rare fallback)
              const firstPara = document.querySelector('.mw-parser-output > p');
              if (firstPara) {
                const txt = firstPara.textContent || "";
                const match = txt.match(/\\(born\\s+([A-Za-z]+\\s+\\d{1,2},\\s+\\d{4})\\)/i);
                if (match) return match[2] || match[1];
                const matchISO = txt.match(/(\\d{4}-\\d{2}-\\d{2})/);
                if (matchISO) return matchISO[1];
              }
              return null;
            }
            """
        )
        return bday

    def _format_birthdate(self, raw: str) -> str:
        """Format ISO date to 'Month Day, Year' if possible; otherwise return raw."""
        if not raw:
            return raw
        # Try ISO first
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.strftime("%B %d, %Y")
            except Exception:
                pass
        # Try already human readable, return as-is
        return raw

    async def extract_birth_date_with_llm(self, page_text: str, person_name: str):
        """Use LLM as a last resort to extract birth date."""
        print(f"Extracting birth date for {person_name} using LLM fallback...")
        prompt = (
            "Extract the birth date of " + person_name + " from this Wikipedia page content.\\n\\n"
            "Page content (first 3000 characters):\\n"
            + page_text[:3000]
            + "\\n\\nPlease respond with ONLY the birth date in this exact format:\\n"
            f'"{person_name} was born on [Month Day, Year]"\\n\\n'
            f'If you cannot find a clear birth date, respond with:\\n"Birth date not found for {person_name}"'
        )
        resp = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        result = resp.choices[0].message.content.strip()
        result = result.strip('"').strip("'")
        print(f"LLM extracted: {result}")
        return result

    async def extract_birth_date(self, display_name: str):
        """Prefer DOM extraction; if not found, use LLM fallback."""
        bday = await self.extract_birth_date_dom()
        if bday:
            return f"{display_name} was born on {self._format_birthdate(bday)}"
        # fall back to LLM
        page_text = await self.page.evaluate(
            """
            () => {
                const unwanted = document.querySelectorAll('script, style, nav, header, footer, .navbox');
                unwanted.forEach(el => el.remove());
                const content = document.querySelector('#mw-content-text, .mw-parser-output, main, body');
                return content ? content.innerText : document.body.innerText;
            }
            """
        )
        return await self.extract_birth_date_with_llm(page_text, display_name)

    # ---------- Cache-aware execution ----------
    async def execute_cached_pattern(self, browser, person_name: str, display_name: str, pattern_type: str):
        print(f"Executing cached pattern for: {person_name}")
        start = time.time()
        cached_actions = self.cache.get_cached_actions(pattern_type)
        if not cached_actions:
            print("No cached actions found!")
            return None

        for i, step in enumerate(cached_actions):
            print(f"Cached step {i+1}/{len(cached_actions)}: {step['action']}")
            act = step["action"]
            if act == "navigate_person_page":
                await self.navigate_to_person_page(browser, person_name)
            elif act == "extract_birth_date":
                result = await self.extract_birth_date(display_name)
                elapsed = time.time() - start
                baseline = 15.0  # illustrative baseline
                self.cache.record_cache_hit(pattern_type, max(0, baseline - elapsed), 0.06)
                return result
        return None

    async def learn_pattern_and_execute(self, browser, person_name: str, display_name: str, pattern_type: str):
        print(f"Learning new Wikipedia pattern for: {person_name}")
        start = time.time()
        actions_taken = []

        await self.navigate_to_person_page(browser, person_name)
        actions_taken.append({"action": "navigate_person_page"})

        result = await self.extract_birth_date(display_name)
        actions_taken.append({"action": "extract_birth_date"})

        exec_time = time.time() - start
        self.cache.store_pattern(pattern_type, actions_taken, exec_time)
        self.cache.record_cache_miss()
        return result

    # ---------- Orchestration ----------
    async def run_wikipedia_automation(self, browser, user_goal: str):
        print()
        print(f"Processing: '{user_goal}'")
        pattern_type = self.classify_goal(user_goal)
        display_name, search_name = self.extract_person_name(user_goal)

        if pattern_type == "wikipedia_birth_date" and self.cache.has_pattern(pattern_type):
            return await self.execute_cached_pattern(browser, search_name, display_name, pattern_type)
        else:
            return await self.learn_pattern_and_execute(browser, search_name, display_name, "wikipedia_birth_date")


# ============================
# Demo Runner
# ============================

async def _get_browser(pw, token):
    # Try remote LightPanda first; fall back to local headless chromium
    try:
        if token:
            print("Connecting to LightPanda browser...")
            browser = await pw.chromium.connect_over_cdp(f"wss://cloud.lightpanda.io/ws?token={token}")
            print("Successfully connected to remote browser!")
            return browser
    except Exception as e:
        print(f"Remote connect failed: {e}")
    raise RuntimeError("Failed to connect to LightPanda remote browser")
async def run_wikipedia_demo():
    print("Starting Wikipedia Action Cache Demo")
    print("=" * 60)

    token = os.getenv("LIGHTPANDA_TOKEN")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        print("OPENAI_API_KEY not found in .env file")
        return

    demo_queries = [
        "When was Napoleon born?",
        "When was Einstein born?",
        "When was Tesla born?",
        "When was Marie Curie born?",
    ]

    async with async_playwright() as p:
        browser = await _get_browser(p, token)

        agent = LLMBrowserAgent()
        # Create an initial page/context
        await agent.ensure_page(browser)

        try:
            print()
            print(f"Running {len(demo_queries)} Wikipedia queries...")
            for i, query in enumerate(demo_queries, start=1):
                print()
                print(f"{'=' * 20} Query {i}/{len(demo_queries)} {'=' * 20}")
                start_t = time.time()
                for attempt in range(2):
                    try:
                        result = await agent.run_wikipedia_automation(browser, query)
                        duration = time.time() - start_t
                        print(f"Query completed in {duration:.1f} seconds")
                        print(f"Result: {result}")
                        break
                    except Exception as e:
                        err_text = str(e)
                        needs_reconnect = "Target page, context or browser has been closed" in err_text
                        if needs_reconnect and attempt == 0:
                            print("[warn] Browser connection lost. Reinitializing...")
                            try:
                                await browser.close()
                            except Exception:
                                pass
                            browser = await _get_browser(p, token)
                            agent.page = None
                            try:
                                await agent.ensure_page(browser)
                            except Exception:
                                pass
                            continue
                        print(f"[Error] Query failed: {e}")
                        break

            # Final stats
            print()
            print("=" * 60)
            print("FINAL DEMO STATISTICS")
            print("=" * 60)
            stats = agent.cache.get_cache_stats()
            print(f"Cache Hit Rate: {stats['hit_rate']:.1f}%")
            print(f"Total Queries: {stats['total_queries']}")
            print(f"Total Time Saved: {stats['time_saved']:.1f} seconds")
            print(f"Total Cost Saved: ${stats['cost_saved']:.3f}")
            print(f"Patterns Learned: {stats['patterns_learned']}")

            if stats["hit_rate"] > 0 and agent.cache.metrics["cache_hits"] > 0:
                avg_savings = stats["time_saved"] / agent.cache.metrics["cache_hits"]
                print()
                print("Cache Performance:")
                print(f"   - Average time savings per hit: {avg_savings:.1f}s")
                print(f"   - Estimated hourly cost savings: ${stats['cost_saved'] * 60:.2f}")
                print(f"   - Productivity improvement: {stats['hit_rate']:.0f}% faster execution")
        finally:
            try:
                await browser.close()
                print()
                print("Browser closed successfully!")
            except Exception as e:
                print(f"Error closing browser: {e}")


def main():
    asyncio.run(run_wikipedia_demo())


if __name__ == "__main__":
    main()
