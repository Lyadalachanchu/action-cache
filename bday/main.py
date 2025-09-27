"""
main.py
-------
Tiny CLI to run the planner agent demo.

Examples:
  python main.py --headful --llm-policy always --goal "When was Napoleon born?"
  python main.py --headful --llm-policy fallback --goal "When was Napoleon born?"
"""

import argparse
import asyncio
import os
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from agent import LLMBrowserAgent

load_dotenv()

async def get_browser(pw, token: str, headless: bool):
    if token:
        try:
            print("Connecting to LightPanda browser...")
            b = await pw.chromium.connect_over_cdp(f"wss://cloud.lightpanda.io/ws?token={token}")
            print("Successfully connected to remote browser!")
            return b
        except Exception as e:
            print(f"Remote connect failed: {e}")
    print(f"Launching local Chromium (headless={headless})...")
    return await pw.chromium.launch(headless=headless)

async def run_once(goal: str, headless: bool, llm_policy: str):
    token = os.getenv("LIGHTPANDA_TOKEN")
    async with async_playwright() as p:
        browser = await get_browser(p, token, headless=headless)
        try:
            page = await browser.new_page()
            agent = LLMBrowserAgent(llm_policy=llm_policy)
            agent.page = page
            print("Navigating to Wikipedia...")
            await page.goto("https://en.wikipedia.org", wait_until="domcontentloaded")
            await agent.run_automation(goal)
        finally:
            try:
                await browser.close()
            except Exception:
                pass

def main():
    ap = argparse.ArgumentParser(description="Planner + Cache Demo (Wikipedia)")
    ap.add_argument("--headful", action="store_true", help="Run with browser UI.")
    ap.add_argument("--llm-policy", choices=["always", "verify", "fallback"], default="fallback",
                    help="LLM usage mode.")
    ap.add_argument("--goal", type=str, required=True, help="User goal/question.")
    args = ap.parse_args()

    asyncio.run(run_once(args.goal, headless=not args.headful, llm_policy=args.llm_policy))

if __name__ == "__main__":
    main()
