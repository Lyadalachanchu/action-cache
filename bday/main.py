"""
main.py
-------
CLI to run the planner+cache demo with HTTP response caching.

Examples:
  python main.py --headful --llm-policy always   --goal "When was Marie Curie born?"
  python main.py --headful --llm-policy fallback --goal "When was Marie Curie born?"
"""

import argparse
import asyncio
import os
import time
import json
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright
from agent import LLMBrowserAgent

load_dotenv()

HTTP_CACHE_TTL = int(os.getenv("HTTP_CACHE_TTL_SECONDS", "86400"))  # 1 day
CACHE_DIR = Path(os.getenv("HTTP_CACHE_DIR", "http_cache"))
CACHE_DIR.mkdir(exist_ok=True)

def _key_from_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()

async def _cached_fetch(route, request):
    """
    Naive HTTP GET cache: fulfills from disk within TTL; otherwise fetches & stores.
    (Upgrade later to handle ETag/If-None-Match for revalidation.)
    """
    if request.method != "GET":
        return await route.continue_()
    k = _key_from_url(request.url)
    p = CACHE_DIR / k
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if time.time() - data["ts"] < HTTP_CACHE_TTL:
                return await route.fulfill(
                    status=data["status"],
                    headers=data["headers"],
                    body=bytes.fromhex(data["body_hex"])
                )
        except Exception:
            pass

    try:
        resp = await route.fetch()
        body = await resp.body()
    except Exception as e:
        # If fetch fails (e.g., unsupported in remote browser), just continue the request.
        return await route.continue_()

    record = {"ts": time.time(), "status": resp.status, "headers": dict(resp.headers), "body_hex": body.hex()}
    try:
        p.write_text(json.dumps(record))
    except Exception:
        pass
    await route.fulfill(status=resp.status, headers=dict(resp.headers), body=body)

async def get_browser_and_page(pw, token: str, headless: bool):
    """
    Create a browser context with a global route handler for HTTP cache.
    Works for both remote LightPanda and local Chromium.
    """
    if token:
        try:
            print("Connecting to LightPanda browser...")
            browser = await pw.chromium.connect_over_cdp(f"wss://cloud.lightpanda.io/ws?token={token}")
            print("Successfully connected to remote browser!")
        except Exception as e:
            print(f"Remote connect failed: {e}")
            browser = await pw.chromium.launch(headless=headless)
    else:
        browser = await pw.chromium.launch(headless=headless)

    ctx = await browser.new_context()

    # Remote LightPanda browsers do not currently support request interception reliably,
    # so skip the HTTP cache route when a token is used.
    if not token:
        await ctx.route("**/*", _cached_fetch)
    page = await ctx.new_page()
    return browser, page

async def run_once(goal: str, headless: bool, llm_policy: str):
    token = os.getenv("LIGHTPANDA_TOKEN")
    async with async_playwright() as p:
        browser, page = await get_browser_and_page(p, token, headless=headless)
        try:
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
    ap = argparse.ArgumentParser(description="Planner + Multi-tier Cache Demo (Wikipedia)")
    ap.add_argument("--headful", action="store_true", help="Run with browser UI.")
    ap.add_argument("--llm-policy", choices=["always", "verify", "fallback"], default="fallback",
                    help="LLM usage mode.")
    ap.add_argument("--goal", type=str, required=True, help="User goal/question.")
    args = ap.parse_args()

    asyncio.run(run_once(args.goal, headless=not args.headful, llm_policy=args.llm_policy))

if __name__ == "__main__":
    main()
