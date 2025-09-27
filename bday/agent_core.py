# agent_core.py - Core browser automation logic
import asyncio
import time
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

from playwright.async_api import async_playwright

# ===================== Globals =====================
STRICT_WIKI_ONLY = True
MAX_PAGE_TEXT_CHARS = 18000
WIKI_HOME = "https://en.wikipedia.org"
WIKI_SEARCH = WIKI_HOME + "/w/index.php?search="

# Common misnamings → correct canonical titles
WIKI_TITLE_FIXES = {
    "Crisis_of_the_Roman_Empire": "Crisis_of_the_Third_Century",
    "Crisis_of_Roman_Empire": "Crisis_of_the_Third_Century",
}

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

class LLMBrowserAgent:
    def __init__(self, page=None, browser=None):
        self.page = page
        self.browser = browser
        self.playwright_context = None
        self.token = None
        self.use_headless = False
        self.connection_attempts = 0
        self.max_connection_attempts = 3

    async def safe_page_operation(self, operation_func, *args, max_retries=2, **kwargs):
        """Safely execute page operations with automatic reconnection"""
        for attempt in range(max_retries):
            try:
                return await operation_func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                if ("has been closed" in error_msg or
                    "Target page" in error_msg or
                    "Target closed" in error_msg):
                    print(f"🔄 Page operation failed (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        if await self.reconnect_browser():
                            print("✅ Reconnected, retrying operation...")
                            continue
                        else:
                            print("❌ Reconnection failed")
                            break
                    else:
                        print("❌ Max reconnection attempts reached")
                        break
                else:
                    # Non-connection error, re-raise immediately
                    raise e

        raise Exception(f"Page operation failed after {max_retries} attempts with reconnection")

    async def reconnect_browser(self):
        """Attempt to reconnect to browser and create new page"""
        self.connection_attempts += 1

        if self.connection_attempts > self.max_connection_attempts:
            print(f"❌ Max connection attempts ({self.max_connection_attempts}) reached")
            return False

        try:
            # Close existing browser if it exists
            if self.browser:
                try:
                    await self.browser.close()
                except:
                    pass

            print(f"🔄 Reconnection attempt {self.connection_attempts}/{self.max_connection_attempts}")

            # Try remote connection first if token is available
            if self.token and self.connection_attempts <= 2:
                try:
                    if not self.playwright_context:
                        self.playwright_context = async_playwright()
                        self.playwright_instance = await self.playwright_context.__aenter__()

                    print("🌐 Attempting Lightpanda reconnection...")
                    self.browser = await self.playwright_instance.chromium.connect_over_cdp(
                        f"wss://cloud.lightpanda.io/ws?token={self.token}"
                    )
                    self.page = await self.browser.new_page()
                    print("✅ Successfully reconnected to Lightpanda")
                    return True

                except Exception as remote_e:
                    print(f"❌ Lightpanda reconnection failed: {remote_e}")

            # Fallback to local browser
            print("🖥️ Falling back to local browser...")
            if not self.playwright_context:
                self.playwright_context = async_playwright()
                self.playwright_instance = await self.playwright_context.__aenter__()

            self.browser = await self.playwright_instance.chromium.launch(
                headless=self.use_headless
            )
            self.page = await self.browser.new_page()
            print("✅ Local browser started successfully")
            return True

        except Exception as e:
            print(f"❌ Browser reconnection failed: {e}")
            return False

    async def check_browser_health(self):
        """Check if browser is accessible and attempt reconnection if needed"""
        try:
            await self.page.title()
            return True
        except Exception as e:
            error_msg = str(e)
            if ("has been closed" in error_msg or
                "Target page" in error_msg or
                "Target closed" in error_msg):
                print(f"⚠️ Browser health check failed: {e}")
                return await self.reconnect_browser()
            else:
                # Some other error, might be temporary
                print(f"⚠️ Browser health check error: {e}")
                return False

    async def _goto_with_retry(self, url: str, retries: int = 2) -> bool:
        """Navigate with retries and automatic reconnection"""
        url = _normalize_wiki_url(url)

        async def goto_operation():
            await self.page.goto(url, wait_until="domcontentloaded")
            await self.page.wait_for_load_state("networkidle")
            return True

        try:
            await self.safe_page_operation(goto_operation, max_retries=retries)
            return True
        except Exception as e:
            print(f"❌ Navigation to {url} failed after retries: {e}")
            return False

    async def _wiki_search(self, query: str) -> bool:
        """Go to Wikipedia search results for query"""
        search_url = WIKI_SEARCH + quote_plus(query)
        print(f"   → WIKI SEARCH '{query}' ({search_url})")
        return await self._goto_with_retry(search_url)

    async def _exec_action(self, a: Dict[str, Any]):
        """Execute a single action with robust error handling"""
        action = a.get("action", "")

        # Safety check for browser health before any action
        if not await self.check_browser_health():
            print("❌ Browser health check failed before action execution")
            return None

        try:
            print(f"   [{action.upper()}] ...")

            if action == "goto":
                target_url = a.get("url", "")
                if not _ensure_wikipedia_url(target_url):
                    print(f"   ❌ BLOCKED: Non-Wikipedia URL: {target_url}")
                    return False

                success = await self._goto_with_retry(target_url)
                if success:
                    # Get page info AFTER navigation
                    try:
                        title = await self.safe_page_operation(self.page.title)
                        url = self.page.url
                        print(f"   ✅ Navigated to '{title}' ({url})")
                    except:
                        print(f"   ✅ Navigated to {target_url}")
                else:
                    # Try wiki search fallback
                    query = a.get("fallback_query", "")
                    if query:
                        print(f"   → GOTO failed; attempting Wikipedia search fallback…")
                        return await self._wiki_search(query)
                return success

            elif action == "read_page":
                # Get current page info
                try:
                    title = await self.safe_page_operation(self.page.title)
                    url = self.page.url
                    print(f"   📖 Reading '{title}' ({url})")
                except Exception as info_error:
                    print(f"   📖 Reading page (could not get info: {info_error})")

                try:
                    # Get page text safely
                    text_content = await self.safe_page_operation(
                        lambda: self.page.evaluate("""
                            () => {
                                const content = document.querySelector('#mw-content-text, .mw-body-content, #content');
                                return content ? content.innerText : document.body.innerText;
                            }
                        """)
                    )

                    if len(text_content) > MAX_PAGE_TEXT_CHARS:
                        text_content = text_content[:MAX_PAGE_TEXT_CHARS] + "... [truncated]"

                    print(f"   → READ {len(text_content)} chars")
                    return text_content

                except Exception as read_error:
                    print(f"   ❌ Failed to read page: {read_error}")
                    return None

            elif action == "scroll":
                direction = a.get("direction", "down")
                # Get current page info for context
                try:
                    title = await self.safe_page_operation(self.page.title)
                    print(f"   🔄 Scrolling {direction} on '{title}'")
                except:
                    print(f"   🔄 Scrolling {direction}")

                try:
                    # Use multiple scroll methods for better compatibility
                    if direction == "down":
                        await self.safe_page_operation(
                            lambda: self.page.evaluate("""
                                () => {
                                    if (typeof window.scrollBy === 'function') {
                                        window.scrollBy(0, 500);
                                    } else if (typeof window.scroll === 'function') {
                                        window.scroll(0, window.pageYOffset + 500);
                                    } else {
                                        document.documentElement.scrollTop += 500;
                                    }
                                }
                            """)
                        )
                    else:
                        await self.safe_page_operation(
                            lambda: self.page.evaluate("""
                                () => {
                                    if (typeof window.scrollBy === 'function') {
                                        window.scrollBy(0, -500);
                                    } else if (typeof window.scroll === 'function') {
                                        window.scroll(0, window.pageYOffset - 500);
                                    } else {
                                        document.documentElement.scrollTop -= 500;
                                    }
                                }
                            """)
                        )
                    print(f"   → SCROLLED {direction}")
                    return True
                except Exception as scroll_error:
                    print(f"   ❌ Scroll failed: {scroll_error}")
                    # Fallback to keyboard scrolling
                    try:
                        if direction == "down":
                            await self.safe_page_operation(lambda: self.page.keyboard.press("PageDown"))
                        else:
                            await self.safe_page_operation(lambda: self.page.keyboard.press("PageUp"))
                        print(f"   → SCROLLED {direction} (keyboard fallback)")
                        return True
                    except Exception as kb_error:
                        print(f"   ❌ Keyboard scroll also failed: {kb_error}")
                        return False

            else:
                print(f"   ❌ Unknown action: {action}")
                return False

        except Exception as e:
            error_msg = str(e)
            if ("has been closed" in error_msg or
                "Target page" in error_msg or
                "Target closed" in error_msg):
                print(f"❌ Browser connection lost during action '{action}': {e}")
                # Try to reconnect for next action
                await self.check_browser_health()
                return None
            else:
                print(f"❌ Action '{action}' failed with error: {e}")
                return None

    def _normalize_action(self, action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize action format to ensure consistency"""
        if not isinstance(action, dict):
            return None

        # If already in correct format
        if "action" in action:
            return action

        # Try to fix common LLM format issues
        for key, value in action.items():
            if key == "goto" and isinstance(value, str):
                return {"action": "goto", "url": value}
            elif key == "goto" and isinstance(value, dict) and "url" in value:
                return {"action": "goto", "url": value["url"]}
            elif key == "read_page":
                return {"action": "read_page"}
            elif key == "scroll" and isinstance(value, str):
                return {"action": "scroll", "direction": value}
            elif key == "scroll" and isinstance(value, dict) and "direction" in value:
                return {"action": "scroll", "direction": value["direction"]}

        # Try to infer from keys
        if "url" in action:
            return {"action": "goto", "url": action["url"]}
        elif "direction" in action:
            return {"action": "scroll", "direction": action["direction"]}
        elif len(action) == 1:
            # Single key might be the action type
            key = list(action.keys())[0]
            if key in ["goto", "read_page", "scroll"]:
                if key == "goto":
                    value = action[key]
                    if isinstance(value, str):
                        return {"action": "goto", "url": value}
                elif key == "read_page":
                    return {"action": "read_page"}
                elif key == "scroll":
                    value = action[key]
                    if isinstance(value, str):
                        return {"action": "scroll", "direction": value}

        print(f"⚠️ Could not normalize action: {action}")
        return None

    async def cleanup(self):
        """Clean up browser and playwright resources"""
        try:
            if self.browser:
                await self.browser.close()
        except Exception as e:
            print(f"⚠️ Browser cleanup warning: {e}")
