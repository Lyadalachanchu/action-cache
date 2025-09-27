from playwright.sync_api import sync_playwright
import dotenv
import os

dotenv.load_dotenv()

LIGHTPANDA_TOKEN = os.getenv("LIGHTPANDA_TOKEN")
if not LIGHTPANDA_TOKEN:
    raise RuntimeError("LIGHTPANDA_TOKEN is not set in the environment")

CDP_ENDPOINT = f"wss://cloud.lightpanda.io/ws?token={LIGHTPANDA_TOKEN}"

with sync_playwright() as playwright:
    browser = playwright.chromium.connect_over_cdp(CDP_ENDPOINT)
    context = browser.contexts[0] if browser.contexts else browser.new_context()
    page = context.pages[0] if context.pages else context.new_page()

    page.goto("https://www.google.com")
    print(page.title())

    context.close()
    browser.close()
