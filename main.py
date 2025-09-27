import asyncio
import os
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv()

async def main():
    token = os.getenv('LIGHTPANDA_TOKEN')
    if not token:
        raise ValueError("LIGHTPANDA_TOKEN environment variable not found. Please check your .env file.")
    
    async with async_playwright() as p:
        try:
            # Try connecting to remote browser via WebSocket
            print("Connecting to LightPanda browser...")
            browser = await p.chromium.connect_over_cdp(f"wss://cloud.lightpanda.io/ws?token={token}")
            print("Successfully connected to remote browser!")
            
        except Exception as e:
            print(f"Failed to connect to remote browser: {e}")
            print("Falling back to local browser...")
            # Fallback to local browser for testing
            browser = await p.chromium.launch(headless=False)
        
        try:
            page = await browser.new_page()
            
            # Example: Navigate to a website
            print("Navigating to page...")
            await page.goto("https://en.wikipedia.org/wiki/Roman_Empire")
            print("Page loaded successfully!")
            
            # Add your additional automation code here
            # Example: Get page title
            title = await page.title()
            print(f"Page title: {title}")
            
            # Wait a bit to see the page
            await asyncio.sleep(5)
        
        except Exception as e:
            print(f"Error during automation: {e}")
        
        finally:
            try:
                await browser.close()
                print("Browser closed successfully!")
            except Exception as e:
                print(f"Error closing browser: {e}")

if __name__ == "__main__":
    asyncio.run(main())