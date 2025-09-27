import asyncio
import os
from pyppeteer import connect
from dotenv import load_dotenv

load_dotenv()

async def main():
    token = os.getenv('LIGHTPANDA_TOKEN')
    if not token:
        raise ValueError("LIGHTPANDA_TOKEN environment variable not found. Please check your .env file.")
    
    browser = await connect({
        'browserWSEndpoint': f"wss://cloud.lightpanda.io/ws?token={token}",
    })
    
    page = await browser.newPage()
    
    # Add your additional automation code here
    
    
    await browser.close()

if __name__ == "__main__":
    asyncio.run(main())