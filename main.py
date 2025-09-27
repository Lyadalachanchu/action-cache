import asyncio
import html
import os
import json
from playwright.async_api import async_playwright
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMBrowserAgent:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.page = None
        self.action_history = []
    
    async def get_page_info(self):
        """Extract key information from the current page"""
        title = await self.page.title()
        url = self.page.url
        
        # Wait for the page to load completely (wait for network to be idle)
        try:
            await self.page.wait_for_load_state('networkidle', timeout=15000)
            # Additional wait for SPAs to render
            await asyncio.sleep(3)
        except:
            print("Network idle timeout, continuing anyway...")
            # Still wait a bit for potential content to load
            await asyncio.sleep(5)
        
        # Get the full HTML content after JavaScript has rendered
        html_content = await self.page.content()
        
        # Also get information about interactive elements that are actually visible
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
                    .slice(0, 20); // Limit to first 20 interactive elements
            }
        """)
        
        print(f"Found {len(interactive_elements)} interactive elements")
        if interactive_elements:
            print("Sample elements:", interactive_elements[:3])
        else:
            # If no interactive elements found, try a broader search
            print("No interactive elements found, trying broader search...")
            all_elements = await self.page.evaluate("""
                () => {
                    const elements = document.querySelectorAll('*');
                    return Array.from(elements)
                        .filter(el => {
                            const rect = el.getBoundingClientRect();
                            return rect.width > 10 && rect.height > 10 && 
                                   el.textContent && el.textContent.trim().length > 0;
                        })
                        .map(el => ({
                            tag: el.tagName.toLowerCase(),
                            text: el.textContent?.trim().slice(0, 50) || '',
                            id: el.id || '',
                            className: el.className || ''
                        }))
                        .slice(0, 10);
                }
            """)
            print(f"Found {len(all_elements)} elements with content:", all_elements[:3])
        
        return {
            "title": title,
            "url": url,
            "html": html_content[:8000] + "..." if len(html_content) > 8000 else html_content,
            "interactive_elements": interactive_elements
        }
    
    async def ask_llm_for_action(self, page_info, user_goal):
        """Ask LLM what action to take based on page content and user goal"""
        
        # Get recent action history
        recent_actions = self.action_history[-5:] if self.action_history else []
        history_text = "\n".join([f"- {action}" for action in recent_actions]) if recent_actions else "None"
        
        prompt = f"""
        You are a browser automation assistant. Based on the current page information and user goal, decide what action to take.
        
        Current page:
        - Title: {page_info['title']}
        - URL: {page_info['url']}
        - Interactive elements found: {json.dumps(page_info['interactive_elements'], indent=2)}
        
        Recent actions taken:
        {history_text}
        
        User goal: {user_goal}
        
        Available actions:
        1. click(selector) - Click an element (use the exact selector from interactive_elements)
        2. type(selector, text) - Type text into an input field
        3. press_enter() - Press Enter key (useful after typing in search boxes)
        4. read_page() - Read and analyze the current page content to answer questions
        5. goto(url) - Navigate to a URL
        6. scroll(direction) - Scroll up or down  
        7. wait(seconds) - Wait for a specified time
        8. done() - Task completed
        
        IMPORTANT: 
        - Only use selectors from the interactive_elements list above. These are guaranteed to exist and be clickable.
        - DON'T repeat the same action - look at your recent actions and choose something different
        - After typing in a search box, ALWAYS use press_enter() to submit the search
        - If you've already typed something, try press_enter() or read_page() instead of typing again
        - If typing fails repeatedly, try read_page() to analyze the current page content directly
        - If you need information that's already on the current page, use read_page() instead of searching
        - If no suitable interactive elements are available for your goal, use scroll(down) to see more content or done() to complete.
        
        Respond with a JSON object containing:
        {{
            "action": "action_name",
            "parameters": {{"selector": "exact_selector_from_list", "other_param": "value"}},
            "reasoning": "explanation of why this action was chosen"
        }}
        """
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        try:
            action_data = json.loads(response.choices[0].message.content)
            return action_data
        except json.JSONDecodeError:
            return {"action": "wait", "parameters": {"seconds": 1}, "reasoning": "Failed to parse LLM response"}
    
    async def execute_action(self, action_data, user_goal=""):
        """Execute the action determined by the LLM"""
        action = action_data.get("action")
        params = action_data.get("parameters", {})
        
        # Add to action history
        action_summary = f"{action}({params.get('selector', '')}) - {action_data.get('reasoning', '')[:50]}..."
        self.action_history.append(action_summary)
        
        print(f"Executing: {action} with params: {params}")
        print(f"Reasoning: {action_data.get('reasoning', 'No reasoning provided')}")
        
        if action == "click":
            selector = params.get("selector")
            try:
                # Wait for element to be visible and clickable
                await self.page.wait_for_selector(selector, timeout=5000, state='visible')
                await self.page.click(selector, timeout=5000)
            except Exception as e:
                print(f"Failed to click {selector}: {e}")
                return False
        elif action == "type":
            selector = params.get("selector")
            text = params.get("text")
            try:
                # Wait for element to exist
                await self.page.wait_for_selector(selector, timeout=5000)
                
                # Use pure JavaScript approach - most compatible
                escaped_text = text.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
                success = await self.page.evaluate(f"""
                    () => {{
                        try {{
                            const element = document.querySelector('{selector}');
                            if (!element) return false;
                            
                            // Clear and set value
                            element.value = '';
                            element.value = '{escaped_text}';
                            
                            // Trigger various events that websites might listen for
                            element.dispatchEvent(new Event('focus', {{bubbles: true}}));
                            element.dispatchEvent(new Event('input', {{bubbles: true}}));
                            element.dispatchEvent(new Event('change', {{bubbles: true}}));
                            element.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Enter', bubbles: true}}));
                            
                            return true;
                        }} catch (e) {{
                            console.error('Typing error:', e);
                            return false;
                        }}
                    }}
                """)
                
                if not success:
                    print(f"JavaScript typing failed for {selector}")
                    return False
                    
                print(f"Successfully typed '{text}' into {selector}")
                
            except Exception as e:
                print(f"Failed to type in {selector}: {e}")
                return False
        elif action == "press_enter":
            try:
                # Try JavaScript form submission first
                success = await self.page.evaluate("""
                    () => {
                        try {
                            // Find and submit search form
                            const searchInput = document.querySelector('#searchInput, input[type="search"]');
                            if (searchInput) {
                                const form = searchInput.closest('form');
                                if (form) {
                                    form.submit();
                                    return true;
                                }
                            }
                            return false;
                        } catch (e) {
                            return false;
                        }
                    }
                """)
                
                if success:
                    print("Form submitted, waiting for navigation...")
                    # Wait for navigation to complete
                    try:
                        await self.page.wait_for_load_state('networkidle', timeout=10000)
                        print("Navigation completed successfully!")
                    except:
                        print("Navigation timeout, but continuing...")
                        await asyncio.sleep(3)  # Give some time for page to load
                else:
                    # Fallback to keyboard press
                    print("Form submission failed, trying keyboard Enter...")
                    await self.page.keyboard.press('Enter')
                    try:
                        await self.page.wait_for_load_state('networkidle', timeout=5000)
                    except:
                        await asyncio.sleep(2)
                    
            except Exception as e:
                # The "Execution context destroyed" error is actually expected during navigation
                if "Execution context was destroyed" in str(e):
                    print("Navigation detected (this is good!), waiting for page to load...")
                    try:
                        await self.page.wait_for_load_state('networkidle', timeout=10000)
                        print("Navigation completed!")
                    except:
                        print("Navigation timeout, but continuing...")
                        await asyncio.sleep(3)
                else:
                    print(f"Failed to press Enter: {e}")
                    return False
        elif action == "read_page":
            try:
                # Get page text content for analysis
                page_text = await self.page.evaluate("""
                    () => {
                        // Remove script and style elements
                        const scripts = document.querySelectorAll('script, style, nav, header, footer');
                        scripts.forEach(el => el.remove());
                        
                        // Get main content
                        const content = document.querySelector('#mw-content-text, main, .content, body');
                        return content ? content.innerText : document.body.innerText;
                    }
                """)
                
                # Ask LLM to analyze the page content and answer the user's question
                analysis_prompt = f"""
                Based on the following page content, please answer this question: {user_goal}
                
                Page content:
                {page_text[:6000]}...
                
                Provide a concise, accurate answer based on the information available on this page.
                """
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.1
                )
                
                answer = response.choices[0].message.content
                print(f"\n=== ANSWER ===")
                print(answer)
                print(f"=============\n")
                
                return True  # Mark as completed since we provided an answer
                
            except Exception as e:
                print(f"Failed to read page: {e}")
                return False
        elif action == "goto":
            await self.page.goto(params.get("url"))
        elif action == "scroll":
            direction = params.get("direction", "down")
            try:
                if direction == "down":
                    await self.page.evaluate("document.documentElement.scrollTop += 500")
                else:
                    await self.page.evaluate("document.documentElement.scrollTop -= 500")
                # Alternative method if the above doesn't work
                await self.page.keyboard.press("PageDown" if direction == "down" else "PageUp")
            except Exception as e:
                print(f"Scroll failed: {e}, trying keyboard method")
                await self.page.keyboard.press("PageDown" if direction == "down" else "PageUp")
        elif action == "wait":
            await asyncio.sleep(params.get("seconds", 1))
        elif action == "done":
            return True
        
        return False
    
    async def run_automation(self, user_goal, max_steps=10):
        """Run the LLM-driven browser automation"""
        print(f"Starting automation with goal: {user_goal}")
        
        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            
            # Get current page information
            page_info = await self.get_page_info()
            
            # Ask LLM for next action
            action_data = await self.ask_llm_for_action(page_info, user_goal)
            
            # Execute the action
            is_done = await self.execute_action(action_data, user_goal)
            
            if is_done:
                print("Task completed!")
                break
            
            # Small delay between actions
            await asyncio.sleep(1)
        
        print("Automation finished!")

async def main():
    token = os.getenv('LIGHTPANDA_TOKEN')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if not token:
        raise ValueError("LIGHTPANDA_TOKEN environment variable not found. Please check your .env file.")
    
    if not openai_key:
        print("Warning: OPENAI_API_KEY not found. LLM features will not work.")
        print("Add your OpenAI API key to the .env file: OPENAI_API_KEY=your_key_here")
    
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
            
            # Create LLM agent
            agent = LLMBrowserAgent()
            agent.page = page
            
            # Example: Navigate to a website and let LLM interact with it
            print("Navigating to page...")
            await page.goto("https://en.wikipedia.org/wiki/Roman_Empire")
            print("Page loaded successfully!")
            
            # Run LLM-driven automation
            if openai_key:
                user_goal = "When did the Crisis of the Roman Republic start?"
                await agent.run_automation(user_goal)
            else:
                print("Skipping LLM automation - add OPENAI_API_KEY to .env file")
                # Just wait a bit to see the page
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