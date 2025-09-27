import asyncio
import html
import os
import json
from playwright.async_api import async_playwright
from openai import AsyncOpenAI
from dotenv import load_dotenv
from memory_system import BrowserMemorySystem

load_dotenv()

class EnhancedLLMBrowserAgent:
    def __init__(self, use_memory=True):
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.page = None
        self.action_history = []
        self.use_memory = use_memory
        
        # Initialize memory system if enabled
        if self.use_memory:
            try:
                self.memory = BrowserMemorySystem()
                print("Memory system initialized successfully")
            except Exception as e:
                print(f"Failed to initialize memory system: {e}")
                print("Continuing without memory system...")
                self.use_memory = False
                self.memory = None
        else:
            self.memory = None
    
    async def get_page_info(self):
        """Extract key information from the current page"""
        title = await self.page.title()
        url = self.page.url
        
        # Wait for the page to load completely
        try:
            await self.page.wait_for_load_state('networkidle', timeout=15000)
            await asyncio.sleep(3)
        except:
            print("Network idle timeout, continuing anyway...")
            await asyncio.sleep(5)
        
        html_content = await self.page.content()
        
        # Get interactive elements
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
        
        print(f"Found {len(interactive_elements)} interactive elements")
        
        return {
            "title": title,
            "url": url,
            "html": html_content[:8000] + "..." if len(html_content) > 8000 else html_content,
            "interactive_elements": interactive_elements
        }
    
    async def get_memory_insights(self, user_goal, current_domain):
        """Get insights from past experiences using semantic search"""
        if not self.use_memory or not self.memory:
            return None
        
        try:
            # Find similar past experiences
            similar_experiences = await self.memory.find_similar_experiences(
                user_goal, 
                domain=current_domain,
                limit=3
            )
            
            if not similar_experiences:
                print("No similar past experiences found")
                return None
            
            # Get starting points from successful experiences
            starting_points = await self.memory.get_starting_points(user_goal, current_domain)
            
            insights = {
                "similar_experiences": similar_experiences,
                "starting_points": starting_points,
                "has_memory": True
            }
            
            print(f"Found {len(similar_experiences)} similar experiences")
            if starting_points:
                print(f"Found {len(starting_points)} suggested starting points")
            
            return insights
            
        except Exception as e:
            print(f"Error getting memory insights: {e}")
            return None
    
    async def ask_llm_for_action(self, page_info, user_goal, memory_insights=None):
        """Ask LLM what action to take, incorporating memory insights"""
        
        # Get recent action history
        recent_actions = self.action_history[-5:] if self.action_history else []
        history_text = "\n".join([f"- {action}" for action in recent_actions]) if recent_actions else "None"
        
        # Build memory context if available
        memory_context = ""
        if memory_insights and memory_insights.get("has_memory"):
            memory_context = f"""
        
        MEMORY INSIGHTS (from past similar tasks):
        - Found {len(memory_insights.get('similar_experiences', []))} similar past experiences
        - Success rates: {[exp.get('success_rate', 0) for exp in memory_insights.get('similar_experiences', [])]}
        - Suggested starting points: {memory_insights.get('starting_points', [])}
        
        Consider using successful patterns from past experiences when available.
        """
        
        prompt = f"""
        You are a browser automation assistant with access to past experiences. Based on the current page information, user goal, and memory insights, decide what action to take.
        
        Current page:
        - Title: {page_info['title']}
        - URL: {page_info['url']}
        - Interactive elements found: {json.dumps(page_info['interactive_elements'], indent=2)}
        
        Recent actions taken:
        {history_text}
        
        User goal: {user_goal}
        {memory_context}
        
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
        - Only use selectors from the interactive_elements list above
        - DON'T repeat the same action - look at your recent actions and choose something different
        - After typing in a search box, ALWAYS use press_enter() to submit the search
        - If you've already typed something, try press_enter() or read_page() instead of typing again
        - If typing fails repeatedly, try read_page() to analyze the current page content directly
        - If you need information that's already on the current page, use read_page() instead of searching
        - If no suitable interactive elements are available for your goal, use scroll(down) to see more content or done() to complete
        - Use memory insights to inform your decisions when available
        
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
                await self.page.wait_for_selector(selector, timeout=5000, state='visible')
                await self.page.click(selector, timeout=5000)
            except Exception as e:
                print(f"Failed to click {selector}: {e}")
                return False
        elif action == "type":
            selector = params.get("selector")
            text = params.get("text")
            try:
                await self.page.wait_for_selector(selector, timeout=5000)
                
                escaped_text = text.replace("'", "\\'").replace('"', '\\"').replace('\n', '\\n')
                success = await self.page.evaluate(f"""
                    () => {{
                        try {{
                            const element = document.querySelector('{selector}');
                            if (!element) return false;
                            
                            element.value = '';
                            element.value = '{escaped_text}';
                            
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
                success = await self.page.evaluate("""
                    () => {
                        try {
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
                    try:
                        await self.page.wait_for_load_state('networkidle', timeout=10000)
                        print("Navigation completed successfully!")
                    except:
                        print("Navigation timeout, but continuing...")
                        await asyncio.sleep(3)
                else:
                    print("Form submission failed, trying keyboard Enter...")
                    await self.page.keyboard.press('Enter')
                    try:
                        await self.page.wait_for_load_state('networkidle', timeout=5000)
                    except:
                        await asyncio.sleep(2)
                    
            except Exception as e:
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
                page_text = await self.page.evaluate("""
                    () => {
                        const scripts = document.querySelectorAll('script, style, nav, header, footer');
                        scripts.forEach(el => el.remove());
                        
                        const content = document.querySelector('#mw-content-text, main, .content, body');
                        return content ? content.innerText : document.body.innerText;
                    }
                """)
                
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
                
                return True
                
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
                await self.page.keyboard.press("PageDown" if direction == "down" else "PageUp")
            except Exception as e:
                print(f"Scroll failed: {e}, trying keyboard method")
                await self.page.keyboard.press("PageDown" if direction == "down" else "PageUp")
        elif action == "wait":
            await asyncio.sleep(params.get("seconds", 1))
        elif action == "done":
            return True
        
        return False
    
    async def store_successful_experience(self, user_goal, extracted_info=""):
        """Store a successful automation experience in memory"""
        if not self.use_memory or not self.memory:
            return
        
        try:
            # Extract domain from current URL
            current_url = self.page.url
            domain = current_url.split('/')[2] if '://' in current_url else "unknown"
            
            # Get successful actions from history
            successful_actions = []
            key_selectors = []
            navigation_path = [current_url]
            
            for action in self.action_history:
                if "click" in action or "type" in action:
                    successful_actions.append({
                        "action": action.split('(')[0],
                        "selector": action.split('(')[1].split(')')[0] if '(' in action else "",
                        "text": action.split('->')[1].strip("'") if '->' in action else ""
                    })
                    if "selector" in action:
                        selector = action.split('(')[1].split(')')[0]
                        if selector not in key_selectors:
                            key_selectors.append(selector)
            
            # Store the experience
            experience_id = await self.memory.store_experience(
                task_goal=user_goal,
                domain=domain,
                successful_actions=successful_actions,
                key_selectors=key_selectors,
                navigation_path=navigation_path,
                extracted_info=extracted_info,
                success_rate=0.9,  # High success rate since task was completed
                context_summary=f"Page: {await self.page.title()}"
            )
            
            print(f"Stored successful experience with ID: {experience_id}")
            
        except Exception as e:
            print(f"Error storing experience: {e}")
    
    async def run_automation(self, user_goal, max_steps=10):
        """Run the LLM-driven browser automation with memory"""
        print(f"Starting automation with goal: {user_goal}")
        
        # Get memory insights at the start
        current_domain = self.page.url.split('/')[2] if '://' in self.page.url else "unknown"
        memory_insights = await self.get_memory_insights(user_goal, current_domain)
        
        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            
            # Get current page information
            page_info = await self.get_page_info()
            
            # Ask LLM for next action (with memory insights)
            action_data = await self.ask_llm_for_action(page_info, user_goal, memory_insights)
            
            # Execute the action
            is_done = await self.execute_action(action_data, user_goal)
            
            if is_done:
                print("Task completed!")
                # Store the successful experience
                await self.store_successful_experience(user_goal)
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
            print("Connecting to LightPanda browser...")
            browser = await p.chromium.connect_over_cdp(f"wss://cloud.lightpanda.io/ws?token={token}")
            print("Successfully connected to remote browser!")
            
        except Exception as e:
            print(f"Failed to connect to remote browser: {e}")
            print("Falling back to local browser...")
            browser = await p.chromium.launch(headless=False)
        
        try:
            page = await browser.new_page()
            
            # Create enhanced LLM agent with memory
            agent = EnhancedLLMBrowserAgent(use_memory=True)
            agent.page = page
            
            # Example: Navigate to a website and let LLM interact with it
            print("Navigating to page...")
            await page.goto("https://en.wikipedia.org/wiki/Roman_Empire")
            print("Page loaded successfully!")
            
            # Run LLM-driven automation with memory
            if openai_key:
                user_goal = "When did the Crisis of the Roman Republic start?"
                await agent.run_automation(user_goal)
            else:
                print("Skipping LLM automation - add OPENAI_API_KEY to .env file")
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


