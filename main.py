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
        self.subgoals = []
        self.current_subgoal_index = 0
        self.collected_information = []  # Store info from each subgoal
    
    async def check_browser_health(self):
        """Check if the browser is still accessible"""
        try:
            await self.page.title()
            return True
        except Exception as e:
            print(f"Browser health check failed: {e}")
            return False
    
    def get_step_title(self, action, params, page_info):
        """Generate a descriptive title for each step"""
        page_title = page_info.get('title', 'Unknown Page')[:50]
        url = page_info.get('url', '')
        
        if action == "type":
            text = params.get('text', '')[:30]
            return f"TYPING '{text}' INTO SEARCH"
        elif action == "click":
            selector = params.get('selector', '')
            return f"CLICKING ELEMENT ({selector})"
        elif action == "press_enter":
            return "SUBMITTING SEARCH"
        elif action == "goto":
            url_target = params.get('url', '')
            if 'wikipedia' in url_target:
                return "NAVIGATING TO WIKIPEDIA"
            else:
                return f"NAVIGATING TO {url_target}"
        elif action == "read_page":
            if 'wikipedia' in url:
                return f"READING WIKIPEDIA ARTICLE: {page_title}"
            else:
                return f"ANALYZING PAGE: {page_title}"
        elif action == "scroll":
            direction = params.get('direction', 'down')
            return f"SCROLLING {direction.upper()} FOR MORE CONTENT"
        elif action == "wait":
            seconds = params.get('seconds', 1)
            return f"WAITING {seconds} SECONDS"
        elif action == "done":
            return "TASK COMPLETED"
        else:
            return f"EXECUTING {action.upper()}"
    
    async def provide_final_answer(self, user_goal):
        """Synthesize all collected information to provide a comprehensive final answer"""
        print(f"\n🧩 SYNTHESIZING INFORMATION FROM ALL SUBGOALS...")
        print("=" * 60)
        
        # Compile all collected information
        all_info = ""
        for info in self.collected_information:
            all_info += f"\n\n--- {info['subgoal_title']} ---\n"
            all_info += f"Source: {info['page_title']} ({info['page_url']})\n"
            all_info += f"Information: {info['information']}\n"
        
        # Create comprehensive analysis prompt
        synthesis_prompt = f"""
        You have completed a multi-step research process to answer this question: {user_goal}
        
        Here is all the information collected from each research step:
        {all_info}
        
        Now provide a comprehensive, well-structured answer that:
        1. Directly answers the original question
        2. Synthesizes information from all research steps
        3. Identifies patterns, connections, or comparisons as requested
        4. Presents the information in a clear, organized manner
        5. Cites the sources when relevant
        
        If any required information was not found, clearly state what is missing.
        """
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.1
        )
        
        final_answer = response.choices[0].message.content
        
        print(f"\n🎯 COMPREHENSIVE FINAL ANSWER")
        print("=" * 80)
        print(final_answer)
        print("=" * 80)
        
        # Show research summary
        print(f"\n📊 RESEARCH SUMMARY:")
        print(f"   • Completed {len(self.collected_information)} research steps")
        print(f"   • Analyzed {len(set([info['page_url'] for info in self.collected_information]))} unique pages")
        for i, info in enumerate(self.collected_information, 1):
            print(f"   • Step {i}: {info['subgoal_title']}")
    
    async def create_subgoals_and_actions(self, user_goal):
        """Create a set of subgoals and map concrete actions to each subgoal"""
        print("\n📋 PLANNING PHASE: Creating subgoals and action plan...")
        print("=" * 60)
        
        planning_prompt = f"""
        You are a browser automation planner for Wikipedia research. Break down the following user goal into specific subgoals and concrete actions.
        
        User Goal: {user_goal}
        
        CONTEXT: You will start on Wikipedia (https://en.wikipedia.org) and must stay on Wikipedia throughout the entire process. Do NOT navigate to Google or any other search engines.
        
        Available actions:
        - click(selector) - Click an element
        - type(selector, text) - Type text into an input field  
        - press_enter() - Press Enter key
        - read_page() - Read and analyze page content
        - goto(url) - Navigate to a URL (ONLY use Wikipedia URLs like https://en.wikipedia.org/wiki/...)
        - scroll(direction) - Scroll up or down
        - wait(seconds) - Wait for specified time
        - done() - Task completed
        
        Create a plan with 3-5 subgoals. For each subgoal, specify the concrete actions needed.
        
        IMPORTANT RULES:
        - NEVER navigate to Google, Bing, or any non-Wikipedia sites
        - Use Wikipedia search functionality only
        - If you need to navigate, only use Wikipedia URLs
        - Focus on finding information within Wikipedia articles
        
        Respond with a JSON object in this format:
        {{
            "subgoals": [
                {{
                    "id": 1,
                    "title": "Search for topic on Wikipedia",
                    "description": "Use Wikipedia search to find the relevant article",
                    "actions": [
                        {{"action": "type", "parameters": {{"selector": "#searchInput", "text": "Crisis of the Roman Republic"}}, "description": "Type search query in Wikipedia search box"}},
                        {{"action": "press_enter", "parameters": {{}}, "description": "Submit Wikipedia search"}}
                    ]
                }},
                {{
                    "id": 2,
                    "title": "Open relevant Wikipedia article",
                    "description": "Click on the most relevant article from search results",
                    "actions": [
                        {{"action": "click", "parameters": {{"selector": "a[href*='Crisis']"}}, "description": "Click on Crisis-related article link"}}
                    ]
                }},
                {{
                    "id": 3,
                    "title": "Extract information from article",
                    "description": "Read the Wikipedia article to find the answer",
                    "actions": [
                        {{"action": "read_page", "parameters": {{}}, "description": "Read and analyze the Wikipedia article content"}}
                    ]
                }}
            ]
        }}
        
        Make the plan specific to answering: {user_goal}
        Focus on Wikipedia-only research and navigation.
        """
        
        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": planning_prompt}],
            temperature=0.1
        )
        
        try:
            plan_data = json.loads(response.choices[0].message.content)
            self.subgoals = plan_data.get("subgoals", [])
            
            print(f"📝 Created {len(self.subgoals)} subgoals:")
            for i, subgoal in enumerate(self.subgoals, 1):
                print(f"\n  {i}. {subgoal['title']}")
                print(f"     Description: {subgoal['description']}")
                print(f"     Actions: {len(subgoal.get('actions', []))} planned actions")
            
            print("\n✅ Planning completed!")
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse planning response: {e}")
            return False
    
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
        if not interactive_elements:
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
        """Ask LLM what action to take based on page content, user goal, and current subgoal"""
        
        # Get recent action history
        recent_actions = self.action_history[-5:] if self.action_history else []
        history_text = "\n".join([f"- {action}" for action in recent_actions]) if recent_actions else "None"
        
        # Check for repeated actions to prevent loops
        last_action = self.action_history[-1] if self.action_history else ""
        repeated_action_warning = ""
        if len(self.action_history) >= 2:
            if "type(#searchInput)" in last_action and "type(#searchInput)" in self.action_history[-2]:
                repeated_action_warning = "\n⚠️ WARNING: You just typed in the search box. DO NOT type again - use press_enter() instead!"
        
        if len([a for a in recent_actions if "type(#searchInput)" in a]) >= 2:
            repeated_action_warning = "\n🚨 CRITICAL: You've typed multiple times. STOP typing and use press_enter() immediately!"
        
        # Get current subgoal context
        current_subgoal = None
        planned_actions = []
        if self.current_subgoal_index < len(self.subgoals):
            current_subgoal = self.subgoals[self.current_subgoal_index]
            planned_actions = current_subgoal.get('actions', [])
        
        subgoal_context = ""
        if current_subgoal:
            subgoal_context = f"""
        
        CURRENT SUBGOAL (#{current_subgoal['id']}): {current_subgoal['title']}
        Description: {current_subgoal['description']}
        Planned actions for this subgoal:
        {json.dumps(planned_actions, indent=2)}
        
        Progress: Working on subgoal {self.current_subgoal_index + 1} of {len(self.subgoals)}
        """
        
        prompt = f"""
        You are a browser automation assistant following a structured plan to answer user questions.
        
        Current page:
        - Title: {page_info['title']}
        - URL: {page_info['url']}
        - Interactive elements found: {json.dumps(page_info['interactive_elements'], indent=2)}
        
        Recent actions taken:
        {history_text}
        {repeated_action_warning}
        
        User goal: {user_goal}
        {subgoal_context}
        
        Available actions:
        1. click(selector) - Click an element (use the exact selector from interactive_elements)
        2. type(selector, text) - Type text into an input field
        3. press_enter() - Press Enter key (useful after typing in search boxes)
        4. read_page() - Read and analyze the current page content to answer questions
        5. goto(url) - Navigate to a URL
        6. scroll(direction) - Scroll up or down  
        7. wait(seconds) - Wait for a specified time
        8. done() - Task completed
        
        WIKIPEDIA-ONLY STRATEGY:
        1. You start on Wikipedia - use the search box to search for the topic (e.g., "Crisis of the Roman Republic")
        2. Press enter to submit the search
        3. Click on the most relevant article from search results
        4. Use read_page() to analyze the content and answer the question
        5. If information is not found, try searching for more specific terms or related topics
        6. NEVER navigate away from Wikipedia - all research must be done within Wikipedia
        
        CRITICAL RULES: 
        - NEVER repeat the same action with the same parameters - this causes infinite loops
        - Look at your recent actions and choose the NEXT logical step
        - If you just typed something, the next action MUST be press_enter() or click() - NEVER type again
        - Follow the planned action sequence for the current subgoal
        - After typing in a search box, you MUST use press_enter() to submit - no exceptions
        - If you've already typed a search query, do NOT type again - press Enter instead
        - Only use selectors from the interactive_elements list above
        - When you complete all actions for a subgoal, move to the next subgoal
        - For questions about historical events, people, or concepts, always try to find the Wikipedia article first
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
    
    def check_subgoal_completion(self, action):
        """Check if current subgoal should be marked as complete"""
        if not self.subgoals or self.current_subgoal_index >= len(self.subgoals):
            return
            
        current_subgoal = self.subgoals[self.current_subgoal_index]
        
        # Mark subgoal complete when read_page extracts information
        if action == "read_page":
            print(f"\n✅ SUBGOAL COMPLETED: {current_subgoal['title']}")
            self.current_subgoal_index += 1
            
            if self.current_subgoal_index < len(self.subgoals):
                next_subgoal = self.subgoals[self.current_subgoal_index]
                print(f"➡️ MOVING TO NEXT SUBGOAL: {next_subgoal['title']}")
            else:
                print("🏁 ALL SUBGOALS COMPLETED - PREPARING FINAL ANSWER!")
    
    async def execute_action(self, action_data, user_goal=""):
        """Execute the action determined by the LLM"""
        action = action_data.get("action")
        params = action_data.get("parameters", {})
        
        # Add to action history
        action_summary = f"{action}({params.get('selector', '')}) - {action_data.get('reasoning', '')[:50]}..."
        self.action_history.append(action_summary)
        
        print(f"Executing: {action} with params: {params}")
        print(f"Reasoning: {action_data.get('reasoning', 'No reasoning provided')}")
        
        # Check for typing loops and block them
        if action == "type" and len(self.action_history) >= 2:
            recent_types = [a for a in self.action_history[-3:] if "type(" in a]
            if len(recent_types) >= 2:
                print("🚨 BLOCKED: Multiple typing actions detected. Forcing press_enter() instead.")
                # Override the action to press_enter
                action = "press_enter"
                params = {}
                action_data = {"action": "press_enter", "parameters": {}, "reasoning": "Auto-corrected from typing loop"}
        
        # Check if this action completes the current subgoal
        self.check_subgoal_completion(action)
        
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
                
                # Get current subgoal for context
                current_subgoal = None
                if self.current_subgoal_index < len(self.subgoals):
                    current_subgoal = self.subgoals[self.current_subgoal_index]
                
                # Extract information relevant to current subgoal
                extraction_prompt = f"""
                Extract information relevant to this subgoal: {current_subgoal['title'] if current_subgoal else 'Information extraction'}
                Description: {current_subgoal['description'] if current_subgoal else 'Extract relevant information'}
                
                Overall goal: {user_goal}
                
                Page content:
                {page_text[:6000]}...
                
                Extract only the information relevant to the current subgoal. Be specific and factual. If the information is not found, say "Information not found on this page."
                """
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.1
                )
                
                extracted_info = response.choices[0].message.content
                
                # Store the information for this subgoal
                subgoal_info = {
                    "subgoal_id": current_subgoal['id'] if current_subgoal else len(self.collected_information) + 1,
                    "subgoal_title": current_subgoal['title'] if current_subgoal else "Information extraction",
                    "information": extracted_info,
                    "page_title": await self.page.title(),
                    "page_url": self.page.url
                }
                
                self.collected_information.append(subgoal_info)
                
                print(f"\n📚 INFORMATION COLLECTED for subgoal: {subgoal_info['subgoal_title']}")
                print(f"📝 Info: {extracted_info[:100]}...")
                
                # Check if all subgoals are completed
                if self.current_subgoal_index >= len(self.subgoals) - 1:
                    await self.provide_final_answer(user_goal)
                    return True
                
                return False  # Continue to next subgoal
                
            except Exception as e:
                print(f"Failed to read page: {e}")
                return False
        elif action == "goto":
            url = params.get("url")
            
            # Validate that we're only navigating to Wikipedia
            if url and "wikipedia.org" not in url.lower():
                print(f"❌ BLOCKED: Attempted to navigate to non-Wikipedia URL: {url}")
                print("🔒 Only Wikipedia navigation is allowed. Staying on current page.")
                return False
                
            try:
                print(f"Navigating to {url}...")
                await self.page.goto(url, wait_until='networkidle', timeout=30000)
                print(f"Successfully navigated to {url}")
            except Exception as e:
                print(f"Failed to navigate to {url}: {e}")
                if "has been closed" in str(e):
                    print("Browser connection lost - this may be a remote browser timeout")
                    return True  # End automation since browser is closed
                return False
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
        print(f"\n🚀 STARTING AUTOMATION")
        print(f"Goal: {user_goal}")
        print("=" * 60)
        
        # Phase 1: Create subgoals and action plan
        if not await self.create_subgoals_and_actions(user_goal):
            print("❌ Planning phase failed. Ending automation.")
            return
        
        print(f"\n🎬 EXECUTION PHASE: Following the plan...")
        print("=" * 60)
        
        for step in range(max_steps):
            # Check if browser is still accessible
            if not await self.check_browser_health():
                print("\n❌ BROWSER CONNECTION LOST")
                print("Browser connection lost. Ending automation.")
                break
            
            try:
                # Get current page information
                page_info = await self.get_page_info()
                
                # Ask LLM for next action
                action_data = await self.ask_llm_for_action(page_info, user_goal)
                
                # Create step title based on the action
                action = action_data.get("action", "unknown")
                step_title = self.get_step_title(action, action_data.get("parameters", {}), page_info)
                
                print(f"\n🔄 STEP {step + 1}: {step_title}")
                print("=" * (len(f"STEP {step + 1}: {step_title}") + 4))
                
                # Execute the action
                is_done = await self.execute_action(action_data, user_goal)
            except Exception as e:
                print(f"\n❌ ERROR IN STEP {step + 1}")
                print(f"Error: {e}")
                if "has been closed" in str(e) or "Target page" in str(e):
                    print("Browser connection lost during step. Ending automation.")
                    break
                continue
            
            if is_done:
                print("\n✅ AUTOMATION COMPLETED SUCCESSFULLY!")
                print("🎯 Task finished!")
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
        browser = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Try connecting to remote browser via WebSocket
                print(f"Connecting to LightPanda browser (attempt {attempt + 1}/{max_retries})...")
                browser = await p.chromium.connect_over_cdp(f"wss://cloud.lightpanda.io/ws?token={token}")
                print("Successfully connected to remote browser!")
                break
                
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("All connection attempts failed. Falling back to local browser...")
                    try:
                        browser = await p.chromium.launch(headless=False)
                        print("Local browser started successfully!")
                    except Exception as local_e:
                        print(f"Failed to start local browser: {local_e}")
                        return
                else:
                    print("Retrying in 2 seconds...")
                    await asyncio.sleep(2)
        
        try:
            page = await browser.new_page()
            
            # Create LLM agent
            agent = LLMBrowserAgent()
            agent.page = page
            
            # Start on Wikipedia since Google search has issues in remote browser
            print("Navigating directly to Wikipedia for reliable search...")
            await page.goto("https://en.wikipedia.org")
            print("Wikipedia loaded - ready for search!")
            
            # Run LLM-driven automation
            if openai_key:
                # user_goal = "When did the Crisis of the Roman Republic start?"
                # user_goal = "When did the Crisis of the Roman Republic end?"
                user_goal = "Explain how the concept of 'justice' differs between Aristotelian philosophy, Confucian ethics, and Islamic jurisprudence, with specific examples."
                await agent.run_automation(user_goal)
            else:
                print("Skipping LLM automation - add OPENAI_API_KEY to .env file")
                print("Add your OpenAI API key to the .env file to enable LLM automation")
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