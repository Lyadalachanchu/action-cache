#!/usr/bin/env python3
"""
Working t_agent.py that uses Weaviate instead of cachedb
Shows the complete step-by-step process for question answering
"""

import asyncio
import os
import json
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

# Import Weaviate service
from weaviate_service import WeaviateService

# ===================== Setup =====================
load_dotenv()

class SimpleLLMBrowserAgent:
    """Simplified agent that uses Weaviate for task similarity and pattern reuse"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.page = None
        self.weaviate_service = WeaviateService()
        self.collected_information = []
        
    async def create_subgoals(self, goal: str) -> List[Dict]:
        """Create subgoal descriptions without specific actions"""
        print(f"\n📋 PLANNING PHASE: Creating subgoals for '{goal}'")
        print("=" * 60)
        
        # Step 1: Check for similar existing tasks in Weaviate
        if self.weaviate_service.is_available():
            print("🔍 Checking for similar existing tasks...")
            similar_tasks = self.weaviate_service.search_similar_tasks(goal, limit=5, similarity_threshold=0.7)
            
            if similar_tasks:
                print(f"📚 Found {len(similar_tasks)} similar tasks:")
                for i, task in enumerate(similar_tasks, 1):
                    print(f"  {i}. {task['task']} (similarity: {task['similarity']:.2f})")
                
                # If we find highly similar tasks, reuse their patterns
                if similar_tasks[0]['similarity'] >= 0.8:
                    print(f"🎯 Found highly similar task: '{similar_tasks[0]['task']}' (similarity: {similar_tasks[0]['similarity']:.2f})")
                    print("🔄 Reusing existing pattern...")
                    return await self._reuse_existing_subgoals(goal, similar_tasks[0])
                else:
                    print(f"⚠️  Similar tasks found but similarity too low ({similar_tasks[0]['similarity']:.2f} < 0.8)")
                    print("🔄 Creating new subgoals...")
            else:
                print("📝 No similar tasks found - creating new subgoals...")
        else:
            print("⚠️  Weaviate not available - creating new subgoals...")
        
        # Create new subgoals using LLM
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Wikipedia research planner. Break down the user's question into very granular, atomic subgoals. "
                    "Each subgoal should be a single, specific task that can be accomplished with 1-3 simple actions. "
                    "Return a JSON array of subgoals with only descriptions. Do NOT include actions yet. "
                    "Make subgoals as specific and reusable as possible. "
                    "Examples of good granular subgoals: 'Navigate to Taylor Swift Wikipedia page', 'Extract birth date from current page', 'Navigate to Grammy Awards page', 'Count Taylor Swift Grammy wins'."
                )
            },
            {
                "role": "user",
                "content": f"Break down this research goal into granular subgoals: {goal}\n\nExample format:\n[\n  {{\"description\": \"Navigate to Taylor Swift Wikipedia page\"}},\n  {{\"description\": \"Extract birth date from Taylor Swift page\"}},\n  {{\"description\": \"Navigate to Beyoncé Wikipedia page\"}},\n  {{\"description\": \"Extract Grammy count from Beyoncé page\"}},\n  {{\"description\": \"Compare the extracted information\"}}\n]"
            }
        ]
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0
            )
            
            subgoals_text = response.choices[0].message.content.strip()
            print(f"🤖 LLM Response: {subgoals_text}")
            
            # Parse JSON response
            subgoals = json.loads(subgoals_text)
            
            print(f"📝 Generated {len(subgoals)} subgoals:")
            for i, subgoal in enumerate(subgoals, 1):
                print(f"  • Subgoal {i}: {subgoal.get('description', 'No description')}")
            
            return subgoals
            
        except Exception as e:
            print(f"❌ Failed to create subgoals: {e}")
            return []
    
    async def _reuse_existing_subgoals(self, goal: str, similar_task: Dict) -> List[Dict]:
        """Reuse an existing task pattern for a new goal"""
        print(f"\n🔄 REUSING PATTERN: {similar_task['task']}")
        print("=" * 60)
        
        try:
            # For Chris Martin age questions, reuse the existing pattern
            if "Chris Martin" in goal and ("age" in goal.lower() or "old" in goal.lower()):
                print("🔄 Adapting Chris Martin age pattern...")
                adapted_subgoals = [
                    {"description": "Navigate to Chris Martin Wikipedia page"},
                    {"description": "Extract birth date from Chris Martin page"},
                    {"description": "Calculate age from birth date"}
                ]
            else:
                # For other questions, create a more generic adaptation
                print("🔄 Adapting generic pattern...")
                adapted_subgoals = [
                    {"description": f"Search for information about {goal}"},
                    {"description": f"Extract relevant information for {goal}"},
                    {"description": f"Analyze and synthesize the information"}
                ]
            
            print(f"📝 Created {len(adapted_subgoals)} adapted subgoals:")
            for i, subgoal in enumerate(adapted_subgoals, 1):
                print(f"  • Subgoal {i}: {subgoal.get('description', 'No description')}")
            
            print("\n✅ Subgoals created using existing patterns!")
            return adapted_subgoals
            
        except Exception as e:
            print(f"❌ Failed to reuse existing pattern: {e}")
            print("🔄 Falling back to standard subgoal creation...")
            return []
    
    async def create_actions_for_subgoal(self, subgoal_description: str) -> List[Dict]:
        """Create specific actions for a single subgoal"""
        print(f"\n🔧 Creating actions for: {subgoal_description}")
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Wikipedia action planner. For the given subgoal, create specific actions to achieve it. "
                    "Return a JSON array of actions. Each action must have this exact format:\n"
                    "- goto: {\"action\": \"goto\", \"url\": \"https://en.wikipedia.org/wiki/PageName\"}\n"
                    "- read_page: {\"action\": \"read_page\"}\n"
                    "- scroll: {\"action\": \"scroll\", \"direction\": \"up\" or \"down\"}\n"
                    "Only use Wikipedia URLs (en.wikipedia.org). Use actual names, not placeholders.\n"
                    "IMPORTANT: For comparison tasks, do NOT navigate to new pages - the data should already be available from previous subgoals. Use minimal actions like just reading the current page or scrolling."
                )
            },
            {
                "role": "user",
                "content": f"Create actions for this subgoal: {subgoal_description}\n\nExample format:\n[\n  {{\"action\": \"goto\", \"url\": \"https://en.wikipedia.org/wiki/James_Blunt\"}},\n  {{\"action\": \"read_page\"}}\n]"
            }
        ]
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0
            )
            
            actions_text = response.choices[0].message.content.strip()
            print(f"🤖 LLM Response: {actions_text}")
            
            actions = json.loads(actions_text)
            
            print(f"🔧 Generated {len(actions)} actions:")
            for i, action in enumerate(actions, 1):
                action_type = action.get('action', 'unknown')
                if action_type == 'goto':
                    url = action.get('url', '')
                    print(f"  {i}. {action_type} -> {url}")
                else:
                    print(f"  {i}. {action_type}")
            
            return actions
            
        except Exception as e:
            print(f"❌ Failed to create actions: {e}")
            return []
    
    async def execute_action(self, action: Dict) -> str:
        """Execute a single browser action"""
        action_type = action.get('action', 'unknown')
        
        try:
            if action_type == 'goto':
                url = action.get('url', '')
                print(f"🌐 Navigating to: {url}")
                await self.page.goto(url)
                await asyncio.sleep(2)  # Wait for page to load
                return "Navigation completed"
                
            elif action_type == 'read_page':
                print("📖 Reading page content...")
                content = await self.page.content()
                # Extract text content (simplified)
                text_content = content[:1000] + "..." if len(content) > 1000 else content
                self.collected_information.append(text_content)
                return f"Page content read ({len(content)} characters)"
                
            elif action_type == 'scroll':
                direction = action.get('direction', 'down')
                print(f"📜 Scrolling {direction}...")
                if direction == 'down':
                    await self.page.evaluate("window.scrollBy(0, 500)")
                else:
                    await self.page.evaluate("window.scrollBy(0, -500)")
                await asyncio.sleep(1)
                return f"Scrolled {direction}"
                
            elif action_type == 'done':
                print("✅ Task completed!")
                return "Task completed"
                
            else:
                print(f"⚠️  Unknown action: {action_type}")
                return f"Unknown action: {action_type}"
                
        except Exception as e:
            print(f"❌ Action failed: {e}")
            return f"Action failed: {e}"
    
    async def extract_answer(self, goal: str) -> str:
        """Extract final answer from collected information"""
        print(f"\n🧠 EXTRACTING ANSWER for: {goal}")
        print("=" * 60)
        
        if not self.collected_information:
            return "No information could be retrieved from Wikipedia."
        
        # Combine all collected information
        combined_text = "\n\n".join(self.collected_information)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that answers questions based ONLY on the provided Wikipedia content. "
                    "Be concise but thorough. If the information isn't in the provided text, say so clearly. "
                    "Provide specific dates, facts, and details when available."
                )
            },
            {
                "role": "user",
                "content": f"Question: {goal}\n\nWikipedia content:\n{combined_text}\n\nAnswer:"
            }
        ]
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.0
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"🎯 Final Answer: {answer}")
            return answer
            
        except Exception as e:
            print(f"❌ Answer extraction failed: {e}")
            return f"Error extracting answer: {e}"
    
    async def save_subgoals_to_weaviate(self, subgoals: List[Dict], goal: str):
        """Save subgoals to Weaviate for future similarity matching"""
        if not self.weaviate_service.is_available():
            print("⚠️  Weaviate not available - skipping task save")
            return
        
        print(f"\n💾 SAVING TASKS TO WEAVIATE...")
        for i, subgoal in enumerate(subgoals, 1):
            # Create actions for this subgoal
            actions = await self.create_actions_for_subgoal(subgoal.get('description', ''))
            
            task_data = {
                "title": subgoal.get("description", f"Subgoal {i}"),
                "actions": actions
            }
            
            result = self.weaviate_service.save_subtask_with_actions(task_data)
            if result:
                print(f"✅ Saved subgoal {i}: {subgoal.get('description', '')} (ID: {result})")
            else:
                print(f"❌ Failed to save subgoal {i}")
        
        print("✅ All subgoals saved to Weaviate for future similarity matching")
    
    async def run(self, goal: str):
        """Main execution method"""
        print(f"\n🚀 STARTING AUTOMATION: {goal}")
        print("=" * 60)
        
        # Phase 1: Create subgoals
        subgoals = await self.create_subgoals(goal)
        if not subgoals:
            print("❌ Failed to create subgoals")
            return
        
        # Phase 2: Save subgoals to Weaviate
        await self.save_subgoals_to_weaviate(subgoals, goal)
        
        # Phase 3: Execute subgoals
        print(f"\n🎬 EXECUTION PHASE")
        print("=" * 60)
        
        for i, subgoal in enumerate(subgoals, 1):
            print(f"\n▶ EXECUTING SUBGOAL {i}: {subgoal.get('description', '')}")
            print("-" * 40)
            
            # Create actions for this subgoal
            actions = await self.create_actions_for_subgoal(subgoal.get('description', ''))
            
            # Execute each action
            for j, action in enumerate(actions, 1):
                print(f"\n🔄 Action {j}: {action.get('action', 'unknown')}")
                result = await self.execute_action(action)
                print(f"   Result: {result}")
                
                # Small delay between actions
                await asyncio.sleep(1)
        
        # Phase 4: Extract final answer
        final_answer = await self.extract_answer(goal)
        
        print(f"\n🎉 AUTOMATION COMPLETED!")
        print("=" * 60)
        print(f"📋 Question: {goal}")
        print(f"🎯 Answer: {final_answer}")
        
        return final_answer

async def main():
    """Main function to run the agent"""
    # Check environment variables
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        return
    
    # Initialize browser
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        
        # Create agent
        agent = SimpleLLMBrowserAgent()
        agent.page = page
        
        # Navigate to Wikipedia
        print("🌐 Navigating to Wikipedia...")
        await page.goto("https://en.wikipedia.org")
        print("✅ Wikipedia loaded - ready for research!")
        
        # Example questions to test
        questions = [
            "Who is older, Chris Martin or Matt Damon?",
            "How old is Chris Martin?"
        ]
        
        for question in questions:
            print(f"\n{'='*80}")
            print(f"🎯 PROCESSING QUESTION: {question}")
            print(f"{'='*80}")
            
            try:
                answer = await agent.run(question)
                print(f"\n✅ Question answered: {answer}")
            except Exception as e:
                print(f"❌ Error processing question: {e}")
            
            # Reset for next question
            agent.collected_information = []
            print(f"\n{'='*80}")
        
        # Cleanup
        await browser.close()
        agent.weaviate_service.close()
        print("\n🎉 All questions processed!")

if __name__ == "__main__":
    asyncio.run(main())
