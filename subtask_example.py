#!/usr/bin/env python3
"""
Example showing how to use the integrated subtask tracking in main.py
"""

import asyncio
import os
from main import LLMBrowserAgent
from dotenv import load_dotenv

load_dotenv()

async def example_usage():
    """Example of how the subtask tracking works in the main automation"""
    
    print("🎯 Subtask Tracking Example")
    print("=" * 40)
    
    # The LLMBrowserAgent now automatically:
    # 1. Connects to Weaviate on initialization
    # 2. Saves all subgoals to Weaviate during automation
    # 3. Provides methods to search for similar subtasks
    
    agent = LLMBrowserAgent()
    
    # Check if Weaviate is available
    if not agent.weaviate_client:
        print("⚠️  Weaviate not configured. Please set WEAVIATE_URL and WEAVIATE_API_KEY")
        print("   The agent will still work, but subtasks won't be saved to Weaviate.")
        return
    
    print("✅ Weaviate connection ready!")
    
    # Example: Search for similar subtasks before starting automation
    print("\n🔍 Looking for similar previous tasks...")
    similar_tasks = agent.search_similar_subtasks("Wikipedia search", limit=3)
    
    if similar_tasks:
        print("Found similar tasks:")
        for task in similar_tasks:
            print(f"  - {task['task']} (similarity: {task['similarity']:.2f})")
    else:
        print("No similar tasks found - this will be a new task type")
    
    # The main automation will now automatically:
    # 1. Create subgoals based on the user's goal
    # 2. Save each subgoal to Weaviate with its actions
    # 3. Allow future searches for similar tasks
    
    print("\n🚀 Ready to run automation with subtask tracking!")
    print("When you run the main automation, it will:")
    print("  - Create subgoals for your task")
    print("  - Save them to Weaviate in the format:")
    print("    {")
    print("      'id': 1,")
    print("      'title': 'Search for topic on Wikipedia',")
    print("      'description': 'Use Wikipedia search to find the relevant article',")
    print("      'actions': [")
    print("        {'action': 'type', 'parameters': {...}, 'description': '...'},")
    print("        {'action': 'press_enter', 'parameters': {}, 'description': '...'}")
    print("      ]")
    print("    }")
    
    # Clean up
    agent.close_weaviate_connection()

if __name__ == "__main__":
    asyncio.run(example_usage())
