#!/usr/bin/env python3
"""
Test script to demonstrate subtask tracking with Weaviate integration
"""

import asyncio
import os
from main import LLMBrowserAgent
from dotenv import load_dotenv

load_dotenv()

async def test_subtask_tracking():
    """Test the subtask tracking functionality"""
    print("🧪 Testing Subtask Tracking Integration")
    print("=" * 50)
    
    # Create agent instance
    agent = LLMBrowserAgent()
    
    # Test 1: Check Weaviate connection
    if agent.weaviate_client and agent.tasks_collection:
        print("✅ Weaviate connection established")
    else:
        print("⚠️  Weaviate not available - check your environment variables")
        return
    
    # Test 2: Create sample subtasks in the format you specified
    sample_subgoals = [
        {
            "id": 1,
            "title": "Search for topic on Wikipedia",
            "description": "Use Wikipedia search to find the relevant article",
            "actions": [
                {"action": "type", "parameters": {"selector": "#searchInput", "text": "Crisis of the Roman Republic"}, "description": "Type search query in Wikipedia search box"},
                {"action": "press_enter", "parameters": {}, "description": "Submit Wikipedia search"}
            ]
        },
        {
            "id": 2,
            "title": "Find information about historical events",
            "description": "Search for specific historical information",
            "actions": [
                {"action": "type", "parameters": {"selector": "#searchInput", "text": "Roman Empire fall"}, "description": "Type search query for Roman Empire"},
                {"action": "press_enter", "parameters": {}, "description": "Submit search"},
                {"action": "click", "parameters": {"selector": "a[href*='Roman_Empire']"}, "description": "Click on Roman Empire article"}
            ]
        }
    ]
    
    # Test 3: Save subtasks to Weaviate
    print("\n💾 Saving sample subtasks...")
    for subgoal in sample_subgoals:
        agent.save_subtask_to_weaviate(subgoal)
    
    # Test 4: Search for similar subtasks
    print("\n🔍 Testing semantic search...")
    search_queries = [
        "Search for stuff on Wikipedia",
        "Find historical information",
        "Wikipedia search tasks"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        similar_tasks = agent.search_similar_subtasks(query, limit=2)
        
        if similar_tasks:
            for task in similar_tasks:
                print(f"  - {task['task']} (similarity: {task['similarity']:.2f})")
                print(f"    Actions: {len(task['actions'])} actions")
        else:
            print("  No similar tasks found")
    
    # Test 5: Close connection
    agent.close_weaviate_connection()
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_subtask_tracking())
