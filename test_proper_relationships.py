#!/usr/bin/env python3
"""
Test script to demonstrate proper Task-Action relationships
"""

import asyncio
from weaviate_service import WeaviateService

async def test_proper_relationships():
    """Test the proper Task-Action relationship structure"""
    print("🧪 Testing Proper Task-Action Relationships")
    print("=" * 60)
    
    # Initialize service
    service = WeaviateService()
    
    if not service.is_available():
        print("❌ Weaviate service not available")
        return
    
    print("✅ Weaviate service ready!")
    
    # Test 1: Get available actions
    print("\n🔧 Available Actions:")
    actions = service.get_available_actions()
    for action in actions:
        print(f"  • {action['name']}: {action['description']}")
    
    # Test 2: Create a task with proper action references
    print("\n💾 Creating task with proper action references...")
    
    # Example subtask data (as it comes from the agent)
    subtask_data = {
        "title": "Navigate to Chris Martin Wikipedia page",
        "actions": [
            {"action": "goto", "url": "https://en.wikipedia.org/wiki/Chris_Martin"},
            {"action": "read_page"}
        ]
    }
    
    # Save the subtask with proper relationships
    result = service.save_subtask_with_actions(subtask_data)
    if result:
        print(f"✅ Created task with ID: {result}")
        
        # Test 3: Retrieve the task with its actions
        print("\n🔍 Retrieving task with actions...")
        task_with_actions = service.get_task_with_actions(result)
        if task_with_actions:
            print(f"📋 Task: {task_with_actions['title']}")
            print(f"🔧 Actions:")
            for action in task_with_actions['actions']:
                print(f"  • {action['name']}: {action['description']}")
        else:
            print("❌ Failed to retrieve task with actions")
    else:
        print("❌ Failed to create task")
    
    # Test 4: Create another task with different actions
    print("\n💾 Creating second task...")
    subtask_data_2 = {
        "title": "Extract birth date from Chris Martin page",
        "actions": [
            {"action": "read_page"}
        ]
    }
    
    result_2 = service.save_subtask_with_actions(subtask_data_2)
    if result_2:
        print(f"✅ Created second task with ID: {result_2}")
    
    # Test 5: Search for similar tasks
    print("\n🔍 Testing similarity search...")
    similar_tasks = service.search_similar_tasks("Chris Martin", limit=3, similarity_threshold=0.5)
    print(f"📚 Found {len(similar_tasks)} similar tasks:")
    for i, task in enumerate(similar_tasks, 1):
        print(f"  {i}. {task['task']} (similarity: {task['similarity']:.2f})")
    
    # Cleanup
    service.close()
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_proper_relationships())
