#!/usr/bin/env python3
"""
Simple test for Weaviate integration without full agent dependencies
"""

import asyncio
from weaviate_service import WeaviateService

async def test_simple_weaviate():
    """Test Weaviate service without agent dependencies"""
    print("🧪 Simple Weaviate Test")
    print("=" * 40)
    
    # Test Weaviate service
    service = WeaviateService()
    print(f"Weaviate Available: {service.is_available()}")
    
    if service.is_available():
        print("✅ Weaviate service is working!")
        
        # Test saving a task
        test_task = {
            "title": "Simple test task",
            "description": "This is a simple test",
            "id": 1,
            "user_goal": "Test goal",
            "success_rate": 0.9
        }
        
        print("💾 Testing task save...")
        result = service.save_task(test_task)
        if result:
            print(f"✅ Task saved with ID: {result}")
        else:
            print("❌ Failed to save task")
        
        # Test similarity search
        print("🔍 Testing similarity search...")
        similar_tasks = service.search_similar_tasks("test query", limit=3)
        print(f"✅ Found {len(similar_tasks)} similar tasks")
        
        if similar_tasks:
            print("📋 Similar tasks found:")
            for i, task in enumerate(similar_tasks, 1):
                print(f"  {i}. {task['task']} (similarity: {task['similarity']:.2f})")
        
    else:
        print("❌ Weaviate service not available")
    
    # Cleanup
    service.close()
    print("\n✅ Simple test completed!")

if __name__ == "__main__":
    asyncio.run(test_simple_weaviate())
