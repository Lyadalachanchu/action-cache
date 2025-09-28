#!/usr/bin/env python3
"""
Test script for Weaviate integration with t_agent
"""

import asyncio
import os
from dotenv import load_dotenv
from weaviate_service import WeaviateService

load_dotenv()

async def test_weaviate_integration():
    """Test the Weaviate integration"""
    print("🧪 Testing Weaviate Integration")
    print("=" * 50)
    
    # Check environment variables
    weaviate_url = os.getenv('WEAVIATE_URL')
    weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    print(f"Environment Variables:")
    print(f"  WEAVIATE_URL: {'Set' if weaviate_url else 'Not set'}")
    print(f"  WEAVIATE_API_KEY: {'Set' if weaviate_api_key else 'Not set'}")
    print(f"  OPENAI_API_KEY: {'Set' if openai_key else 'Not set'}")
    print()
    
    # Test Weaviate service
    service = WeaviateService()
    print(f"Weaviate Service Available: {service.is_available()}")
    
    if service.is_available():
        print("✅ Weaviate service is ready!")
        
        # Test saving a task
        test_task = {
            "title": "Test task",
            "actions": [{"action": "goto", "url": "https://example.com"}],
            "description": "This is a test task",
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
        
    else:
        print("❌ Weaviate service not available")
        print("   Please check your environment variables:")
        print("   - WEAVIATE_URL")
        print("   - WEAVIATE_API_KEY") 
        print("   - OPENAI_API_KEY")
    
    # Cleanup
    service.close()
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_weaviate_integration())
