#!/usr/bin/env python3
"""
Test the memory system without using OpenAI tokens.
This creates a mock Weaviate setup for testing the structure.
"""

import asyncio
import json
from datetime import datetime

class MockMemorySystem:
    """Mock memory system that doesn't use OpenAI tokens"""
    
    def __init__(self):
        self.experiences = []
        print("🧪 Mock Memory System (No OpenAI tokens used)")
    
    async def store_experience(self, task_goal, domain, successful_actions, 
                             key_selectors, navigation_path, extracted_info="", 
                             success_rate=1.0, context_summary=""):
        """Store experience without using OpenAI embeddings"""
        experience = {
            "id": f"mock_{len(self.experiences)}",
            "task_goal": task_goal,
            "domain": domain,
            "successful_actions": successful_actions,
            "key_selectors": key_selectors,
            "navigation_path": navigation_path,
            "extracted_info": extracted_info,
            "success_rate": success_rate,
            "context_summary": context_summary,
            "timestamp": datetime.now().isoformat()
        }
        self.experiences.append(experience)
        print(f"✅ Stored mock experience: {task_goal}")
        return experience["id"]
    
    async def find_similar_experiences(self, task_goal, domain=None, limit=5, similarity_threshold=0.7):
        """Find similar experiences using simple text matching (no embeddings)"""
        similar = []
        for exp in self.experiences:
            # Simple keyword matching instead of semantic search
            if any(word in exp["task_goal"].lower() for word in task_goal.lower().split()):
                similar.append({
                    **exp,
                    "similarity": 0.8  # Mock similarity score
                })
        
        return similar[:limit]
    
    async def get_starting_points(self, task_goal, domain=None):
        """Get starting points without using embeddings"""
        similar = await self.find_similar_experiences(task_goal, domain)
        starting_points = []
        
        for exp in similar:
            if exp["success_rate"] > 0.7:
                starting_points.append({
                    "domain": exp["domain"],
                    "suggested_actions": exp["successful_actions"][:3],
                    "key_selectors": exp["key_selectors"],
                    "navigation_path": exp["navigation_path"][:2],
                    "similarity": exp["similarity"],
                    "success_rate": exp["success_rate"]
                })
        
        return starting_points
    
    def get_stats(self):
        """Get statistics without using tokens"""
        return {
            "total_experiences": len(self.experiences),
            "average_success_rate": sum(exp["success_rate"] for exp in self.experiences) / len(self.experiences) if self.experiences else 0
        }

async def test_mock_system():
    """Test the mock system without using any OpenAI tokens"""
    print("🧪 Testing Mock Memory System (No OpenAI tokens)")
    print("=" * 50)
    
    memory = MockMemorySystem()
    
    # Store some mock experiences
    await memory.store_experience(
        task_goal="Find information about Python programming",
        domain="stackoverflow.com",
        successful_actions=[
            {"action": "click", "selector": "#search input"},
            {"action": "type", "selector": "#search input", "text": "python"},
            {"action": "press_enter"}
        ],
        key_selectors=["#search input", ".question-summary"],
        navigation_path=["https://stackoverflow.com/questions/tagged/python"],
        extracted_info="Found Python programming questions",
        success_rate=0.9
    )
    
    await memory.store_experience(
        task_goal="Search for web development tutorials",
        domain="youtube.com",
        successful_actions=[
            {"action": "click", "selector": "input[name='search_query']"},
            {"action": "type", "selector": "input[name='search_query']", "text": "web development"},
            {"action": "press_enter"}
        ],
        key_selectors=["input[name='search_query']", ".ytd-video-renderer"],
        navigation_path=["https://youtube.com/results?search_query=web+development"],
        extracted_info="Found web development video tutorials",
        success_rate=0.85
    )
    
    # Test finding similar experiences
    print("\n🔍 Testing similarity search...")
    similar = await memory.find_similar_experiences("Find Python help")
    print(f"Found {len(similar)} similar experiences:")
    for exp in similar:
        print(f"  - {exp['task_goal']} (similarity: {exp['similarity']})")
    
    # Test starting points
    print("\n🎯 Testing starting points...")
    starting_points = await memory.get_starting_points("Learn programming")
    print(f"Found {len(starting_points)} starting points:")
    for point in starting_points:
        print(f"  - {point['domain']}: {point['suggested_actions'][:2]}")
    
    # Get stats
    stats = memory.get_stats()
    print(f"\n📊 Stats: {stats['total_experiences']} experiences, avg success: {stats['average_success_rate']:.2f}")
    
    print("\n✅ Mock system test completed (No OpenAI tokens used!)")

if __name__ == "__main__":
    asyncio.run(test_mock_system())

