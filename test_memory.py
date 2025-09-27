#!/usr/bin/env python3
"""
Test script for the Browser Memory System.
This demonstrates how the memory system works with semantic search.
"""

import asyncio
import os
from memory_system import BrowserMemorySystem

async def test_memory_system():
    """Test the memory system with sample data"""
    print("🧠 Testing Browser Memory System")
    print("=" * 40)
    
    try:
        # Initialize memory system
        memory = BrowserMemorySystem()
        print("✅ Memory system initialized")
        
        # Store some sample experiences
        print("\n📝 Storing sample experiences...")
        
        # Experience 1: Wikipedia search
        exp1_id = await memory.store_experience(
            task_goal="Find information about Roman Empire history",
            domain="wikipedia.org",
            successful_actions=[
                {"action": "click", "selector": "#searchInput"},
                {"action": "type", "selector": "#searchInput", "text": "Roman Empire"},
                {"action": "press_enter"},
                {"action": "read_page"}
            ],
            key_selectors=["#searchInput", ".mw-search-results", ".mw-content-text"],
            navigation_path=["https://en.wikipedia.org/wiki/Roman_Empire"],
            extracted_info="The Roman Empire was the post-Republican period of ancient Rome",
            success_rate=0.9,
            context_summary="Wikipedia page with search functionality"
        )
        print(f"   Stored experience 1: {exp1_id}")
        
        # Experience 2: GitHub search
        exp2_id = await memory.store_experience(
            task_goal="Find Python web scraping libraries on GitHub",
            domain="github.com",
            successful_actions=[
                {"action": "click", "selector": "input[placeholder*='Search']"},
                {"action": "type", "selector": "input[placeholder*='Search']", "text": "python web scraping"},
                {"action": "press_enter"},
                {"action": "click", "selector": ".repo-list-item"}
            ],
            key_selectors=["input[placeholder*='Search']", ".repo-list-item", ".repo-list-item h3 a"],
            navigation_path=["https://github.com/search?q=python+web+scraping"],
            extracted_info="Found libraries like BeautifulSoup, Scrapy, Selenium",
            success_rate=0.85,
            context_summary="GitHub search results page"
        )
        print(f"   Stored experience 2: {exp2_id}")
        
        # Experience 3: Stack Overflow search
        exp3_id = await memory.store_experience(
            task_goal="Find solutions for Python async programming issues",
            domain="stackoverflow.com",
            successful_actions=[
                {"action": "click", "selector": "#search input"},
                {"action": "type", "selector": "#search input", "text": "python async await"},
                {"action": "press_enter"},
                {"action": "click", "selector": ".question-summary .question-hyperlink"}
            ],
            key_selectors=["#search input", ".question-summary", ".question-hyperlink"],
            navigation_path=["https://stackoverflow.com/questions/tagged/python+asyncio"],
            extracted_info="Found solutions for async/await patterns and event loops",
            success_rate=0.8,
            context_summary="Stack Overflow Q&A page"
        )
        print(f"   Stored experience 3: {exp3_id}")
        
        # Test semantic search
        print("\n🔍 Testing semantic search...")
        
        # Search for similar experiences
        similar_experiences = await memory.find_similar_experiences(
            "Learn about ancient Roman history",
            limit=3
        )
        
        print(f"Found {len(similar_experiences)} similar experiences:")
        for i, exp in enumerate(similar_experiences, 1):
            print(f"   {i}. {exp['task_goal']} (similarity: {exp['similarity']:.2f})")
            print(f"      Domain: {exp['domain']}")
            print(f"      Success rate: {exp['success_rate']}")
            print(f"      Key selectors: {exp['key_selectors'][:2]}...")
            print()
        
        # Test starting points
        print("🎯 Testing starting points...")
        starting_points = await memory.get_starting_points(
            "Research historical information",
            domain="wikipedia.org"
        )
        
        print(f"Found {len(starting_points)} starting points:")
        for i, point in enumerate(starting_points, 1):
            print(f"   {i}. Domain: {point['domain']}")
            print(f"      Suggested actions: {point['suggested_actions'][:2]}")
            print(f"      Key selectors: {point['key_selectors'][:2]}")
            print(f"      Similarity: {point['similarity']:.2f}")
            print()
        
        # Test cross-domain search
        print("🌐 Testing cross-domain search...")
        cross_domain_experiences = await memory.find_similar_experiences(
            "Search for programming libraries",
            limit=2
        )
        
        print(f"Found {len(cross_domain_experiences)} cross-domain experiences:")
        for i, exp in enumerate(cross_domain_experiences, 1):
            print(f"   {i}. {exp['task_goal']} on {exp['domain']}")
            print(f"      Similarity: {exp['similarity']:.2f}")
            print()
        
        # Get statistics
        print("📊 Memory system statistics:")
        stats = memory.get_stats()
        print(f"   Total experiences: {stats['total_experiences']}")
        print(f"   Average success rate: {stats['average_success_rate']:.2f}")
        
        print("\n✅ Memory system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing memory system: {e}")
        print("\nMake sure Weaviate is running:")
        print("1. Run: python setup_weaviate.py")
        print("2. Or start manually: docker compose up -d")

async def test_semantic_similarity():
    """Test semantic similarity with various queries"""
    print("\n🔬 Testing Semantic Similarity")
    print("=" * 30)
    
    try:
        memory = BrowserMemorySystem()
        
        test_queries = [
            "Find information about ancient Rome",
            "Search for Python programming help",
            "Look up web development tutorials",
            "Find solutions to coding problems",
            "Research historical facts"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            experiences = await memory.find_similar_experiences(query, limit=2)
            
            if experiences:
                for exp in experiences:
                    print(f"   → {exp['task_goal']} (similarity: {exp['similarity']:.2f})")
            else:
                print("   → No similar experiences found")
        
    except Exception as e:
        print(f"❌ Error testing semantic similarity: {e}")

if __name__ == "__main__":
    print("🚀 Browser Memory System Test Suite")
    print("=" * 50)
    
    # Check if Weaviate is running
    try:
        import requests
        response = requests.get('http://localhost:8080/v1/meta', timeout=5)
        if response.status_code != 200:
            raise Exception("Weaviate not responding")
    except:
        print("❌ Weaviate is not running!")
        print("Please start Weaviate first:")
        print("1. Run: python setup_weaviate.py")
        print("2. Or manually: docker compose up -d")
        exit(1)
    
    # Run tests
    asyncio.run(test_memory_system())
    asyncio.run(test_semantic_similarity())


