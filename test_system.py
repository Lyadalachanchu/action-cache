#!/usr/bin/env python3
"""
Comprehensive system testing for the Weaviate-integrated agent
"""

import asyncio
import requests
import json
from weaviate_service import WeaviateService

async def test_weaviate_connection():
    """Test 1: Weaviate connection and basic operations"""
    print("🧪 TEST 1: Weaviate Connection")
    print("=" * 50)
    
    service = WeaviateService()
    
    if not service.is_available():
        print("❌ Weaviate not available")
        return False
    
    print("✅ Weaviate connection successful")
    
    # Test saving a task
    test_task = {
        "title": "Test connection task",
        "actions": [
            {"action": "read_page"},
            {"action": "done"}
        ]
    }
    
    result = service.save_subtask_with_actions(test_task)
    if result:
        print(f"✅ Task saved successfully (ID: {result})")
    else:
        print("❌ Failed to save task")
        return False
    
    # Test similarity search
    similar_tasks = service.search_similar_tasks("test connection", limit=3, similarity_threshold=0.5)
    print(f"✅ Found {len(similar_tasks)} similar tasks")
    
    service.close()
    return True

def test_web_ui_api():
    """Test 2: Web UI API endpoints"""
    print("\n🧪 TEST 2: Web UI API")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test if server is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Web UI server is running")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Web UI server is not running")
        print("💡 Start it with: python start_web_ui.py")
        return False
    
    # Test search endpoint
    try:
        search_data = {
            "question": "Who is older, Chris Martin or Matt Damon?"
        }
        
        response = requests.post(
            f"{base_url}/search",
            json=search_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Search endpoint working")
            print(f"   📋 Question: {result.get('question', 'N/A')}")
            print(f"   🧠 Similarity: {result.get('similarity_analysis', {}).get('reuse_decision', 'N/A')}")
            print(f"   ⚡ Plan: {result.get('execution_plan', {}).get('type', 'N/A')}")
            return True
        else:
            print(f"❌ Search failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

async def test_agent_planning():
    """Test 3: Agent planning without browser execution"""
    print("\n🧪 TEST 3: Agent Planning")
    print("=" * 50)
    
    try:
        from t_agent_working import SimpleLLMBrowserAgent
        
        agent = SimpleLLMBrowserAgent()
        
        # Test subgoal creation
        print("📋 Testing subgoal creation...")
        subgoals = await agent.create_subgoals("Who is older, Chris Martin or Matt Damon?")
        
        if subgoals:
            print(f"✅ Created {len(subgoals)} subgoals:")
            for i, subgoal in enumerate(subgoals, 1):
                print(f"  {i}. {subgoal.get('description', 'No description')}")
        else:
            print("❌ Failed to create subgoals")
            return False
        
        # Test action creation for first subgoal
        if subgoals:
            print("\n🔧 Testing action creation...")
            actions = await agent.create_actions_for_subgoal(subgoals[0].get('description', ''))
            
            if actions:
                print(f"✅ Created {len(actions)} actions:")
                for i, action in enumerate(actions, 1):
                    action_type = action.get('action', 'unknown')
                    if action_type == 'goto':
                        url = action.get('url', '')
                        print(f"  {i}. {action_type} -> {url}")
                    else:
                        print(f"  {i}. {action_type}")
            else:
                print("❌ Failed to create actions")
                return False
        
        # Test Weaviate integration
        print("\n💾 Testing Weaviate integration...")
        await agent.save_subgoals_to_weaviate(subgoals, "Test question")
        
        agent.weaviate_service.close()
        return True
        
    except Exception as e:
        print(f"❌ Agent planning test failed: {e}")
        return False

def test_similarity_learning():
    """Test 4: Similarity learning and pattern reuse"""
    print("\n🧪 TEST 4: Similarity Learning")
    print("=" * 50)
    
    service = WeaviateService()
    
    if not service.is_available():
        print("❌ Weaviate not available")
        return False
    
    # Test questions that should show similarity
    test_questions = [
        "Who is older, Chris Martin or Matt Damon?",
        "How old is Chris Martin?",
        "Which has more people, Paris or London?",
        "How many people does Paris have?"
    ]
    
    print("🔍 Testing similarity detection...")
    for i, question in enumerate(test_questions, 1):
        print(f"\n📋 Question {i}: {question}")
        
        similar_tasks = service.search_similar_tasks(question, limit=3, similarity_threshold=0.5)
        
        if similar_tasks:
            print(f"   📚 Found {len(similar_tasks)} similar tasks:")
            for j, task in enumerate(similar_tasks, 1):
                similarity = task.get('similarity', 0)
                task_name = task.get('task', 'Unknown')
                print(f"      {j}. {task_name} (similarity: {similarity:.2f})")
                
                if similarity >= 0.8:
                    print(f"         🎯 Would REUSE pattern (high similarity)")
                elif similarity >= 0.7:
                    print(f"         🔄 Would ADAPT pattern (medium similarity)")
                else:
                    print(f"         📝 Would CREATE new plan (low similarity)")
        else:
            print("   📝 No similar tasks found - would create new plan")
    
    service.close()
    return True

async def test_full_workflow():
    """Test 5: Full workflow simulation"""
    print("\n🧪 TEST 5: Full Workflow Simulation")
    print("=" * 50)
    
    try:
        from t_agent_working import SimpleLLMBrowserAgent
        
        agent = SimpleLLMBrowserAgent()
        
        # Simulate the complete workflow
        goal = "Who is older, Chris Martin or Matt Damon?"
        
        print(f"🎯 Goal: {goal}")
        print("\n📋 PHASE 1: Planning")
        subgoals = await agent.create_subgoals(goal)
        
        if not subgoals:
            print("❌ Planning failed")
            return False
        
        print(f"✅ Created {len(subgoals)} subgoals")
        
        print("\n💾 PHASE 2: Learning")
        await agent.save_subgoals_to_weaviate(subgoals, goal)
        print("✅ Subgoals saved to Weaviate")
        
        print("\n🔍 PHASE 3: Similarity Check")
        similar_tasks = agent.weaviate_service.search_similar_tasks(goal, limit=3, similarity_threshold=0.5)
        print(f"✅ Found {len(similar_tasks)} similar tasks")
        
        print("\n🎬 PHASE 4: Execution Planning")
        for i, subgoal in enumerate(subgoals[:2], 1):  # Test first 2 subgoals
            print(f"\n  Subgoal {i}: {subgoal.get('description', '')}")
            actions = await agent.create_actions_for_subgoal(subgoal.get('description', ''))
            print(f"    ✅ Generated {len(actions)} actions")
        
        print("\n🧠 PHASE 5: Answer Extraction")
        print("✅ Would extract answer from collected information")
        
        agent.weaviate_service.close()
        return True
        
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 COMPREHENSIVE SYSTEM TESTING")
    print("=" * 60)
    
    tests = [
        ("Weaviate Connection", test_weaviate_connection()),
        ("Web UI API", test_web_ui_api()),
        ("Agent Planning", test_agent_planning()),
        ("Similarity Learning", test_similarity_learning()),
        ("Full Workflow", test_full_workflow())
    ]
    
    results = []
    
    for test_name, test_coro in tests:
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())
