#!/usr/bin/env python3
"""
Demo script to show the Web UI functionality
"""

import asyncio
import json
from weaviate_service import WeaviateService

async def demo_web_ui_functionality():
    """Demonstrate the Web UI functionality"""
    print("🎭 Web UI Demo - Weaviate Agent Functionality")
    print("=" * 60)
    
    # Initialize Weaviate service
    service = WeaviateService()
    
    if not service.is_available():
        print("❌ Weaviate not available - cannot run demo")
        return
    
    print("✅ Weaviate service ready!")
    
    # Demo questions
    demo_questions = [
        "Who is older, Chris Martin or Matt Damon?",
        "How old is Chris Martin?",
        "Which has more people, Paris or London?",
        "How many people does Paris have?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n🔍 DEMO QUESTION {i}: {question}")
        print("-" * 50)
        
        # Simulate the Web UI's analysis process
        print("🧠 AI Thinking Process:")
        
        # Step 1: Similarity Analysis
        similar_tasks = service.search_similar_tasks(question, limit=5, similarity_threshold=0.5)
        
        if similar_tasks:
            best_similarity = similar_tasks[0]['similarity']
            print(f"   📚 Found {len(similar_tasks)} similar tasks")
            print(f"   🎯 Best match: '{similar_tasks[0]['task']}' (similarity: {best_similarity:.2f})")
            
            if best_similarity >= 0.8:
                print("   🔄 Decision: REUSE_EXISTING_PATTERN (High similarity)")
                print("   ⚡ Estimated time: 2-3 minutes")
                print("   🎯 Confidence: High (using proven pattern)")
            elif best_similarity >= 0.7:
                print("   🔄 Decision: ADAPT_EXISTING_PATTERN (Medium similarity)")
                print("   ⚡ Estimated time: 3-4 minutes")
                print("   🎯 Confidence: Medium (adapting existing pattern)")
            else:
                print("   🔄 Decision: CREATE_NEW_PLAN (Low similarity)")
                print("   ⚡ Estimated time: 5-7 minutes")
                print("   🎯 Confidence: Medium (creating new plan)")
        else:
            print("   📝 No similar tasks found")
            print("   🔄 Decision: CREATE_NEW_PLAN")
            print("   ⚡ Estimated time: 5-7 minutes")
            print("   🎯 Confidence: Medium (creating new plan)")
        
        # Step 2: Save task for learning
        task_data = {
            'title': f"Research: {question}",
            'actions': [
                {"action": "read_page"},
                {"action": "done"}
            ]
        }
        
        result = service.save_subtask_with_actions(task_data)
        if result:
            print(f"   💾 Task saved for future learning (ID: {result})")
        else:
            print("   ❌ Failed to save task")
        
        print(f"   ✅ Analysis complete for question {i}")
    
    # Show cross-question similarity
    print(f"\n🔍 CROSS-QUESTION SIMILARITY ANALYSIS")
    print("-" * 50)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n📋 Similar tasks for: '{question}'")
        similar_tasks = service.search_similar_tasks(question, limit=3, similarity_threshold=0.5)
        
        for j, task in enumerate(similar_tasks, 1):
            print(f"   {j}. {task['task']} (similarity: {task['similarity']:.2f})")
    
    # Cleanup
    service.close()
    print(f"\n✅ Demo completed!")
    print(f"\n🌐 To use the Web UI, run: python start_web_ui.py")
    print(f"📱 Then open: http://localhost:5000")

if __name__ == "__main__":
    asyncio.run(demo_web_ui_functionality())
