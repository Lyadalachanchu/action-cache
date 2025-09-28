#!/usr/bin/env python3
"""
Interactive testing script - ask questions and see the AI's thinking process
"""

import asyncio
from weaviate_service import WeaviateService

async def interactive_test():
    """Interactive test where you can ask questions and see the AI's process"""
    print("🤖 INTERACTIVE AI AGENT TEST")
    print("=" * 50)
    print("Ask questions and see how the AI thinks through them!")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    service = WeaviateService()
    
    if not service.is_available():
        print("❌ Weaviate not available - cannot test similarity")
        return
    
    print("✅ Weaviate connected - similarity detection enabled")
    print()
    
    while True:
        try:
            question = input("\n❓ Ask a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print(f"\n🧠 AI THINKING PROCESS for: '{question}'")
            print("-" * 60)
            
            # Step 1: Check for similar tasks
            print("🔍 Step 1: Checking for similar existing tasks...")
            similar_tasks = service.search_similar_tasks(question, limit=5, similarity_threshold=0.5)
            
            if similar_tasks:
                print(f"📚 Found {len(similar_tasks)} similar tasks:")
                for i, task in enumerate(similar_tasks, 1):
                    similarity = task.get('similarity', 0)
                    task_name = task.get('task', 'Unknown')
                    print(f"  {i}. {task_name} (similarity: {similarity:.2f})")
                
                best_match = similar_tasks[0]
                best_similarity = best_match.get('similarity', 0)
                
                print(f"\n🎯 Best match: '{best_match.get('task', '')}' (similarity: {best_similarity:.2f})")
                
                if best_similarity >= 0.8:
                    print("🔄 DECISION: REUSE existing pattern (high similarity)")
                    print("⚡ Estimated time: 2-3 minutes")
                    print("🎯 Confidence: High (using proven pattern)")
                elif best_similarity >= 0.7:
                    print("🔄 DECISION: ADAPT existing pattern (medium similarity)")
                    print("⚡ Estimated time: 3-4 minutes")
                    print("🎯 Confidence: Medium (adapting existing pattern)")
                else:
                    print("📝 DECISION: CREATE new plan (low similarity)")
                    print("⚡ Estimated time: 5-7 minutes")
                    print("🎯 Confidence: Medium (creating new plan)")
            else:
                print("📝 No similar tasks found")
                print("📝 DECISION: CREATE new plan")
                print("⚡ Estimated time: 5-7 minutes")
                print("🎯 Confidence: Medium (creating new plan)")
            
            # Step 2: Show what the AI would do
            print(f"\n🎬 EXECUTION PLAN:")
            if similar_tasks and similar_tasks[0].get('similarity', 0) >= 0.8:
                print("1. Load existing pattern from similar task")
                print("2. Adapt pattern for current question")
                print("3. Execute adapted pattern")
                print("4. Extract answer from results")
            elif similar_tasks and similar_tasks[0].get('similarity', 0) >= 0.7:
                print("1. Analyze similar task pattern")
                print("2. Modify pattern for current question")
                print("3. Execute modified pattern")
                print("4. Extract answer from results")
            else:
                print("1. Generate new subgoals")
                print("2. Generate actions for each subgoal")
                print("3. Execute actions sequentially")
                print("4. Extract answer from results")
            
            # Step 3: Save for learning
            print(f"\n💾 LEARNING:")
            task_data = {
                'title': f"Research: {question}",
                'actions': [
                    {"action": "read_page"},
                    {"action": "done"}
                ]
            }
            
            result = service.save_subtask_with_actions(task_data)
            if result:
                print(f"✅ Task saved for future learning (ID: {result})")
            else:
                print("❌ Failed to save task")
            
            print("\n" + "=" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    service.close()

if __name__ == "__main__":
    asyncio.run(interactive_test())
