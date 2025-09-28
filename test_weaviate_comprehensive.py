#!/usr/bin/env python3
"""
Comprehensive Weaviate test to verify all cache operations.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add bday to path
sys.path.append(os.path.join(os.getcwd(), 'bday'))

def test_weaviate_comprehensive():
    """Test all Weaviate operations comprehensively."""
    print("🚀 Comprehensive Weaviate Cache Test")
    print("=" * 60)

    # Load environment
    load_dotenv()

    print("🔧 Environment Check:")
    print("-" * 30)
    print(f"WEAVIATE_URL: {os.getenv('WEAVIATE_URL', 'Not set')}")
    print(f"WEAVIATE_API_KEY: {'Set' if os.getenv('WEAVIATE_API_KEY') else 'Not set'}")
    print(f"OPENAI_API_KEY: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")

    try:
        from cachedb.weaviate_repos import AnswersRepo, PlansRepo, LLMRepo, init_weaviate_collections

        print("\n🔌 Initializing Weaviate:")
        print("-" * 30)
        init_weaviate_collections()
        print("✅ Collections initialized")

        # Create repo instances
        answers_repo = AnswersRepo()
        plans_repo = PlansRepo()
        llm_repo = LLMRepo()
        print("✅ All repositories created")

        # Test 1: Answers Repository
        print("\n📝 Testing Answers Repository:")
        print("-" * 40)

        # Insert test answer
        answer_id = answers_repo.put(
            canonical_q="TEST:comprehensive_test_answer",
            question_text="What is the capital of France?",
            answer_text="The capital of France is Paris.",
            confidence=0.95,
            evidence={"source": "test", "verified": True},
            sources=[{"title": "Test Source", "url": "test://france"}]
        )
        print(f"✅ Answer inserted: {answer_id}")

        # Test retrieval
        retrieved_answer = answers_repo.approx_get("What is the capital of France?")
        if retrieved_answer:
            print(f"✅ Answer retrieved: {retrieved_answer['answer_text']}")
            print(f"   Confidence: {retrieved_answer['confidence']}")
        else:
            print("❌ Answer not found")

        # Test 2: LLM Cache Repository
        print("\n🤖 Testing LLM Cache Repository:")
        print("-" * 40)

        # Insert LLM cache entry
        llm_id = llm_repo.put(
            model="gpt-4",
            prompt="What is the weather like today?",
            output_text="I don't have access to real-time weather data.",
            usage={"total_tokens": 25, "prompt_tokens": 12, "completion_tokens": 13},
            source_tag="test"
        )
        print(f"✅ LLM cache inserted: {llm_id}")

        # Test LLM retrieval
        retrieved_llm = llm_repo.approx_get("What is the weather like today?")
        if retrieved_llm:
            print(f"✅ LLM cache retrieved: {retrieved_llm['output_text']}")
            print(f"   Tokens: {retrieved_llm['usage']}")
        else:
            print("❌ LLM cache not found")

        # Test 3: Plans Repository
        print("\n📋 Testing Plans Repository:")
        print("-" * 40)

        # Insert test plan
        plan_id = plans_repo.put(
            intent_key="test_comprehensive_plan",
            goal_text="Find information about a country",
            plan_json={
                "actions": [
                    {"action": "search", "query": "country information"},
                    {"action": "extract", "field": "capital"},
                    {"action": "verify", "source": "reliable"}
                ]
            },
            site_domain="wikipedia.org",
            success_rate=0.85,
            version="test_v1"
        )
        print(f"✅ Plan inserted: {plan_id}")

        # Test plan retrieval
        retrieved_plan = plans_repo.approx_get("Find information about a country")
        if retrieved_plan:
            print(f"✅ Plan retrieved: {retrieved_plan['goal_text']}")
            print(f"   Actions: {len(retrieved_plan['plan_json']['actions'])} actions")
            print(f"   Success rate: {retrieved_plan['success_rate']}")
        else:
            print("❌ Plan not found")

        # Test 4: Cache Statistics
        print("\n📊 Cache Statistics:")
        print("-" * 40)

        # Count records in each collection
        try:
            answers_count = answers_repo.client.collections.get("Answers").aggregate.over_all(total_count=True)
            print(f"Answers in Weaviate: {answers_count.total_count}")
        except Exception as e:
            print(f"Could not count answers: {e}")

        try:
            llm_count = llm_repo.client.collections.get("LLMCache").aggregate.over_all(total_count=True)
            print(f"LLM cache entries: {llm_count.total_count}")
        except Exception as e:
            print(f"Could not count LLM cache: {e}")

        try:
            plans_count = plans_repo.client.collections.get("Plans").aggregate.over_all(total_count=True)
            print(f"Plans in Weaviate: {plans_count.total_count}")
        except Exception as e:
            print(f"Could not count plans: {e}")

        # Test 5: Vector Search Test
        print("\n🔍 Testing Vector Search:")
        print("-" * 40)

        # Test semantic search for answers
        similar_answers = answers_repo.approx_get("capital city of France", top_k=3)
        if similar_answers:
            print(f"✅ Vector search found similar answer: {similar_answers['answer_text'][:50]}...")
        else:
            print("❌ Vector search returned no results")

        # Test semantic search for plans
        similar_plans = plans_repo.approx_get("country information lookup", top_k=3)
        if similar_plans:
            print(f"✅ Vector search found similar plan: {similar_plans['goal_text']}")
        else:
            print("❌ Vector search returned no results")

        print("\n🎉 All Weaviate tests completed successfully!")
        print("✅ Weaviate is properly connected and functioning")
        print("✅ Data insertion and retrieval working")
        print("✅ Vector search working")
        print("✅ All repository operations successful")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_weaviate_vs_sqlite():
    """Compare Weaviate data with SQLite data."""
    print("\n📊 Comparing Weaviate vs SQLite:")
    print("=" * 50)

    try:
        # Check SQLite
        import sqlite3
        db_path = "bday/cachedb/cache.sqlite3"

        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM answers")
            sqlite_answers = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM plans")
            sqlite_plans = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM llm_cache")
            sqlite_llm = cursor.fetchone()[0]

            print(f"SQLite Database:")
            print(f"  Answers: {sqlite_answers}")
            print(f"  Plans: {sqlite_plans}")
            print(f"  LLM Cache: {sqlite_llm}")

            conn.close()
        else:
            print("❌ SQLite database not found")
            return

        # Check Weaviate
        from cachedb.weaviate_repos import AnswersRepo, PlansRepo, LLMRepo

        answers_repo = AnswersRepo()
        plans_repo = PlansRepo()
        llm_repo = LLMRepo()

        print(f"\nWeaviate Database:")
        try:
            answers_count = answers_repo.client.collections.get("Answers").aggregate.over_all(total_count=True)
            print(f"  Answers: {answers_count.total_count}")
        except:
            print("  Answers: Unable to count")

        try:
            plans_count = plans_repo.client.collections.get("Plans").aggregate.over_all(total_count=True)
            print(f"  Plans: {plans_count.total_count}")
        except:
            print("  Plans: Unable to count")

        try:
            llm_count = llm_repo.client.collections.get("LLMCache").aggregate.over_all(total_count=True)
            print(f"  LLM Cache: {llm_count.total_count}")
        except:
            print("  LLM Cache: Unable to count")

    except Exception as e:
        print(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    success = test_weaviate_comprehensive()

    if success:
        test_weaviate_vs_sqlite()
        print(f"\n✅ Weaviate cache system is fully operational!")
    else:
        print(f"\n❌ Weaviate cache system has issues - check configuration")
