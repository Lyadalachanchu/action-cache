#!/usr/bin/env python3
"""
Test script to verify Weaviate connection and cache operations.
"""

import os
import sys
import time
import json
from datetime import datetime

# Add bday to path
sys.path.append(os.path.join(os.getcwd(), 'bday'))

def test_weaviate_connection():
    """Test if Weaviate connection works."""
    print("🔌 Testing Weaviate Connection:")
    print("=" * 50)

    try:
        from cachedb.weaviate_repos import (
            AnswersRepo,
            PlansRepo,
            LLMRepo,
            init_weaviate_collections
        )

        print("✅ Successfully imported Weaviate repos")

        # Initialize collections
        print("🔄 Initializing Weaviate collections...")
        init_weaviate_collections()
        print("✅ Collections initialized")

        # Test creating repo instances
        answers_repo = AnswersRepo()
        plans_repo = PlansRepo()
        llm_repo = LLMRepo()

        print("✅ All repo instances created successfully")

        return True, answers_repo, plans_repo, llm_repo

    except Exception as e:
        print(f"❌ Weaviate connection failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False, None, None, None

def test_weaviate_operations(answers_repo, plans_repo, llm_repo):
    """Test basic Weaviate operations."""
    print("\n🧪 Testing Weaviate Operations:")
    print("=" * 50)

    if not all([answers_repo, plans_repo, llm_repo]):
        print("❌ Cannot test operations - repos not available")
        return False

    try:
        # Test 1: Add a test answer
        print("1. Testing answer insertion...")
        test_answer_id = answers_repo.put(
            canonical_q="TEST:weaviate_connection_test",
            question_text="Is Weaviate connection working?",
            answer_text="Yes, Weaviate connection is working!",
            confidence=1.0,
            evidence={"test": "weaviate_connection"},
            sources=[{"title": "Test", "url": "test://weaviate"}]
        )
        print(f"   ✅ Answer inserted with ID: {test_answer_id}")

        # Test 2: Retrieve the answer
        print("2. Testing answer retrieval...")
        retrieved_answer = answers_repo.get_by_canonical_q("TEST:weaviate_connection_test")
        if retrieved_answer:
            print(f"   ✅ Answer retrieved: {retrieved_answer['answer_text']}")
        else:
            print("   ❌ Answer not found")

        # Test 3: Add a test plan
        print("3. Testing plan insertion...")
        test_plan_id = plans_repo.put(
            intent_key="test_weaviate_plan",
            goal_text="Test Weaviate plan functionality",
            plan_json={
                "actions": [
                    {"action": "test", "description": "Test Weaviate connection"}
                ]
            },
            site_domain="test.com",
            success_rate=1.0,
            version="test_v1"
        )
        print(f"   ✅ Plan inserted with ID: {test_plan_id}")

        # Test 4: Retrieve the plan
        print("4. Testing plan retrieval...")
        retrieved_plan = plans_repo.get_by_intent_key("test_weaviate_plan")
        if retrieved_plan:
            print(f"   ✅ Plan retrieved: {retrieved_plan['goal_text']}")
        else:
            print("   ❌ Plan not found")

        # Test 5: Add a test LLM cache entry
        print("5. Testing LLM cache insertion...")
        test_llm_id = llm_repo.put(
            model="test-model",
            prompt_text="Test Weaviate LLM cache",
            output_text="Weaviate LLM cache is working!",
            usage={"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
            source_tag="test"
        )
        print(f"   ✅ LLM cache entry inserted with ID: {test_llm_id}")

        # Test 6: Retrieve the LLM cache entry
        print("6. Testing LLM cache retrieval...")
        retrieved_llm = llm_repo.get_by_prompt_hash(
            llm_repo._hash_prompt("Test Weaviate LLM cache")
        )
        if retrieved_llm:
            print(f"   ✅ LLM cache retrieved: {retrieved_llm['output_text']}")
        else:
            print("   ❌ LLM cache not found")

        return True

    except Exception as e:
        print(f"❌ Weaviate operations failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def test_weaviate_vs_sqlite():
    """Compare Weaviate vs SQLite data."""
    print("\n📊 Comparing Weaviate vs SQLite Data:")
    print("=" * 50)

    try:
        # Check SQLite data
        import sqlite3
        db_path = "bday/cachedb/cache.sqlite3"

        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Count SQLite records
            cursor.execute("SELECT COUNT(*) FROM answers")
            sqlite_answers = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM plans")
            sqlite_plans = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM llm_cache")
            sqlite_llm = cursor.fetchone()[0]

            print(f"SQLite records:")
            print(f"  Answers: {sqlite_answers}")
            print(f"  Plans: {sqlite_plans}")
            print(f"  LLM Cache: {sqlite_llm}")

            conn.close()
        else:
            print("❌ SQLite database not found")
            return False

        # Check Weaviate data
        from cachedb.weaviate_repos import AnswersRepo, PlansRepo, LLMRepo

        answers_repo = AnswersRepo()
        plans_repo = PlansRepo()
        llm_repo = LLMRepo()

        # Count Weaviate records (approximate)
        print(f"\nWeaviate records (approximate):")
        try:
            weaviate_answers = answers_repo.client.collections.get("Answers").aggregate.over_all(total_count=True)
            print(f"  Answers: {weaviate_answers.total_count}")
        except:
            print("  Answers: Unable to count")

        try:
            weaviate_plans = plans_repo.client.collections.get("Plans").aggregate.over_all(total_count=True)
            print(f"  Plans: {weaviate_plans.total_count}")
        except:
            print("  Plans: Unable to count")

        try:
            weaviate_llm = llm_repo.client.collections.get("LLMCache").aggregate.over_all(total_count=True)
            print(f"  LLM Cache: {weaviate_llm.total_count}")
        except:
            print("  LLM Cache: Unable to count")

        return True

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False

def test_weaviate_environment():
    """Check Weaviate environment configuration."""
    print("\n🔧 Checking Weaviate Environment:")
    print("=" * 50)

    # Check environment variables
    env_vars = [
        "WEAVIATE_URL",
        "WEAVIATE_API_KEY",
        "WEAVIATE_CLUSTER_URL",
        "WEAVIATE_CLUSTER_API_KEY"
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            # Mask the API key for security
            if "API_KEY" in var:
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"✅ {var}: {masked_value}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")

    # Check if .env file exists
    env_file = "bday/.env"
    if os.path.exists(env_file):
        print(f"✅ .env file found: {env_file}")

        # Read .env file and check for Weaviate config
        with open(env_file, 'r') as f:
            content = f.read()
            if "WEAVIATE" in content.upper():
                print("✅ Weaviate configuration found in .env")
            else:
                print("⚠️  No Weaviate configuration found in .env")
    else:
        print(f"❌ .env file not found: {env_file}")

def main():
    """Main test function."""
    print("🚀 Weaviate Cache Testing Tool")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check environment
    test_weaviate_environment()

    # Test connection
    success, answers_repo, plans_repo, llm_repo = test_weaviate_connection()

    if success:
        # Test operations
        operations_success = test_weaviate_operations(answers_repo, plans_repo, llm_repo)

        if operations_success:
            print("\n✅ All Weaviate tests passed!")
        else:
            print("\n❌ Some Weaviate operations failed")

        # Compare with SQLite
        test_weaviate_vs_sqlite()

    else:
        print("\n❌ Weaviate connection failed - check configuration")

    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
