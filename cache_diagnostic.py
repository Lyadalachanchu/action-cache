#!/usr/bin/env python3
"""
Cache diagnostic tool to identify why cache performance is poor.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add bday to path
sys.path.append(os.path.join(os.getcwd(), 'bday'))

def diagnose_cache_performance():
    """Diagnose why cache performance is poor."""
    print("🔍 Cache Performance Diagnostic")
    print("=" * 60)

    load_dotenv()

    try:
        from cachedb.weaviate_repos import AnswersRepo, PlansRepo, LLMRepo, init_weaviate_collections
        from cachedb_integrations.cache_adapters import LLMCacheAdapter as LLMCache

        # Initialize
        init_weaviate_collections()
        answers_repo = AnswersRepo()
        plans_repo = PlansRepo()
        llm_repo = LLMRepo()
        llm_cache = LLMCache()

        print("✅ Weaviate connection established")

        # Test 1: Check cache hit rates
        print("\n📊 Cache Hit Rate Analysis:")
        print("-" * 40)

        test_queries = [
            "What is the population of Paris?",
            "What is the capital of France?",
            "When was Marie Curie born?",
            "What is the weather like today?"
        ]

        for query in test_queries:
            print(f"\nTesting: '{query}'")

            # Test LLM cache
            llm_hit = llm_cache.approx_get(query)
            print(f"  LLM Cache: {'HIT' if llm_hit else 'MISS'}")

            # Test answer cache
            answer_hit = answers_repo.approx_get(query)
            print(f"  Answer Cache: {'HIT' if answer_hit else 'MISS'}")

            # Test plan cache
            plan_hit = plans_repo.approx_get(query)
            print(f"  Plan Cache: {'HIT' if plan_hit else 'MISS'}")

        # Test 2: Check cache similarity thresholds
        print("\n🎯 Cache Similarity Analysis:")
        print("-" * 40)

        # Test with slight variations
        base_query = "What is the population of Paris?"
        variations = [
            "What is the population of Paris?",
            "What's the population of Paris?",
            "Population of Paris",
            "How many people live in Paris?",
            "Paris population"
        ]

        for variation in variations:
            llm_hit = llm_cache.approx_get(variation)
            print(f"'{variation}' -> LLM: {'HIT' if llm_hit else 'MISS'}")

        # Test 3: Check cache content quality
        print("\n📝 Cache Content Analysis:")
        print("-" * 40)

        # Get recent LLM cache entries
        try:
            recent_entries = llm_repo.client.collections.get('LLMCache').query.fetch_objects(
                limit=3,
                sort=wq.Sort.by_property("ts", ascending=False)
            )

            print("Recent LLM cache entries:")
            for i, entry in enumerate(recent_entries.objects, 1):
                prompt = entry.properties.get('prompt', 'No prompt')
                output = entry.properties.get('output_text', 'No output')
                model = entry.properties.get('model', 'Unknown')

                print(f"\n  Entry {i}:")
                print(f"    Model: {model}")
                print(f"    Prompt: {prompt[:100]}...")
                print(f"    Output: {output[:100]}...")

        except Exception as e:
            print(f"Error fetching LLM entries: {e}")

        # Test 4: Check cache configuration
        print("\n⚙️ Cache Configuration:")
        print("-" * 40)

        # Check similarity thresholds
        from cachedb.weaviate_repos import COSINE_THRESHOLD_PROMPT, COSINE_THRESHOLD_QA, COSINE_THRESHOLD_PLAN
        print(f"LLM Cache Threshold: {COSINE_THRESHOLD_PROMPT}")
        print(f"Answer Cache Threshold: {COSINE_THRESHOLD_QA}")
        print(f"Plan Cache Threshold: {COSINE_THRESHOLD_PLAN}")

        # Test 5: Simulate cache behavior
        print("\n🔄 Cache Behavior Simulation:")
        print("-" * 40)

        def simulate_cache_flow(question):
            print(f"\nProcessing: '{question}'")

            # Step 1: Check answer cache
            answer_hit = answers_repo.approx_get(question)
            if answer_hit:
                print("  ✅ Answer cache HIT - returning cached answer")
                return answer_hit['answer_text']

            # Step 2: Check plan cache
            plan_hit = plans_repo.approx_get(question)
            if plan_hit:
                print("  ✅ Plan cache HIT - using cached plan")
            else:
                print("  ❌ Plan cache MISS - need to create plan")

            # Step 3: Check LLM cache for intermediate steps
            llm_hit = llm_cache.approx_get(question)
            if llm_hit:
                print("  ✅ LLM cache HIT - using cached LLM response")
            else:
                print("  ❌ LLM cache MISS - need to call LLM")

            print("  🔄 Would execute full workflow...")
            return None

        # Test the simulation
        simulate_cache_flow("What is the population of Paris?")
        simulate_cache_flow("What is the population of Paris?")  # Same question

        return True

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_optimizations():
    """Suggest cache optimizations."""
    print("\n💡 Cache Optimization Suggestions:")
    print("=" * 60)

    print("1. 🎯 Lower Similarity Thresholds:")
    print("   - Current LLM threshold: 0.97 (very strict)")
    print("   - Try: 0.85-0.90 for better hit rates")
    print("   - Set: export COSINE_THRESHOLD_PROMPT=0.85")

    print("\n2. 🔧 Fix Schema Issues:")
    print("   - Answers collection has schema problems")
    print("   - Recreate collections with proper schema")

    print("\n3. 📝 Improve Cache Keys:")
    print("   - Use more consistent question normalization")
    print("   - Store both exact and semantic matches")

    print("\n4. 🚀 Enable Answer Caching:")
    print("   - Currently only LLM and Plans are cached")
    print("   - Enable answer cache for direct responses")

    print("\n5. 🔄 Test Cache Flow:")
    print("   - Run same question multiple times")
    print("   - Check if subsequent runs use cache")
    print("   - Monitor token usage reduction")

if __name__ == "__main__":
    success = diagnose_cache_performance()

    if success:
        suggest_optimizations()
        print(f"\n🎯 Run this to test cache improvements:")
        print(f"   cd /Users/mari/Desktop/action-cache/bday")
        print(f"   export COSINE_THRESHOLD_PROMPT=0.85")
        print(f"   python3 t_agent.py --show-counts 'Your question'")
    else:
        print(f"\n❌ Diagnostic failed - check Weaviate connection")
