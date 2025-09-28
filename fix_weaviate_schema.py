#!/usr/bin/env python3
"""
Fix Weaviate schema issues by recreating broken collections.
"""

import os
import sys
from dotenv import load_dotenv

# Add bday to path
sys.path.append(os.path.join(os.getcwd(), 'bday'))

def fix_weaviate_schema():
    """Fix Weaviate schema by recreating broken collections."""
    print("🔧 Fixing Weaviate Schema Issues")
    print("=" * 60)

    load_dotenv()

    try:
        import weaviate
        import weaviate.classes.config as wc
        import weaviate.classes.init as wvc
        from weaviate.auth import AuthApiKey

        # Connect to Weaviate
        weaviate_url = os.getenv("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

        if not weaviate_url or not weaviate_api_key:
            print("❌ Missing Weaviate credentials")
            return False

        print(f"🔌 Connecting to Weaviate: {weaviate_url}")

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=AuthApiKey(weaviate_api_key)
        )

        print("✅ Connected to Weaviate")

        # Get OpenAI API key for vectorization
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            client.collections.create(
                name="Answers",
                vectorizer_config=wvc.Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small",
                    vectorize_collection_name=False
                ),
                properties=[
                    wc.Property(name="canonical_q", data_type=wc.DataType.TEXT),
                    wc.Property(name="question_text", data_type=wc.DataType.TEXT),
                    wc.Property(name="answer_text", data_type=wc.DataType.TEXT),
                    wc.Property(name="confidence", data_type=wc.DataType.NUMBER),
                    wc.Property(name="freshness_horizon", data_type=wc.DataType.INT),
                    wc.Property(name="ts", data_type=wc.DataType.NUMBER),
                    wc.Property(name="evidence_json", data_type=wc.DataType.TEXT),
                    wc.Property(name="sources_json", data_type=wc.DataType.TEXT),
                ]
            )
            print("✅ Created Answers collection")

            client.collections.create(
                name="Plans",
                vectorizer_config=wvc.Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small",
                    vectorize_collection_name=False
                ),
                properties=[
                    wc.Property(name="intent_key", data_type=wc.DataType.TEXT),
                    wc.Property(name="goal_text", data_type=wc.DataType.TEXT),
                    wc.Property(name="site_domain", data_type=wc.DataType.TEXT),
                    wc.Property(name="env_fingerprint", data_type=wc.DataType.TEXT),
                    wc.Property(name="plan_json", data_type=wc.DataType.TEXT),
                    wc.Property(name="success_rate", data_type=wc.DataType.NUMBER),
                    wc.Property(name="version", data_type=wc.DataType.TEXT),
                    wc.Property(name="ts", data_type=wc.DataType.NUMBER),
                ]
            )
            print("✅ Created Plans collection")

        else:
            print("❌ No OpenAI API key found - cannot create vectorized collections")
            return False

        # Test the collections
        print("\n🧪 Testing Collections:")
        print("-" * 30)

        try:
            answers_count = client.collections.get("Answers").aggregate.over_all(total_count=True)
            print(f"✅ Answers: {answers_count.total_count} entries")
        except Exception as e:
            print(f"❌ Answers: {e}")

        try:
            plans_count = client.collections.get("Plans").aggregate.over_all(total_count=True)
            print(f"✅ Plans: {plans_count.total_count} entries")
        except Exception as e:
            print(f"❌ Plans: {e}")

        try:
            llm_count = client.collections.get("LLMCache").aggregate.over_all(total_count=True)
            print(f"✅ LLMCache: {llm_count.total_count} entries")
        except Exception as e:
            print(f"❌ LLMCache: {e}")

        client.close()

        print("\n🎉 Schema fix completed!")
        print("Now test with: python3 t_agent.py --show-counts 'Your question'")

        return True

    except Exception as e:
        print(f"❌ Schema fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    fix_weaviate_schema()
