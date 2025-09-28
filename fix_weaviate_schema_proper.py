#!/usr/bin/env python3
"""
Fix Weaviate schema issues by recreating broken collections using the irvin branch approach.
"""

import os
import weaviate
import json
import uuid
from weaviate.auth import AuthApiKey
import weaviate.classes.config as wc
import weaviate.classes.init as wvc
from dotenv import load_dotenv

load_dotenv()

def ensure_drop(collection_name, client):
    """Drop collection if it exists"""
    try:
        client.collections.delete(collection_name)
        print(f"🗑️ Dropped existing collection: {collection_name}")
    except Exception as e:
        print(f"ℹ️ Collection {collection_name} doesn't exist or couldn't be dropped: {e}")

def main():
    print("🔧 Fixing Weaviate Schema (Based on irvin branch)")
    print("=" * 60)

    # Connect to Weaviate Cloud
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not weaviate_url or not weaviate_api_key:
        print("❌ Missing Weaviate credentials")
        return False

    if not openai_api_key:
        print("❌ Missing OpenAI API key for vectorization")
        return False

    print(f"🔌 Connecting to Weaviate: {weaviate_url}")

    try:
        # Configure additional settings to handle gRPC issues (from irvin branch)
        additional_config = wvc.AdditionalConfig(
            timeout=wvc.Timeout(init=30)
        )

        headers = {"X-OpenAI-Api-Key": openai_api_key}

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=AuthApiKey(weaviate_api_key),
            additional_config=additional_config,
            headers=headers,
            skip_init_checks=True  # Skip gRPC health checks
        )
        print("✅ Connected to Weaviate Cloud!")

    except Exception as e:
        print(f"❌ Failed to connect to Weaviate Cloud: {e}")
        return False

    try:
        # --- Fix Answers Collection ---
        print("\n📝 Fixing Answers collection...")
        ensure_drop("Answers", client)

        client.collections.create(
            name="Answers",
            vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(
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
        print("✅ Answers collection created successfully!")

        # --- Fix Plans Collection ---
        print("\n📋 Fixing Plans collection...")
        ensure_drop("Plans", client)

        client.collections.create(
            name="Plans",
            vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(
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
        print("✅ Plans collection created successfully!")

        # --- Test Collections ---
        print("\n🧪 Testing collections...")

        try:
            answers_count = client.collections.get("Answers").aggregate.over_all(total_count=True)
            print(f"✅ Answers: {answers_count.total_count} entries")
        except Exception as e:
            print(f"❌ Answers test failed: {e}")

        try:
            plans_count = client.collections.get("Plans").aggregate.over_all(total_count=True)
            print(f"✅ Plans: {plans_count.total_count} entries")
        except Exception as e:
            print(f"❌ Plans test failed: {e}")

        try:
            llm_count = client.collections.get("LLMCache").aggregate.over_all(total_count=True)
            print(f"✅ LLMCache: {llm_count.total_count} entries")
        except Exception as e:
            print(f"❌ LLMCache test failed: {e}")

        client.close()

        print("\n🎉 Schema fix completed successfully!")
        print("Now test with: cd bday && python3 t_agent.py --show-counts 'Your question'")

        return True

    except Exception as e:
        print(f"❌ Schema fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
