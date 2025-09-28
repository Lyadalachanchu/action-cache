#!/usr/bin/env python3
"""
Fix Weaviate schema to match irvin branch structure: Task + Action collections with cross-references.
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
    print("🔧 Fixing Weaviate Schema (Irvin Branch Style)")
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
        # --- Create Action Collection (like irvin branch) ---
        print("\n🎬 Creating Action collection...")
        ensure_drop("Action", client)

        client.collections.create(
            name="Action",
            vector_config=wc.Configure.Vectors.self_provided(),
            properties=[
                wc.Property(name="name", data_type=wc.DataType.TEXT, skip_vectorization=True),
                wc.Property(name="description", data_type=wc.DataType.TEXT, skip_vectorization=True),
                wc.Property(name="parameters_schema", data_type=wc.DataType.TEXT, skip_vectorization=True),
            ]
        )
        print("✅ Action collection created successfully!")

        # --- Create Task Collection (like irvin branch) ---
        print("\n📋 Creating Task collection...")
        ensure_drop("Task", client)

        client.collections.create(
            name="Task",
            vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(
                model="text-embedding-3-large",
                vectorize_collection_name=False
            ),
            properties=[
                wc.Property(name="title", data_type=wc.DataType.TEXT),
            ],
            references=[
                wc.ReferenceProperty(name="actions", target_collection="Action")
            ]
        )
        print("✅ Task collection created successfully!")

        # --- Insert Default Actions (like irvin branch) ---
        print("\n🎯 Inserting default Actions...")
        Action = client.collections.get("Action")

        actions = [
            {
                "name": "click",
                "description": "Click a DOM element identified by a CSS selector.",
                "parameters_schema": json.dumps({"selector": "string"}),
            },
            {
                "name": "type",
                "description": "Type text into an input field identified by a CSS selector.",
                "parameters_schema": json.dumps({"selector": "string", "text": "string"}),
            },
            {
                "name": "press_enter",
                "description": "Press the Enter key on the currently focused element.",
                "parameters_schema": json.dumps({}),
            },
            {
                "name": "read_page",
                "description": "Read and extract text content from the current page.",
                "parameters_schema": json.dumps({}),
            },
            {
                "name": "goto",
                "description": "Navigate to a specified URL.",
                "parameters_schema": json.dumps({"url": "string"}),
            },
            {
                "name": "scroll",
                "description": "Scroll the page in a specified direction.",
                "parameters_schema": json.dumps({"direction": "string"}),
            },
            {
                "name": "wait",
                "description": "Wait for a specified number of seconds.",
                "parameters_schema": json.dumps({"seconds": "number"}),
            },
            {
                "name": "done",
                "description": "Mark the task as completed and return results.",
                "parameters_schema": json.dumps({}),
            }
        ]

        # Insert all actions and collect their UUIDs
        action_uuids = {}
        for action in actions:
            result = Action.data.insert(
                action,
                uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"action:{action['name']}:v1").hex
            )
            action_uuids[action['name']] = result
            print(f"✅ Inserted action: {action['name']} (UUID: {result})")

        print(f"✅ All {len(actions)} Actions inserted successfully!")

        # --- Insert Example Tasks (like irvin branch) ---
        print("\n📝 Inserting example Tasks...")
        Task = client.collections.get("Task")

        def create_task(title, action_names, action_uuids, task_collection):
            """Create a task with the given title and actions."""
            # Get the UUIDs for the specified actions
            action_refs = [action_uuids[name] for name in action_names if name in action_uuids]

            # Create the task
            task_uuid = task_collection.data.insert(
                properties={"title": title},
                references={"actions": action_refs},
                uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"task:{title.lower().replace(' ', '_')}:v1").hex
            )

            print(f"✅ Created task: {title} with {len(action_refs)} actions")
            return task_uuid

        # Create example tasks using the function
        example_tasks = [
            {
                "title": "Search for celebrity birth date",
                "actions": ["goto", "type", "press_enter", "wait", "read_page", "done"]
            },
            {
                "title": "Fill out contact form",
                "actions": ["goto", "type", "type", "type", "click", "wait", "done"]
            },
            {
                "title": "Navigate and scroll through page",
                "actions": ["goto", "wait", "scroll", "wait", "scroll", "read_page", "done"]
            }
        ]

        # Create all tasks using the function
        created_tasks = []
        for task_config in example_tasks:
            task_uuid = create_task(
                title=task_config["title"],
                action_names=task_config["actions"],
                action_uuids=action_uuids,
                task_collection=Task
            )
            created_tasks.append(task_uuid)

        print(f"✅ All {len(created_tasks)} Tasks created successfully!")

        # --- Test Collections ---
        print("\n🧪 Testing collections...")

        try:
            action_count = client.collections.get("Action").aggregate.over_all(total_count=True)
            print(f"✅ Actions: {action_count.total_count} entries")
        except Exception as e:
            print(f"❌ Actions test failed: {e}")

        try:
            task_count = client.collections.get("Task").aggregate.over_all(total_count=True)
            print(f"✅ Tasks: {task_count.total_count} entries")
        except Exception as e:
            print(f"❌ Tasks test failed: {e}")

        client.close()

        print("\n🎉 Irvin-style schema fix completed successfully!")
        print("Now the database has Task + Action collections with cross-references")
        print("This matches the irvin branch structure for better caching")

        return True

    except Exception as e:
        print(f"❌ Schema fix failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
