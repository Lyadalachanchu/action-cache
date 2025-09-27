import weaviate
import os
import json
import uuid
from weaviate.auth import AuthApiKey
import weaviate.classes.config as wc
import weaviate.classes.init as wvc
from dotenv import load_dotenv

load_dotenv()

def ensure_drop(collection_name):
    """Drop collection if it exists"""
    try:
        client.collections.delete(collection_name)
        print(f"Dropped existing collection: {collection_name}")
    except Exception as e:
        print(f"Collection {collection_name} doesn't exist or couldn't be dropped: {e}")

# Connect to Weaviate Cloud
weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

if weaviate_url and weaviate_api_key:
    print(f"Connecting to Weaviate Cloud: {weaviate_url}")
    try:
        # Configure additional settings to handle gRPC issues
        additional_config = wvc.AdditionalConfig(
            timeout=wvc.Timeout(init=30)  # Increase timeout
        )
        
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url, 
            auth_credentials=AuthApiKey(weaviate_api_key),
            additional_config=additional_config,
            skip_init_checks=True  # Skip gRPC health checks
        )
        print("Successfully connected to Weaviate Cloud!")
    except Exception as e:
        print(f"Failed to connect to Weaviate Cloud: {e}")
        print("Falling back to local connection...")
        try:
            client = weaviate.connect_to_local()
            print("Successfully connected to local Weaviate!")
        except Exception as local_e:
            print(f"Failed to connect to local Weaviate: {local_e}")
            print("Please ensure Weaviate is running locally or check your cloud credentials.")
            exit(1)
else:
    print("No Weaviate Cloud credentials found. Trying local connection...")
    try:
        client = weaviate.connect_to_local()
        print("Successfully connected to local Weaviate!")
    except Exception as e:
        print(f"Failed to connect to local Weaviate: {e}")
        print("Please set WEAVIATE_URL and WEAVIATE_API_KEY environment variables or start a local Weaviate instance.")
        exit(1)

# --- Action Collection ---
print("\nCreating Action collection...")
ensure_drop("Action")
client.collections.create(
    name="Action",
    vector_config=wc.Configure.Vectors.self_provided(),
    properties=[
        wc.Property(name="name", data_type=wc.DataType.TEXT, skip_vectorization=True,),
        wc.Property(name="description", data_type=wc.DataType.TEXT, skip_vectorization=True,),
        wc.Property(name="parameters_schema", data_type=wc.DataType.TEXT, skip_vectorization=True,),
    ],
)
print("✅ Action collection created successfully!")

# --- Task Collection ---
print("\nCreating Task collection...")
ensure_drop("Task")
client.collections.create(
    name="Task",
    vector_config=wc.Configure.Vectors.self_provided(),
    properties=[
        wc.Property(name="title", data_type=wc.DataType.TEXT),
    ],
    references=[
        wc.ReferenceProperty(name="actions", target_collection="Action")
    ]
)
print("✅ Task collection created successfully!")

# --- Insert All Supported Actions ---
print("\nInserting all supported Actions...")
Action = client.collections.get("Action")

# Define all supported actions with their parameters
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

# Insert all actions
for action in actions:
    Action.data.insert(action, uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"action:{action['name']}:v1").hex)
    print(f"✅ Inserted action: {action['name']}")

print(f"✅ All {len(actions)} Actions inserted successfully!")

# --- Insert Example Tasks ---
print("\nInserting example Tasks...")
Task = client.collections.get("Task")

# Get action UUIDs for referencing
action_uuids = {}
for action in actions:
    action_uuid = uuid.uuid5(uuid.NAMESPACE_URL, f"action:{action['name']}:v1").hex
    action_uuids[action['name']] = action_uuid

# Task 1: Search for celebrity birth date
task_search_actions = [
    action_uuids["goto"],
    action_uuids["type"],
    action_uuids["press_enter"],
    action_uuids["wait"],
    action_uuids["read_page"],
    action_uuids["done"]
]

task_search = {
    "title": "Search for celebrity birth date"
}

# Task 2: Fill out a contact form
task_form_actions = [
    action_uuids["goto"],
    action_uuids["type"],
    action_uuids["type"],
    action_uuids["type"],
    action_uuids["click"],
    action_uuids["wait"],
    action_uuids["done"]
]

task_form = {
    "title": "Fill out contact form"
}

# Task 3: Navigate and scroll through content
task_scroll_actions = [
    action_uuids["goto"],
    action_uuids["wait"],
    action_uuids["scroll"],
    action_uuids["wait"],
    action_uuids["scroll"],
    action_uuids["read_page"],
    action_uuids["done"]
]

task_scroll = {
    "title": "Navigate and scroll through page"
}

# Insert all tasks
tasks_data = [
    (task_search, task_search_actions),
    (task_form, task_form_actions),
    (task_scroll, task_scroll_actions)
]

for task_data, actions_refs in tasks_data:
    Task.data.insert(
        properties=task_data,
        references={"actions": actions_refs},
        uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"task:{task_data['title'].lower().replace(' ', '_')}:v1").hex
    )
    print(f"✅ Inserted task: {task_data['title']}")

print(f"✅ All {len(tasks_data)} Tasks inserted successfully!")

print("\n🎉 Collections created with text-embedding-3-large and sample objects inserted.")
client.close()
print("Connection closed.")
