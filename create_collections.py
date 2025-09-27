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
        
        # Get OpenAI API key for vectorizer
        openai_api_key = os.getenv("OPENAI_TOKEN")
        headers = {}
        if openai_api_key:
            headers["X-OpenAI-Api-Key"] = openai_api_key
            print(f"✅ OpenAI API key found and will be used for vectorization")
        else:
            print("⚠️  No OPENAI_TOKEN found - vectorization may fail")
        
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url, 
            auth_credentials=AuthApiKey(weaviate_api_key),
            additional_config=additional_config,
            headers=headers,
            skip_init_checks=True  # Skip gRPC health checks
        )
        print("Successfully connected to Weaviate Cloud!")
    except Exception as e:
        print(f"Failed to connect to Weaviate Cloud: {e}")
        print("Falling back to local connection...")
        try:
            # Get OpenAI API key for vectorizer
            openai_api_key = os.getenv("OPENAI_TOKEN")
            headers = {}
            if openai_api_key:
                headers["X-OpenAI-Api-Key"] = openai_api_key
                print(f"✅ OpenAI API key found and will be used for vectorization")
            else:
                print("⚠️  No OPENAI_TOKEN found - vectorization may fail")
            
            client = weaviate.connect_to_local(headers=headers)
            print("Successfully connected to local Weaviate!")
        except Exception as local_e:
            print(f"Failed to connect to local Weaviate: {local_e}")
            print("Please ensure Weaviate is running locally or check your cloud credentials.")
            exit(1)
else:
    print("No Weaviate Cloud credentials found. Trying local connection...")
    try:
        # Get OpenAI API key for vectorizer
        openai_api_key = os.getenv("OPENAI_TOKEN")
        headers = {}
        if openai_api_key:
            headers["X-OpenAI-Api-Key"] = openai_api_key
            print(f"✅ OpenAI API key found and will be used for vectorization")
        else:
            print("⚠️  No OPENAI_TOKEN found - vectorization may fail")
        
        client = weaviate.connect_to_local(headers=headers)
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
    vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-large"
    ),
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

# Insert all actions and collect their actual UUIDs
action_uuids = {}
for action in actions:
    result = Action.data.insert(action, uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"action:{action['name']}:v1").hex)
    action_uuids[action['name']] = result  # Store the actual UUID returned by Weaviate
    print(f"✅ Inserted action: {action['name']} (UUID: {result})")

print(f"✅ All {len(actions)} Actions inserted successfully!")

def create_task(title, action_names, action_uuids, task_collection):
    """
    Create a task with the given title and actions.
    
    Args:
        title (str): The title of the task
        action_names (list): List of action names to include in the task
        action_uuids (dict): Dictionary mapping action names to their UUIDs
        task_collection: The Weaviate Task collection object
    
    Returns:
        str: The UUID of the created task
    """
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

# --- Insert Example Tasks ---
print("\nInserting example Tasks...")
Task = client.collections.get("Task")

# Create example tasks using the new function
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

print("\n🎉 Collections created with text-embedding-3-large and sample objects inserted.")
client.close()
print("Connection closed.")
