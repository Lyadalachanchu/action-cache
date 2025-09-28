"""
Weaviate utility functions for the t_agent.py
Contains only the necessary functions without collection creation/deletion
"""
import weaviate
import os
import json
import uuid
from weaviate.auth import AuthApiKey
import weaviate.classes.config as wc
import weaviate.classes.init as wvc
import weaviate.classes.query as wq
from dotenv import load_dotenv

load_dotenv()

def connect_to_weaviate():
    """Connect to Weaviate Cloud"""
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    if weaviate_url and weaviate_api_key:
        try:
            additional_config = wvc.AdditionalConfig(
                timeout=wvc.Timeout(init=30)
            )
            
            # Get OpenAI API key for vectorization
            openai_api_key = os.getenv("OPENAI_TOKEN")
            headers = {}
            if openai_api_key:
                headers["X-OpenAI-Api-Key"] = openai_api_key
            else:
                print("⚠️  No OPENAI_TOKEN found - vectorization may fail")
            
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url, 
                auth_credentials=AuthApiKey(weaviate_api_key),
                additional_config=additional_config,
                headers=headers,
                skip_init_checks=True
            )
            return client
        except Exception as e:
            print(f"Failed to connect to Weaviate Cloud: {e}")
            return None
    else:
        print("No Weaviate Cloud credentials found.")
        return None

def find_nearest_task_by_text(client, query_text):
    """Find the nearest Task using vector similarity"""
    try:
        Task = client.collections.get("Task")
        response = Task.query.near_text(
            query=query_text,
            limit=1,
            return_metadata=wq.MetadataQuery(distance=True)
        )
        return response
    except Exception as e:
        print(f"Error: {e}")
        return None

def retrieve_task_with_actions(client, task_uuid):
    """Retrieve a Task with its Actions"""
    try:
        Task = client.collections.get("Task")
        task_response = Task.query.fetch_objects(
            limit=100,
            return_references=wq.QueryReference(
                link_on="actions",
                return_properties=["name", "description", "parameters_schema"]
            )
        )
        
        # Find the specific task
        for task in task_response.objects:
            if str(task.uuid) == str(task_uuid):
                class MockResponse:
                    def __init__(self, task_obj):
                        self.objects = [task_obj]
                return MockResponse(task)
        return None
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def find_nearest_task_and_actions(client, query_text):
    """Find the nearest Task and retrieve its Actions"""
    nearest_tasks = find_nearest_task_by_text(client, query_text)
    
    if not nearest_tasks or not nearest_tasks.objects:
        return None
    
    task = nearest_tasks.objects[0]
    task_with_actions = retrieve_task_with_actions(client, task.uuid)
    
    if task_with_actions:
        return {
            'task': task,
            'task_with_actions': task_with_actions,
            'distance': task.metadata.distance if hasattr(task.metadata, 'distance') else None
        }
    return None

def create_task_in_weaviate(client, title, action_names):
    """
    Create a task with the given title and actions in Weaviate.
    
    Args:
        client: Weaviate client instance
        title (str): The title of the task
        action_names (list): List of action names to include in the task
    
    Returns:
        str: The UUID of the created task, or None if failed
    """
    try:
        # Get existing action UUIDs from the Action collection
        action_collection = client.collections.get("Action")
        action_objects = action_collection.query.fetch_objects(limit=100)
        action_uuids = {}
        for obj in action_objects.objects:
            name = obj.properties.get('name')
            if name:
                action_uuids[name] = obj.uuid
        
        # Get the UUIDs for the specified actions
        action_refs = [action_uuids[name] for name in action_names if name in action_uuids]
        
        if not action_refs:
            print(f"⚠️ No valid actions found for task '{title}'")
            return None
        
        # Create the task
        task_collection = client.collections.get("Task")
        task_uuid = task_collection.data.insert(
            properties={"title": title},
            references={"actions": action_refs},
            uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"task:{title.lower().replace(' ', '_')}:v1").hex
        )
        
        print(f"✅ Created task: {title} with {len(action_refs)} actions")
        return task_uuid
        
    except Exception as e:
        print(f"❌ Failed to create task '{title}': {e}")
        return None

def ensure_collections_exist(client):
    """
    Ensure that Task and Action collections exist, create them if they don't.
    Does NOT drop existing collections.
    """
    try:
        # Check if collections exist
        collections = client.collections.list_all()
        # collections.list_all() returns a dict with collection names as keys
        collection_names = list(collections.keys()) if collections else []
        
        # Create Action collection if it doesn't exist
        if "Action" not in collection_names:
            print("Creating Action collection...")
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
            
            # Insert default actions
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
            
            for action in actions:
                Action.data.insert(action, uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"action:{action['name']}:v1").hex)
            print(f"✅ Inserted {len(actions)} default actions")
        
        # Create Task collection if it doesn't exist
        if "Task" not in collection_names:
            print("Creating Task collection...")
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
        
        return True
        
    except Exception as e:
        print(f"❌ Error ensuring collections exist: {e}")
        return False
