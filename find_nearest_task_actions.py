import weaviate
import os
import json
import numpy as np
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
        print(f"Connecting to Weaviate Cloud: {weaviate_url}")
        try:
            additional_config = wvc.AdditionalConfig(
                timeout=wvc.Timeout(init=30)
            )
            
            # Get OpenAI API key for vectorization
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
                skip_init_checks=True
            )
            print("Successfully connected to Weaviate Cloud!")
            return client
        except Exception as e:
            print(f"Failed to connect to Weaviate Cloud: {e}")
            return None
    else:
        print("No Weaviate Cloud credentials found.")
        return None

def find_nearest_task_by_vector(query_vector, limit=1):
    """
    Find the nearest Task(s) in vector space using cosine similarity
    
    Args:
        query_vector: The query vector to search with
        limit: Number of nearest tasks to return (default: 1)
    
    Returns:
        Query response with nearest tasks
    """
    client = connect_to_weaviate()
    if not client:
        return None
    
    try:
        # Get Task collection
        Task = client.collections.get("Task")
        
        # Perform vector similarity search
        print(f"Searching for nearest Task(s) in vector space...")
        response = Task.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            return_metadata=wq.MetadataQuery(distance=True)  # Include distance in results
        )
        
        return response
        
    except Exception as e:
        print(f"Error performing vector search: {e}")
        return None
    finally:
        client.close()

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

def format_result(result):
    """Format the result for output"""
    if not result:
        return "No results found."
    
    task = result['task']
    task_with_actions = result['task_with_actions']
    distance = result['distance']
    
    output = []
    output.append(f"Task: {task.properties.get('title', 'N/A')}")
    if distance is not None:
        output.append(f"Distance: {distance:.4f}")
    output.append("")
    
    # Get actions
    task_obj = task_with_actions.objects[0]
    if task_obj and hasattr(task_obj, 'references') and task_obj.references and 'actions' in task_obj.references:
        actions_ref = task_obj.references['actions']
        if hasattr(actions_ref, 'objects'):
            actions = actions_ref.objects
            output.append("Actions:")
            for i, action_ref in enumerate(actions, 1):
                if hasattr(action_ref, 'properties'):
                    output.append(f"{i}. {action_ref.properties.get('name', 'Unknown')}")
    
    return "\n".join(output)

def main():
    """Find the nearest Task and show its Actions"""
    client = connect_to_weaviate()
    if not client:
        return
    
    try:
        query_text = "Fill out contact form"
        result = find_nearest_task_and_actions(client, query_text)
        
        if result:
            formatted_output = format_result(result)
            print(formatted_output)
            
            # Save to file
            with open("nearest_task_actions.txt", 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            print("Results saved to nearest_task_actions.txt")
        else:
            print("No results found.")
    finally:
        client.close()

if __name__ == "__main__":
    main()
