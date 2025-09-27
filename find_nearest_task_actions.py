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
            
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url, 
                auth_credentials=AuthApiKey(weaviate_api_key),
                additional_config=additional_config,
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

def find_nearest_task_by_text(query_text, limit=1):
    """
    Find the nearest Task(s) in vector space using text query
    
    Args:
        query_text: The text query to search with
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
        
        print(f"Searching for Task(s) matching: '{query_text}'")
        
        # First, get all tasks to compute similarities
        print("Fetching all tasks to compute similarities...")
        all_tasks_response = Task.query.fetch_objects(limit=100)
        all_tasks = all_tasks_response.objects
        
        # Try BM25 search for the query
        try:
            # Try BM25 search first
            bm25_response = Task.query.bm25(
                query=query_text,
                limit=limit,
                return_metadata=wq.MetadataQuery(score=True)
            )
            print(f"BM25 search found {len(bm25_response.objects)} tasks")
        except Exception as bm25_error:
            print(f"BM25 search failed: {bm25_error}")
            bm25_response = None
        
        # Compute similarities with all tasks
        print("\n" + "="*60)
        print("SIMILARITY SCORES WITH ALL TASKS")
        print("="*60)
        
        similarities = []
        for i, task in enumerate(all_tasks, 1):
            task_title = task.properties.get('title', 'N/A')
            
            # Simple text similarity (Jaccard similarity on words)
            query_words = set(query_text.lower().split())
            task_words = set(task_title.lower().split())
            
            if query_words and task_words:
                intersection = query_words.intersection(task_words)
                union = query_words.union(task_words)
                jaccard_similarity = len(intersection) / len(union) if union else 0
            else:
                jaccard_similarity = 0
            
            # Check if it's an exact match
            exact_match = query_text.lower() in task_title.lower()
            
            # Check if it's a partial match
            partial_match = any(word in task_title.lower() for word in query_text.lower().split())
            
            similarities.append({
                'task': task,
                'jaccard_similarity': jaccard_similarity,
                'exact_match': exact_match,
                'partial_match': partial_match
            })
            
            print(f"{i}. Task: {task_title}")
            print(f"   UUID: {task.uuid}")
            print(f"   Jaccard Similarity: {jaccard_similarity:.4f}")
            print(f"   Exact Match: {exact_match}")
            print(f"   Partial Match: {partial_match}")
            print()
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['jaccard_similarity'], reverse=True)
        
        print("="*60)
        print("TASKS RANKED BY SIMILARITY")
        print("="*60)
        for i, sim in enumerate(similarities, 1):
            print(f"{i}. {sim['task'].properties.get('title', 'N/A')} (Similarity: {sim['jaccard_similarity']:.4f})")
        
        # Return the top matching tasks with similarity scores
        top_tasks = []
        for sim in similarities[:limit]:
            task = sim['task']
            # Add similarity score to task metadata
            if not hasattr(task, 'metadata') or task.metadata is None:
                class MockMetadata:
                    def __init__(self, similarity):
                        self.similarity = similarity
                task.metadata = MockMetadata(sim['jaccard_similarity'])
            else:
                task.metadata.similarity = sim['jaccard_similarity']
            top_tasks.append(task)
        
        # Create a mock response object
        class MockResponse:
            def __init__(self, objects):
                self.objects = objects
        
        response = MockResponse(top_tasks)
        
        return response
        
    except Exception as e:
        print(f"Error performing text search: {e}")
        return None
    finally:
        client.close()

def retrieve_task_with_actions(task_uuid):
    """
    Retrieve a specific Task with its cross-referenced Actions
    
    Args:
        task_uuid: UUID of the task to retrieve
    
    Returns:
        Task with its actions
    """
    client = connect_to_weaviate()
    if not client:
        return None
    
    try:
        # Get Task collection
        Task = client.collections.get("Task")
        
        # Retrieve task with its referenced actions
        print(f"Retrieving Task {task_uuid} with its Actions...")
        # Convert UUID to string if needed
        uuid_str = str(task_uuid)
        
        # Use the same pattern as retrieve_actions_from_task.py
        task_response = Task.query.fetch_objects(
            limit=100,  # Get all tasks
            return_references=wq.QueryReference(
                link_on="actions",
                return_properties=["name", "description", "parameters_schema"]
            )
        )
        
        # Filter to find the specific task
        found_task = None
        for task in task_response.objects:
            if str(task.uuid) == uuid_str:
                found_task = task
                break
        
        if found_task:
            # Create a mock response with just this task
            class MockResponse:
                def __init__(self, task_obj):
                    self.objects = [task_obj]
            
            task_response = MockResponse(found_task)
        else:
            task_response = None
        
        return task_response
        
    except Exception as e:
        print(f"Error retrieving task with actions: {e}")
        return None
    finally:
        client.close()

def find_nearest_task_and_actions(query_text=None, query_vector=None, limit=1):
    """
    Find the nearest Task(s) and retrieve their Actions
    
    Args:
        query_text: Text query for semantic search (optional)
        query_vector: Vector query for similarity search (optional)
        limit: Number of nearest tasks to return (default: 1)
    
    Returns:
        List of tasks with their actions
    """
    # Determine search method
    if query_text:
        nearest_tasks = find_nearest_task_by_text(query_text, limit)
    elif query_vector is not None:
        nearest_tasks = find_nearest_task_by_vector(query_vector, limit)
    else:
        print("Error: Either query_text or query_vector must be provided")
        return None
    
    if not nearest_tasks or not hasattr(nearest_tasks, 'objects') or not nearest_tasks.objects:
        print("No tasks found")
        return None
    
    # Retrieve actions for each nearest task
    results = []
    for task in nearest_tasks.objects:
        print(f"Retrieving actions for task: {task.properties.get('title', 'Unknown')}")
        task_with_actions = retrieve_task_with_actions(task.uuid)
        if task_with_actions:
            results.append({
                'task': task,
                'task_with_actions': task_with_actions,
                'distance': task.metadata.distance if hasattr(task.metadata, 'distance') else None
            })
        else:
            print(f"No actions found for task {task.uuid}")
    
    return results

def format_nearest_task_results(results):
    """Format the nearest task results for output"""
    output = []
    output.append("=" * 80)
    output.append("NEAREST TASK(S) WITH ACTIONS")
    output.append("=" * 80)
    output.append("")
    
    if not results:
        output.append("No results found.")
        return "\n".join(output)
    
    for i, result in enumerate(results, 1):
        task = result['task']
        task_with_actions = result['task_with_actions']
        distance = result['distance']
        
        output.append(f"RESULT {i}:")
        output.append("-" * 40)
        output.append(f"Task Title: {task.properties.get('title', 'N/A')}")
        output.append(f"Task UUID: {task.uuid}")
        if distance is not None:
            output.append(f"Similarity Distance: {distance:.4f}")
        if hasattr(task, 'metadata') and hasattr(task.metadata, 'similarity'):
            output.append(f"Jaccard Similarity: {task.metadata.similarity:.4f}")
        output.append("")
        
        # Format referenced actions
        # Get the task object from the response
        task_obj = task_with_actions.objects[0] if hasattr(task_with_actions, 'objects') and task_with_actions.objects else None
        
        if task_obj and hasattr(task_obj, 'references') and task_obj.references and 'actions' in task_obj.references:
            actions_ref = task_obj.references['actions']
            output.append("ACTION SEQUENCE:")
            output.append("-" * 30)
            
            # Handle the cross-reference object
            if hasattr(actions_ref, 'objects'):
                actions = actions_ref.objects
                output.append(f"Total Actions: {len(actions)}")
                output.append("")
                
                for j, action_ref in enumerate(actions, 1):
                    if hasattr(action_ref, 'properties'):
                        output.append(f"{j}. Action: {action_ref.properties.get('name', 'Unknown Action')}")
                        output.append(f"   UUID: {action_ref.uuid}")
                        output.append(f"   Description: {action_ref.properties.get('description', 'N/A')}")
                        output.append(f"   Parameters: {action_ref.properties.get('parameters_schema', 'N/A')}")
                        output.append("")
            else:
                output.append("Cross-reference object found but no actions accessible.")
                output.append("")
        else:
            output.append("No actions referenced in this task.")
            output.append("")
        
        output.append("")  # Add spacing between results
    
    return "\n".join(output)

def main():
    """Main function with example usage"""
    print("Finding nearest Task(s) and retrieving their Actions...")
    
    # Example 1: Search by text query
    print("\n" + "="*50)
    print("EXAMPLE 1: Search by text query")
    print("="*50)
    
    query_text = "Fill out contact form"
    results = find_nearest_task_and_actions(query_text=query_text, limit=2)
    
    if results:
        formatted_output = format_nearest_task_results(results)
        print(formatted_output)
        
        # Save to file
        output_file = "nearest_task_actions.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"Error saving to file: {e}")
    else:
        print("No results found for text query.")
    
    # # Example 2: Search by vector (you would need to provide an actual vector)
    # print("\n" + "="*50)
    # print("EXAMPLE 2: Search by vector (commented out)")
    # print("="*50)
    # print("# To search by vector, uncomment and provide a vector:")
    # print("# query_vector = [0.1, 0.2, 0.3, ...]  # Your query vector")
    # print("# results = find_nearest_task_and_actions(query_vector=query_vector)")
    
    # # Example 3: Interactive search
    # print("\n" + "="*50)
    # print("EXAMPLE 3: Try your own search")
    # print("="*50)
    
    # Uncomment the following lines to make it interactive:
    # user_query = input("Enter a task description to search for: ")
    # if user_query.strip():
    #     results = find_nearest_task_and_actions(query_text=user_query.strip())
    #     if results:
    #         formatted_output = format_nearest_task_results(results)
    #         print(formatted_output)

if __name__ == "__main__":
    main()
