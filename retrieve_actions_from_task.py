import weaviate
import os
import json
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

def retrieve_task_with_actions():
    """Retrieve a Task with its cross-referenced Actions"""
    client = connect_to_weaviate()
    if not client:
        return None
    
    try:
        # Get Task collection
        Task = client.collections.get("Task")
        
        # Retrieve first task with its referenced actions
        print("Retrieving a Task with its Actions...")
        task_response = Task.query.fetch_objects(
            limit=1,
            return_references=wq.QueryReference(
                link_on="actions",
                return_properties=["name", "description", "parameters_schema"]
            )
        )
        
        return task_response
        
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None
    finally:
        client.close()

def format_task_with_actions(task_response):
    """Format the task with its actions for output"""
    output = []
    output.append("=" * 80)
    output.append("TASK WITH CROSS-REFERENCED ACTIONS")
    output.append("=" * 80)
    output.append("")
    
    if task_response and task_response.objects:
        task = task_response.objects[0]
        output.append("TASK:")
        output.append("-" * 40)
        output.append(f"Title: {task.properties.get('title', 'N/A')}")
        output.append(f"UUID: {task.uuid}")
        output.append("")
        
        # Format referenced actions
        if hasattr(task, 'references') and task.references and 'actions' in task.references:
            actions_ref = task.references['actions']
            output.append("ACTION SEQUENCE:")
            output.append("-" * 40)
            
            # Handle the cross-reference object
            if hasattr(actions_ref, 'objects'):
                actions = actions_ref.objects
                output.append(f"Total Actions: {len(actions)}")
                output.append("")
                
                for i, action_ref in enumerate(actions, 1):
                    if hasattr(action_ref, 'properties'):
                        output.append(f"{i}. Action: {action_ref.properties.get('name', 'Unknown Action')}")
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
    else:
        output.append("No tasks found.")
        output.append("")
    
    return "\n".join(output)

def main():
    """Main function"""
    print("Testing cross-references in Weaviate...")
    
    # Retrieve task with actions
    task_response = retrieve_task_with_actions()
    
    if not task_response:
        print("Failed to retrieve any data.")
        return
    
    # Format results
    formatted_output = format_task_with_actions(task_response)
    
    # Save to file
    output_file = "task_with_actions.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        print(f"Results saved to {output_file}")
        
        # Also print to console
        print("\n" + formatted_output)
        
    except Exception as e:
        print(f"Error saving to file: {e}")
        print("\n" + formatted_output)

if __name__ == "__main__":
    main()
