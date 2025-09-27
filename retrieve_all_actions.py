import weaviate
import os
import json
from weaviate.auth import AuthApiKey
import weaviate.classes.config as wc
import weaviate.classes.init as wvc
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

def retrieve_data():
    """Retrieve Actions and Tasks from Weaviate"""
    client = connect_to_weaviate()
    if not client:
        return None, None
    
    try:
        # Get collections
        Action = client.collections.get("Action")
        Task = client.collections.get("Task")
        
        # Retrieve all actions
        print("Retrieving all Actions...")
        actions_response = Action.query.fetch_objects()
        
        # Retrieve all tasks
        print("Retrieving all Tasks...")
        tasks_response = Task.query.fetch_objects()
        
        return actions_response, tasks_response
        
    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None, None
    finally:
        client.close()

def format_results(actions_response, tasks_response):
    """Format the results for output"""
    output = []
    output.append("=" * 80)
    output.append("WEAVIATE DATABASE RETRIEVAL RESULTS")
    output.append("=" * 80)
    output.append("")
    
    # Format Actions
    if actions_response and actions_response.objects:
        output.append("ACTIONS COLLECTION:")
        output.append("-" * 40)
        output.append(f"Total Actions: {len(actions_response.objects)}")
        output.append("")
        
        for i, action in enumerate(actions_response.objects, 1):
            output.append(f"{i}. Action: {action.properties.get('name', 'N/A')}")
            output.append(f"   UUID: {action.uuid}")
            output.append(f"   Description: {action.properties.get('description', 'N/A')}")
            output.append(f"   Parameters: {action.properties.get('parameters_schema', 'N/A')}")
            output.append("")
    
    # Format Tasks
    if tasks_response and tasks_response.objects:
        output.append("TASKS COLLECTION:")
        output.append("-" * 40)
        output.append(f"Total Tasks: {len(tasks_response.objects)}")
        output.append("")
        
        for i, task in enumerate(tasks_response.objects, 1):
            output.append(f"{i}. Task: {task.properties.get('title', 'N/A')}")
            output.append(f"   UUID: {task.uuid}")
            output.append("")
    
    return "\n".join(output)

def main():
    """Main function"""
    print("Starting Weaviate data retrieval...")
    
    # Retrieve data
    actions_response, tasks_response = retrieve_data()
    
    if not actions_response and not tasks_response:
        print("Failed to retrieve any data.")
        return
    
    # Format results
    formatted_output = format_results(actions_response, tasks_response)
    
    # Save to file
    output_file = "weaviate_results.txt"
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
