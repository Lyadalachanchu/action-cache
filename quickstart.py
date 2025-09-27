import weaviate
from weaviate.classes.init import Auth
import os
from weaviate.classes.config import Property, DataType, Configure
import json


# Best practice: store your credentials in environment variables
weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
)

# Delete existing collection if it exists
try:
    client.collections.delete("Subtask")
    print("🗑️  Deleted existing Subtask collection")
except:
    pass  # Collection doesn't exist, that's fine
tasks = client.collections.create(
    name="Subtask",
    properties=[
        Property(name="task", data_type=DataType.TEXT),
        Property(name="actions", data_type=DataType.TEXT_ARRAY),
    ],
    vector_config=[
        Configure.Vectors.text2vec_openai(
            name="task_embedding",
            source_properties=["task"],
            model="text-embedding-3-large",
            dimensions=1024
        )
    ],
    )


# Sample data
subgoals = [
    {
        "id": 1,
        "title": "Search for topic on Wikipedia",
        "description": "Use Wikipedia search to find the relevant article",
        "actions": [
            {"action": "type", "parameters": {"selector": "#searchInput", "text": "Crisis of the Roman Republic"}, "description": "Type search query in Wikipedia search box"},
            {"action": "press_enter", "parameters": {}, "description": "Submit Wikipedia search"}
        ]
    },
     {
        "id": 1,
        "title": "Dogs and cats",
        "description": "Use Wikipedia search to find the relevant article",
        "actions": [
            {"action": "type", "parameters": {"selector": "#searchInput", "text": "Crisis of the Roman Republic"}, "description": "Type search query in Wikipedia search box"},
            {"action": "press_enter", "parameters": {}, "description": "Submit Wikipedia search"}
        ]
    }
]

# Insert data
for subgoal in subgoals:
    # Prepare data for Weaviate
    weaviate_data = {
        "task": subgoal["title"],  # This will be vectorized
        "actions": [json.dumps(action) for action in subgoal["actions"]]  # Convert to string array
    }
    
    # Insert into collection
    result = tasks.data.insert(weaviate_data)
    print(f"Inserted: {subgoal['title']} (ID: {result})")

print("\n🔍 Testing semantic search...")
response = tasks.query.near_text(
    query="Dogs and cats",
    limit=3,
    return_metadata=["certainty"]
)

print(f"Found {len(response.objects)} similar tasks:")
for obj in response.objects:
    similarity = obj.metadata.certainty if obj.metadata else 0.0
    print(f"  - {obj.properties['task']} (similarity: {similarity:.2f})")
    print(obj.properties['actions'])

response = tasks.query.near_text(
    query="Search for stuff on Wikipedia",
    limit=3,
    return_metadata=["certainty"]
)

print(f"Found {len(response.objects)} similar tasks:")
for obj in response.objects:
    similarity = obj.metadata.certainty if obj.metadata else 0.0
    print(f"  - {obj.properties['task']} (similarity: {similarity:.2f})")
    print(obj.properties['actions'])


client.close()