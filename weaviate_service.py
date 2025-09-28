#!/usr/bin/env python3
"""
Weaviate Service for task encoding and similarity extraction
"""

import os
import json
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from dotenv import load_dotenv

load_dotenv()

class WeaviateService:
    """Service class for Weaviate operations"""
    
    def __init__(self):
        self.client = None
        self.tasks_collection = None
        self.is_connected = False
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup Weaviate connection and create/connect to collections"""
        try:
            # Get environment variables
            weaviate_url = os.getenv('WEAVIATE_URL')
            weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
            openai_key = os.getenv('OPENAI_API_KEY')
            
            if not weaviate_url or not weaviate_api_key:
                print("⚠️  Weaviate credentials not found. Service disabled.")
                return
            
            if not openai_key:
                print("⚠️  OpenAI API key not found. Vectorization disabled.")
                return
            
            print(f"🔗 Connecting to Weaviate...")
            print(f"   URL: {weaviate_url}")
            print(f"   API Key: {'Set' if weaviate_api_key else 'Not set'}")
            
            # Set OpenAI API key for Weaviate
            os.environ['OPENAI_APIKEY'] = openai_key
            
            # Connect to Weaviate
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_url,
                auth_credentials=Auth.api_key(weaviate_api_key),
                headers={"X-Openai-Api-Key": openai_key}
            )
            
            # Setup collections
            self._setup_collections()
            self.is_connected = True
            print("✅ Weaviate service connected successfully!")
            
        except Exception as e:
            print(f"❌ Failed to setup Weaviate: {e}")
            self.is_connected = False
    
    def _setup_collections(self):
        """Setup or connect to existing collections"""
        try:
            # Check if Task collection exists
            try:
                self.tasks_collection = self.client.collections.get("Task")
                print("📚 Found existing Task collection")
            except:
                print("🔧 Creating new Task collection...")
                self.tasks_collection = self.client.collections.create(
                    name="Task",
                    properties=[
                        Property(name="task", data_type=DataType.TEXT),
                        Property(name="actions", data_type=DataType.TEXT_ARRAY),
                        Property(name="description", data_type=DataType.TEXT),
                        Property(name="subgoal_id", data_type=DataType.INT),
                        Property(name="user_goal", data_type=DataType.TEXT),
                        Property(name="success_rate", data_type=DataType.NUMBER),
                    ],
                    vector_config=[
                        Configure.Vectors.text2vec_openai(
                            name="task_embedding",
                            source_properties=["task", "description"],
                            model="text-embedding-3-large",
                            dimensions=1024
                        )
                    ],
                )
                print("✅ Subtask collection created successfully!")
                
        except Exception as e:
            print(f"❌ Failed to setup collections: {e}")
            self.tasks_collection = None
    
    def save_task(self, task_data):
        """Save a task to Weaviate"""
        if not self.is_connected or not self.tasks_collection:
            print("⚠️  Weaviate not available - skipping task save")
            return None
        
        try:
            # Prepare data for Weaviate (matching existing schema)
            weaviate_data = {
                "title": task_data.get("title", ""),
                "task": task_data.get("title", ""),  # Also store in task field
                "description": task_data.get("description", ""),
                "subgoal_id": task_data.get("id", 0),
                "user_goal": task_data.get("user_goal", ""),
                "success_rate": task_data.get("success_rate", 0.0)
            }
            
            # Insert into collection
            result = self.tasks_collection.data.insert(weaviate_data)
            print(f"💾 Saved task: {task_data.get('title', 'Unknown')} (ID: {result})")
            return result
            
        except Exception as e:
            print(f"❌ Failed to save task: {e}")
            return None
    
    def search_similar_tasks(self, query, limit=5, similarity_threshold=0.7):
        """Search for similar tasks in Weaviate"""
        if not self.is_connected or not self.tasks_collection:
            print("⚠️  Weaviate not available - skipping similarity search")
            return []
        
        try:
            response = self.tasks_collection.query.near_text(
                query=query,
                limit=limit,
                return_metadata=["certainty"]
            )
            
            similar_tasks = []
            for obj in response.objects:
                similarity = obj.metadata.certainty if obj.metadata else 0.0
                
                # Filter by similarity threshold
                if similarity >= similarity_threshold:
                    similar_tasks.append({
                        "task": obj.properties.get('task', obj.properties.get('title', '')),
                        "actions": [],  # Actions are stored as references, not in this collection
                        "description": obj.properties.get('description', ''),
                        "user_goal": obj.properties.get('user_goal', ''),
                        "similarity": similarity,
                        "subgoal_id": obj.properties.get('subgoal_id', 0)
                    })
            
            return similar_tasks
            
        except Exception as e:
            print(f"❌ Failed to search similar tasks: {e}")
            return []
    
    def get_task_by_id(self, task_id):
        """Get a specific task by ID"""
        if not self.is_connected or not self.tasks_collection:
            return None
        
        try:
            response = self.tasks_collection.query.fetch_object_by_id(task_id)
            if response:
                return {
                    "task": response.properties['task'],
                    "actions": response.properties['actions'],
                    "description": response.properties['description'],
                    "user_goal": response.properties.get('user_goal', ''),
                    "subgoal_id": response.properties.get('subgoal_id', 0)
                }
            return None
            
        except Exception as e:
            print(f"❌ Failed to get task by ID: {e}")
            return None
    
    def update_task_success_rate(self, task_id, success_rate):
        """Update the success rate of a task"""
        if not self.is_connected or not self.tasks_collection:
            return False
        
        try:
            self.tasks_collection.data.update(
                uuid=task_id,
                properties={"success_rate": success_rate}
            )
            print(f"✅ Updated success rate for task {task_id}: {success_rate}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to update success rate: {e}")
            return False
    
    def close(self):
        """Close Weaviate connection"""
        if self.client:
            try:
                self.client.close()
                print("🔌 Weaviate connection closed")
            except Exception as e:
                print(f"⚠️  Error closing Weaviate: {e}")
    
    def is_available(self):
        """Check if Weaviate service is available"""
        return self.is_connected and self.tasks_collection is not None
