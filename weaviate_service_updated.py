#!/usr/bin/env python3
"""
Updated Weaviate Service for existing database schema
Works with Action and Task collections as defined in create_collections.py
"""

import os
import json
import weaviate
from weaviate.classes.init import Auth
from dotenv import load_dotenv

load_dotenv()

class WeaviateService:
    """Service class for Weaviate operations with existing schema"""
    
    def __init__(self):
        self.client = None
        self.tasks_collection = None
        self.actions_collection = None
        self.is_connected = False
        self._setup_connection()
    
    def _setup_connection(self):
        """Setup Weaviate connection and connect to existing collections"""
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
            
            # Connect to existing collections
            self._connect_to_collections()
            self.is_connected = True
            print("✅ Weaviate service connected successfully!")
            
        except Exception as e:
            print(f"❌ Failed to setup Weaviate: {e}")
            self.is_connected = False
    
    def _connect_to_collections(self):
        """Connect to existing Action and Task collections"""
        try:
            # Connect to Task collection
            self.tasks_collection = self.client.collections.get("Task")
            print("📚 Connected to existing Task collection")
            
            # Connect to Action collection
            self.actions_collection = self.client.collections.get("Action")
            print("📚 Connected to existing Action collection")
                
        except Exception as e:
            print(f"❌ Failed to connect to collections: {e}")
            print("Please run create_collections.py first to create the collections")
            self.tasks_collection = None
            self.actions_collection = None
    
    def save_task(self, task_data):
        """Save a task to Weaviate using existing schema"""
        if not self.is_connected or not self.tasks_collection:
            print("⚠️  Weaviate not available - skipping task save")
            return None
        
        try:
            # Prepare data for Weaviate (matching existing schema)
            weaviate_data = {
                "title": task_data.get("title", "")
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
                        "task": obj.properties.get('title', ''),
                        "actions": [],  # Actions are stored as references, not in this collection
                        "description": obj.properties.get('title', ''),
                        "user_goal": "",
                        "similarity": similarity,
                        "subgoal_id": 0
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
                    "task": response.properties.get('title', ''),
                    "actions": [],
                    "description": response.properties.get('title', ''),
                    "user_goal": "",
                    "subgoal_id": 0
                }
            return None
            
        except Exception as e:
            print(f"❌ Failed to get task by ID: {e}")
            return None
    
    def get_available_actions(self):
        """Get all available actions from the Action collection"""
        if not self.is_connected or not self.actions_collection:
            print("⚠️  Weaviate not available - cannot get actions")
            return []
        
        try:
            response = self.actions_collection.query.fetch_objects(limit=100)
            actions = []
            for obj in response.objects:
                actions.append({
                    "name": obj.properties.get('name', ''),
                    "description": obj.properties.get('description', ''),
                    "parameters_schema": obj.properties.get('parameters_schema', '{}'),
                    "uuid": str(obj.uuid)
                })
            return actions
            
        except Exception as e:
            print(f"❌ Failed to get actions: {e}")
            return []
    
    def create_task_with_actions(self, title, action_names):
        """Create a task with specific actions"""
        if not self.is_connected or not self.tasks_collection or not self.actions_collection:
            print("⚠️  Weaviate not available - cannot create task")
            return None
        
        try:
            # Get all available actions
            available_actions = self.get_available_actions()
            action_map = {action['name']: action['uuid'] for action in available_actions}
            
            # Get UUIDs for requested actions
            action_uuids = []
            for action_name in action_names:
                if action_name in action_map:
                    action_uuids.append(action_map[action_name])
                else:
                    print(f"⚠️  Action '{action_name}' not found in database")
            
            # Create the task with action references
            result = self.tasks_collection.data.insert(
                properties={"title": title},
                references={"actions": action_uuids}
            )
            
            print(f"💾 Created task: {title} with {len(action_uuids)} actions (ID: {result})")
            return result
            
        except Exception as e:
            print(f"❌ Failed to create task with actions: {e}")
            return None
    
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
