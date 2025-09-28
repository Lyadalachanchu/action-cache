#!/usr/bin/env python3
# t_agent_irvin_style.py — Main automation script with irvin-style Weaviate caching
import argparse
import asyncio
import os
import json
import re
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from playwright.async_api import async_playwright

# --- Weaviate imports (irvin style)
import weaviate
from weaviate.auth import AuthApiKey
import weaviate.classes.config as wc
import weaviate.classes.init as wvc
import weaviate.classes.query as wq

# --- Provider-agnostic LLM client
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bday'))
from llm_client import LLMClient

# --- Import our core agent
from agent_core import LLMBrowserAgent, MAX_PAGE_TEXT_CHARS
from urllib.parse import quote

# ===================== Setup =====================
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATHS = [
    SCRIPT_DIR / ".env",
    SCRIPT_DIR / "bday" / ".env",
    Path.cwd() / ".env",
]
for env_path in DEFAULT_ENV_PATHS:
    if env_path.exists():
        load_dotenv(env_path, override=False)
        break
else:
    load_dotenv()  # fallback to standard search

# Initialize LLM client and Weaviate connection
llm_client = LLMClient()
weaviate_client = None

# Prefer OpenAI as LLM (for reliable usage accounting and JSON)
if llm_client.provider.lower() != "openai":
    print(f"⚠️ LLM provider is '{llm_client.provider}'. To force OpenAI, set:")
    print("   export OPENAI_API_KEY=sk-... ; export OPENAI_MODEL=gpt-4o-mini ; export FORCE_PROVIDER=openai")

# Global tracking
RUN_TOKENS = {"prompt": 0, "completion": 0, "total": 0}
EXECUTION_START_TIME = 0
TOTAL_EXECUTION_TIME = 0

# ===================== Weaviate Functions (Irvin Style) =====================

def connect_to_weaviate():
    """Connect to Weaviate using irvin branch approach"""
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    if weaviate_url and weaviate_api_key:
        try:
            additional_config = wvc.AdditionalConfig(
                timeout=wvc.Timeout(init=30)
            )

            # Get OpenAI API key for vectorization
            openai_api_key = os.getenv("OPENAI_API_KEY")
            headers = {}
            if openai_api_key:
                headers["X-OpenAI-Api-Key"] = openai_api_key

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
    """Find the nearest Task using vector similarity (irvin style)"""
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
    """Retrieve a Task with its Actions (irvin style)"""
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
    """Find the nearest Task and retrieve its Actions (irvin style)"""
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

def _serialize_action(action: Dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True)


def _ensure_action_object(client, action: Dict[str, Any]) -> Optional[str]:
    """Return UUID of action object matching the payload, creating it if needed."""
    try:
        action_collection = client.collections.get("Action")
        payload_json = _serialize_action(action)

        existing = action_collection.query.fetch_objects(
            filters=wq.Filter.by_property("parameters_schema").equal(payload_json),
            limit=1,
        )
        existing_objects = existing.objects or []
        if existing_objects:
            return str(existing_objects[0].uuid)

        name = action.get("action", "unknown")
        description = ""
        if name == "goto":
            description = action.get("url", "")
        elif name == "scroll":
            description = f"direction={action.get('direction', 'down')}"

        new_uuid = action_collection.data.insert(
            properties={
                "name": name,
                "description": description,
                "parameters_schema": payload_json,
            }
        )
        return str(new_uuid)
    except Exception as exc:
        print(f"❌ Failed to upsert action payload {action}: {exc}")
        return None


def create_task_in_weaviate(client, title, actions: List[Dict[str, Any]]):
    """Create a task with the given title and actions in Weaviate (irvin style)"""
    try:
        # Get or create Action objects for each payload
        action_refs: List[str] = []
        for action in actions:
            action_uuid = _ensure_action_object(client, action)
            if action_uuid:
                action_refs.append(action_uuid)

        if not action_refs:
            print(f"⚠️ No valid actions found for task '{title}'")
            return None

        # Check if task already exists by searching for similar title
        task_collection = client.collections.get("Task")
        existing_tasks = task_collection.query.near_text(
            query=title,
            limit=1,
            return_metadata=wq.MetadataQuery(distance=True)
        )

        # Check if we found an exact match (very high similarity)
        if existing_tasks.objects and existing_tasks.objects[0].metadata.distance < 0.1:
            print(f"📦 Task already exists: {title}")
            return existing_tasks.objects[0].uuid

        # Create the task with a unique UUID
        task_uuid = task_collection.data.insert(
            properties={"title": title},
            references={"actions": action_refs}
        )

        print(f"✅ Created new task: {title} with {len(action_refs)} actions")
        return task_uuid

    except Exception as e:
        print(f"❌ Failed to create task '{title}': {e}")
        return None

def ensure_collections_exist(client):
    """Ensure that Task and Action collections exist (irvin style)"""
    try:
        # Check if collections exist
        collections = client.collections.list_all()
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

# ===================== LLM Functions =====================

async def _chat_async(messages, temperature=0.0, model_hint="", bypass_cache=False):
    """Call LLMClient and print token usage with timing"""

    start_time = time.time()

    # Make LLM call
    loop = asyncio.get_event_loop()

    def _call():
        return llm_client.chat(messages, temperature=temperature)

    text, usage = await loop.run_in_executor(None, _call)

    end_time = time.time()
    duration = end_time - start_time

    # tokens
    pt = int(usage.get("prompt_tokens", 0))
    ct = int(usage.get("completion_tokens", 0))
    tt = int(usage.get("total_tokens", pt + ct))
    RUN_TOKENS["prompt"] += pt
    RUN_TOKENS["completion"] += ct
    RUN_TOKENS["total"] += tt

    print(f"[LLM {llm_client.provider.upper()}] {model_hint or 'call'} tokens: prompt={pt} completion={ct} total={tt} in {duration:.2f}s")
    return text

# ===================== Helper Functions =====================

def _loose_json_parse(text: str):
    """Extract a JSON object/array from arbitrary LLM output."""
    raw = text.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    for start_pos in range(len(raw)):
        if raw[start_pos] in "{[":
            for end_pos in range(len(raw) - 1, start_pos - 1, -1):
                if raw[end_pos] in "}]":
                    try:
                        return json.loads(raw[start_pos:end_pos + 1])
                    except:
                        continue
    raise ValueError(f"No valid JSON found in: {text[:200]}...")

def _print_actions(title: str, subgoals: List[Dict]):
    """Pretty-print subgoals and their actions."""
    print(f"\n🧭 {title}:")
    for i, sg in enumerate(subgoals, 1):
        print(f"  • Subgoal {i}: {sg.get('description', 'No description')}")
        for j, act in enumerate(sg.get("actions", []), 1):
            print(f"    {j}. {act.get('action', 'unknown')} {act.get('url', '')} {act.get('text', '')} {act.get('direction', '')}")

def print_db_stats():
    """Print Weaviate database statistics."""
    try:
        client = connect_to_weaviate()
        if client:
            # Get collection info
            try:
                task_collection = client.collections.get("Task")
                action_collection = client.collections.get("Action")

                task_count = task_collection.aggregate.over_all(total_count=True).total_count
                action_count = action_collection.aggregate.over_all(total_count=True).total_count

                print(f"\n📊 WEAVIATE DATABASE STATS")
                print(f"   Tasks: {task_count}")
                print(f"   Actions: {action_count}")

            except Exception as e:
                print(f"❌ Error getting database stats: {e}")
            finally:
                client.close()
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")

# ===================== Main Agent Class =====================

class LLMBrowserAgentIrvin:
    def __init__(self, page=None, browser=None):
        self.page = page
        self.browser = browser
        self.playwright_context = None
        self.token = None
        self.connection_attempts = 0
        self.max_connection_attempts = 3

    async def safe_page_operation(self, operation_func, *args, max_retries=5, **kwargs):
        """Safely execute page operations with automatic reconnection"""
        for attempt in range(max_retries):
            try:
                return await operation_func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                if ("has been closed" in error_msg or
                    "Target page" in error_msg or
                    "Target closed" in error_msg):
                    print(f"🔄 Page operation failed (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        if await self.reconnect_browser():
                            print("✅ Reconnected, retrying operation...")
                            continue
                        else:
                            print("❌ Reconnection failed")
                            break
                    else:
                        print("❌ Max reconnection attempts reached")
                        break
                else:
                    # Non-connection error, re-raise immediately
                    raise e

        raise Exception(f"Page operation failed after {max_retries} attempts with reconnection")

    async def reconnect_browser(self):
        """Attempt to reconnect to browser and create new page"""
        try:
            if self.playwright_context:
                await self.playwright_context.close()

            playwright = await async_playwright().start()
            self.playwright_context = await playwright.chromium.launch_persistent_context(
                user_data_dir="/tmp/playwright_persistent",
                headless=False,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )

            pages = self.playwright_context.pages
            if pages:
                self.page = pages[0]
            else:
                self.page = await self.playwright_context.new_page()

            self.connection_attempts += 1
            return True

        except Exception as e:
            print(f"❌ Reconnection failed: {e}")
            return False

    async def _create_subgoals(self, goal: str) -> List[Dict]:
        """Create subgoal descriptions without specific actions"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Wikipedia automation planner. Break down the given goal into 2-4 clear subgoals. "
                    "Each subgoal should be a specific, actionable step that can be accomplished on Wikipedia. "
                    "Return a JSON array of subgoal objects with 'description' field. "
                    "Focus on Wikipedia-specific tasks like searching, reading articles, comparing information, etc."
                )
            },
            {
                "role": "user",
                "content": f"Break down this goal into subgoals: {goal}\n\nExample format:\n[\n  {{\"description\": \"Search for information about X\"}},\n  {{\"description\": \"Read the main article about X\"}},\n  {{\"description\": \"Compare X with Y\"}}\n]"
            }
        ]

        response = await _chat_async(messages, model_hint="create_subgoals")
        subgoals = _loose_json_parse(response)

        if not isinstance(subgoals, list):
            raise ValueError("Subgoals must be a list")

        return subgoals

    async def _create_actions_for_subgoal(self, subgoal_description: str) -> Tuple[List[Dict], bool]:
        """Create specific actions for a single subgoal using Weaviate to find similar tasks."""
        try:
            # First, try to find a similar task in Weaviate
            client = connect_to_weaviate()
            if client:
                result = find_nearest_task_and_actions(client, subgoal_description)
                if result:
                    distance = result.get('distance', 1.0)
                    similarity = 1.0 - distance if distance is not None else 0.0

                    # Use cached actions if similarity is high enough
                    if similarity >= 0.7:  # 70% similarity threshold
                        task_with_actions = result['task_with_actions']
                        if task_with_actions and task_with_actions.objects:
                            task_obj = task_with_actions.objects[0]
                            if hasattr(task_obj, 'references') and task_obj.references and 'actions' in task_obj.references:
                                actions_ref = task_obj.references['actions']
                                if hasattr(actions_ref, 'objects'):
                                    cached_actions = []
                                    for action_ref in actions_ref.objects:
                                        if hasattr(action_ref, 'properties'):
                                            payload_raw = action_ref.properties.get('parameters_schema')
                                            action_payload = None
                                            if payload_raw:
                                                try:
                                                    action_payload = json.loads(payload_raw)
                                                except json.JSONDecodeError:
                                                    action_payload = None
                                            if not action_payload:
                                                action_name = action_ref.properties.get('name', 'unknown')
                                                if action_name == 'goto':
                                                    action_payload = {"action": "goto", "url": "https://en.wikipedia.org/wiki/Main_Page"}
                                                elif action_name == 'read_page':
                                                    action_payload = {"action": "read_page"}
                                                elif action_name == 'scroll':
                                                    action_payload = {"action": "scroll", "direction": "down"}
                                                else:
                                                    action_payload = {"action": action_name}

                                            if action_payload:
                                                cached_actions.append(action_payload)

                                    if cached_actions:
                                        print(f"   📦 Using cached actions from Weaviate (sim={similarity:.3f})")
                                        client.close()
                                        return cached_actions, True

                client.close()

            # Fallback to LLM generation if no good match found
            print(f"   🔨 Generating new actions via LLM...")
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Wikipedia action planner. For the given subgoal, create specific actions to achieve it. "
                        "Return a JSON array of actions. Each action must have this exact format:\n"
                        "- goto: {\"action\": \"goto\", \"url\": \"https://en.wikipedia.org/wiki/PageName\"}\n"
                        "- read_page: {\"action\": \"read_page\"}\n"
                        "- scroll: {\"action\": \"scroll\", \"direction\": \"up\" or \"down\"}\n"
                        "Only use Wikipedia URLs (en.wikipedia.org). Use actual names, not placeholders.\n"
                        "IMPORTANT: For comparison tasks, do NOT navigate to new pages - the data should already be available from previous subgoals. Use minimal actions like just reading the current page or scrolling."
                    )
                },
                {
                    "role": "user",
                    "content": f"Create actions for this subgoal: {subgoal_description}\n\nExample format:\n[\n  {{\"action\": \"goto\", \"url\": \"https://en.wikipedia.org/wiki/James_Blunt\"}},\n  {{\"action\": \"read_page\"}}\n]"
                }
            ]

            response = await _chat_async(messages, model_hint="create_actions")
            actions = _loose_json_parse(response)

            if not isinstance(actions, list):
                raise ValueError("Actions must be a list")

            # Fix action format if needed
            fixed_actions = []
            for action in actions:
                fixed_action = self._normalize_action(action)
                if fixed_action:
                    fixed_actions.append(fixed_action)

            return fixed_actions, False

        except Exception as e:
            print(f"❌ Action creation failed for subgoal '{subgoal_description}': {e}")
            return [], False

    def _normalize_action(self, action: Dict) -> Optional[Dict]:
        """Normalize action format"""
        if not isinstance(action, dict) or "action" not in action:
            return None

        action_type = action["action"]

        if action_type == "goto":
            if "url" not in action:
                return None
            return {"action": "goto", "url": action["url"]}
        elif action_type == "read_page":
            return {"action": "read_page"}
        elif action_type == "scroll":
            direction = action.get("direction", "down")
            return {"action": "scroll", "direction": direction}
        elif action_type in ["type", "click", "press_enter", "wait", "done"]:
            # Skip these for Wikipedia-only browsing
            return None
        else:
            return action

    async def run(self, goal: str, no_cache=False, plan_preview=False, force_plan=False):
        """Main execution method with robust error handling"""
        global EXECUTION_START_TIME, TOTAL_EXECUTION_TIME
        EXECUTION_START_TIME = time.time()

        print(f"\n🚀 STARTING AUTOMATION: {goal}")
        print("=" * 60)
        print(f"provider={llm_client.provider.upper()}  db=Weaviate (Irvin Style)")

        # Ensure Weaviate collections exist (create only if needed)
        print("\n🔌 WEAVIATE CONNECTION")
        print("=" * 60)
        client = connect_to_weaviate()
        if not client:
            print("❌ Failed to connect to Weaviate")
            return

        if not ensure_collections_exist(client):
            print("❌ Failed to ensure Weaviate collections exist")
            client.close()
            return

        client.close()
        print("✅ Weaviate connection verified")

        # Planning phase: Two-phase approach for better caching
        print(f"\n📋 PLANNING PHASE")
        print("=" * 60)

        # Phase 1: Create subgoal descriptions
        print("📝 [CREATING SUBGOALS]")
        subgoal_creation_start = time.time()
        subgoals = await self._create_subgoals(goal)
        subgoal_creation_time = time.time() - subgoal_creation_start
        if not subgoals:
            print("❌ Failed to create subgoals")
            return

        print(f"📝 Generated {len(subgoals)} subgoals:")
        for i, sg in enumerate(subgoals, 1):
            print(f"  • Subgoal {i}: {sg.get('description', 'No description')}")
        print(f"📝 [SUBGOALS CREATED] in {subgoal_creation_time:.3f}s")

        # Phase 2: Get or create actions for each subgoal
        print(f"\n🔧 ACTION PLANNING PHASE")
        print("=" * 60)

        action_planning_start = time.time()
        for i, subgoal in enumerate(subgoals, 1):
            desc = subgoal.get("description", "")
            print(f"\n▶ Planning actions for subgoal {i}: {desc}")

            # Generate actions for this subgoal (Weaviate lookup is now handled inside the method)
            actions, from_cache = await self._create_actions_for_subgoal(desc)
            if actions:
                subgoal["actions"] = actions
                # Store the new actions as a task in Weaviate, but avoid caching overly complex sequences
                if len(actions) <= 5 and not force_plan and not from_cache:  # Only cache simple new sequences
                    try:
                        client = connect_to_weaviate()
                        if client:
                            # Create action names list for the create_task function
                            task_uuid = create_task_in_weaviate(client, desc, actions)
                            if task_uuid:
                                print(f"   💾 Stored new task in Weaviate")
                            else:
                                print(f"   ⚠️ Failed to store task in Weaviate")

                            client.close()
                    except Exception as e:
                        print(f"   ⚠️ Failed to store task in Weaviate: {e}")
                elif len(actions) > 5:
                    print(f"   ⚠️ Skipping Weaviate storage - action sequence too complex ({len(actions)} actions)")
            else:
                print(f"   ❌ Failed to generate actions for subgoal")
                subgoal["actions"] = []

        action_planning_time = time.time() - action_planning_start
        print(f"🔧 [ACTION PLANNING COMPLETED] in {action_planning_time:.3f}s")

        # Print final plan summary
        _print_actions("Final execution plan", subgoals)

        # Execution phase
        print(f"\n🎬 EXECUTION PHASE")
        print("=" * 60)

        execution_start = time.time()

        try:
            # Initialize browser
            playwright = await async_playwright().start()
            self.playwright_context = await playwright.chromium.launch_persistent_context(
                user_data_dir="/tmp/playwright_persistent",
                headless=False,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )

            pages = self.playwright_context.pages
            if pages:
                self.page = pages[0]
            else:
                self.page = await self.playwright_context.new_page()

            print("✅ Browser initialized")

            # Execute each subgoal
            for i, subgoal in enumerate(subgoals, 1):
                desc = subgoal.get("description", "")
                actions = subgoal.get("actions", [])

                print(f"\n▶ Executing subgoal {i}: {desc}")

                if not actions:
                    print(f"   ⚠️ No actions to execute for subgoal {i}")
                    continue

                # Execute actions
                for j, action in enumerate(actions, 1):
                    action_type = action.get("action", "unknown")
                    print(f"   {j}. {action_type} {action.get('url', '')} {action.get('text', '')} {action.get('direction', '')}")

                    try:
                        if action_type == "goto":
                            url = action.get("url", "")
                            if url:
                                await self.safe_page_operation(self.page.goto, url)
                                await self.safe_page_operation(self.page.wait_for_load_state, "networkidle")
                        elif action_type == "read_page":
                            # Read page content
                            content = await self.safe_page_operation(self.page.content)
                            print(f"      📄 Page content length: {len(content)} chars")
                        elif action_type == "scroll":
                            direction = action.get("direction", "down")
                            if direction == "down":
                                await self.safe_page_operation(self.page.evaluate, "window.scrollBy(0, 500)")
                            else:
                                await self.safe_page_operation(self.page.evaluate, "window.scrollBy(0, -500)")
                            await asyncio.sleep(1)
                        elif action_type == "wait":
                            seconds = action.get("seconds", 2)
                            await asyncio.sleep(seconds)
                        elif action_type == "done":
                            print(f"      ✅ Subgoal {i} completed")
                            break

                    except Exception as e:
                        print(f"      ❌ Action failed: {e}")
                        continue

                print(f"   ✅ Subgoal {i} completed")

            execution_time = time.time() - execution_start
            print(f"\n🎬 [EXECUTION COMPLETED] in {execution_time:.3f}s")

        except Exception as e:
            print(f"❌ Execution failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up browser
            if self.playwright_context:
                await self.playwright_context.close()
            print("🧹 Browser cleaned up")

        # Final summary
        TOTAL_EXECUTION_TIME = time.time() - EXECUTION_START_TIME
        print(f"\n📊 FINAL SUMMARY")
        print("=" * 60)
        print(f"Total execution time: {TOTAL_EXECUTION_TIME:.3f}s")
        print(f"Total tokens used: {RUN_TOKENS['total']} (prompt: {RUN_TOKENS['prompt']}, completion: {RUN_TOKENS['completion']})")
        print(f"Average tokens per second: {RUN_TOKENS['total'] / TOTAL_EXECUTION_TIME:.2f}")

        # Print database stats
        print_db_stats()

# ===================== Main Execution =====================

async def main():
    parser = argparse.ArgumentParser(description="LLM Browser Agent with Irvin-Style Weaviate Caching")
    parser.add_argument("goal", help="The goal to accomplish")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--plan-preview", action="store_true", help="Show plan preview only")
    parser.add_argument("--force-plan", action="store_true", help="Force plan regeneration")
    parser.add_argument("--show-counts", action="store_true", help="Show token counts")

    args = parser.parse_args()

    agent = LLMBrowserAgentIrvin()
    await agent.run(
        goal=args.goal,
        no_cache=args.no_cache,
        plan_preview=args.plan_preview,
        force_plan=args.force_plan
    )

if __name__ == "__main__":
    asyncio.run(main())
