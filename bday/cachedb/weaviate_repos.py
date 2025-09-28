# cachedb/weaviate_repos.py
"""
Weaviate-based repository classes that replace SQLite repos.
Drop-in replacements for AnswersRepo, PlansRepo, and LLMRepo.
"""
import json
import time
import os
from typing import Optional, Dict, Any, List, Iterable
from dotenv import load_dotenv

import weaviate
import weaviate.classes.config as wc
import weaviate.classes.query as wq
from weaviate.auth import AuthApiKey
import weaviate.classes.init as wvc

load_dotenv()

# Use same thresholds as SQLite version
COSINE_THRESHOLD_PROMPT = float(os.getenv("COSINE_THRESHOLD_PROMPT", "0.97"))
COSINE_THRESHOLD_QA = float(os.getenv("COSINE_THRESHOLD_QA", "0.92"))
COSINE_THRESHOLD_PLAN = float(os.getenv("COSINE_THRESHOLD_PLAN", "0.70"))
MAX_LLM_CACHE_CHARS = int(os.getenv("LLM_CACHE_MAX_PROMPT_CHARS", "12000"))

_COUNT_FALLBACK = ("match_count", "matches", "deleted")


def connect_to_weaviate():
    """Connect to Weaviate using exact same logic as irvin branch"""
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    if weaviate_url and weaviate_api_key:
        try:
            additional_config = wvc.AdditionalConfig(
                timeout=wvc.Timeout(init=30)
            )

            # Get OpenAI API key for vectorization - same as irvin branch
            openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_TOKEN")
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


def ensure_cache_collections_exist(client):
    """Ensure cache collections exist - Answer, Plan, LLMCache"""
    try:
        collections = client.collections.list_all()
        collection_names = list(collections.keys()) if collections else []

        # Create Answer collection
        if "Answer" not in collection_names:
            print("Creating Answer collection...")
            client.collections.create(
                name="Answer",
                vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-large"  # Same as irvin branch
                ),
                properties=[
                    wc.Property(name="canonical_q", data_type=wc.DataType.TEXT),
                    wc.Property(name="question_text", data_type=wc.DataType.TEXT),
                    wc.Property(name="answer_text", data_type=wc.DataType.TEXT, skip_vectorization=True),
                    wc.Property(name="confidence", data_type=wc.DataType.NUMBER),
                    wc.Property(name="freshness_horizon", data_type=wc.DataType.INT),
                    wc.Property(name="ts", data_type=wc.DataType.NUMBER),
                    wc.Property(name="evidence_json", data_type=wc.DataType.TEXT, skip_vectorization=True),
                    wc.Property(name="sources_json", data_type=wc.DataType.TEXT, skip_vectorization=True),
                ]
            )
            print("✅ Answer collection created")

        # Create Plan collection
        if "Plan" not in collection_names:
            print("Creating Plan collection...")
            client.collections.create(
                name="Plan",
                vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-large"  # Same as irvin branch
                ),
                properties=[
                    wc.Property(name="intent_key", data_type=wc.DataType.TEXT),
                    wc.Property(name="goal_text", data_type=wc.DataType.TEXT),
                    wc.Property(name="site_domain", data_type=wc.DataType.TEXT),
                    wc.Property(name="env_fingerprint", data_type=wc.DataType.TEXT),
                    wc.Property(name="plan_json", data_type=wc.DataType.TEXT, skip_vectorization=True),
                    wc.Property(name="success_rate", data_type=wc.DataType.NUMBER),
                    wc.Property(name="version", data_type=wc.DataType.TEXT),
                    wc.Property(name="ts", data_type=wc.DataType.NUMBER),
                ]
            )
            print("✅ Plan collection created")

        # Create LLMCache collection
        if "LLMCache" not in collection_names:
            print("Creating LLMCache collection...")
            client.collections.create(
                name="LLMCache",
                vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-large"  # Same as irvin branch
                ),
                properties=[
                    wc.Property(name="model", data_type=wc.DataType.TEXT),
                    wc.Property(name="prompt_norm_hash", data_type=wc.DataType.TEXT),
                    wc.Property(name="prompt_text", data_type=wc.DataType.TEXT),
                    wc.Property(name="tool_state", data_type=wc.DataType.TEXT),
                    wc.Property(name="output_text", data_type=wc.DataType.TEXT, skip_vectorization=True),
                    wc.Property(name="usage_json", data_type=wc.DataType.TEXT, skip_vectorization=True),
                    wc.Property(name="ts", data_type=wc.DataType.NUMBER),
                    wc.Property(name="ttl", data_type=wc.DataType.INT),
                    wc.Property(name="source_tag", data_type=wc.DataType.TEXT),
                    wc.Property(name="version", data_type=wc.DataType.TEXT),
                ]
            )
            print("✅ LLMCache collection created")

        return True
    except Exception as e:
        print(f"❌ Error creating cache collections: {e}")
        return False


class _WeaviateRepoBase:
    """Shared helpers for Weaviate-backed repositories."""

    collection_name: str = ""

    def __init__(self):
        self.client: Optional[weaviate.WeaviateClient] = None
        self._ensure_connection()

    # NOTE: keep connection cached per-instance because instantiating a client is expensive.
    def _ensure_connection(self):
        if self.client:
            return
        self.client = connect_to_weaviate()
        if self.client:
            ensure_cache_collections_exist(self.client)

    def _get_collection(self):
        self._ensure_connection()
        if not self.client:
            raise RuntimeError("Weaviate client unavailable - ensure credentials are set.")
        return self.client.collections.get(self.collection_name)

    def count(self) -> int:
        try:
            coll = self._get_collection()
            result = coll.aggregate.over_all(total_count=True)
            return int(getattr(result, "total_count", 0) or 0)
        except Exception as exc:
            print(f"❌ Error counting {self.collection_name}: {exc}")
            return 0


def _extract_delete_count(result: Any) -> int:
    for attr in _COUNT_FALLBACK:
        value = getattr(result, attr, None)
        if value is not None:
            return int(value)
    if isinstance(result, dict):
        for key in _COUNT_FALLBACK:
            if key in result:
                return int(result[key])
    return 0


def _load_json_field(value: Any, default: Any) -> Any:
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return default

def safe_get_property(obj, key: str, default=None):
    """Safely get property from Weaviate object"""
    if obj is None:
        return default
    if hasattr(obj, 'properties') and obj.properties:
        return obj.properties.get(key, default)
    return default

def distance_to_similarity(distance: float) -> float:
    """Convert Weaviate distance to similarity (1 - distance)"""
    if distance is None:
        return 1.0
    return 1.0 - float(distance)

# ------------------------------ LLM Cache -----------------------------------

class LLMRepo(_WeaviateRepoBase):
    """Weaviate-based LLM repository - drop-in replacement for SQLite version"""

    def __init__(self):
        self.collection_name = "LLMCache"
        super().__init__()

    def put(self, model: str, prompt: str, output_text: str,
            usage: Optional[dict] = None, tool_state: Optional[str] = None,
            ttl: Optional[int] = None, source_tag: Optional[str] = None,
            version: Optional[str] = None) -> Optional[str]:
        """Store LLM response in Weaviate"""
        if len(prompt) > MAX_LLM_CACHE_CHARS:
            print("[llm-cache] Skipping store: prompt too long for embedding")
            return None

        self._ensure_connection()
        if not self.client:
            return None

        try:
            llm_collection = self._get_collection()

            properties = {
                "model": model,
                "prompt_norm_hash": str(hash(prompt)),
                "prompt_text": prompt,
                "tool_state": tool_state,
                "output_text": output_text,
                "usage_json": json.dumps(usage or {}),
                "ts": time.time(),
                "ttl": ttl,
                "source_tag": source_tag,
                "version": version
            }

            llm_uuid = llm_collection.data.insert(properties)
            return str(llm_uuid)

        except Exception as e:
            print(f"❌ Error storing LLM cache: {e}")
            return None

    def approx_get(self, prompt: str, top_k: int = 5) -> Optional[dict]:
        """Get LLM response from Weaviate using vector similarity"""
        if len(prompt) > MAX_LLM_CACHE_CHARS:
            return None

        self._ensure_connection()
        if not self.client:
            return None

        try:
            llm_collection = self._get_collection()

            response = llm_collection.query.near_text(
                query=prompt,
                limit=1,
                return_metadata=wq.MetadataQuery(distance=True)
            )

            if response.objects:
                cached_response = response.objects[0]
                distance = getattr(cached_response.metadata, 'distance', 0.0)
                similarity = distance_to_similarity(distance)

                if similarity >= COSINE_THRESHOLD_PROMPT:
                    return {
                        "id": str(cached_response.uuid),
                        "model": safe_get_property(cached_response, "model"),
                        "prompt_text": safe_get_property(cached_response, "prompt_text"),
                        "output_text": safe_get_property(cached_response, "output_text"),
                        "usage": _load_json_field(safe_get_property(cached_response, "usage_json"), {}),
                        "ts": _to_float(safe_get_property(cached_response, "ts")),
                        "similarity": similarity
                    }

            return None

        except Exception as e:
            print(f"❌ Error getting LLM cache: {e}")
            return None

# ------------------------------ Answers -------------------------------------

class AnswersRepo(_WeaviateRepoBase):
    """Weaviate-based Answers repository - drop-in replacement for SQLite version"""

    def __init__(self):
        self.collection_name = "Answer"
        super().__init__()

    def put(self, canonical_q: str, question_text: str, answer_text: str,
            confidence: float = 1.0, freshness_horizon: Optional[int] = None,
            evidence: Optional[dict] = None, sources: Optional[list] = None) -> Optional[str]:
        """Store answer in Weaviate"""
        self._ensure_connection()
        if not self.client:
            return None

        try:
            answer_collection = self._get_collection()

            properties = {
                "canonical_q": canonical_q,
                "question_text": question_text,
                "answer_text": answer_text,
                "confidence": confidence,
                "freshness_horizon": freshness_horizon,
                "ts": time.time(),
                "evidence_json": json.dumps(evidence or {}),
                "sources_json": json.dumps(sources or [])
            }

            answer_uuid = answer_collection.data.insert(properties)
            return str(answer_uuid)

        except Exception as e:
            print(f"❌ Error storing answer: {e}")
            return None

    def approx_get(self, question_text: str, canonical_q: Optional[str] = None,
                   top_k: int = 5) -> Optional[dict]:
        """Get answer from Weaviate using vector similarity"""
        self._ensure_connection()
        if not self.client:
            return None

        try:
            answer_collection = self._get_collection()

            # Search using the combined key like SQLite version
            search_text = (canonical_q or "") + " " + question_text
            response = answer_collection.query.near_text(
                query=search_text,
                limit=1,
                return_metadata=wq.MetadataQuery(distance=True)
            )

            if response.objects:
                answer = response.objects[0]
                distance = getattr(answer.metadata, 'distance', 0.0)
                similarity = distance_to_similarity(distance)

                # Check freshness like SQLite version
                fh_raw = safe_get_property(answer, "freshness_horizon")
                fh = _to_float(fh_raw, 0.0) if fh_raw is not None else None
                if fh:
                    ts = _to_float(safe_get_property(answer, "ts"))
                    expired = (time.time() - ts) > fh
                    if expired:
                        similarity *= 0.8  # light penalty like SQLite version

                if similarity >= COSINE_THRESHOLD_QA:
                    return {
                        "id": str(answer.uuid),
                        "canonical_q": safe_get_property(answer, "canonical_q"),
                        "question_text": safe_get_property(answer, "question_text"),
                        "answer_text": safe_get_property(answer, "answer_text"),
                        "confidence": safe_get_property(answer, "confidence"),
                        "sources": _load_json_field(safe_get_property(answer, "sources_json"), []),
                        "evidence": _load_json_field(safe_get_property(answer, "evidence_json"), {}),
                        "similarity": similarity,
                        "ts": _to_float(safe_get_property(answer, "ts"))
                    }

            return None

        except Exception as e:
            print(f"❌ Error getting answer: {e}")
            return None

    def delete_by_canonical(self, canonical_q: str) -> int:
        self._ensure_connection()
        if not self.client:
            return 0
        try:
            collection = self._get_collection()
            result = collection.data.delete_many(
                where=wq.Filter.by_property("canonical_q").equal(canonical_q)
            )
            return _extract_delete_count(result)
        except Exception as exc:
            print(f"❌ Error deleting answers for {canonical_q}: {exc}")
            return 0

    def iter_question_answer_pairs(
        self, return_properties: Optional[List[str]] = None
    ) -> Iterable[Dict[str, Any]]:
        self._ensure_connection()
        if not self.client:
            return
        props = return_properties or ["canonical_q", "question_text", "answer_text", "ts", "confidence"]
        try:
            collection = self._get_collection()
            iterator = collection.iterator(return_properties=props)
            for item in iterator:
                yield {"uuid": str(item.uuid), **(item.properties or {})}
        except Exception as exc:
            print(f"❌ Error iterating answers: {exc}")
            return

    def fetch_by_canonical(self, canonical_q: str, return_properties: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        self._ensure_connection()
        if not self.client:
            return []
        props = return_properties or ["canonical_q", "question_text", "answer_text", "ts"]
        try:
            collection = self._get_collection()
            response = collection.query.fetch_objects(
                filters=wq.Filter.by_property("canonical_q").equal(canonical_q),
                return_properties=props,
            )
            matches: List[Dict[str, Any]] = []
            for obj in response.objects or []:
                matches.append({"uuid": str(obj.uuid), **(obj.properties or {})})
            return matches
        except Exception as exc:
            print(f"❌ Error fetching answers for {canonical_q}: {exc}")
            return []

# ------------------------------ Plans ---------------------------------------

class PlansRepo(_WeaviateRepoBase):
    """Weaviate-based Plans repository - drop-in replacement for SQLite version"""

    def __init__(self):
        self.collection_name = "Plan"
        super().__init__()

    def put(self, intent_key: str, goal_text: str, plan_json: dict,
            site_domain: Optional[str] = None, env_fingerprint: Optional[str] = None,
            success_rate: float = 0.5, version: Optional[str] = None) -> Optional[str]:
        """Store plan in Weaviate"""
        self._ensure_connection()
        if not self.client:
            return None

        try:
            plan_collection = self._get_collection()

            properties = {
                "intent_key": intent_key,
                "goal_text": goal_text,
                "site_domain": site_domain,
                "env_fingerprint": env_fingerprint,
                "plan_json": json.dumps(plan_json, ensure_ascii=False),
                "success_rate": success_rate,
                "version": version,
                "ts": time.time()
            }

            plan_uuid = plan_collection.data.insert(properties)
            return str(plan_uuid)

        except Exception as e:
            print(f"❌ Error storing plan: {e}")
            return None

    def approx_get(self, goal_text: str, top_k: int = 5) -> Optional[dict]:
        """Get plan from Weaviate using vector similarity"""
        self._ensure_connection()
        if not self.client:
            return None

        try:
            plan_collection = self._get_collection()

            # First try exact match like SQLite version
            exact_response = plan_collection.query.fetch_objects(
                filters=wq.Filter.by_property("goal_text").equal(goal_text),
                limit=1,
            )

            exact_objects = exact_response.objects or []
            if exact_objects:
                plan = exact_objects[0]
                return {
                    "id": str(plan.uuid),
                    "intent_key": safe_get_property(plan, "intent_key"),
                    "goal_text": safe_get_property(plan, "goal_text"),
                    "site_domain": safe_get_property(plan, "site_domain"),
                    "env_fingerprint": safe_get_property(plan, "env_fingerprint"),
                    "plan_json": _load_json_field(safe_get_property(plan, "plan_json"), {}),
                    "success_rate": safe_get_property(plan, "success_rate"),
                    "version": safe_get_property(plan, "version"),
                    "similarity": 1.0,
                    "ts": _to_float(safe_get_property(plan, "ts"))
                }

            # Fall back to vector similarity search
            response = plan_collection.query.near_text(
                query=goal_text,
                limit=1,
                return_metadata=wq.MetadataQuery(distance=True)
            )

            if response.objects:
                plan = response.objects[0]
                distance = getattr(plan.metadata, 'distance', 0.0)
                similarity = distance_to_similarity(distance)

                if similarity >= COSINE_THRESHOLD_PLAN:
                    return {
                        "id": str(plan.uuid),
                        "intent_key": safe_get_property(plan, "intent_key"),
                        "goal_text": safe_get_property(plan, "goal_text"),
                        "site_domain": safe_get_property(plan, "site_domain"),
                        "env_fingerprint": safe_get_property(plan, "env_fingerprint"),
                        "plan_json": _load_json_field(safe_get_property(plan, "plan_json"), {}),
                        "success_rate": safe_get_property(plan, "success_rate"),
                        "version": safe_get_property(plan, "version"),
                        "similarity": similarity,
                        "ts": _to_float(safe_get_property(plan, "ts"))
                    }

            return None

        except Exception as e:
            print(f"❌ Error getting plan: {e}")
            return None

# Helper function to initialize Weaviate collections
def init_weaviate_collections():
    """Initialize Weaviate collections - call this once at startup"""
    client = connect_to_weaviate()
    if client:
        success = ensure_cache_collections_exist(client)
        client.close()
        return success
    return False
