"""Bootstrap Action and Task collections in Weaviate with sample entries."""
from __future__ import annotations

import json
import os
import uuid
from urllib.parse import urlparse

from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, DataType, Property
from weaviate.exceptions import WeaviateStartUpError


def connect_client() -> weaviate.WeaviateClient:
    """Instantiate a Weaviate client using environment configuration."""

    load_dotenv()

    cluster_url = os.getenv("WEAVIATE_URL")
    api_key = os.getenv("WEAVIATE_API_KEY")

    if cluster_url:
        if api_key:
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=cluster_url,
                auth_credentials=AuthApiKey(api_key=api_key),
            )

        parsed = urlparse(cluster_url)
        if not parsed.scheme or not parsed.hostname:
            raise ValueError(
                "WEAVIATE_URL must include scheme and host, e.g. https://example.weaviate.network"
            )

        http_secure = parsed.scheme == "https"
        http_port = parsed.port or (443 if http_secure else 8080)
        grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT") or (443 if http_secure else 50051))

        return weaviate.connect_to_custom(
            http_host=parsed.hostname,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=parsed.hostname,
            grpc_port=grpc_port,
            grpc_secure=http_secure,
        )

    http_port = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

    try:
        return weaviate.connect_to_local(
            host=os.getenv("WEAVIATE_HOST", "localhost"),
            port=http_port,
            grpc_port=grpc_port,
        )
    except WeaviateStartUpError as exc:
        raise RuntimeError(
            "Could not connect to local Weaviate. Set WEAVIATE_URL for remote clusters."
        ) from exc


def ensure_drop(client: weaviate.WeaviateClient, collection_name: str) -> None:
    """Remove an existing collection so it can be recreated."""

    if collection_name in client.collections.list_all():
        client.collections.delete(collection_name)


def create_action_collection(client: weaviate.WeaviateClient) -> None:
    ensure_drop(client, "Action")
    client.collections.create(
        name="Action",
        vector_configs=[
            Configure.Vectors.text2vec_openai(
                name="action_vector",
                source_properties=["name", "description", "search_text"],
                model="text-embedding-3-large",
                dimensions=3072,
            )
        ],
        properties=[
            Property(name="name", data_type=DataType.TEXT),
            Property(name="description", data_type=DataType.TEXT),
            Property(name="parameters_schema", data_type=DataType.TEXT),
            Property(name="preconditions", data_type=DataType.TEXT),
            Property(name="effects", data_type=DataType.TEXT),
            Property(name="success_criteria", data_type=DataType.TEXT),
            Property(name="failure_modes", data_type=DataType.TEXT),
            Property(name="observations", data_type=DataType.TEXT),
            Property(name="implementation_notes", data_type=DataType.TEXT),
            Property(name="version", data_type=DataType.TEXT),
            Property(name="search_text", data_type=DataType.TEXT),
        ],
    )

    action_collection = client.collections.get("Action")
    action_click = {
        "name": "click",
        "description": "Click a DOM element identified by a selector or handle.",
        "parameters_schema": json.dumps(
            {
                "selector": "string",
                "button": "left|middle|right",
                "clickCount": "int=1",
            }
        ),
        "preconditions": "Element exists, is visible, is interactable.",
        "effects": "Triggers click event; may navigate or open UI.",
        "success_criteria": "No exception; optional: DOM change or navigation observed.",
        "failure_modes": "Timeout; detached element; obscured element.",
        "observations": "Returns boolean success; may capture screenshot/DOM diff.",
        "implementation_notes": "Use page.click(selector) with retries and waits.",
        "version": "1.0",
        "search_text": "click element button link press select",
    }
    action_collection.data.insert(
        action_click,
        uuid=uuid.uuid5(uuid.NAMESPACE_URL, "action:click:v1").hex,
    )


def create_task_collection(client: weaviate.WeaviateClient) -> None:
    ensure_drop(client, "Task")
    client.collections.create(
        name="Task",
        vector_configs=[
            Configure.Vectors.text2vec_openai(
                name="task_vector",
                source_properties=["title", "goal", "search_text"],
                model="text-embedding-3-large",
                dimensions=3072,
            )
        ],
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="goal", data_type=DataType.TEXT),
            Property(name="inputs", data_type=DataType.TEXT),
            Property(name="outputs", data_type=DataType.TEXT),
            Property(name="steps_json", data_type=DataType.TEXT),
            Property(name="assertions", data_type=DataType.TEXT),
            Property(name="search_text", data_type=DataType.TEXT),
        ],
    )

    task_collection = client.collections.get("Task")
    steps = [
        {"action": "new_tab", "params": {"url": "https://www.google.com"}},
        {
            "action": "type",
            "params": {"selector": "input[name=q]", "text": "{{celebrity}} birth date"},
        },
        {"action": "keypress", "params": {"key": "Enter"}},
        {"action": "wait_for", "params": {"selector": "#search"}},
        {
            "action": "extract_text",
            "params": {"selector": "#search [data-attrid='kc:/people/person:born']"},
            "assign": "birth_date",
        },
        {
            "action": "extract_attr",
            "params": {"selector": "#search a", "attr": "href"},
            "assign": "source_url",
        },
        {"action": "assert", "params": {"condition": "birth_date != null"}},
    ]

    task_find_dob = {
        "title": "Find birth date via Google",
        "goal": "Return the celebrity's date of birth.",
        "inputs": json.dumps({"celebrity": "string"}),
        "outputs": json.dumps({"birth_date": "string", "source_url": "string"}),
        "steps_json": json.dumps(steps),
        "assertions": "birth_date is non-empty; source_url is a valid URL.",
        "search_text": "google search find birth date celebrity when was X born",
    }
    task_collection.data.insert(
        task_find_dob,
        uuid=uuid.uuid5(uuid.NAMESPACE_URL, "task:find_birth_date_via_google:v1").hex,
    )


def main() -> None:
    client = connect_client()
    try:
        create_action_collection(client)
        create_task_collection(client)
        print("✅ Collections created with text-embedding-3-large and sample objects inserted.")
    finally:
        if client.is_connected():
            client.close()


if __name__ == "__main__":
    main()
