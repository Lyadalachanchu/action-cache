
"""Simple key-value interface backed by a Weaviate vector database."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.exceptions import WeaviateStartUpError

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from weaviate import WeaviateClient


@dataclass
class LookupResult:
    """Represents the value retrieved for a key (or nearest key)."""

    key: str
    value: str
    distance: Optional[float] = None


class WeaviateKeyValueStore:
    """Stores string key/value pairs in Weaviate with semantic lookup."""

    def __init__(self, class_name: Optional[str] = None) -> None:
        load_dotenv()

        self._class_name = class_name or os.getenv("WEAVIATE_CLASS_NAME", "DictionaryEntry")
        self._vectorizer = os.getenv("WEAVIATE_VECTORIZER", "text2vec-openai").lower()
        self._client = self._connect()
        self._collection = self._ensure_collection()

    def close(self) -> None:
        if self._client.is_connected():
            self._client.close()

    def __enter__(self) -> "WeaviateKeyValueStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def upsert(self, key: str, value: str) -> None:
        """Create or replace a key/value pair."""

        existing = self._collection.query.fetch_objects(
            filters=Filter.by_property("key").equal(key),
            limit=1,
        )

        if existing.objects:
            self._collection.data.update(
                uuid=existing.objects[0].uuid,
                properties={"key": key, "value": value},
            )
        else:
            self._collection.data.insert({"key": key, "value": value})

    def get(self, key: str) -> Optional[LookupResult]:
        """Return the value for a key; fall back to nearest semantic match."""

        exact = self._collection.query.fetch_objects(
            filters=Filter.by_property("key").equal(key),
            limit=1,
        )
        if exact.objects:
            obj = exact.objects[0]
            return LookupResult(
                key=obj.properties["key"],
                value=obj.properties["value"],
                distance=0.0,
            )

        nearest = self._collection.query.near_text(
            query=key,
            limit=1,
            return_metadata=MetadataQuery(distance=True),
        )

        if not nearest.objects:
            return None

        obj = nearest.objects[0]
        return LookupResult(
            key=obj.properties["key"],
            value=obj.properties["value"],
            distance=getattr(obj.metadata, "distance", None),
        )

    def _connect(self) -> "WeaviateClient":
        """Connect to Weaviate using env configuration."""

        cluster_url = os.getenv("WEAVIATE_URL")
        api_key = os.getenv("WEAVIATE_API_KEY")

        if cluster_url:
            if api_key:
                credentials = AuthApiKey(api_key=api_key)
                return weaviate.connect_to_weaviate_cloud(
                    cluster_url=cluster_url,
                    auth_credentials=credentials,
                )

            from urllib.parse import urlparse

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

    def _ensure_collection(self):
        if self._class_name not in self._client.collections.list_all():
            self._client.collections.create(
                name=self._class_name,
                vectorizer_config=self._vectorizer_config(),
                properties=[
                    Property(name="key", data_type=DataType.TEXT),
                    Property(name="value", data_type=DataType.TEXT),
                ],
            )

        return self._client.collections.get(self._class_name)

    def _vectorizer_config(self):
        if self._vectorizer in {"text2vec-openai", "openai"}:
            return Configure.Vectorizer.text2vec_openai()
        if self._vectorizer in {"text2vec-transformers", "transformers"}:
            return Configure.Vectorizer.text2vec_transformers()
        return Configure.Vectorizer.none()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic key/value store using Weaviate.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    insert_parser = subparsers.add_parser("set", help="Create or replace a key/value pair.")
    insert_parser.add_argument("key", help="Key to store")
    insert_parser.add_argument("value", help="Value to associate with the key")

    get_parser = subparsers.add_parser("get", help="Fetch a key or nearest semantic match.")
    get_parser.add_argument("key", help="Key to retrieve")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with WeaviateKeyValueStore() as store:
        if args.command == "set":
            store.upsert(args.key, args.value)
            print(f"Stored '{args.key}'.")
        elif args.command == "get":
            result = store.get(args.key)
            if not result:
                print("No entries found. You may need to add data first.")
                return

            if result.distance is None or result.distance <= 0.0:
                print(f"Exact match: {result.key} -> {result.value}")
            else:
                print(
                    f"Nearest match (distance {result.distance:.4f}):\n"
                    f"  {result.key} -> {result.value}"
                )


if __name__ == "__main__":
    main()
