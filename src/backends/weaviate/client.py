from __future__ import annotations

import json
import time
import uuid
from typing import Any

import requests

from ...generation.equivalent import Expr


def _value_key(value: Any) -> str:
    if isinstance(value, bool):
        return "valueBoolean"
    if isinstance(value, int):
        return "valueInt"
    if isinstance(value, float):
        return "valueNumber"
    return "valueText"


def render_where(expr: Expr) -> dict[str, Any]:
    """Render the backend-neutral filter AST as a Weaviate where filter."""
    op, args = expr.op, expr.args
    if op in {"and", "or"}:
        return {
            "operator": "And" if op == "and" else "Or",
            "operands": [render_where(child) for child in args],
        }
    if op == "not":
        return {"operator": "Not", "operands": [render_where(args[0])]}

    field, values = str(args[0]), args[1:]
    if op == "in":
        # The generated metadata columns are scalar.  Weaviate's ContainsAny
        # targets array properties, so scalar membership is represented as OR
        # of Equal leaves.  Duplicate list entries intentionally remain as
        # duplicate OR operands for the set-idempotence tests.
        leaves = [
            {"path": [field], "operator": "Equal", _value_key(value): value}
            for value in values
        ]
        return leaves[0] if len(leaves) == 1 else {"operator": "Or", "operands": leaves}
    if op == "between":
        low, high = values
        return {
            "operator": "And",
            "operands": [
                {"path": [field], "operator": "GreaterThanEqual", _value_key(low): low},
                {"path": [field], "operator": "LessThanEqual", _value_key(high): high},
            ],
        }
    operators = {
        "eq": "Equal",
        "lt": "LessThan",
        "lte": "LessThanEqual",
        "gt": "GreaterThan",
        "gte": "GreaterThanEqual",
    }
    value = values[0]
    return {"path": [field], "operator": operators[op], _value_key(value): value}


def graphql_literal(value: Any) -> str:
    if isinstance(value, dict):
        fields = []
        for key, item in value.items():
            # GraphQL input enum values are identifiers, not JSON strings.
            # Weaviate's where.operator is an enum (Equal, And, ...), while
            # paths and property values remain ordinary string literals.
            rendered = item if key == "operator" and isinstance(item, str) else graphql_literal(item)
            fields.append(f"{key}:{rendered}")
        return "{" + ",".join(fields) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(graphql_literal(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return repr(value)


class WeaviateBackend:
    def __init__(self, url: str, timeout: int = 300):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def wait_until_ready(self, timeout_s: int = 180) -> None:
        """Wait until the REST API is accepting requests after container start."""
        deadline = time.monotonic() + timeout_s
        last_error = "Weaviate did not report ready"
        while time.monotonic() < deadline:
            try:
                response = self.session.get(
                    self.url + "/v1/.well-known/ready", timeout=5
                )
                if response.ok:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(1)
        raise TimeoutError(
            f"Weaviate API was not ready within {timeout_s}s: {last_error}"
        )

    def request(self, method: str, path: str, body: dict | None = None) -> Any:
        response = self.session.request(
            method, self.url + path, json=body, timeout=self.timeout
        )
        if not response.ok:
            raise RuntimeError(
                f"Weaviate {method} {path}: {response.status_code} {response.text[:2000]}"
            )
        if not response.content:
            return None
        return response.json()

    def graphql(self, query: str) -> dict:
        response = self.request("POST", "/v1/graphql", {"query": query})
        if response.get("errors"):
            raise RuntimeError(json.dumps(response["errors"], ensure_ascii=False)[:4000])
        return response["data"]

    def meta(self) -> dict:
        return self.request("GET", "/v1/meta")

    def recreate_collection(self, class_name: str, dimensions: int, index: dict, ef: int,
                            distance: str = "cosine") -> None:
        response = self.session.get(self.url + f"/v1/schema/{class_name}", timeout=30)
        if response.status_code == 200:
            self.request("DELETE", f"/v1/schema/{class_name}")
        elif response.status_code != 404:
            raise RuntimeError(f"schema lookup failed: {response.status_code} {response.text[:1000]}")

        numeric_properties = ["point_id", "tenant", "level", "category", "category5", "age"]
        properties = [
            {
                "name": name,
                "dataType": ["int"],
                "indexFilterable": True,
                "indexSearchable": False,
                "indexRangeFilters": True,
            }
            for name in numeric_properties
        ]
        properties.append(
            {
                "name": "parity",
                "dataType": ["text"],
                "tokenization": "field",
                "indexFilterable": True,
                "indexSearchable": False,
            }
        )
        self.request(
            "POST",
            "/v1/schema",
            {
                "class": class_name,
                "description": f"Equivalent-filter campaign; fixed HNSW ef={ef}",
                "vectorizer": "none",
                "vectorIndexType": "hnsw",
                "vectorIndexConfig": {
                    "distance": distance,
                    "ef": int(ef),
                    "efConstruction": int(index["ef_construct"]),
                    "maxConnections": int(index["hnsw_m"]),
                    "flatSearchCutoff": 0,
                    "vectorCacheMaxObjects": 500000,
                },
                "invertedIndexConfig": {
                    "cleanupIntervalSeconds": 300,
                    "indexTimestamps": False,
                    "indexNullState": False,
                    "indexPropertyLength": False,
                },
                "replicationConfig": {"factor": 1},
                "shardingConfig": {"desiredCount": 1},
                "properties": properties,
            },
        )

    @staticmethod
    def object_uuid(point_id: int) -> str:
        return str(uuid.UUID(int=int(point_id) + 1))

    def upload(self, class_name: str, ids, vectors, payloads: list[dict], batch_size: int = 400) -> None:
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            objects = []
            for offset in range(start, end):
                point_id = int(ids[offset])
                objects.append(
                    {
                        "class": class_name,
                        "id": self.object_uuid(point_id),
                        "properties": {"point_id": point_id, **payloads[offset]},
                        "vector": vectors[offset].astype(float).tolist(),
                    }
                )
            response = self.request("POST", "/v1/batch/objects", {"objects": objects})
            errors = [
                item for item in response
                if (item.get("result") or {}).get("status") not in {"SUCCESS", None}
            ]
            if errors:
                raise RuntimeError(f"batch upload errors: {json.dumps(errors[:3])[:3000]}")
            if start == 0 or end % 10000 == 0 or end == len(ids):
                print(f"uploaded {end}/{len(ids)} to {class_name}", flush=True)

    def aggregate_count(self, class_name: str, where: dict | None = None) -> tuple[int, float]:
        where_clause = "" if where is None else f"(where:{graphql_literal(where)})"
        query = f"{{Aggregate{{{class_name}{where_clause}{{meta{{count}}}}}}}}"
        started = time.perf_counter_ns()
        data = self.graphql(query)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        rows = data["Aggregate"][class_name]
        count = int(rows[0]["meta"]["count"]) if rows else 0
        return count, elapsed

    def wait_ready(self, class_name: str, expected: int, timeout_s: int = 1800) -> dict:
        deadline = time.time() + timeout_s
        last: dict[str, Any] = {}
        while time.time() < deadline:
            count, _ = self.aggregate_count(class_name)
            nodes = self.request("GET", "/v1/nodes?output=verbose")
            schema = self.request("GET", f"/v1/schema/{class_name}")
            physical = set((schema.get("shardingState") or {}).get("physical", {}).keys())
            shards = [
                shard
                for node in nodes.get("nodes", [])
                for shard in node.get("shards", [])
                if not physical or shard.get("name") in physical
            ]
            queue = sum(int(shard.get("vectorQueueLength") or 0) for shard in shards)
            statuses = sorted(
                {
                    str(shard.get("vectorIndexingStatus") or shard.get("status"))
                    for shard in shards
                }
            )
            last = {"count": count, "vector_queue_length": queue, "shard_statuses": statuses}
            if count == expected and queue == 0 and all(status == "READY" for status in statuses):
                time.sleep(3)
                return last
            time.sleep(3)
        raise TimeoutError(f"Weaviate collection did not become ready: {last}")

    def reusable_collection(self, class_name: str, expected: int, dimensions: int,
                            index: dict, ef: int, timeout_s: int = 30) -> dict | None:
        """Return state only when schema, count, vector size, and HNSW are ready."""
        try:
            schema = self.request("GET", f"/v1/schema/{class_name}")
            if str(schema.get("vectorIndexType", "")).lower() != "hnsw":
                return None
            vector_index = schema.get("vectorIndexConfig") or {}
            if str(vector_index.get("distance", "")).lower() != "cosine":
                return None
            if int(vector_index.get("ef", -1)) != int(ef):
                return None
            if int(vector_index.get("efConstruction", -1)) != int(index["ef_construct"]):
                return None
            if int(vector_index.get("maxConnections", -1)) != int(index["hnsw_m"]):
                return None
            required_properties = {
                "point_id", "tenant", "level", "category", "category5", "age", "parity"
            }
            actual_properties = {
                str(item.get("name")) for item in schema.get("properties", [])
            }
            if not required_properties.issubset(actual_properties):
                return None
            state = self.wait_ready(class_name, int(expected), timeout_s=timeout_s)
            sample = self.request(
                "GET", f"/v1/objects?class={class_name}&limit=1&include=vector"
            )
            objects = sample.get("objects") or []
            if expected and (not objects or len(objects[0].get("vector") or []) != int(dimensions)):
                return None
            return state
        except Exception:
            return None

    def search(self, class_name: str, vector: list[float], where: dict | None, limit: int) -> tuple[list[int], dict]:
        where_clause = "" if where is None else f",where:{graphql_literal(where)}"
        query = (
            f"{{Get{{{class_name}(nearVector:{{vector:{graphql_literal(vector)}}},"
            f"limit:{int(limit)}{where_clause})"
            "{point_id _additional{distance id}}}}"
        )
        started = time.perf_counter_ns()
        data = self.graphql(query)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        rows = data["Get"].get(class_name) or []
        return [int(row["point_id"]) for row in rows], {
            "client_wall_ms": elapsed,
            "distances": [float(row["_additional"]["distance"]) for row in rows],
        }
