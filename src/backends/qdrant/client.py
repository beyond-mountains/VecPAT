from __future__ import annotations

import time
from typing import Any, Iterable

import requests


class Qdrant:
    def __init__(self, url: str, timeout: int = 360):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        last_error: requests.RequestException | None = None
        for attempt in range(6):
            try:
                response = requests.request(
                    method, self.url + path, json=body, timeout=self.timeout
                )
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt == 5:
                    raise
                time.sleep(attempt + 1)
        else:  # pragma: no cover - defensive; the loop either breaks or raises.
            raise RuntimeError(f"Qdrant request failed: {last_error}")
        if not response.ok:
            raise RuntimeError(f"Qdrant {method} {path}: {response.status_code} {response.text[:1000]}")
        return response.json()

    def recreate_collection(self, name: str, dimensions: int, index: dict) -> None:
        exists = requests.get(f"{self.url}/collections/{name}", timeout=10).status_code == 200
        if exists:
            self.request("DELETE", f"/collections/{name}")
        self.request("PUT", f"/collections/{name}", {
            "vectors": {"size": dimensions, "distance": "Cosine"},
            "hnsw_config": {
                "m": index["hnsw_m"],
                "ef_construct": index["ef_construct"],
                "full_scan_threshold": index.get("full_scan_threshold", 10000),
            },
            # Avoid repeatedly rebuilding HNSW while batches are still arriving.
            "optimizers_config": {"indexing_threshold": 0, "default_segment_number": 1},
        })

    def enable_indexing(self, name: str, index: dict) -> None:
        self.request("PATCH", f"/collections/{name}", {
            "optimizers_config": {
                "indexing_threshold": index["indexing_threshold"],
                "default_segment_number": 1,
                "max_optimization_threads": 4,
            }
        })

    def create_payload_indexes(self, name: str) -> None:
        for field, schema in (
            ("tenant", "integer"), ("level", "integer"), ("category", "integer"),
            ("category5", "integer"), ("age", "integer"), ("parity", "keyword"),
        ):
            self.request("PUT", f"/collections/{name}/index?wait=true", {
                "field_name": field, "field_schema": schema,
            })

    def upload(self, name: str, ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
        self.request("PUT", f"/collections/{name}/points?wait=true", {
            "batch": {"ids": ids, "vectors": vectors, "payloads": payloads}
        })

    def wait_ready(self, name: str, expected: int, require_index: bool = True, timeout_s: int = 3600) -> dict:
        deadline = time.time() + timeout_s
        last = {}
        while time.time() < deadline:
            last = self.request("GET", f"/collections/{name}")["result"]
            optimizer = last.get("optimizer_status")
            ok = optimizer == "ok" or (isinstance(optimizer, dict) and optimizer.get("ok"))
            indexed = int(last.get("indexed_vectors_count") or 0)
            index_ready = (not require_index) or indexed >= int(expected * 0.90)
            if last.get("status") == "green" and ok and last.get("points_count") == expected and index_ready:
                # A short quiet period prevents the benchmark from racing a just-finished optimizer.
                time.sleep(3)
                return last
            time.sleep(2)
        raise TimeoutError(f"collection did not become ready: {last}")

    def reusable_collection(self, name: str, expected: int, dimensions: int,
                            index: dict) -> dict | None:
        """Return collection state only when data, schema, and HNSW match."""
        try:
            state = self.request("GET", f"/collections/{name}")["result"]
            # Do not enter the long index-readiness wait for a partially
            # uploaded collection left by an interrupted setup.  Such a
            # collection must be recreated immediately.
            if int(state.get("points_count") or 0) != int(expected):
                return None
            config = state.get("config") or {}
            params = config.get("params") or {}
            vectors = params.get("vectors") or {}
            hnsw = config.get("hnsw_config") or {}
            if int(vectors.get("size", -1)) != int(dimensions):
                return None
            if str(vectors.get("distance", "")).lower() != "cosine":
                return None
            if int(hnsw.get("m", -1)) != int(index["hnsw_m"]):
                return None
            if int(hnsw.get("ef_construct", -1)) != int(index["ef_construct"]):
                return None
            if int(hnsw.get("full_scan_threshold", -1)) != int(
                index.get("full_scan_threshold", 10000)
            ):
                return None
            required_payload = {
                "tenant", "level", "category", "category5", "age", "parity"
            }
            if not required_payload.issubset(set((state.get("payload_schema") or {}).keys())):
                return None
            return self.wait_ready(name, int(expected))
        except Exception:
            return None

    def scroll_ids(self, name: str, query_filter: dict, page_size: int = 10000) -> list[int]:
        ids: list[int] = []
        offset: Any = None
        while True:
            body: dict[str, Any] = {
                "filter": query_filter, "limit": page_size,
                "with_payload": False, "with_vector": False,
            }
            if offset is not None:
                body["offset"] = offset
            result = self.request("POST", f"/collections/{name}/points/scroll", body)["result"]
            ids.extend(int(p["id"]) for p in result["points"])
            offset = result.get("next_page_offset")
            if offset is None:
                break
        return sorted(ids)

    def search(self, name: str, vector: list[float], query_filter: dict | None, limit: int,
               ef_search: int, exact: bool = False) -> tuple[list[int], dict]:
        body = {
            "vector": vector,
            "limit": limit,
            "params": {"hnsw_ef": ef_search, "exact": exact},
            "with_payload": False,
            "with_vector": False,
        }
        if query_filter is not None:
            body["filter"] = query_filter
        started = time.perf_counter_ns()
        # Qdrant has a separate server-side search timeout.  Keep it aligned
        # with the HTTP client timeout so expensive exact-search oracles on
        # large datasets are not aborted by the server's 60-second default.
        response = self.request(
            "POST", f"/collections/{name}/points/search?timeout={self.timeout}", body
        )
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        ids = [int(hit["id"]) for hit in response["result"]]
        metrics = {"client_wall_ms": wall_ms, "server_time_ms": response.get("time", 0) * 1000}
        usage = response.get("usage") or {}
        hardware = usage.get("hardware", usage)
        metrics.update({f"usage_{k}": v for k, v in hardware.items() if isinstance(v, (int, float))})
        return ids, metrics
