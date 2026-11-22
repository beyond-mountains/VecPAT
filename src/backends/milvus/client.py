from __future__ import annotations

import time

from pymilvus import DataType, MilvusClient


class MilvusBackend:
    def __init__(self, uri: str):
        self.client = MilvusClient(uri=uri)

    @staticmethod
    def exact_name(base_name: str) -> str:
        return f"{base_name}_exact"

    def _schema(self, dimensions: int):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimensions)
        for field in ("tenant", "level", "category", "category5", "age"):
            schema.add_field(field_name=field, datatype=DataType.INT64)
        schema.add_field(field_name="parity", datatype=DataType.VARCHAR, max_length=8)
        return schema

    def recreate_collections(self, base_name: str, dimensions: int) -> None:
        for name in (base_name, self.exact_name(base_name)):
            if self.client.has_collection(name):
                self.client.drop_collection(name)
            self.client.create_collection(collection_name=name, schema=self._schema(dimensions), consistency_level="Strong")

    def insert(self, base_name: str, rows: list[dict]) -> None:
        self.client.insert(collection_name=base_name, data=rows)
        self.client.insert(collection_name=self.exact_name(base_name), data=rows)

    def build_indexes(self, base_name: str, index_cfg: dict) -> None:
        for name, vector_index in ((base_name, "HNSW"), (self.exact_name(base_name), "FLAT")):
            self.client.flush(collection_name=name)
            params = self.client.prepare_index_params()
            vector_params = ({"M": int(index_cfg["hnsw_m"]), "efConstruction": int(index_cfg["ef_construct"])}
                             if vector_index == "HNSW" else {})
            params.add_index(field_name="vector", index_name="vector_idx", index_type=vector_index,
                             metric_type="COSINE", params=vector_params)
            for field in ("tenant", "level", "category", "category5", "age"):
                params.add_index(field_name=field, index_name=f"{field}_idx", index_type="INVERTED")
            params.add_index(field_name="parity", index_name="parity_idx", index_type="TRIE")
            self.client.create_index(collection_name=name, index_params=params, sync=True)
            self.client.load_collection(collection_name=name)

    def collection_state(self, base_name: str) -> dict:
        return {
            "ann": self.client.describe_collection(collection_name=base_name),
            "ann_indexes": self.client.list_indexes(collection_name=base_name),
            "ann_load": self.client.get_load_state(collection_name=base_name),
            "exact": self.client.describe_collection(collection_name=self.exact_name(base_name)),
            "exact_indexes": self.client.list_indexes(collection_name=self.exact_name(base_name)),
            "exact_load": self.client.get_load_state(collection_name=self.exact_name(base_name)),
        }

    def reusable_collections(self, base_name: str, expected_rows: int,
                             dimensions: int, index_cfg: dict) -> dict | None:
        """Return state only when both ANN/exact collections are compatible."""
        try:
            required_indexes = {
                "vector_idx", "tenant_idx", "level_idx", "category_idx",
                "category5_idx", "age_idx", "parity_idx",
            }
            for name, expected_type in (
                (base_name, "HNSW"), (self.exact_name(base_name), "FLAT")
            ):
                if not self.client.has_collection(name):
                    return None
                description = self.client.describe_collection(collection_name=name)
                vector_field = next(
                    field for field in description.get("fields", [])
                    if field.get("name") == "vector"
                )
                params = vector_field.get("params") or vector_field.get("type_params") or {}
                if int(params.get("dim", -1)) != int(dimensions):
                    return None
                stats = self.client.get_collection_stats(collection_name=name)
                if int(stats.get("row_count", -1)) != int(expected_rows):
                    return None
                if not required_indexes.issubset(
                    set(self.client.list_indexes(collection_name=name))
                ):
                    return None
                vector_index = self.client.describe_index(
                    collection_name=name, index_name="vector_idx"
                )
                if str(vector_index.get("index_type", "")).upper() != expected_type:
                    return None
                if str(vector_index.get("metric_type", "")).upper() != "COSINE":
                    return None
                if expected_type == "HNSW":
                    index_params = vector_index.get("params") or {}
                    actual_m = index_params.get("M", index_params.get("m"))
                    actual_ef = index_params.get(
                        "efConstruction", index_params.get("ef_construction")
                    )
                    if int(actual_m or -1) != int(index_cfg["hnsw_m"]):
                        return None
                    if int(actual_ef or -1) != int(index_cfg["ef_construct"]):
                        return None
            self.ensure_ready(base_name)
            return self.collection_state(base_name)
        except Exception:
            return None

    def ensure_ready(self, base_name: str, timeout_seconds: float = 300.0) -> None:
        names = (base_name, self.exact_name(base_name))
        for name in names:
            self.client.load_collection(collection_name=name)
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                loaded = all(self.client.get_load_state(collection_name=name).get("state") == 3 for name in names)
                if loaded:
                    for name in names:
                        self.client.query(collection_name=name, filter="id >= 0", output_fields=["id"], limit=1,
                                          consistency_level="Strong")
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(2.0)
        raise TimeoutError(f"Milvus collections did not become query-ready: {last_error}")

    def query_ids(self, collection_name: str, expression: str, batch_size: int = 4096) -> list[int]:
        iterator = self.client.query_iterator(collection_name=collection_name, batch_size=batch_size,
                                              filter=expression, output_fields=["id"], consistency_level="Strong")
        ids: list[int] = []
        try:
            while True:
                page = iterator.next()
                if not page:
                    break
                ids.extend(int(row["id"]) for row in page)
        finally:
            iterator.close()
        return sorted(ids)

    def search(self, collection_name: str, vector: list[float], expression: str, limit: int,
               ef: int | None = None) -> tuple[list[int], dict]:
        params = {"metric_type": "COSINE", "params": {}}
        if ef is not None:
            params["params"]["ef"] = int(ef)
        start = time.perf_counter_ns()
        result = self.client.search(collection_name=collection_name, anns_field="vector", data=[vector],
                                    filter=expression, limit=limit, search_params=params,
                                    output_fields=[], consistency_level="Strong")
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        ids = [int(hit["id"]) for hit in result[0]]
        return ids, {"client_wall_ms": elapsed_ms}
