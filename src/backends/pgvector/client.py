from __future__ import annotations

import json
import time

import psycopg2
from psycopg2.extras import execute_values


class PgvectorBackend:
    name = "pgvector"

    def __init__(self, dsn: str, table: str):
        self.table = table
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("SELECT version(), extversion FROM pg_extension JOIN LATERAL (SELECT version()) x(version) ON true WHERE extname='vector'")
            row = cur.fetchone()
            self.version = f"{row[0]}; pgvector {row[1]}" if row else "unknown"

    @staticmethod
    def _vector(vector: list[float]) -> str:
        return "[" + ",".join(f"{value:.8g}" for value in vector) + "]"

    def recreate(self, cfg: dict) -> None:
        dim = int(cfg["dataset"]["dimensions"])
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            cur.execute(f"""CREATE TABLE {self.table} (
                id BIGINT PRIMARY KEY, embedding vector({dim}) NOT NULL,
                tenant INTEGER NOT NULL, level INTEGER NOT NULL, category INTEGER NOT NULL,
                category5 INTEGER NOT NULL, age INTEGER NOT NULL, parity TEXT NOT NULL)""")

    def insert_batch(self, start: int, vectors, payloads: list[dict]) -> None:
        rows = [(start + index, self._vector(vector.tolist()), payload["tenant"], payload["level"],
                 payload["category"], payload["category5"], payload["age"], payload["parity"])
                for index, (vector, payload) in enumerate(zip(vectors, payloads))]
        with self.conn.cursor() as cur:
            execute_values(cur, f"INSERT INTO {self.table} (id,embedding,tenant,level,category,category5,age,parity) VALUES %s", rows, page_size=500)

    def build_indexes(self, cfg: dict) -> None:
        index = cfg["index"]
        with self.conn.cursor() as cur:
            cur.execute("SET maintenance_work_mem='2GB'")
            cur.execute(f"CREATE INDEX {self.table}_hnsw ON {self.table} USING hnsw (embedding vector_cosine_ops) WITH (m={int(index['hnsw_m'])}, ef_construction={int(index['ef_construct'])})")
            for field in ("tenant", "level", "category", "category5", "age", "parity"):
                cur.execute(f"CREATE INDEX {self.table}_{field} ON {self.table} ({field})")
            cur.execute(f"ANALYZE {self.table}")

    def ensure_ready(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            if cur.fetchone()[0] == 0:
                raise RuntimeError("table is empty")

    def state(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return {"table": self.table, "rows": cur.fetchone()[0], "version": self.version}

    def reusable_state(self, cfg: dict) -> dict | None:
        """Return state only when row count, vector type, and indexes match."""
        try:
            expected = int(cfg["dataset"]["points"])
            dimensions = int(cfg["dataset"]["dimensions"])
            required_indexes = {
                f"{self.table}_hnsw",
                *(f"{self.table}_{field}" for field in
                  ("tenant", "level", "category", "category5", "age", "parity")),
            }
            with self.conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s)", (self.table,))
                if cur.fetchone()[0] is None:
                    return None
                cur.execute(f"SELECT count(*) FROM {self.table}")
                if int(cur.fetchone()[0]) != expected:
                    return None
                cur.execute(
                    "SELECT format_type(a.atttypid,a.atttypmod) "
                    "FROM pg_attribute a WHERE a.attrelid=%s::regclass "
                    "AND a.attname='embedding' AND NOT a.attisdropped",
                    (self.table,),
                )
                vector_type = str(cur.fetchone()[0]).replace(" ", "").lower()
                if vector_type != f"vector({dimensions})":
                    return None
                cur.execute(
                    "SELECT indexname,indexdef FROM pg_indexes "
                    "WHERE schemaname=current_schema() AND tablename=%s",
                    (self.table,),
                )
                indexes = {name: definition for name, definition in cur.fetchall()}
                if not required_indexes.issubset(indexes):
                    return None
                hnsw_definition = indexes[f"{self.table}_hnsw"].lower()
                if " using hnsw " not in hnsw_definition or "vector_cosine_ops" not in hnsw_definition:
                    return None
                cur.execute(
                    "SELECT reloptions FROM pg_class WHERE oid=%s::regclass",
                    (f"{self.table}_hnsw",),
                )
                options = set(cur.fetchone()[0] or [])
                if f"m={int(cfg['index']['hnsw_m'])}" not in options:
                    return None
                if f"ef_construction={int(cfg['index']['ef_construct'])}" not in options:
                    return None
            return self.state()
        except Exception:
            return None

    def query_ids(self, expression: str) -> list[int]:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {self.table} WHERE {expression} ORDER BY id")
            return [int(row[0]) for row in cur.fetchall()]

    def search_ann(self, vector: list[float], expression: str, top_k: int, ef: int):
        with self.conn.cursor() as cur:
            cur.execute("SET enable_seqscan=off; SET enable_indexscan=on; SET hnsw.ef_search=%s", (ef,))
            started = time.perf_counter()
            cur.execute(f"SELECT id FROM {self.table} WHERE {expression} ORDER BY embedding <=> %s::vector LIMIT %s",
                        (self._vector(vector), top_k))
            ids = [int(row[0]) for row in cur.fetchall()]
            elapsed = (time.perf_counter() - started) * 1000
        return ids, elapsed

    def search_exact(self, vector: list[float], expression: str, top_k: int):
        with self.conn.cursor() as cur:
            cur.execute("SET enable_seqscan=on; SET enable_indexscan=off; SET enable_bitmapscan=off")
            started = time.perf_counter()
            cur.execute(f"SELECT id FROM {self.table} WHERE {expression} ORDER BY embedding <=> %s::vector LIMIT %s",
                        (self._vector(vector), top_k))
            ids = [int(row[0]) for row in cur.fetchall()]
            elapsed = (time.perf_counter() - started) * 1000
        return ids, elapsed
