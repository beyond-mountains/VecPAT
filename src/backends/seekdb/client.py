from __future__ import annotations

import time

import pymysql


class SeekDBBackend:
    name = "seekdb"

    def __init__(self, host: str, port: int, user: str, password: str, database: str, table: str):
        self.table = table
        deadline = time.time() + 600
        last_error = None
        while time.time() < deadline:
            try:
                bootstrap = pymysql.connect(host=host, port=port, user=user, password=password,
                                            autocommit=True, connect_timeout=10)
                with bootstrap.cursor() as cur:
                    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
                bootstrap.close()
                self.conn = pymysql.connect(host=host, port=port, user=user, password=password,
                                            database=database, autocommit=True, connect_timeout=30,
                                            read_timeout=1800, write_timeout=1800)
                break
            except Exception as exc:
                last_error = exc; time.sleep(5)
        else:
            raise RuntimeError(f"seekDB did not become ready: {last_error}")
        with self.conn.cursor() as cur:
            cur.execute("SELECT VERSION()")
            self.version = str(cur.fetchone()[0])

    @staticmethod
    def _vector(vector: list[float]) -> str:
        return "[" + ",".join(f"{value:.8g}" for value in vector) + "]"

    def recreate(self, cfg: dict) -> None:
        dim = int(cfg["dataset"]["dimensions"])
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table}")
            cur.execute(f"""CREATE TABLE {self.table} (
                id BIGINT NOT NULL PRIMARY KEY, embedding VECTOR({dim}) NOT NULL,
                tenant INT NOT NULL, level INT NOT NULL, category INT NOT NULL,
                category5 INT NOT NULL, age INT NOT NULL, parity VARCHAR(8) NOT NULL,
                KEY idx_tenant(tenant), KEY idx_level(level), KEY idx_category(category),
                KEY idx_category5(category5), KEY idx_age(age), KEY idx_parity(parity)
            ) ORGANIZATION HEAP""")

    def insert_batch(self, start: int, vectors, payloads: list[dict]) -> None:
        rows = [(start + index, self._vector(vector.tolist()), payload["tenant"], payload["level"],
                 payload["category"], payload["category5"], payload["age"], payload["parity"])
                for index, (vector, payload) in enumerate(zip(vectors, payloads))]
        sql = f"INSERT INTO {self.table} (id,embedding,tenant,level,category,category5,age,parity) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"
        with self.conn.cursor() as cur:
            for offset in range(0, len(rows), 200):
                cur.executemany(sql, rows[offset:offset + 200])

    def build_indexes(self, cfg: dict) -> None:
        index = cfg["index"]
        with self.conn.cursor() as cur:
            cur.execute(f"""CREATE VECTOR INDEX {self.table}_hnsw ON {self.table}(embedding)
                WITH (distance=l2, type=hnsw, lib=vsag, m={int(index['hnsw_m'])},
                      ef_construction={int(index['ef_construct'])}, sync_mode=immediate)""")
            try:
                cur.execute("CALL DBMS_INDEX_MANAGER.REFRESH()")
            except Exception:
                pass
            cur.execute(f"ANALYZE TABLE {self.table}")

    def ensure_ready(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            if int(cur.fetchone()[0]) == 0:
                raise RuntimeError("table is empty")

    def state(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            return {"table": self.table, "rows": int(cur.fetchone()[0]), "version": self.version}

    def _reusable_data_details(self, cfg: dict) -> dict | None:
        """Inspect the uploaded table independently from its vector index."""
        expected = int(cfg["dataset"]["points"])
        dimensions = int(cfg["dataset"]["dimensions"])
        required_metadata_indexes = {
            "idx_tenant", "idx_level", "idx_category", "idx_category5",
            "idx_age", "idx_parity",
        }
        with self.conn.cursor() as cur:
            cur.execute("SHOW TABLES LIKE %s", (self.table,))
            if cur.fetchone() is None:
                return None
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            if int(cur.fetchone()[0]) != expected:
                return None
            cur.execute(f"SHOW CREATE TABLE {self.table}")
            create_sql = str(cur.fetchone()[1]).lower()
            normalized = "".join(
                create_sql.replace("`", "").replace("'", "").split()
            )
            if f"vector({dimensions})" not in normalized:
                return None
            cur.execute(f"SHOW INDEX FROM {self.table}")
            index_names = {str(row[2]) for row in cur.fetchall()}
            if not required_metadata_indexes.issubset(index_names):
                return None
        return {"normalized_create_sql": normalized, "index_names": index_names}

    def reusable_data_without_vector_index(self, cfg: dict) -> dict | None:
        """Reuse a complete data table when only the HNSW index is absent."""
        try:
            details = self._reusable_data_details(cfg)
            if details is None:
                return None
            if f"{self.table}_hnsw" in details["index_names"]:
                return None
            return self.state()
        except Exception:
            return None

    def reusable_state(self, cfg: dict) -> dict | None:
        """Return state only when row count, vector schema, and indexes match."""
        try:
            details = self._reusable_data_details(cfg)
            if details is None:
                return None
            normalized = details["normalized_create_sql"]
            if f"{self.table}_hnsw" not in details["index_names"]:
                return None
            if "type=hnsw" not in normalized:
                return None
            if f"m={int(cfg['index']['hnsw_m'])}" not in normalized:
                return None
            if f"ef_construction={int(cfg['index']['ef_construct'])}" not in normalized:
                return None
            return self.state()
        except Exception:
            return None

    def query_ids(self, expression: str) -> list[int]:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {self.table} WHERE {expression} ORDER BY id")
            return [int(row[0]) for row in cur.fetchall()]

    def search_ann(self, vector: list[float], expression: str, top_k: int, ef: int):
        sql = (f"SELECT id FROM {self.table} WHERE {expression} "
               f"ORDER BY l2_distance(embedding, %s) APPROXIMATE LIMIT {int(top_k)}")
        with self.conn.cursor() as cur:
            cur.execute("SET ob_hnsw_ef_search = %s", (int(ef),))
            started = time.perf_counter(); cur.execute(sql, (self._vector(vector),))
            ids = [int(row[0]) for row in cur.fetchall()]
            elapsed = (time.perf_counter() - started) * 1000
        return ids, elapsed

    def search_exact(self, vector: list[float], expression: str, top_k: int):
        sql = (f"SELECT id FROM {self.table} WHERE {expression} "
               f"ORDER BY l2_distance(embedding, %s) LIMIT {int(top_k)}")
        with self.conn.cursor() as cur:
            started = time.perf_counter(); cur.execute(sql, (self._vector(vector),))
            ids = [int(row[0]) for row in cur.fetchall()]
            elapsed = (time.perf_counter() - started) * 1000
        return ids, elapsed
