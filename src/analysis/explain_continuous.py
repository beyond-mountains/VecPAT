from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from ..core.dataset import generate_dataset


def _measured_row(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    measured = [row for row in rows if "latency_ratio" in row]
    if not measured:
        return rows[0] if rows else {}
    return max(measured, key=lambda row: float(row.get("latency_ratio") or 0))


def _vector_text(vector) -> str:
    return "[" + ",".join(f"{float(value):.8g}" for value in vector) + "]"


def explain_sql(backend: str, cfg: dict, row: dict) -> dict:
    _, _, _, centers = generate_dataset(cfg)
    vector = centers[0]
    table = cfg["experiment"]["table"]
    k, ef = int(row.get("top_k") or cfg["workload"]["top_k"][0]), int(row.get("ef_search") or cfg["workload"]["ef_search"][0])
    plans = {}
    if backend == "pgvector":
        import psycopg2
        conn = psycopg2.connect(os.environ["PGVECTOR_DSN"]); conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan=off; SET enable_indexscan=on; SET hnsw.ef_search=%s", (max(k, ef),))
            for label, expression in (("original", row["base_filter"]), ("equivalent", row["variant_filter"])):
                sql = f"SELECT id FROM {table} WHERE {expression} ORDER BY embedding <=> %s::vector LIMIT {k}"
                cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, (_vector_text(vector),))
                plans[label] = {"sql": sql.replace("%s", "<query_vector>"), "plan": cur.fetchone()[0]}
        conn.close()
    elif backend == "seekdb":
        import pymysql
        conn = pymysql.connect(host=os.getenv("SEEKDB_HOST", "seekdb"), port=int(os.getenv("SEEKDB_PORT", "2881")),
                               user=os.getenv("SEEKDB_USER", "root"), password=os.environ["SEEKDB_PASSWORD"],
                               database=os.getenv("SEEKDB_DATABASE", "filter_anomaly"), autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SET ob_hnsw_ef_search=%s", (max(k, ef),))
            for label, expression in (("original", row["base_filter"]), ("equivalent", row["variant_filter"])):
                sql = f"SELECT id FROM {table} WHERE {expression} ORDER BY l2_distance(embedding, %s) APPROXIMATE LIMIT {k}"
                cur.execute("EXPLAIN " + sql, (_vector_text(vector),))
                plans[label] = {"sql": sql.replace("%s", "<query_vector>"), "plan": [list(value) for value in cur.fetchall()]}
        conn.close()
    return {"backend": backend, "case_id": cfg.get("continuous_case", {}).get("case_id"),
            "explain_available": True, "plans": plans}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("milvus", "pgvector", "qdrant", "seekdb", "weaviate"))
    parser.add_argument("--config", required=True); parser.add_argument("--result", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); row = _measured_row(Path(args.result))
    if args.backend in {"pgvector", "seekdb"} and row.get("base_filter") and row.get("variant_filter"):
        payload = explain_sql(args.backend, cfg, row)
    else:
        payload = {"backend": args.backend, "case_id": cfg.get("continuous_case", {}).get("case_id"),
                   "explain_available": False,
                   "reason": "This backend API does not expose a SQL-style EXPLAIN plan for this query path.",
                   "request_metadata": {key: row.get(key) for key in ("collection", "table", "pattern_id", "case_name", "top_k", "ef_search")}}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output), "explain_available": payload["explain_available"]}))


if __name__ == "__main__":
    main()
