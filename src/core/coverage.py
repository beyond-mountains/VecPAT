from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def case_id(config: dict[str, Any]) -> str:
    """Stable identity for c=(r,t,s,e,b)."""
    identity = {key: config[key] for key in ("rule_id", "transformation", "semantic", "execution", "backend")}
    return hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:20]


def physical_dataset_spec(config: dict[str, Any]) -> dict[str, Any]:
    """Return only inputs that change generated vectors or metadata."""
    semantic = config["semantic"]
    execution = config["execution"]
    spec: dict[str, Any] = {
        "generator": "core.dataset.generate_dataset:v1",
        "seed": int(semantic.get("dataset_seed", 20261002)),
        "points": int(semantic["points"]),
        "dimensions": int(execution["dimensions"]),
        "correlation": semantic.get("correlation"),
        "tenants": int(semantic.get("tenants", 100)),
        "levels": int(semantic.get("levels", 32)),
        "age_universe": int(semantic.get("age_universe", 1000)),
    }
    if config.get("layer") == "ann":
        # ANN campaigns use a separate generator whose distribution changes
        # the physical vectors. Query-only rewrites do not belong here.
        spec.update({
            "generator": "campaigns.ann._data:v2-no-exact-ties",
            "distribution": config.get("transformation", {}).get("distribution", "gaussian"),
        })
    return spec


def dataset_signature(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(physical_dataset_spec(config)).encode()).hexdigest()[:12]


class CoverageLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS configurations(
          case_id TEXT PRIMARY KEY, backend TEXT NOT NULL, layer TEXT NOT NULL,
          rule_id TEXT NOT NULL, case_name TEXT NOT NULL, priority INTEGER NOT NULL,
          config_json TEXT NOT NULL, dataset_signature TEXT,
          state TEXT NOT NULL DEFAULT 'pending',
          attempts INTEGER NOT NULL DEFAULT 0, confirmations INTEGER NOT NULL DEFAULT 0,
          latency_ratio REAL, recall_gap REAL, anomaly INTEGER,
          observed_selectivity REAL, eligible_count INTEGER,
          base_p50_ms REAL, variant_p50_ms REAL, exact_latency_ratio REAL,
          base_recall REAL, variant_recall REAL,
          result_path TEXT, explain_path TEXT, run_id TEXT,
          created_at REAL NOT NULL, updated_at REAL NOT NULL, last_error TEXT
        );
        CREATE INDEX IF NOT EXISTS configs_pick ON configurations(backend,state,priority,case_id);
        CREATE INDEX IF NOT EXISTS configs_pattern_pick ON configurations(backend,rule_id,state,priority,case_id);
        CREATE TABLE IF NOT EXISTS attempts(
          id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, run_id TEXT NOT NULL,
          mode TEXT NOT NULL, started_at REAL NOT NULL, finished_at REAL,
          status TEXT NOT NULL, result_path TEXT, latency_ratio REAL, recall_gap REAL, detail TEXT
        );
        """)
        existing = {row[1] for row in self.db.execute("PRAGMA table_info(configurations)")}
        for name, kind in (("observed_selectivity", "REAL"), ("eligible_count", "INTEGER"),
                           ("base_p50_ms", "REAL"), ("variant_p50_ms", "REAL"),
                           ("exact_latency_ratio", "REAL"), ("base_recall", "REAL"),
                           ("variant_recall", "REAL")):
            if name not in existing:
                self.db.execute(f"ALTER TABLE configurations ADD COLUMN {name} {kind}")
        if "dataset_signature" not in existing:
            self.db.execute("ALTER TABLE configurations ADD COLUMN dataset_signature TEXT")
        # Recompute signatures on startup so a generator-version change cannot
        # silently reuse physical collections created by an older generator.
        for row in self.db.execute(
            "SELECT case_id,config_json,dataset_signature FROM configurations"
        ).fetchall():
            config = json.loads(row["config_json"])
            expected_signature = dataset_signature(config)
            if row["dataset_signature"] != expected_signature:
                self.db.execute(
                    "UPDATE configurations SET dataset_signature=?,updated_at=? WHERE case_id=?",
                    (expected_signature, time.time(), row["case_id"]),
                )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS configs_dataset_pick "
            "ON configurations(backend,rule_id,state,dataset_signature,priority,case_id)"
        )
        self.db.commit()

    def add(self, config: dict[str, Any], priority: int) -> str:
        identifier = case_id(config)
        now = time.time()
        self.db.execute(
            "INSERT OR IGNORE INTO configurations(case_id,backend,layer,rule_id,case_name,priority,config_json,dataset_signature,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (identifier, config["backend"], config["layer"], config["rule_id"], config["case_name"],
             priority, canonical_json(config), dataset_signature(config), now, now),
        )
        return identifier

    def commit(self) -> None:
        self.db.commit()

    def recover(self, stale_seconds: int = 21600, pattern: str | None = None) -> int:
        cutoff = time.time() - stale_seconds
        # Retry transient infrastructure failures only when a new campaign or
        # explicit resume starts. This avoids a tight retry loop in one run.
        query = ("UPDATE configurations SET state='pending',run_id=NULL,updated_at=? "
                 "WHERE (state='retryable_failed' OR (state='running' AND updated_at<?))")
        parameters: list[Any] = [time.time(), cutoff]
        if pattern:
            query += " AND rule_id=?"
            parameters.append(pattern)
        cursor = self.db.execute(query, parameters)
        self.db.commit()
        return cursor.rowcount

    def claim(self, backend: str, run_id: str, deadline: float,
              pattern: str | None = None) -> tuple[dict[str, Any], str] | None:
        if time.time() >= deadline:
            return None
        self.db.execute("BEGIN IMMEDIATE")
        query = "SELECT * FROM configurations WHERE backend=? AND state IN ('candidate','pending')"
        parameters: list[Any] = [backend]
        if pattern:
            query += " AND rule_id=?"
            parameters.append(pattern)
        query += (" ORDER BY CASE state WHEN 'candidate' THEN 0 ELSE 1 END,"
                  "dataset_signature,priority,case_id LIMIT 1")
        row = self.db.execute(query, parameters).fetchone()
        if row is None:
            self.db.commit()
            return None
        mode = "confirmation" if row["state"] == "candidate" else "discovery"
        now = time.time()
        self.db.execute("UPDATE configurations SET state='running',run_id=?,attempts=attempts+1,updated_at=? WHERE case_id=?",
                        (run_id, now, row["case_id"]))
        self.db.execute("INSERT INTO attempts(case_id,run_id,mode,started_at,status) VALUES(?,?,?,?,?)",
                        (row["case_id"], run_id, mode, now, "running"))
        self.db.commit()
        return json.loads(row["config_json"]), mode

    def finish(self, identifier: str, run_id: str, mode: str, result_path: str,
               status: str, latency_ratio: float | None, recall_gap: float | None,
               detail: str = "", metrics: dict[str, Any] | None = None) -> None:
        metrics = metrics or {}
        # Recall remains a diagnostic measurement, but it does not promote a
        # case to candidate/anomaly.  VecPAT's default performance oracle is
        # exclusively the symmetric end-to-end latency ratio.
        anomaly = bool((latency_ratio or 0) > 2.0)
        if status in {"unsupported", "semantic_mismatch", "failed", "retryable_failed"}:
            state = status
        elif mode == "discovery" and anomaly:
            state = "candidate"
        else:
            state = "complete"
        now = time.time()
        self.db.execute(
            "UPDATE configurations SET state=?,latency_ratio=?,recall_gap=?,anomaly=?,result_path=?,"
            "confirmations=confirmations+?,updated_at=?,last_error=?,observed_selectivity=?,eligible_count=?,"
            "base_p50_ms=?,variant_p50_ms=?,exact_latency_ratio=?,base_recall=?,variant_recall=? WHERE case_id=?",
            (state, latency_ratio, recall_gap, int(anomaly), result_path,
             1 if mode == "confirmation" else 0, now, detail or None,
             metrics.get("observed_selectivity"), metrics.get("base_count"), metrics.get("base_p50_ms"),
             metrics.get("variant_p50_ms"), metrics.get("exact_latency_ratio"),
             metrics.get("base_recall"), metrics.get("variant_recall"), identifier),
        )
        self.db.execute(
            "UPDATE attempts SET finished_at=?,status=?,result_path=?,latency_ratio=?,recall_gap=?,detail=? "
            "WHERE id=(SELECT id FROM attempts WHERE case_id=? AND run_id=? AND mode=? ORDER BY id DESC LIMIT 1)",
            (now, status, result_path, latency_ratio, recall_gap, detail, identifier, run_id, mode),
        )
        self.db.commit()

    def set_explain(self, identifier: str, path: str) -> None:
        self.db.execute("UPDATE configurations SET explain_path=?,updated_at=? WHERE case_id=?", (path, time.time(), identifier))
        self.db.commit()

    def snapshot(self, path: str | Path, anomalies_path: str | Path | None = None,
                 pattern: str | None = None) -> None:
        query = "SELECT * FROM configurations"
        parameters: tuple[str, ...] = ()
        if pattern:
            query += " WHERE rule_id=?"
            parameters = (pattern,)
        query += " ORDER BY backend,priority,case_id"
        rows = self.db.execute(query, parameters).fetchall()
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        database_fields = [column[1] for column in self.db.execute("PRAGMA table_info(configurations)")]
        axis_fields = ["transformation", "semantic", "execution", "strength", "ast_shape", "selectivity_target", "eligible_set_size",
                       "correlation", "points", "index_type", "top_k", "ef_search", "dimensions"]
        fields = database_fields + axis_fields
        expanded = []
        for raw in rows:
            row = dict(raw); config = json.loads(row["config_json"])
            transformation, semantic, execution = config["transformation"], config["semantic"], config["execution"]
            row.update({
                "transformation": canonical_json(transformation), "semantic": canonical_json(semantic),
                "execution": canonical_json(execution), "strength": transformation.get("strength"),
                "ast_shape": transformation.get("ast") or transformation.get("variant"),
                "selectivity_target": semantic.get("selectivity_target"), "eligible_set_size": semantic.get("eligible_set_size"),
                "correlation": semantic.get("correlation"),
                "points": semantic.get("points"), "index_type": execution.get("index"),
                "top_k": execution.get("top_k"), "ef_search": execution.get("ef_search"),
                "dimensions": execution.get("dimensions"),
            })
            expanded.append(row)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(expanded)
        if anomalies_path is not None:
            anomaly_path = Path(anomalies_path); anomaly_path.parent.mkdir(parents=True, exist_ok=True)
            with anomaly_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader(); writer.writerows(row for row in expanded if row.get("anomaly"))

    def summary(self, pattern: str | None = None) -> dict[str, Any]:
        query = "SELECT backend,state,count(*) n FROM configurations"
        parameters: tuple[str, ...] = ()
        if pattern:
            query += " WHERE rule_id=?"
            parameters = (pattern,)
        query += " GROUP BY backend,state"
        rows = self.db.execute(query, parameters).fetchall()
        return {"ledger": str(self.path), "coverage": [dict(row) for row in rows]}
