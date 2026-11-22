from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ...core.confirmation import balanced_random_orders, serialize_orders
from ...core.dataset import generate_dataset
from ...generation.equivalent import Expr, _features
from ...generation.campaign_1002 import build_filter_instances_1002
from ...generation.continuous import build_continuous_filter_instances
from ...generation.comparative import build_comparative_instances
from .client import WeaviateBackend, render_where


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values), q))


def _recall(found: list[int], truth: list[int]) -> float:
    denominator = len(truth)
    return len(set(found) & set(truth)) / denominator if denominator else 1.0


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _expression_evaluator(payloads: list[dict]):
    columns = {
        key: np.asarray([row[key] for row in payloads])
        for key in ("tenant", "level", "category", "category5", "age", "parity")
    }

    @lru_cache(maxsize=None)
    def evaluate(expr: Expr) -> np.ndarray:
        op, args = expr.op, expr.args
        if op == "and":
            result = np.ones(len(payloads), dtype=bool)
            for child in args:
                result &= evaluate(child)
            return result
        if op == "or":
            result = np.zeros(len(payloads), dtype=bool)
            for child in args:
                result |= evaluate(child)
            return result
        if op == "not":
            return ~evaluate(args[0])
        values = columns[str(args[0])]
        operands = args[1:]
        if op == "eq":
            return values == operands[0]
        if op == "in":
            return np.isin(values, operands)
        if op == "between":
            return (values >= operands[0]) & (values <= operands[1])
        if op == "lt":
            return values < operands[0]
        if op == "lte":
            return values <= operands[0]
        if op == "gt":
            return values > operands[0]
        if op == "gte":
            return values >= operands[0]
        raise ValueError(op)

    return evaluate


def setup(backend: WeaviateBackend, cfg: dict) -> dict:
    exp, ds, index = cfg["experiment"], cfg["dataset"], cfg["index"]
    vectors, payloads, _, _ = generate_dataset(cfg)
    ids = np.arange(int(ds["points"]), dtype=np.int64)
    states = {}
    for ef in cfg["workload"]["ef_search"]:
        class_name = f"{exp['class_prefix']}Ef{int(ef)}"
        existing = backend.reusable_collection(
            class_name,
            int(ds["points"]),
            int(ds["dimensions"]),
            index,
            int(ef),
            timeout_s=30,
        )
        if existing is not None:
            states[class_name] = existing
            print(json.dumps({"collection_reused": class_name, "state": existing}), flush=True)
            continue
        backend.recreate_collection(class_name, int(ds["dimensions"]), index, int(ef))
        backend.upload(class_name, ids, vectors, payloads, int(ds["batch_size"]))
        states[class_name] = backend.wait_ready(class_name, int(ds["points"]))
        print(json.dumps({"collection_ready": class_name, "state": states[class_name]}), flush=True)
    return states


def _candidate_keys(path: Path | None, threshold: float) -> set[tuple[str, int, int, int]]:
    if path is None or not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "executed" and float(row.get("latency_ratio") or 0) > threshold:
            keys.add((str(row["case_name"]), int(row["scale"]), int(row["top_k"]), int(row["ef_search"])))
    return keys


def run(
    backend: WeaviateBackend,
    cfg: dict,
    output: Path,
    mode: str = "discovery",
    candidate_source: Path | None = None,
) -> list[dict]:
    exp, ds, oracle = cfg["experiment"], cfg["dataset"], cfg["oracle"]
    work = cfg["workload"] if mode == "discovery" else cfg["confirmation"]
    vectors, payloads, tenants, centers = generate_dataset(cfg)
    evaluate = _expression_evaluator(payloads)
    if cfg["workload"].get("comparative_method"):
        pairs = build_comparative_instances(
            cfg["workload"]["comparative_method"],
            cfg["workload"]["transformation_strengths"],
            int(ds.get("age_universe", 1000)),
            generation_round=int(cfg["workload"].get("comparative_generation_round", 0)),
        )
        # Weaviate's property Filter DSL cannot evaluate field/constant
        # arithmetic expressions.
        pairs = [item for item in pairs if item.name not in {
            "constant_expression_reduction",
            "arithmetic_constant_folding",
            "nested_expression_folding",
        }]
    else:
        builder = build_continuous_filter_instances if cfg["workload"].get("campaign_continuous") else build_filter_instances_1002
        pairs = builder(cfg["workload"]["transformation_strengths"], int(ds.get("age_universe", 1000)))
    if cfg["workload"].get("patterns"):
        patterns = set(cfg["workload"]["patterns"])
        pairs = [pair for pair in pairs if pair.name in patterns or pair.rule_id in patterns]
    if cfg["workload"].get("case_names"):
        case_names = set(cfg["workload"]["case_names"])
        pairs = [pair for pair in pairs if pair.case_name in case_names]
    candidate_keys = _candidate_keys(candidate_source, float(oracle["latency_ratio"]))
    if mode == "confirmation" and not candidate_keys:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")
        return []

    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    count_validation_class = f"{exp['class_prefix']}Ef{int(cfg['workload']['ef_search'][0])}"
    with output.open("w", encoding="utf-8") as stream:
        for pair_index, item in enumerate(pairs):
            base_where = render_where(item.original)
            variant_where = render_where(item.equivalent)
            base_mask = evaluate(item.original)
            variant_mask = evaluate(item.equivalent)
            semantic_equal = bool(np.array_equal(base_mask, variant_mask))
            eligible_ids = np.flatnonzero(base_mask)
            expected_count = int(len(eligible_ids))
            count_error = ""
            count_base = count_variant = -1
            count_base_ms = count_variant_ms = None
            try:
                count_base, count_base_ms = backend.aggregate_count(count_validation_class, base_where)
                count_variant, count_variant_ms = backend.aggregate_count(count_validation_class, variant_where)
                backend_semantic_count_equal = (
                    count_base == count_variant == expected_count
                )
            except Exception as exc:
                backend_semantic_count_equal = False
                count_error = f"{type(exc).__name__}: {exc}"

            common = {
                "timestamp": time.time(),
                "backend": "weaviate",
                "experiment": exp["name"],
                "mode": mode,
                "pattern_id": item.rule_id,
                "pattern": item.name,
                "case_name": item.case_name,
                "scale": item.strength,
                "layer": item.layer,
                "preconditions": item.preconditions,
                "equivalence": item.equivalence,
                "query_signature": {**_features(item.equivalent), "strength": item.strength},
                "selectivity_label": item.selectivity_label,
                "observed_selectivity": expected_count / int(ds["points"]),
                "base_count": expected_count,
                "variant_count": int(np.count_nonzero(variant_mask)),
                "semantic_equal": semantic_equal,
                "backend_base_count": count_base,
                "backend_variant_count": count_variant,
                "backend_semantic_count_equal": backend_semantic_count_equal,
                "backend_count_base_ms": count_base_ms,
                "backend_count_variant_ms": count_variant_ms,
                "base_filter": base_where,
                "variant_filter": variant_where,
            }
            if not semantic_equal or not backend_semantic_count_equal:
                row = {
                    **common,
                    "status": "unsupported" if count_error else "semantic_mismatch",
                    "error": count_error or "eligible counts do not match reference interpreter",
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(row)
                print(
                    f"[{pair_index + 1}/{len(pairs)}] {item.rule_id}/{item.case_name}: "
                    f"{row['status']} {row['error'][:180]}",
                    flush=True,
                )
                continue

            eligible_tenants = sorted(set(int(tenants[index]) for index in eligible_ids))
            pair_max_ratio = 0.0
            for ef in cfg["workload"]["ef_search"]:
                class_name = f"{exp['class_prefix']}Ef{int(ef)}"
                for top_k in cfg["workload"]["top_k"]:
                    key = (item.case_name, int(item.strength), int(top_k), int(ef))
                    if mode == "confirmation" and key not in candidate_keys:
                        continue
                    query_rows = []
                    for query_index in range(int(work["queries_per_case"])):
                        target = eligible_tenants[query_index % len(eligible_tenants)] if eligible_tenants else 0
                        query_rng = np.random.default_rng(
                            _stable_seed(exp["seed"], item.case_name, item.strength, query_index)
                        )
                        query = centers[target] + query_rng.normal(
                            0, 0.03, size=int(ds["dimensions"])
                        )
                        query = (query / np.linalg.norm(query)).astype(np.float32)
                        if len(eligible_ids):
                            scores = vectors[eligible_ids] @ query
                            order = np.argsort(-scores, kind="stable")[: min(int(top_k), len(eligible_ids))]
                            truth = [int(value) for value in eligible_ids[order]]
                        else:
                            truth = []

                        warmup_orders = balanced_random_orders(
                            int(work["warmups"]), int(work["order_seed"]),
                            "weaviate", item.case_name, item.strength, top_k, ef,
                            query_index, mode, "warmup",
                        )
                        for execution_order in warmup_orders:
                            for label in execution_order:
                                filt = base_where if label == "base" else variant_where
                                backend.search(class_name, query.astype(float).tolist(), filt, int(top_k))

                        samples = {"base": [], "variant": []}
                        recalls = {"base": [], "variant": []}
                        ids_by_side: dict[str, list[int]] = {"base": [], "variant": []}
                        measurement_orders = balanced_random_orders(
                            int(work["repetitions"]), int(work["order_seed"]),
                            "weaviate", item.case_name, item.strength, top_k, ef,
                            query_index, mode, "measurement",
                        )
                        for execution_order in measurement_orders:
                            for label in execution_order:
                                filt = base_where if label == "base" else variant_where
                                found, metrics = backend.search(
                                    class_name, query.astype(float).tolist(), filt, int(top_k)
                                )
                                ids_by_side[label] = found
                                samples[label].append(float(metrics["client_wall_ms"]))
                                recalls[label].append(_recall(found, truth))
                        query_rows.append(
                            {
                                "query_index": query_index,
                                "base_ms": statistics.median(samples["base"]),
                                "variant_ms": statistics.median(samples["variant"]),
                                "base_recall": statistics.mean(recalls["base"]),
                                "variant_recall": statistics.mean(recalls["variant"]),
                                "base_ids": ids_by_side["base"],
                                "variant_ids": ids_by_side["variant"],
                                "truth_ids": truth,
                                "samples": samples,
                                "recalls": recalls,
                                "warmup_orders": serialize_orders(warmup_orders),
                                "measurement_orders": serialize_orders(measurement_orders),
                            }
                        )

                    base_values = [row["base_ms"] for row in query_rows]
                    variant_values = [row["variant_ms"] for row in query_rows]
                    base_p50 = statistics.median(base_values)
                    variant_p50 = statistics.median(variant_values)
                    slower = max(base_p50, variant_p50)
                    faster = max(min(base_p50, variant_p50), 1e-9)
                    ratio = slower / faster
                    pair_max_ratio = max(pair_max_ratio, ratio)
                    base_recall = statistics.mean(row["base_recall"] for row in query_rows)
                    variant_recall = statistics.mean(row["variant_recall"] for row in query_rows)
                    row = {
                        **common,
                        "status": "executed",
                        "collection": class_name,
                        "top_k": int(top_k),
                        "ef_search": int(ef),
                        "base_p50_ms": base_p50,
                        "variant_p50_ms": variant_p50,
                        "base_p95_ms": _quantile(base_values, 0.95),
                        "variant_p95_ms": _quantile(variant_values, 0.95),
                        "latency_ratio": ratio,
                        "variant_over_base": variant_p50 / max(base_p50, 1e-9),
                        "base_recall": base_recall,
                        "variant_recall": variant_recall,
                        "recall_gap": abs(base_recall - variant_recall),
                        "performance_anomaly": ratio > float(oracle["latency_ratio"]),
                        "exact_oracle": "NumPy exact cosine Top-K over reference eligible IDs",
                        "exact_results_equal": semantic_equal,
                        "order_protocol": "balanced_random_ab_ba",
                        "internal_metrics_available": False,
                        "internal_metric_note": (
                            "Weaviate public GraphQL API does not expose request-scoped visited nodes, "
                            "distance computations, or candidate counts."
                        ),
                        "query_details": query_rows,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    records.append(row)
                    if ratio > float(oracle["latency_ratio"]):
                        print(
                            f"candidate {item.rule_id}/{item.case_name} scale={item.strength} "
                            f"k={top_k} ef={ef} ratio={ratio:.2f}",
                            flush=True,
                        )
            print(
                f"[{pair_index + 1}/{len(pairs)}] {item.rule_id}/{item.case_name} "
                f"eligible={expected_count} max_ratio={pair_max_ratio:.2f}",
                flush=True,
            )
    return records


def report(jsonl: Path, output_prefix: Path, threshold: float = 2.0) -> dict:
    rows = [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if jsonl.exists() else []
    valid = [row for row in rows if row.get("status") == "executed"]
    anomalies = sorted(
        [row for row in valid if float(row["latency_ratio"]) > threshold],
        key=lambda row: float(row["latency_ratio"]),
        reverse=True,
    )
    fields = [
        "backend", "mode", "pattern_id", "pattern", "case_name", "scale",
        "observed_selectivity", "top_k", "ef_search", "semantic_equal",
        "backend_semantic_count_equal", "base_p50_ms", "variant_p50_ms",
        "latency_ratio", "variant_over_base", "base_p95_ms", "variant_p95_ms",
        "base_recall", "variant_recall", "recall_gap", "performance_anomaly",
        "collection", "status",
    ]
    _write_csv(output_prefix.with_suffix(".csv"), valid, fields)
    _write_csv(output_prefix.with_name(output_prefix.name + "_latency_gt_2.csv"), anomalies, fields)
    errors = [row for row in rows if row.get("status") != "executed"]
    _write_csv(
        output_prefix.with_name(output_prefix.name + "_unsupported.csv"),
        errors,
        ["pattern_id", "pattern", "case_name", "scale", "status", "error"],
    )
    summary = {
        "source": str(jsonl),
        "total_records": len(rows),
        "executed_configurations": len(valid),
        "unsupported_or_semantic_failures": len(errors),
        "latency_ratio_gt_2": len(anomalies),
        "maximum_latency_ratio": max((float(row["latency_ratio"]) for row in valid), default=None),
    }
    output_prefix.with_name(output_prefix.name + "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary
