from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from ...core.dataset import generate_dataset
from .filters import build_sql_campaign_pairs
from ...generation.equivalent import emit_filter_pairs
from ...generation.campaign_1002 import emit_filter_pairs_1002
from ...generation.continuous import emit_filter_pairs_continuous
from ...generation.comparative import emit_comparative_pairs
from ...core.confirmation import (
    balanced_random_orders,
    filter_candidate_key,
    load_candidate_keys,
    serialize_orders,
)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _hash_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def _recall(found: list[int], truth: list[int], limit: int) -> float:
    denominator = min(limit, len(truth))
    return len(set(found) & set(truth)) / denominator if denominator else 1.0


def setup(backend, cfg: dict) -> dict:
    reusable = getattr(backend, "reusable_state", None)
    if reusable is not None:
        existing = reusable(cfg)
        if existing is not None:
            print(json.dumps({"database_reused": existing}, default=str), flush=True)
            return existing
    reusable_data = getattr(backend, "reusable_data_without_vector_index", None)
    if reusable_data is not None:
        existing_data = reusable_data(cfg)
        if existing_data is not None:
            print(json.dumps({"database_data_reused": existing_data,
                              "action": "build_missing_vector_index"}, default=str), flush=True)
            backend.build_indexes(cfg)
            state = backend.state()
            print(json.dumps({"database_ready": state}, default=str), flush=True)
            return state
    vectors, payloads, _, _ = generate_dataset(cfg)
    backend.recreate(cfg)
    batch_size = int(cfg["dataset"]["batch_size"])
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        backend.insert_batch(start, vectors[start:end], payloads[start:end])
        if start % (batch_size * 10) == 0:
            print(f"uploaded {end}/{len(vectors)}", flush=True)
    backend.build_indexes(cfg)
    state = backend.state()
    print(json.dumps({"database_ready": state}, default=str), flush=True)
    return state


def run(backend, cfg: dict, output: Path) -> list[dict]:
    exp, ds, work, oracle = cfg["experiment"], cfg["dataset"], cfg["workload"], cfg["oracle"]
    backend.ensure_ready()
    _, _, tenant, centers = generate_dataset(cfg)
    if work.get("comparative_method"):
        pairs = emit_comparative_pairs(
            backend.name, work["comparative_method"],
            work.get("transformation_strengths", [2, 8, 32, 128]),
            int(ds.get("age_universe", 1000)),
            relation_name=exp["table"],
            generation_round=int(work.get("comparative_generation_round", 0)),
        )
    elif work.get("campaign_continuous"):
        pairs = emit_filter_pairs_continuous(backend.name, work.get("transformation_strengths", [2, 8, 32, 128]),
                                             int(ds.get("age_universe", 1000)))
    elif work.get("campaign_1002"):
        pairs = emit_filter_pairs_1002(backend.name, work.get("transformation_strengths", [2, 8, 32, 128]),
                                       int(ds.get("age_universe", 1000)))
    elif work.get("campaign_1001"):
        pairs = emit_filter_pairs(backend.name, work.get("transformation_strengths", [2, 8, 32]),
                                  int(ds.get("age_universe", 1000)))
    else:
        pairs = build_sql_campaign_pairs(work["redundancy_scales"], work["set_widths"],
                                         int(ds.get("age_universe", 1000)))
    if work.get("patterns"):
        pairs = [pair for pair in pairs if pair.pattern in set(work["patterns"])]
    if work.get("case_names"):
        pairs = [pair for pair in pairs if pair.case_name in set(work["case_names"])]
    candidate_keys = load_candidate_keys(cfg, backend.name, "filter")
    if candidate_keys:
        candidate_case_names = {key[1] for key in candidate_keys}
        pairs = [pair for pair in pairs if pair.case_name in candidate_case_names]
    candidates = {tuple(item) for item in work.get("candidate_cases", [])}
    rng = np.random.default_rng(int(work.get("query_seed", int(exp["seed"]) + 1)))
    records: list[dict] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for pair_index, pair in enumerate(pairs):
            try:
                base_ids = backend.query_ids(pair.base)
                variant_ids = backend.query_ids(pair.variant)
            except Exception as exc:
                row = {"backend": backend.name, "pattern_id": pair.pattern_id,
                       "pattern": pair.pattern, "case_name": pair.case_name, "scale": pair.scale,
                       "base_filter": pair.base, "variant_filter": pair.variant,
                       "error": "semantic_query_failed", "detail": str(exc)}
                stream.write(json.dumps(row) + "\n"); stream.flush(); records.append(row)
                print(f"[{pair_index + 1}/{len(pairs)}] {pair.pattern} unsupported: {exc}", flush=True)
                continue
            semantic_equal = base_ids == variant_ids
            semantic = {"base_count": len(base_ids), "variant_count": len(variant_ids),
                        "base_hash": _hash_ids(base_ids), "variant_hash": _hash_ids(variant_ids),
                        "semantic_equal": semantic_equal}
            if not semantic_equal:
                row = {"backend": backend.name, "pattern_id": pair.pattern_id,
                       "pattern": pair.pattern, "case_name": pair.case_name, "scale": pair.scale,
                       **semantic, "base_filter": pair.base, "variant_filter": pair.variant,
                       "error": "semantic_mismatch"}
                stream.write(json.dumps(row) + "\n"); stream.flush(); records.append(row)
                continue
            eligible_tenants = sorted(set(int(tenant[index]) for index in base_ids))
            for top_k in work["top_k"]:
                for ef in work["ef_search"]:
                    if candidate_keys and filter_candidate_key(pair, top_k, ef) not in candidate_keys:
                        continue
                    if not candidate_keys and candidates and (pair.pattern, pair.scale, top_k, ef) not in candidates:
                        continue
                    effective_ef = max(int(ef), int(top_k))
                    query_rows = []
                    for query_index in range(int(work["queries_per_case"])):
                        target = eligible_tenants[query_index % len(eligible_tenants)] if eligible_tenants else 0
                        query = centers[target] + rng.normal(0, 0.03, size=int(ds["dimensions"]))
                        query = (query / np.linalg.norm(query)).astype(np.float32).tolist()
                        exact_samples = {"base": [], "variant": []}; exact_results = {}
                        order_seed = int(cfg.get("confirmation", {}).get("order_seed", exp["seed"]))
                        balanced_mode = bool(cfg.get("confirmation", {}).get("balanced_random_order"))
                        exact_count = int(work.get("exact_repetitions", 1))
                        exact_orders = (
                            balanced_random_orders(
                                exact_count, order_seed,
                                backend.name, pair.case_name, pair.scale, top_k, ef, query_index, "exact",
                            )
                            if balanced_mode
                            else [("base", "variant") if index % 2 == 0 else ("variant", "base") for index in range(exact_count)]
                        )
                        for order in exact_orders:
                            for label in order:
                                expression = pair.base if label == "base" else pair.variant
                                found, elapsed = backend.search_exact(query, expression, int(top_k))
                                exact_samples[label].append(elapsed); exact_results[label] = found
                        truth = exact_results["base"]
                        samples = {"base": [], "variant": []}; recalls = {"base": [], "variant": []}
                        warmup_count = int(work["warmups"])
                        warmup_orders = (
                            balanced_random_orders(
                                warmup_count, order_seed,
                                backend.name, pair.case_name, pair.scale, top_k, ef, query_index, "warmup",
                            )
                            if balanced_mode
                            else [("base", "variant") if index % 2 == 0 else ("variant", "base") for index in range(warmup_count)]
                        )
                        for order in warmup_orders:
                            for label in order:
                                backend.search_ann(query, pair.base if label == "base" else pair.variant,
                                                   int(top_k), effective_ef)
                        repetition_count = int(work["repetitions"])
                        measurement_orders = (
                            balanced_random_orders(
                                repetition_count, order_seed,
                                backend.name, pair.case_name, pair.scale, top_k, ef, query_index, "measurement",
                            )
                            if balanced_mode
                            else [("base", "variant") if index % 2 == 0 else ("variant", "base") for index in range(repetition_count)]
                        )
                        for order in measurement_orders:
                            for label in order:
                                found, elapsed = backend.search_ann(
                                    query, pair.base if label == "base" else pair.variant,
                                    int(top_k), effective_ef)
                                samples[label].append(elapsed); recalls[label].append(_recall(found, truth, int(top_k)))
                        query_rows.append({
                            "query_index": query_index,
                            "base_ms": statistics.median(samples["base"]),
                            "variant_ms": statistics.median(samples["variant"]),
                            "base_recall": statistics.mean(recalls["base"]),
                            "variant_recall": statistics.mean(recalls["variant"]),
                            "exact_base_ms": statistics.median(exact_samples["base"]),
                            "exact_variant_ms": statistics.median(exact_samples["variant"]),
                            "exact_results_equal": exact_results["base"] == exact_results["variant"],
                            "samples": samples, "exact_samples": exact_samples,
                            "recalls": recalls,
                            "exact_orders": serialize_orders(exact_orders),
                            "warmup_orders": serialize_orders(warmup_orders),
                            "measurement_orders": serialize_orders(measurement_orders),
                        })
                    base_ms = [row["base_ms"] for row in query_rows]
                    variant_ms = [row["variant_ms"] for row in query_rows]
                    base_p50, variant_p50 = statistics.median(base_ms), statistics.median(variant_ms)
                    ratio = max(base_p50, variant_p50) / max(min(base_p50, variant_p50), 1e-9)
                    exact_base = statistics.median(row["exact_base_ms"] for row in query_rows)
                    exact_variant = statistics.median(row["exact_variant_ms"] for row in query_rows)
                    exact_ratio = max(exact_base, exact_variant) / max(min(exact_base, exact_variant), 1e-9)
                    base_recall = statistics.mean(row["base_recall"] for row in query_rows)
                    variant_recall = statistics.mean(row["variant_recall"] for row in query_rows)
                    recall_gap = abs(base_recall - variant_recall)
                    result = {
                        "timestamp": time.time(), "backend": backend.name, "experiment": exp["name"],
                        "table": backend.table, "database_version": backend.version,
                        "pattern_id": pair.pattern_id, "pattern": pair.pattern, "case_name": pair.case_name,
                        "scale": pair.scale, "selectivity_label": pair.selectivity_label,
                        "layer": getattr(pair, "layer", "filter"),
                        "preconditions": getattr(pair, "preconditions", ""),
                        "equivalence": getattr(pair, "equivalence", ""),
                        "query_signature": getattr(pair, "signature", {}),
                        "observed_selectivity": len(base_ids) / int(ds["points"]),
                        "top_k": top_k, "ef_search": ef, "effective_ef_search": effective_ef, **semantic,
                        "base_filter": pair.base, "variant_filter": pair.variant,
                        "base_p50_ms": base_p50, "variant_p50_ms": variant_p50,
                        "base_p95_ms": float(np.quantile(base_ms, .95)),
                        "variant_p95_ms": float(np.quantile(variant_ms, .95)),
                        "latency_ratio": ratio, "variant_over_base": variant_p50 / max(base_p50, 1e-9),
                        "exact_base_p50_ms": exact_base, "exact_variant_p50_ms": exact_variant,
                        "exact_latency_ratio": exact_ratio,
                        "exact_variant_over_base": exact_variant / max(exact_base, 1e-9),
                        "ann_amplification": ratio / max(exact_ratio, 1e-9),
                        "exact_results_equal": all(row["exact_results_equal"] for row in query_rows),
                        "base_recall": base_recall, "variant_recall": variant_recall,
                        "recall_gap": recall_gap,
                        "performance_anomaly": ratio > float(oracle["latency_ratio"]),
                        "internal_metrics_available": False,
                        "internal_metric_note": f"{backend.name} SQL interface does not expose request-scoped visited-node, distance, or candidate counters.",
                        "query_details": query_rows,
                        "order_protocol": "balanced_random_ab_ba" if balanced_mode else "alternating",
                    }
                    stream.write(json.dumps(result) + "\n"); stream.flush(); records.append(result)
                    print(f"[{pair_index + 1}/{len(pairs)}] {pair.pattern} scale={pair.scale} k={top_k} ef={ef} ratio={ratio:.2f} recall_gap={recall_gap:.3f}", flush=True)
    return records


def report(jsonl: Path, output_prefix: Path, backend_name: str) -> None:
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [row for row in rows if "latency_ratio" in row]
    fields = ["pattern_id", "pattern", "case_name", "scale", "selectivity_label", "observed_selectivity",
              "top_k", "ef_search", "semantic_equal", "base_p50_ms", "variant_p50_ms", "latency_ratio",
              "variant_over_base", "exact_base_p50_ms", "exact_variant_p50_ms", "exact_latency_ratio",
              "ann_amplification", "exact_results_equal", "base_recall", "variant_recall", "recall_gap",
              "performance_anomaly"]
    with output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(valid)
    print(json.dumps({"backend": backend_name, "rows": len(valid),
                      "errors": sum("error" in row for row in rows),
                      "anomalies": sum(row["performance_anomaly"] for row in valid)}))
