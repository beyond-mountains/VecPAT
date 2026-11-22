from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .filters import FilterPair, build_campaign_pairs, build_pairs
from .client import Qdrant
from ...core.dataset import generate_dataset
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
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup(q: Qdrant, cfg: dict) -> dict:
    name = cfg["experiment"]["collection"]
    existing = q.reusable_collection(
        name,
        int(cfg["dataset"]["points"]),
        int(cfg["dataset"]["dimensions"]),
        cfg["index"],
    )
    if existing is not None:
        print(json.dumps({"collection_reused": name, "state": existing}, default=str), flush=True)
        return existing
    vectors, payloads, _, _ = generate_dataset(cfg)
    q.recreate_collection(name, cfg["dataset"]["dimensions"], cfg["index"])
    q.create_payload_indexes(name)
    batch = cfg["dataset"]["batch_size"]
    for start in range(0, len(vectors), batch):
        end = min(start + batch, len(vectors))
        q.upload(name, list(range(start, end)), vectors[start:end].tolist(), payloads[start:end])
        if start % (batch * 10) == 0:
            print(f"uploaded {end}/{len(vectors)}", flush=True)
    q.enable_indexing(name, cfg["index"])
    print("upload complete; HNSW indexing enabled", flush=True)
    state = q.wait_ready(name, len(vectors))
    print(json.dumps({"collection_ready": name, "state": state}, default=str), flush=True)
    return state


def _hash_ids(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def _recall(found: list[int], truth: list[int], k: int) -> float:
    denominator = min(k, len(truth))
    return len(set(found) & set(truth)) / denominator if denominator else 1.0


def _quantile(xs: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(xs), q))


def run(q: Qdrant, cfg: dict, output: Path) -> list[dict]:
    exp, ds, work, oracle = cfg["experiment"], cfg["dataset"], cfg["workload"], cfg["oracle"]
    name = exp["collection"]
    q.wait_ready(name, ds["points"])
    vectors, _, tenant, centers = generate_dataset(cfg)
    if work.get("comparative_method"):
        pairs = emit_comparative_pairs(
            "qdrant", work["comparative_method"],
            work.get("transformation_strengths", [2, 8, 32, 128]),
            int(ds.get("age_universe", 1000)),
            generation_round=int(work.get("comparative_generation_round", 0)),
        )
    elif work.get("campaign_continuous"):
        pairs = emit_filter_pairs_continuous("qdrant", work.get("transformation_strengths", [2, 8, 32, 128]),
                                             int(ds.get("age_universe", 1000)))
    elif work.get("campaign_1002"):
        pairs = emit_filter_pairs_1002("qdrant", work.get("transformation_strengths", [2, 8, 32, 128]),
                                       int(ds.get("age_universe", 1000)))
    elif work.get("campaign_1001"):
        pairs = emit_filter_pairs("qdrant", work.get("transformation_strengths", [2, 8, 32]),
                                  int(ds.get("age_universe", 1000)))
    elif work.get("campaign_0924"):
        pairs = build_campaign_pairs(work["redundancy_scales"], work.get("set_widths", [10]),
                                     int(ds.get("age_universe", 1000)))
    else:
        pairs = build_pairs(
            ds["tenants"], work["tenant_widths"], work["redundancy_scales"],
            work.get("set_widths", [10]), int(ds.get("age_universe", 1000)),
        )
    if work.get("patterns"):
        pairs = [p for p in pairs if p.pattern in set(work["patterns"])]
    if work.get("case_names"):
        pairs = [p for p in pairs if p.case_name in set(work["case_names"])]
    candidate_keys = load_candidate_keys(cfg, "qdrant", "filter")
    if candidate_keys:
        candidate_case_names = {key[1] for key in candidate_keys}
        pairs = [pair for pair in pairs if pair.case_name in candidate_case_names]
    rng = np.random.default_rng(exp["seed"] + 1)
    records: list[dict] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        candidates = {tuple(item) for item in work.get("candidate_cases", [])}
        for pair_index, pair in enumerate(pairs):
            try:
                base_ids = q.scroll_ids(name, pair.base)
                variant_ids = q.scroll_ids(name, pair.variant)
            except Exception as exc:
                row = {"backend": "qdrant", "pattern_id": pair.pattern_id, "pattern": pair.pattern,
                       "case_name": pair.case_name, "scale": pair.scale,
                       "base_filter": pair.base, "variant_filter": pair.variant,
                       "error": "semantic_query_failed", "detail": str(exc)}
                stream.write(json.dumps(row) + "\n"); stream.flush(); records.append(row)
                print(f"[{pair_index+1}/{len(pairs)}] {pair.pattern} unsupported: {exc}", flush=True)
                continue
            semantic_equal = base_ids == variant_ids
            semantic = {
                "base_count": len(base_ids), "variant_count": len(variant_ids),
                "base_hash": _hash_ids(base_ids), "variant_hash": _hash_ids(variant_ids),
                "semantic_equal": semantic_equal,
            }
            if not semantic_equal:
                row = {"backend": "qdrant", "pattern_id": pair.pattern_id, "pattern": pair.pattern,
                       "case_name": pair.case_name, "scale": pair.scale, **semantic,
                       "base_filter": pair.base, "variant_filter": pair.variant,
                       "error": "semantic_mismatch"}
                stream.write(json.dumps(row) + "\n"); stream.flush(); records.append(row)
                continue
            eligible_tenants = sorted(set(int(tenant[i]) for i in base_ids))
            for top_k in work["top_k"]:
                for ef in work["ef_search"]:
                    if candidate_keys and filter_candidate_key(pair, top_k, ef) not in candidate_keys:
                        continue
                    if not candidate_keys and candidates and (pair.pattern, pair.scale, top_k, ef) not in candidates:
                        continue
                    query_rows = []
                    for query_index in range(work["queries_per_case"]):
                        target_tenant = eligible_tenants[query_index % len(eligible_tenants)] if eligible_tenants else 0
                        query = centers[target_tenant] + rng.normal(0, 0.03, size=ds["dimensions"])
                        query = (query / np.linalg.norm(query)).astype(np.float32).tolist()
                        exact_samples = {"base": [], "variant": []}
                        exact_usages: dict[str, list[dict]] = {"base": [], "variant": []}
                        exact_results_equal = True
                        truth: list[int] = []
                        order_seed = int(cfg.get("confirmation", {}).get("order_seed", exp["seed"]))
                        balanced_mode = bool(cfg.get("confirmation", {}).get("balanced_random_order"))
                        exact_count = int(work.get("exact_repetitions", 1))
                        exact_orders = (
                            balanced_random_orders(
                                exact_count, order_seed,
                                "qdrant", pair.case_name, pair.scale, top_k, ef, query_index, "exact",
                            )
                            if balanced_mode
                            else [("base", "variant") if index % 2 == 0 else ("variant", "base") for index in range(exact_count)]
                        )
                        for order in exact_orders:
                            exact_found: dict[str, list[int]] = {}
                            for label in order:
                                filt = pair.base if label == "base" else pair.variant
                                found, metrics = q.search(name, query, filt, top_k, ef, exact=True)
                                exact_found[label] = found
                                exact_samples[label].append(metrics["client_wall_ms"])
                                exact_usages[label].append(metrics)
                            if not truth:
                                truth = exact_found["base"]
                            exact_results_equal = exact_results_equal and exact_found["base"] == exact_found["variant"]
                        warmup_count = int(work["warmups"])
                        warmup_orders = (
                            balanced_random_orders(
                                warmup_count, order_seed,
                                "qdrant", pair.case_name, pair.scale, top_k, ef, query_index, "warmup",
                            )
                            if balanced_mode
                            else [("base", "variant")] * warmup_count
                        )
                        for order in warmup_orders:
                            for label in order:
                                filt = pair.base if label == "base" else pair.variant
                                q.search(name, query, filt, top_k, ef)
                        samples = {"base": [], "variant": []}
                        recalls = {"base": [], "variant": []}
                        usages: dict[str, list[dict]] = {"base": [], "variant": []}
                        repetition_count = int(work["repetitions"])
                        measurement_orders = (
                            balanced_random_orders(
                                repetition_count, order_seed,
                                "qdrant", pair.case_name, pair.scale, top_k, ef, query_index, "measurement",
                            )
                            if balanced_mode
                            else [("base", "variant") if index % 2 == 0 else ("variant", "base") for index in range(repetition_count)]
                        )
                        for order in measurement_orders:
                            for label in order:
                                filt = pair.base if label == "base" else pair.variant
                                found, metrics = q.search(name, query, filt, top_k, ef)
                                samples[label].append(metrics["client_wall_ms"])
                                recalls[label].append(_recall(found, truth, top_k))
                                usages[label].append(metrics)
                        query_rows.append({
                            "query_index": query_index,
                            "base_ms": statistics.median(samples["base"]),
                            "variant_ms": statistics.median(samples["variant"]),
                            "base_recall": statistics.mean(recalls["base"]),
                            "variant_recall": statistics.mean(recalls["variant"]),
                            "exact_base_ms": statistics.median(exact_samples["base"]),
                            "exact_variant_ms": statistics.median(exact_samples["variant"]),
                            "exact_results_equal": exact_results_equal,
                            "samples": samples,
                            "recalls": recalls,
                            "exact_usage": exact_usages,
                            "usage": usages,
                            "exact_orders": serialize_orders(exact_orders),
                            "warmup_orders": serialize_orders(warmup_orders),
                            "measurement_orders": serialize_orders(measurement_orders),
                        })
                    base_ms = [x["base_ms"] for x in query_rows]
                    variant_ms = [x["variant_ms"] for x in query_rows]
                    base_recall = statistics.mean(x["base_recall"] for x in query_rows)
                    variant_recall = statistics.mean(x["variant_recall"] for x in query_rows)
                    slower = max(statistics.median(base_ms), statistics.median(variant_ms))
                    faster = max(min(statistics.median(base_ms), statistics.median(variant_ms)), 1e-9)
                    ratio = slower / faster
                    recall_gap = abs(base_recall - variant_recall)
                    exact_base_ms = statistics.median(x["exact_base_ms"] for x in query_rows)
                    exact_variant_ms = statistics.median(x["exact_variant_ms"] for x in query_rows)
                    exact_ratio = max(exact_base_ms, exact_variant_ms) / max(min(exact_base_ms, exact_variant_ms), 1e-9)
                    instrument_names = (
                        "hnsw_visited_nodes", "hnsw_neighbor_checks", "distance_computations",
                        "filter_evaluations", "filter_acceptances", "filter_rejections",
                    )
                    instrumentation: dict[str, dict[str, float]] = {}
                    for metric_name in instrument_names:
                        key = f"usage_{metric_name}"
                        ann_values = {
                            label: [float(sample[key]) for detail in query_rows for sample in detail["usage"][label] if key in sample]
                            for label in ("base", "variant")
                        }
                        exact_values = {
                            label: [float(sample[key]) for detail in query_rows for sample in detail["exact_usage"][label] if key in sample]
                            for label in ("base", "variant")
                        }
                        if ann_values["base"] and ann_values["variant"]:
                            ann_base = statistics.median(ann_values["base"])
                            ann_variant = statistics.median(ann_values["variant"])
                            instrumentation[metric_name] = {
                                "ann_base_p50": ann_base,
                                "ann_variant_p50": ann_variant,
                                "ann_ratio": ann_variant / max(ann_base, 1e-9),
                                "exact_base_p50": statistics.median(exact_values["base"]) if exact_values["base"] else 0.0,
                                "exact_variant_p50": statistics.median(exact_values["variant"]) if exact_values["variant"] else 0.0,
                            }
                    row = {
                        "timestamp": time.time(), "backend": "qdrant", "experiment": exp["name"], "collection": name,
                        "pattern_id": pair.pattern_id, "pattern": pair.pattern, "case_name": pair.case_name,
                        "scale": pair.scale,
                        "layer": getattr(pair, "layer", "filter"),
                        "preconditions": getattr(pair, "preconditions", ""),
                        "equivalence": getattr(pair, "equivalence", ""),
                        "query_signature": getattr(pair, "signature", {}),
                        "selectivity_label": pair.selectivity_label,
                        "observed_selectivity": len(base_ids) / ds["points"],
                        "top_k": top_k, "ef_search": ef, **semantic,
                        "base_filter": pair.base, "variant_filter": pair.variant,
                        "base_p50_ms": statistics.median(base_ms), "variant_p50_ms": statistics.median(variant_ms),
                        "base_p95_ms": _quantile(base_ms, .95), "variant_p95_ms": _quantile(variant_ms, .95),
                        "latency_ratio": ratio,
                        "variant_over_base": statistics.median(variant_ms) / max(statistics.median(base_ms), 1e-9),
                        "exact_base_p50_ms": exact_base_ms, "exact_variant_p50_ms": exact_variant_ms,
                        "exact_latency_ratio": exact_ratio,
                        "exact_variant_over_base": exact_variant_ms / max(exact_base_ms, 1e-9),
                        "ann_amplification": ratio / max(exact_ratio, 1e-9),
                        "exact_results_equal": all(x["exact_results_equal"] for x in query_rows),
                        "base_recall": base_recall, "variant_recall": variant_recall,
                        "recall_gap": recall_gap,
                        "performance_anomaly": ratio > oracle["latency_ratio"],
                        "query_details": query_rows,
                        "order_protocol": "balanced_random_ab_ba" if balanced_mode else "alternating",
                        "instrumentation": instrumentation,
                        "internal_metric_note": "Instrumented Qdrant reports request-scoped HNSW, distance, and filter counters; stock Qdrant leaves instrumentation empty.",
                    }
                    for metric_name, metric in instrumentation.items():
                        row[f"base_{metric_name}"] = metric["ann_base_p50"]
                        row[f"variant_{metric_name}"] = metric["ann_variant_p50"]
                        row[f"{metric_name}_ratio"] = metric["ann_ratio"]
                    stream.write(json.dumps(row) + "\n"); stream.flush(); records.append(row)
                    print(f"[{pair_index+1}/{len(pairs)}] {pair.pattern} scale={pair.scale} k={top_k} ef={ef} ratio={ratio:.2f} recall_gap={recall_gap:.3f}", flush=True)
    return records


def report(jsonl: Path, output_prefix: Path) -> None:
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [r for r in rows if "latency_ratio" in r]
    fields = [
        "pattern", "scale", "selectivity_label", "observed_selectivity", "top_k", "ef_search",
        "semantic_equal", "base_p50_ms", "variant_p50_ms", "latency_ratio",
        "exact_base_p50_ms", "exact_variant_p50_ms", "exact_latency_ratio", "ann_amplification",
        "base_hnsw_visited_nodes", "variant_hnsw_visited_nodes", "hnsw_visited_nodes_ratio",
        "base_distance_computations", "variant_distance_computations", "distance_computations_ratio",
        "base_filter_evaluations", "variant_filter_evaluations", "filter_evaluations_ratio",
        "base_filter_rejections", "variant_filter_rejections", "filter_rejections_ratio",
        "base_recall", "variant_recall", "recall_gap", "performance_anomaly",
    ]
    with output_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(valid)
    anomalies = sorted((r for r in valid if r["performance_anomaly"]),
                       key=lambda r: r["latency_ratio"], reverse=True)
    md = [
        "# Equivalent-filter anomaly report", "",
        f"- Evaluated configurations: {len(valid)}",
        f"- Semantic mismatches: {sum(not r.get('semantic_equal', False) for r in rows)}",
        f"- Performance anomalies: {sum(r['performance_anomaly'] for r in valid)}", "",
        "|pattern|scale|selectivity|K|ef|base ms|variant ms|ANN ratio|exact ratio|ANN amplification|visited ratio|distance ratio|filter-eval ratio|base recall|variant recall|flags|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in anomalies[:30]:
        flags = "latency" if r["performance_anomaly"] else ""
        md.append(f"|{r['pattern']}|{r['scale']}|{r['observed_selectivity']:.4f}|{r['top_k']}|{r['ef_search']}|{r['base_p50_ms']:.3f}|{r['variant_p50_ms']:.3f}|{r['latency_ratio']:.2f}|{r.get('exact_latency_ratio', float('nan')):.2f}|{r.get('ann_amplification', float('nan')):.2f}|{r.get('hnsw_visited_nodes_ratio', float('nan')):.2f}|{r.get('distance_computations_ratio', float('nan')):.2f}|{r.get('filter_evaluations_ratio', float('nan')):.2f}|{r['base_recall']:.3f}|{r['variant_recall']:.3f}|{flags}|")
    md.extend(["", "## Interpretation note", "", "Latency flags are screening signals. Re-run candidates with more repetitions and an idle host before treating them as confirmed defects. With the instrumented Qdrant image, request-scoped HNSW visited-node, neighbor-check, distance-computation, and filter accept/reject counters are retained in JSONL query details. Exact-search filter/distance counters are not yet hooked; exact latency remains the control baseline."])
    report_path = None
    if not output_prefix.name.endswith("_discovery"):
        report_path = output_prefix.with_suffix(".md")
        report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(valid), "anomalies": len(anomalies),
                      "report": str(report_path) if report_path else None}))
