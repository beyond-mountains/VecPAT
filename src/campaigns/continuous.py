from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from ..core.coverage import (
    CoverageLedger,
    canonical_json,
    case_id,
    dataset_signature,
    physical_dataset_spec,
)
from ..generation.continuous import TARGET_FILTER_RULES, build_continuous_filter_instances
from ..generation.equivalent import _features, evaluate

BACKENDS = ("milvus", "pgvector", "qdrant", "seekdb", "weaviate")
FILTER_RULE_BACKENDS = {
    # Qdrant and Weaviate filter DSLs compare stored properties directly and
    # cannot represent a computed operand such as ``age + 5``.
    "F12": frozenset({"milvus", "pgvector", "seekdb"}),
}
ANN_CASES = {
    "A1": ("dimension_permutation",),
    "A2": ("positive_global_scaling",),
    "A3": ("cosine_vs_inner_product", "cosine_vs_inner_product_desc", "cosine_vs_l2"),
    "A4": ("l2_vs_squared_l2",),
    "FA1": ("no_filter_vs_true_filter",),
    "FA2": ("filtered_full_vs_materialized_subset",),
}
SUPPORTED_PATTERNS = frozenset(TARGET_FILTER_RULES) | frozenset(ANN_CASES)


def parse_pattern(value: str) -> str:
    pattern = value.strip().upper()
    if pattern not in SUPPORTED_PATTERNS:
        supported = ", ".join(sorted(SUPPORTED_PATTERNS))
        raise argparse.ArgumentTypeError(f"unsupported pattern {value!r}; choose one of: {supported}")
    return pattern


def _config(backend: str, layer: str, rule_id: str, case_name: str,
            transformation: dict, semantic: dict, execution: dict) -> dict:
    return {"backend": backend, "layer": layer, "rule_id": rule_id, "case_name": case_name,
            "transformation": transformation, "semantic": semantic, "execution": execution}


def populate(ledger: CoverageLedger, strengths: list[int], pattern: str | None = None,
             backend_only: str | None = None) -> int:
    filters = build_continuous_filter_instances(strengths)
    if pattern:
        filters = [item for item in filters if item.rule_id == pattern]
    semantics = [
        {"profile": "baseline", "correlation": 0.85, "points": 200000},
        {"profile": "uncorrelated", "correlation": 0.0, "points": 200000},
        {"profile": "high_correlation", "correlation": 0.99, "points": 200000},
        {"profile": "small", "correlation": 0.85, "points": 50000},
        {"profile": "large", "correlation": 0.85, "points": 1000000},
    ]
    executions = [
        {"index": "hnsw", "top_k": 10, "ef_search": 16, "dimensions": 128},
        {"index": "hnsw", "top_k": 100, "ef_search": 16, "dimensions": 128},
        {"index": "hnsw", "top_k": 10, "ef_search": 64, "dimensions": 128},
        {"index": "hnsw", "top_k": 100, "ef_search": 64, "dimensions": 128},
    ]
    inserted = 0
    selected_backends = (backend_only,) if backend_only else BACKENDS
    for backend in selected_backends:
        for item in filters:
            if backend not in FILTER_RULE_BACKENDS.get(item.rule_id, BACKENDS):
                continue
            # Keep the generator inputs separate from the complexity reported
            # by a generated case.  Some rules (for example F4 double-NOT)
            # derive an output depth from the requested strength, so replaying
            # item.strength as the generator input can regenerate a different
            # case and leave the backend runner with zero matching pairs.
            syntax = {
                "strength": item.strength,
                "ast": item.case_name,
                "generator_strengths": sorted({int(value) for value in strengths}),
            }
            case_semantics = [{**semantic, "selectivity_target": item.selectivity_label,
                               "eligible_set_size": "measured_by_semantic_oracle"} for semantic in semantics]
            # First cover every high-complexity rewrite at the baseline semantic/execution point.
            priority0 = max(0, 2048 - min(item.strength, 2048))
            ledger.add(_config(backend, "filter", item.rule_id, item.case_name, syntax, case_semantics[0], executions[0]), priority0)
            inserted += 1
            for semantic_index, semantic in enumerate(case_semantics[1:], 1):
                ledger.add(_config(backend, "filter", item.rule_id, item.case_name, syntax, semantic, executions[0]),
                           10000 + semantic_index * 1000 + priority0)
                inserted += 1
            for execution_index, execution in enumerate(executions[1:], 1):
                ledger.add(_config(backend, "filter", item.rule_id, item.case_name, syntax, case_semantics[0], execution),
                           20000 + execution_index * 1000 + priority0)
                inserted += 1

        for rule_id, names in ANN_CASES.items():
            if pattern and rule_id != pattern:
                continue
            for name in names:
                # inner_product(... ) DESC APPROXIMATE is a seekDB-specific
                # SQL formulation. Keep it as an independent case instead of
                # replacing the supported negative_inner_product ASC form.
                if name == "cosine_vs_inner_product_desc" and backend != "seekdb":
                    continue
                transforms = [{"variant": name}]
                if rule_id == "A1":
                    transforms = [
                        {"variant": name, "permutation": "random", "distribution": "gaussian"},
                        {"variant": name, "permutation": "reverse", "distribution": "gaussian"},
                        {"variant": name, "permutation": "block8", "distribution": "gaussian"},
                        {"variant": name, "permutation": "block16", "distribution": "gaussian"},
                        {"variant": name, "permutation": "block32", "distribution": "gaussian"},
                        {"variant": name, "permutation": "reverse", "distribution": "sparse_continuous"},
                        {"variant": name, "permutation": "variance", "distribution": "high_variance"},
                    ]
                elif rule_id == "A2":
                    transforms = [{"variant": name, "scale_factor": value, "distribution": distribution}
                                  for distribution in ("gaussian", "clustered", "single_cluster")
                                  for value in (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e4, 1e6)]
                elif rule_id == "A3":
                    transforms = [{"variant": name, "distribution": value}
                                  for value in ("gaussian", "clustered", "single_cluster")]
                elif rule_id == "A4":
                    transforms = [{"variant": name, "distribution": value}
                                  for value in ("gaussian", "clustered", "single_cluster")]
                elif rule_id in {"FA1", "FA2"}:
                    transforms = [{"variant": name, "distribution": "gaussian"}]
                for transform_index, transformation in enumerate(transforms):
                    ann_executions = [
                        {"index": "hnsw", "top_k": 10, "ef_search": 16, "dimensions": 128},
                        {"index": "hnsw", "top_k": 10, "ef_search": 16, "dimensions": 64},
                        {"index": "hnsw", "top_k": 10, "ef_search": 16, "dimensions": 768},
                        {"index": "hnsw", "top_k": 10, "ef_search": 16, "dimensions": 1536},
                        {"index": "hnsw", "top_k": 100, "ef_search": 16, "dimensions": 128},
                        {"index": "hnsw", "top_k": 10, "ef_search": 64, "dimensions": 128},
                        {"index": "hnsw", "top_k": 100, "ef_search": 64, "dimensions": 128},
                    ]
                    for execution_index, execution in enumerate(ann_executions):
                        semantic = {"profile": "unit_vectors", "correlation": None, "points": 20000}
                        priority = (3000 + transform_index * 100 if execution_index == 0
                                    else 24000 + execution_index * 1000 + transform_index * 100)
                        ledger.add(_config(backend, "ann", rule_id, name, transformation, semantic, execution),
                                   priority)
                        inserted += 1
    ledger.commit()
    return inserted


def validate_cases(strengths: list[int], output: Path, pattern: str | None = None) -> dict:
    cases = build_continuous_filter_instances(strengths)
    if pattern:
        cases = [item for item in cases if item.rule_id == pattern]
    rows = []
    for item in cases:
        mismatch = None
        for age in range(1, 1001):
            sample = {"age": age, "category5": age % 5 + 1, "category": age % 100 + 1,
                      "tenant": age % 100, "level": age % 32,
                      "parity": "even" if age % 2 == 0 else "odd"}
            if evaluate(item.original, sample) != evaluate(item.equivalent, sample):
                mismatch = sample; break
        rows.append({"rule_id": item.rule_id, "case_name": item.case_name, "strength": item.strength,
                     "semantic_equal": mismatch is None, "counterexample": mismatch,
                     "original_features": _features(item.original), "equivalent_features": _features(item.equivalent),
                     "preconditions": item.preconditions})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"cases": len(rows), "semantic_failures": sum(not row["semantic_equal"] for row in rows), "output": str(output)}


def render_config(base_path: Path, config: dict, mode: str, destination: Path) -> dict:
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    identifier = case_id(config); signature = dataset_signature(config)
    base["experiment"]["name"] = f"continuous_{config['backend']}_{identifier}"
    if "collection" in base["experiment"]:
        base["experiment"]["collection"] = f"efa_cont_{signature}"
    if "table" in base["experiment"]:
        base["experiment"]["table"] = f"efa_cont_{signature}"
    if "class_prefix" in base["experiment"]:
        base["experiment"]["class_prefix"] = "EfaCont" + signature
    ds = base["dataset"]
    ds["points"] = int(config["semantic"]["points"])
    ds["correlation"] = float(config["semantic"].get("correlation") or 0.0)
    ds["dimensions"] = int(config["execution"]["dimensions"])
    work = base["workload"]
    work.update({"campaign_continuous": True, "campaign_1002": False,
                 "case_names": [config["case_name"]],
                 "patterns": [],
                 "transformation_strengths": [
                     int(value) for value in config["transformation"].get(
                         "generator_strengths",
                         [config["transformation"].get("strength", 2)],
                     )
                 ],
                 "top_k": [int(config["execution"]["top_k"])],
                 "ef_search": [int(config["execution"]["ef_search"])]})
    if mode == "confirmation":
        work.update({"queries_per_case": 8, "warmups": 4, "repetitions": 12, "exact_repetitions": 4})
        base.setdefault("confirmation", {}).update({"balanced_random_order": True, "order_seed": 20261011})
    ann = base.setdefault("ann_campaign", {})
    # Keep discovery runnable even when a backend-specific base config omits
    # the ANN campaign block. Confirmation overrides the sampling counts below.
    for key, value in {
        "queries": 12,
        "warmups": 1,
        "repetitions": 5,
        "subset_category5": 1,
    }.items():
        ann.setdefault(key, value)
    ann.update({"cases": [config["case_name"]], "points": int(config["semantic"]["points"]),
                "dimensions": int(config["execution"]["dimensions"]),
                "top_k": [int(config["execution"]["top_k"])],
                "ef_search": [int(config["execution"]["ef_search"]),],
                "scale_factor": float(config["transformation"].get("scale_factor", 3.0)),
                "permutation": config["transformation"].get("permutation", "random"),
                "distribution": config["transformation"].get("distribution", "gaussian")})
    if mode == "confirmation":
        ann.update({"queries": 24, "warmups": 4, "repetitions": 12})
        base.setdefault("confirmation", {}).update({"balanced_random_order": True, "order_seed": 20261011})
    import hashlib
    setup_spec: dict[str, Any] = {"dataset_signature": signature, "index": base["index"]}
    if config["backend"] == "weaviate" and config["layer"] == "filter":
        # Weaviate stores ef in the collection's HNSW configuration, so Ef16
        # and Ef64 are distinct prepared resources over the same data.
        setup_spec["fixed_ef"] = int(config["execution"]["ef_search"])
    setup_signature = hashlib.sha256(canonical_json(setup_spec).encode()).hexdigest()[:12]
    base["continuous_case"] = {
        **config,
        "case_id": identifier,
        "dataset_signature": signature,
        "dataset_spec": physical_dataset_spec(config),
        "setup_signature": setup_signature,
        "mode": mode,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return base["continuous_case"]


def _result_summary(path: Path) -> tuple[str, float | None, float | None, str, dict]:
    if not path.exists():
        return "retryable_failed", None, None, "result file was not created", {}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    measured = [row for row in rows if "latency_ratio" in row]
    if measured:
        row = max(measured, key=lambda value: float(value.get("latency_ratio") or 0))
        status = "semantic_mismatch" if row.get("semantic_equal") is False else "executed"
        return status, float(row.get("latency_ratio") or 0), float(row.get("recall_gap") or 0), "", row
    if not rows:
        return "retryable_failed", None, None, "no measurement row", {}

    row = rows[0]
    reported_status = str(
        row.get("status")
        or ("semantic_mismatch" if row.get("error") == "semantic_mismatch" else "failed")
    )
    detail = str(row.get("detail") or row.get("error") or "no measurement row")
    if reported_status == "semantic_mismatch":
        return reported_status, None, None, detail, row

    # Preserve an explicit capability decision, but never turn a missing
    # measurement into a successful/complete case. Connection loss and
    # timeouts are transient and become pending again on the next recover.
    lowered = detail.lower()
    transient_markers = (
        "timed out", "timeout", "lost connection", "interfaceerror",
        "connection reset", "connection refused", "server has gone away",
        "temporarily unavailable",
    )
    if any(marker in lowered for marker in transient_markers):
        return "retryable_failed", None, None, detail, row
    if reported_status == "unsupported":
        return "unsupported", None, None, detail, row
    return "failed", None, None, detail, row


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable equivalent-query campaign control plane")
    parser.add_argument("--ledger", default="continuous_state/coverage.sqlite")
    parser.add_argument("--pattern", type=parse_pattern,
                        help="only populate, validate, claim, and snapshot this rule (for example F1 or A2)")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init"); init.add_argument("--strengths", default="1,2,8,32,128,512,1024")
    init.add_argument("--backend", choices=BACKENDS,
                      help="only initialize configurations for this backend")
    validate = sub.add_parser("validate"); validate.add_argument("--strengths", default="1,2,8,32,128,512,1024"); validate.add_argument("--output", required=True)
    claim = sub.add_parser("claim"); claim.add_argument("--backend", choices=BACKENDS, required=True); claim.add_argument("--run-id", required=True)
    claim.add_argument("--deadline", type=float, required=True); claim.add_argument("--base-config", required=True); claim.add_argument("--output", required=True)
    done = sub.add_parser("complete"); done.add_argument("--case-id", required=True); done.add_argument("--run-id", required=True)
    done.add_argument("--mode", choices=("discovery", "confirmation"), required=True); done.add_argument("--result", required=True)
    skip = sub.add_parser("skip"); skip.add_argument("--case-id", required=True); skip.add_argument("--run-id", required=True); skip.add_argument("--mode", required=True); skip.add_argument("--reason", required=True)
    fail = sub.add_parser("fail"); fail.add_argument("--case-id", required=True); fail.add_argument("--run-id", required=True); fail.add_argument("--mode", required=True)
    fail.add_argument("--status", choices=("failed", "retryable_failed"), required=True); fail.add_argument("--reason", required=True)
    snap = sub.add_parser("snapshot"); snap.add_argument("--output", required=True); snap.add_argument("--anomalies-output")
    args = parser.parse_args(); ledger = CoverageLedger(args.ledger)
    if args.command == "init":
        recovered = ledger.recover(0, args.pattern)
        count = populate(ledger, [int(value) for value in args.strengths.split(",")],
                         args.pattern, args.backend)
        print(json.dumps({"pattern": args.pattern or "all", "considered": count,
                          "recovered_interrupted": recovered, **ledger.summary(args.pattern)})); return
    if args.command == "validate":
        report = validate_cases([int(value) for value in args.strengths.split(",")], Path(args.output), args.pattern)
        report["pattern"] = args.pattern or "all"
        print(json.dumps(report)); raise SystemExit(1 if report["semantic_failures"] else 0)
    if args.command == "claim":
        claimed = ledger.claim(args.backend, args.run_id, args.deadline, args.pattern)
        if claimed is None:
            raise SystemExit(3)
        config, mode = claimed; metadata = render_config(Path(args.base_config), config, mode, Path(args.output))
        print(json.dumps({**metadata, "mode": mode, "config_path": args.output})); return
    if args.command == "complete":
        status, ratio, gap, detail, metrics = _result_summary(Path(args.result))
        ledger.finish(args.case_id, args.run_id, args.mode, args.result, status, ratio, gap, detail, metrics)
        print(json.dumps({"case_id": args.case_id, "status": status, "latency_ratio": ratio, "recall_gap": gap})); return
    if args.command == "skip":
        ledger.finish(args.case_id, args.run_id, args.mode, "", "unsupported", None, None, args.reason)
        print(json.dumps({"case_id": args.case_id, "status": "unsupported", "reason": args.reason})); return
    if args.command == "fail":
        ledger.finish(args.case_id, args.run_id, args.mode, "", args.status, None, None, args.reason)
        print(json.dumps({"case_id": args.case_id, "status": args.status, "reason": args.reason})); return
    ledger.snapshot(args.output, args.anomalies_output, args.pattern)
    print(json.dumps({"pattern": args.pattern or "all", **ledger.summary(args.pattern)}, indent=2))


if __name__ == "__main__":
    main()
