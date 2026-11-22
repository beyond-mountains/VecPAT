from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def analyze(input_path: Path, output_prefix: Path) -> dict:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    valid = [row for row in rows if "latency_ratio" in row]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in valid:
        groups[row["pattern"]].append(row)
    patterns = []
    for pattern, group in sorted(groups.items()):
        top = max(group, key=lambda row: row["latency_ratio"])
        patterns.append({
            "pattern": pattern,
            "configurations": len(group),
            "performance_anomalies": sum(row["performance_anomaly"] for row in group),
            "max_latency_ratio": top["latency_ratio"],
            "max_exact_ratio": max(row["exact_latency_ratio"] for row in group),
            "max_ann_amplification": max(row["ann_amplification"] for row in group),
            "top_case": {key: top[key] for key in (
                "scale", "observed_selectivity", "top_k", "ef_search", "effective_ef_search",
                "base_p50_ms", "variant_p50_ms", "exact_latency_ratio", "ann_amplification",
                "base_recall", "variant_recall",
            )},
        })
    summary = {
        "backend": "milvus",
        "configurations": len(valid),
        "execution_errors": sum("error" in row for row in rows),
        "semantic_mismatches": sum(row.get("error") == "semantic_mismatch" for row in rows),
        "exact_result_mismatches": sum(not row.get("exact_results_equal", True) for row in valid),
        "performance_anomalies": sum(row["performance_anomaly"] for row in valid),
        "ef_adjusted_configurations": sum(row["effective_ef_search"] != row["ef_search"] for row in valid),
        "internal_metrics_available": False,
        "patterns": patterns,
    }
    output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = ["# Milvus 实验归因汇总", "",
          f"- 有效配置：{summary['configurations']}",
          f"- 执行失败：{summary['execution_errors']}",
          f"- Semantic mismatch：{summary['semantic_mismatches']}",
          f"- Exact Top-K mismatch：{summary['exact_result_mismatches']}",
          f"- 性能异常：{summary['performance_anomalies']}",
          f"- 因 Milvus 要求 ef>=K 而调整 ef 的配置：{summary['ef_adjusted_configurations']}", "",
          "|pattern|配置数|性能异常数|最大 ANN ratio|最大 exact ratio|最大 amplification|top scale|selectivity|K|requested ef|effective ef|",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for item in patterns:
        case = item["top_case"]
        md.append(f"|{item['pattern']}|{item['configurations']}|{item['performance_anomalies']}|"
                  f"{item['max_latency_ratio']:.2f}|{item['max_exact_ratio']:.2f}|{item['max_ann_amplification']:.2f}|"
                  f"{case['scale']}|{case['observed_selectivity']:.4f}|{case['top_k']}|"
                  f"{case['ef_search']}|{case['effective_ef_search']}|")
    md.extend(["", "## 归因限制", "",
               "Milvus 公共 API 未提供 request-scoped visited nodes、distance computations 和 candidate count。"
               "因此这里的 ANN amplification 只用于筛选 filter–ANN interaction，不能证明 traversal 工作量发生变化。"])
    output_prefix.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "configurations", "execution_errors", "semantic_mismatches", "exact_result_mismatches",
        "performance_anomalies", "ef_adjusted_configurations",
    )}))
    return summary

