from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path


def balanced_random_orders(
    count: int,
    seed: int,
    *context: object,
    labels: tuple[str, str] = ("base", "variant"),
) -> list[tuple[str, str]]:
    """Return a deterministic random permutation with exactly half AB and half BA."""
    if count <= 0 or count % 2:
        raise ValueError(f"balanced order count must be a positive even integer, got {count}")
    material = "\x1f".join([str(seed), *(str(item) for item in context)]).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    orders = [labels] * (count // 2) + [(labels[1], labels[0])] * (count // 2)
    random.Random(derived_seed).shuffle(orders)
    return orders


def serialize_orders(orders: list[tuple[str, str]]) -> list[str]:
    return [f"{first}-{second}" for first, second in orders]


def load_candidate_keys(cfg: dict, backend: str, layer: str) -> set[tuple[str, str, str, int, int]]:
    """Load precise result_1002 anomaly keys from the confirmation manifest."""
    confirmation = cfg.get("confirmation", {})
    manifest_value = confirmation.get("candidate_manifest")
    if not manifest_value:
        return set()
    path = Path(manifest_value)
    if not path.exists():
        raise FileNotFoundError(f"confirmation candidate manifest not found: {path}")

    keys: set[tuple[str, str, str, int, int]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            if row.get("backend") != backend or row.get("layer") != layer:
                continue
            rule_id = row.get("rule_id") or row.get("pattern_id") or ""
            keys.add(
                (
                    rule_id,
                    row.get("case_name") or "",
                    row.get("scale") or "",
                    int(row["top_k"]),
                    int(row["ef_search"]),
                )
            )
    return keys


def filter_candidate_key(pair, top_k: int, ef_search: int) -> tuple[str, str, str, int, int]:
    return (
        str(pair.pattern_id),
        str(pair.case_name),
        str(pair.scale) if pair.scale is not None else "",
        int(top_k),
        int(ef_search),
    )


def ann_candidate_key(rule_id: str, case_name: str, top_k: int, ef_search: int) -> tuple[str, str, str, int, int]:
    return (str(rule_id), str(case_name), "", int(top_k), int(ef_search))
