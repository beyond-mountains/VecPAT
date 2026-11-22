from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SqlFilterPair:
    pattern: str
    base: str
    variant: str
    scale: int
    selectivity_label: str
    pattern_id: str
    case_name: str


def _eq(field: str, value: int) -> str:
    return f"{field} = {value}"


def _in(field: str, values: list[int]) -> str:
    return f"{field} IN ({', '.join(map(str, values))})"


def _join(operator: str, expressions: list[str]) -> str:
    return f" {operator} ".join(f"({expression})" for expression in expressions)


def build_sql_campaign_pairs(redundancy_scales: list[int], set_widths: list[int],
                             age_universe: int) -> list[SqlFilterPair]:
    pairs: list[SqlFilterPair] = []
    atom = _eq("category", 1)
    group = _join("OR", [_eq("category", 1), _eq("category", 2)])
    for scale in redundancy_scales:
        scale = int(scale)
        pairs.extend([
            SqlFilterPair("p1_duplicate", _in("category5", [1, 2, 3]),
                          _in("category5", [1, 2, 3] * scale), scale,
                          "category5_1_3", "P1", "duplicate_membership"),
            SqlFilterPair("p2_or_atom", atom, _join("OR", [atom] * scale), scale,
                          "category_1", "P2", "or_idempotent_atom"),
            SqlFilterPair("p2_or_group", group, _join("OR", [group] * scale), scale,
                          "category_1_2", "P2", "or_idempotent_group"),
            SqlFilterPair("p3_and_atom", atom, _join("AND", [atom] * scale), scale,
                          "category_1", "P3", "and_idempotent_atom"),
        ])
    for requested_width in set_widths:
        width = min(int(requested_width), age_universe)
        values = list(range(1, width + 1))
        age_range = f"age >= 1 AND age <= {width}"
        pairs.extend([
            SqlFilterPair("p4_range_to_in", age_range, _in("age", values), width,
                          f"age_1_{width}", "P4", "range_to_membership"),
            SqlFilterPair("p4_range_to_or", age_range,
                          _join("OR", [_eq("age", value) for value in values]), width,
                          f"age_1_{width}", "P4", "range_to_or"),
        ])
    pairs.extend([
        SqlFilterPair("p5_membership_to_or", _in("category5", [1, 2, 3]),
                      _join("OR", [_eq("category5", value) for value in [1, 2, 3]]), 3,
                      "category5_1_3", "P5", "membership_to_or"),
        SqlFilterPair("p6_dominated_lt", "age < 10", "(age < 10) AND (age < 20)", 2,
                      "age_lt_10", "P6", "dominated_less_than"),
        SqlFilterPair("p6_dominated_gt", "age > 10",
                      "(age > 10) AND (age > 5) AND (age > 0)", 3,
                      "age_gt_10", "P6", "dominated_greater_than_corrected"),
    ])
    identities = [
        ("category_1", atom),
        ("category5_1_3", _in("category5", [1, 2, 3])),
        ("age_lte_100", "age <= 100"),
    ]
    for label, expression in identities:
        pairs.extend([
            SqlFilterPair(f"p7_or_false_{label}", expression,
                          f"({expression}) OR (category = -1)", 1, label,
                          "P7", "boolean_or_false"),
            SqlFilterPair(f"p8_and_true_{label}", expression,
                          f"({expression}) AND (age >= 1 AND age <= {age_universe})", 1, label,
                          "P8", "boolean_and_true"),
        ])
    pairs.extend([
        SqlFilterPair("p9_de_morgan", "NOT ((category5 = 4) OR (category5 = 5))",
                      "(category5 <> 4) AND (category5 <> 5)", 2,
                      "category5_not_4_5", "P9", "de_morgan_not_or"),
        SqlFilterPair("p10_complement", _in("category5", [1, 2, 3]),
                      "category5 NOT IN (4, 5)", 3,
                      "category5_1_3", "P10", "finite_universe_complement"),
    ])
    return pairs
