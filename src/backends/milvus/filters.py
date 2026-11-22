from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MilvusFilterPair:
    pattern: str
    base: str
    variant: str
    scale: int
    selectivity_label: str
    pattern_id: str = ""
    case_name: str = ""


def _eq(field: str, value: int) -> str:
    return f"{field} == {value}"


def _in(field: str, values: list[int]) -> str:
    return f"{field} in [{', '.join(map(str, values))}]"


def _join(operator: str, expressions: list[str]) -> str:
    return f" {operator} ".join(f"({expression})" for expression in expressions)


def build_milvus_pairs(tenants: int, tenant_widths: list[int], redundancy_scales: list[int],
                       set_widths: list[int], age_universe: int) -> list[MilvusFilterPair]:
    pairs: list[MilvusFilterPair] = []
    for width in tenant_widths:
        values = list(range(min(width, tenants)))
        membership = _in("tenant", values)
        pairs.extend([
            MilvusFilterPair("in_vs_or", membership, _join("or", [_eq("tenant", value) for value in values]),
                             width, f"tenant_width_{width}"),
            MilvusFilterPair("range_vs_in", f"tenant >= 0 and tenant <= {len(values) - 1}", membership,
                             width, f"tenant_width_{width}"),
        ])

    for requested_width in set_widths:
        width = min(int(requested_width), age_universe)
        values = list(range(1, width + 1))
        age_range = f"age <= {width}"
        age_membership = _in("age", values)
        age_or = _join("or", [_eq("age", value) for value in values])
        pairs.extend([
            MilvusFilterPair("set_range_vs_membership", age_range, age_membership, width, f"age_1_{width}"),
            MilvusFilterPair("set_range_vs_or", age_range, age_or, width, f"age_1_{width}"),
        ])

    membership = _in("category5", [1, 2, 3])
    pairs.extend([
        MilvusFilterPair("representation_membership_vs_or", membership,
                         _join("or", [_eq("category5", value) for value in [1, 2, 3]]),
                         3, "category5_1_3"),
        MilvusFilterPair("representation_membership_vs_range", membership,
                         "category5 >= 1 and category5 <= 3", 3, "category5_1_3"),
        MilvusFilterPair("representation_membership_vs_negation", membership,
                         "category5 not in [4, 5]", 3, "category5_1_3"),
    ])

    atom = _eq("category", 10)
    for scale in redundancy_scales:
        bounds = [f"age <= {min(age_universe, 100 + 100 * index)}" for index in range(1, scale + 1)]
        pairs.extend([
            MilvusFilterPair("duplicate_in", atom, _in("category", [10] * scale), scale, "category_10"),
            MilvusFilterPair("or_idempotence", atom, _join("or", [atom] * scale), scale, "category_10"),
            MilvusFilterPair("and_idempotence", atom, _join("and", [atom] * scale), scale, "category_10"),
            MilvusFilterPair("redundant_containment", atom,
                             _join("and", [atom] + [f"tenant >= 0 and tenant <= {tenants - 1}"] * scale),
                             scale, "category_10"),
            MilvusFilterPair("subsumption_redundancy", "age <= 100",
                             _join("and", ["age <= 100"] + bounds), scale, "age_lte_100"),
        ])
    return pairs


def build_milvus_campaign_pairs(redundancy_scales: list[int], set_widths: list[int],
                                age_universe: int) -> list[MilvusFilterPair]:
    pairs: list[MilvusFilterPair] = []
    atom = _eq("category", 1)
    group = _join("or", [_eq("category", 1), _eq("category", 2)])
    for scale in redundancy_scales:
        pairs.extend([
            MilvusFilterPair("p1_duplicate", _in("category5", [1, 2, 3]),
                             _in("category5", [1, 2, 3] * int(scale)), scale,
                             "category5_1_3", "P1", "duplicate_membership"),
            MilvusFilterPair("p2_or_atom", atom, _join("or", [atom] * int(scale)), scale,
                             "category_1", "P2", "or_idempotent_atom"),
            MilvusFilterPair("p2_or_group", group, _join("or", [group] * int(scale)), scale,
                             "category_1_2", "P2", "or_idempotent_group"),
            MilvusFilterPair("p3_and_atom", atom, _join("and", [atom] * int(scale)), scale,
                             "category_1", "P3", "and_idempotent_atom"),
        ])
    for requested_width in set_widths:
        width = min(int(requested_width), age_universe)
        values = list(range(1, width + 1))
        age_range = f"age >= 1 and age <= {width}"
        pairs.extend([
            MilvusFilterPair("p4_range_to_in", age_range, _in("age", values), width,
                             f"age_1_{width}", "P4", "range_to_membership"),
            MilvusFilterPair("p4_range_to_or", age_range,
                             _join("or", [_eq("age", value) for value in values]), width,
                             f"age_1_{width}", "P4", "range_to_or"),
        ])
    pairs.extend([
        MilvusFilterPair("p5_membership_to_or", _in("category5", [1, 2, 3]),
                         _join("or", [_eq("category5", value) for value in [1, 2, 3]]), 3,
                         "category5_1_3", "P5", "membership_to_or"),
        MilvusFilterPair("p6_dominated_lt", "age < 10", "(age < 10) and (age < 20)", 2,
                         "age_lt_10", "P6", "dominated_less_than"),
        MilvusFilterPair("p6_dominated_gt", "age > 10",
                         "(age > 10) and (age > 5) and (age > 0)", 3,
                         "age_gt_10", "P6", "dominated_greater_than_corrected"),
    ])
    identities = [
        ("category_1", atom),
        ("category5_1_3", _in("category5", [1, 2, 3])),
        ("age_lte_100", "age <= 100"),
    ]
    for label, expression in identities:
        pairs.extend([
            MilvusFilterPair(f"p7_or_false_{label}", expression,
                             f"({expression}) or (category == -1)", 1, label,
                             "P7", "boolean_or_false"),
            MilvusFilterPair(f"p8_and_true_{label}", expression,
                             f"({expression}) and (age >= 1 and age <= {age_universe})", 1, label,
                             "P8", "boolean_and_true"),
        ])
    pairs.extend([
        MilvusFilterPair("p9_de_morgan", "not ((category5 == 4) or (category5 == 5))",
                         "(category5 != 4) and (category5 != 5)", 2,
                         "category5_not_4_5", "P9", "de_morgan_not_or"),
        MilvusFilterPair("p10_complement", _in("category5", [1, 2, 3]),
                         "category5 not in [4, 5]", 3,
                         "category5_1_3", "P10", "finite_universe_complement"),
    ])
    return pairs
