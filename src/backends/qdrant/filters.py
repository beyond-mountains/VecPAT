from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterPair:
    pattern: str
    base: dict
    variant: dict
    scale: int
    selectivity_label: str
    pattern_id: str = ""
    case_name: str = ""


def match_value(key: str, value: int | str) -> dict:
    return {"key": key, "match": {"value": value}}


def match_any(key: str, values: list[int | str]) -> dict:
    return {"key": key, "match": {"any": values}}


def build_pairs(tenants: int, tenant_widths: list[int], redundancy_scales: list[int],
                set_widths: list[int], age_universe: int) -> list[FilterPair]:
    pairs: list[FilterPair] = []
    for width in tenant_widths:
        values = list(range(min(width, tenants)))
        base = {"must": [match_any("tenant", values)]}
        pairs.append(FilterPair(
            "in_vs_or", base,
            {"should": [match_value("tenant", v) for v in values]},
            width, f"tenant_width_{width}",
        ))
        pairs.append(FilterPair(
            "range_vs_in",
            {"must": [{"key": "tenant", "range": {"gte": 0, "lte": len(values) - 1}}]},
            base, width, f"tenant_width_{width}",
        ))

    # Set expansion: the same finite integer set expressed as a range, membership, or OR.
    for width in set_widths:
        width = min(int(width), age_universe)
        values = list(range(1, width + 1))
        age_range = {"must": [{"key": "age", "range": {"lte": width}}]}
        age_membership = {"must": [match_any("age", values)]}
        age_or = {"should": [match_value("age", value) for value in values]}
        pairs.extend([
            FilterPair("set_range_vs_membership", age_range, age_membership, width, f"age_1_{width}"),
            FilterPair("set_range_vs_or", age_range, age_or, width, f"age_1_{width}"),
        ])

    # Representation equivalence on a small, closed category universe.
    category_values = [1, 2, 3]
    category_membership = {"must": [match_any("category5", category_values)]}
    pairs.extend([
        FilterPair("representation_membership_vs_or", category_membership,
                   {"should": [match_value("category5", value) for value in category_values]},
                   len(category_values), "category5_1_3"),
        FilterPair("representation_membership_vs_range", category_membership,
                   {"must": [
                       {"key": "category5", "range": {"gte": 1}},
                       {"key": "category5", "range": {"lte": 3}},
                   ]}, len(category_values), "category5_1_3"),
        FilterPair("representation_membership_vs_negation", category_membership,
                   {"must_not": [match_any("category5", [4, 5])]},
                   len(category_values), "category5_1_3"),
    ])

    atom = match_value("category", 10)
    base_atom = {"must": [atom]}
    for scale in redundancy_scales:
        subsuming_bounds = [
            {"key": "age", "range": {"lte": min(age_universe, 100 + 100 * i)}}
            for i in range(1, scale + 1)
        ]
        pairs.extend([
            FilterPair("duplicate_in", base_atom,
                       {"must": [match_any("category", [10] * scale)]}, scale, "category_10"),
            FilterPair("or_idempotence", base_atom,
                       {"should": [atom] * scale}, scale, "category_10"),
            FilterPair("and_idempotence", base_atom,
                       {"must": [atom] * scale}, scale, "category_10"),
            FilterPair("redundant_containment", base_atom,
                       {"must": [atom] + [{"key": "tenant", "range": {"gte": 0, "lte": tenants - 1}}] * scale},
                       scale, "category_10"),
            FilterPair("subsumption_redundancy",
                       {"must": [{"key": "age", "range": {"lte": 100}}]},
                       {"must": [{"key": "age", "range": {"lte": 100}}] + subsuming_bounds},
                       scale, "age_lte_100"),
        ])
    return pairs


def build_campaign_pairs(redundancy_scales: list[int], set_widths: list[int],
                         age_universe: int) -> list[FilterPair]:
    """Build the P1-P10 equivalent-filter campaign requested for results_0924."""
    pairs: list[FilterPair] = []
    category_atom = match_value("category", 1)
    category_atom_filter = {"must": [category_atom]}
    category_pair = {"should": [match_value("category", 1), match_value("category", 2)]}

    for scale in redundancy_scales:
        repeated_members = [1, 2, 3] * int(scale)
        pairs.extend([
            FilterPair("p1_duplicate", {"must": [match_any("category5", [1, 2, 3])]},
                       {"must": [match_any("category5", repeated_members)]}, scale,
                       "category5_1_3", "P1", "duplicate_membership"),
            FilterPair("p2_or_atom", category_atom_filter,
                       {"should": [category_atom] * int(scale)}, scale,
                       "category_1", "P2", "or_idempotent_atom"),
            FilterPair("p2_or_group", category_pair,
                       {"should": [category_pair] * int(scale)}, scale,
                       "category_1_2", "P2", "or_idempotent_group"),
            FilterPair("p3_and_atom", category_atom_filter,
                       {"must": [category_atom] * int(scale)}, scale,
                       "category_1", "P3", "and_idempotent_atom"),
        ])

    for requested_width in set_widths:
        width = min(int(requested_width), age_universe)
        values = list(range(1, width + 1))
        age_range = {"must": [{"key": "age", "range": {"gte": 1, "lte": width}}]}
        pairs.extend([
            FilterPair("p4_range_to_in", age_range, {"must": [match_any("age", values)]}, width,
                       f"age_1_{width}", "P4", "range_to_membership"),
            FilterPair("p4_range_to_or", age_range,
                       {"should": [match_value("age", value) for value in values]}, width,
                       f"age_1_{width}", "P4", "range_to_or"),
        ])

    pairs.extend([
        FilterPair("p5_membership_to_or", {"must": [match_any("category5", [1, 2, 3])]},
                   {"should": [match_value("category5", value) for value in [1, 2, 3]]}, 3,
                   "category5_1_3", "P5", "membership_to_or"),
        FilterPair("p6_dominated_lt", {"must": [{"key": "age", "range": {"lt": 10}}]},
                   {"must": [{"key": "age", "range": {"lt": 10}},
                              {"key": "age", "range": {"lt": 20}}]}, 2,
                   "age_lt_10", "P6", "dominated_less_than"),
        # Corrected equivalence: age>10 dominates the weaker age>5 and age>0 predicates.
        FilterPair("p6_dominated_gt", {"must": [{"key": "age", "range": {"gt": 10}}]},
                   {"must": [{"key": "age", "range": {"gt": 10}},
                              {"key": "age", "range": {"gt": 5}},
                              {"key": "age", "range": {"gt": 0}}]}, 3,
                   "age_gt_10", "P6", "dominated_greater_than_corrected"),
    ])

    false_condition = match_value("category", -1)
    true_condition = {"key": "age", "range": {"gte": 1, "lte": age_universe}}
    identity_atoms = [
        ("category_1", category_atom),
        ("category5_1_3", match_any("category5", [1, 2, 3])),
        ("age_lte_100", {"key": "age", "range": {"lte": 100}}),
    ]
    for label, atom in identity_atoms:
        pairs.extend([
            FilterPair(f"p7_or_false_{label}", {"must": [atom]},
                       {"should": [atom, false_condition]}, 1, label,
                       "P7", "boolean_or_false"),
            FilterPair(f"p8_and_true_{label}", {"must": [atom]},
                       {"must": [atom, true_condition]}, 1, label,
                       "P8", "boolean_and_true"),
        ])

    a = match_value("category5", 4)
    b = match_value("category5", 5)
    pairs.extend([
        FilterPair("p9_de_morgan", {"must_not": [{"should": [a, b]}]},
                   {"must_not": [a, b]}, 2, "category5_not_4_5",
                   "P9", "de_morgan_not_or"),
        FilterPair("p10_complement", {"must": [match_any("category5", [1, 2, 3])]},
                   {"must_not": [match_any("category5", [4, 5])]}, 3,
                   "category5_1_3", "P10", "finite_universe_complement"),
    ])
    return pairs
