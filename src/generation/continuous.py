from __future__ import annotations

"""High-yield equivalent-filter cases for the resumable campaign.

This module deliberately stays on the existing backend-neutral AST.  It adds
value substitutions and alternative tree shapes without bypassing the semantic
oracle used by the backend runners.
"""

import random

from .equivalent import (
    AND, NOT, OR, EmittedPair, Expr, RuleInstance, _features, atom,
    build_arithmetic_filter_instances, render_qdrant, render_text,
)

TARGET_FILTER_RULES = (
    "F1", "F2", "F3", "F4", "F5", "F6",
    "F7", "F8", "F9", "F10", "F11", "F12",
)


def _tree(op: str, leaves: list[Expr], shape: str, seed: int = 0) -> Expr:
    if not leaves:
        raise ValueError("a Boolean tree needs at least one leaf")
    xs = list(leaves)
    if shape == "random":
        random.Random(seed).shuffle(xs)
        shape = "balanced"
    if shape == "flat":
        return Expr(op, tuple(xs))
    if shape == "right":
        node = xs[-1]
        for item in reversed(xs[:-1]):
            node = Expr(op, (item, node))
        return node
    if shape == "left":
        node = xs[0]
        for item in xs[1:]:
            node = Expr(op, (node, item))
        return node
    if shape == "balanced":
        if len(xs) == 1:
            return xs[0]
        middle = len(xs) // 2
        return Expr(op, (_tree(op, xs[:middle], "balanced"), _tree(op, xs[middle:], "balanced")))
    raise ValueError(shape)


def _ri(rule: str, name: str, case: str, original: Expr, equivalent: Expr, strength: int,
        label: str, preconditions: str, equivalence: str = "guaranteed") -> RuleInstance:
    return RuleInstance(rule, "filter", name, case, original, equivalent, strength,
                        label, preconditions, equivalence)


def _disjoint_segments(low: int, high: int, parts: int) -> list[Expr]:
    """Partition an inclusive integer range into non-empty disjoint ranges."""
    width = high - low + 1
    parts = max(1, min(int(parts), width))
    base, remainder = divmod(width, parts)
    cursor = low
    segments: list[Expr] = []
    for index in range(parts):
        size = base + (1 if index < remainder else 0)
        end = cursor + size - 1
        segments.append(atom("between", "age", cursor, end))
        cursor = end + 1
    return segments


def build_continuous_filter_instances(strengths: list[int], age_universe: int = 1000) -> list[RuleInstance]:
    requested_strengths = sorted({max(1, int(value)) for value in strengths})
    strengths = sorted({max(2, int(value)) for value in strengths})
    out: list[RuleInstance] = []

    # F1: range/membership/OR, sparse values, boundary values, and conjunctions.
    for width in sorted({2, 8, 32, 128, min(age_universe, 512), age_universe}):
        low, high = 1, min(age_universe, width)
        values = list(range(low, high + 1))
        out.extend([
            _ri("F1", "operator_equivalence", f"range_{low}_{high}_to_in",
                atom("between", "age", low, high), atom("in", "age", *values), len(values),
                f"age_{low}_{high}", "age is a non-null integer", "domain-dependent"),
            _ri("F1", "operator_equivalence", f"range_{low}_{high}_to_or_balanced",
                atom("between", "age", low, high),
                _tree("or", [atom("eq", "age", value) for value in values], "balanced"), len(values),
                f"age_{low}_{high}", "age is a non-null integer", "domain-dependent"),
        ])
    for count in strengths:
        sparse = sorted({1 + (index * 37) % age_universe for index in range(min(count, age_universe))})
        original = atom("in", "age", *sparse)
        out.append(_ri("F1", "operator_equivalence", f"sparse_in_to_or_{len(sparse)}", original,
                       _tree("or", [atom("eq", "age", value) for value in sparse], "balanced"),
                       len(sparse), "sparse_age_set", "age is a non-null integer"))
    left = atom("in", "category5", 1, 3, 5)
    right = atom("between", "age", 1, min(100, age_universe))
    out.append(_ri("F1", "operator_equivalence", "multicolumn_in_range_to_or_bounds",
                   AND(left, right), AND(_tree("or", [atom("eq", "category5", x) for x in (1, 3, 5)], "balanced"),
                                         atom("gte", "age", 1), atom("lte", "age", min(100, age_universe))),
                   5, "multi_column", "age and category5 are non-null"))
    out.append(_ri("F1", "operator_equivalence", "boundary_min_max_in_to_or",
                   atom("in", "age", 1, age_universe), OR(atom("eq", "age", 1), atom("eq", "age", age_universe)),
                   2, "age_boundary", "age is a non-null integer"))

    # F2: P = P AND ... AND P = P OR ... OR P. Cover atomic/grouped
    # predicates and both flat and balanced AST representations.
    idempotent_targets = (
        ("category_atom", atom("eq", "category", 1)),
        ("age_range_atom", atom("lte", "age", min(100, age_universe))),
        ("category_or_group", OR(atom("eq", "category5", 1), atom("eq", "category5", 2))),
        ("mixed_and_group", AND(atom("eq", "parity", "even"), atom("lte", "age", min(500, age_universe)))),
    )
    for repeat in strengths:
        shapes = ("flat", "balanced") if repeat >= 8 else ("flat",)
        for label, target in idempotent_targets:
            for op in ("and", "or"):
                for shape in shapes:
                    out.append(_ri(
                        "F2", "boolean_idempotence",
                        f"{label}_{op}_{shape}_repeat_{repeat}",
                        target, _tree(op, [target] * repeat, shape), repeat, label,
                        "standard two-valued Boolean logic; referenced fields are non-null",
                    ))

    # F3: F AND TRUE = F and F OR FALSE = F. TRUE/FALSE are encoded with
    # finite-domain predicates so every backend can use the common AST.
    true_age = atom("between", "age", 1, age_universe)
    true_category5 = atom("in", "category5", 1, 2, 3, 4, 5)
    false_age = atom("gt", "age", age_universe)
    false_category = atom("eq", "category", -1)
    identity_bases = (
        ("category_atom", atom("eq", "category", 1)),
        ("age_range", atom("between", "age", min(10, age_universe), min(90, age_universe))),
        ("mixed_group", AND(atom("eq", "parity", "even"), atom("in", "category5", 1, 3, 5))),
    )
    identities = (
        ("and_true_age", "and", true_age, f"age universe is exactly [1,{age_universe}]"),
        ("and_true_category5", "and", true_category5, "category5 universe is exactly {1,2,3,4,5}"),
        ("or_false_age", "or", false_age, f"age universe is exactly [1,{age_universe}]"),
        ("or_false_category", "or", false_category, "category is always positive"),
    )
    for repeat in strengths:
        for label, base in identity_bases:
            for identity_name, op, identity, precondition in identities:
                out.append(_ri(
                    "F3", "boolean_identity", f"{label}_{identity_name}_{repeat}",
                    base, _tree(op, [base] + [identity] * repeat, "balanced"), repeat, label,
                    precondition, "domain-dependent",
                ))

    # F4: even-depth negation and large De Morgan trees.
    # Very deep unary NOT chains hit parser/transport recursion limits before
    # they become useful.  Keep that axis at 256 AST levels; large Boolean
    # fan-out is still explored up to the requested 1024 branches below.
    for depth in sorted({min(value, 128) for value in strengths}):
        base = atom("lte", "age", min(500, age_universe))
        wrapped = base
        for _ in range(2 * depth):
            wrapped = NOT(wrapped)
        out.append(_ri("F4", "negation_equivalence", f"double_not_depth_{2 * depth}", base, wrapped,
                       2 * depth, "age_mid", "age is non-null"))
        leaves = [atom("eq", "age", 1 + index % age_universe) for index in range(depth)]
        out.append(_ri("F4", "negation_equivalence", f"de_morgan_or_{depth}", NOT(_tree("or", leaves, "balanced")),
                       _tree("and", [NOT(leaf) for leaf in leaves], "balanced"), depth,
                       "not_age_set", "age is non-null"))
    negated_values = [1, 2, 3, 4]
    out.extend([
        _ri("F4", "negation_equivalence", "not_in_to_conjunction",
            NOT(atom("in", "category5", *negated_values)),
            _tree("and", [NOT(atom("eq", "category5", value)) for value in negated_values], "balanced"),
            len(negated_values), "not_category_set", "category5 is non-null"),
        _ri("F4", "negation_equivalence", "not_range_to_outside_bounds",
            NOT(atom("between", "age", 20, 40)), OR(atom("lt", "age", 20), atom("gt", "age", 40)),
            2, "outside_age_range", "age is non-null"),
        _ri("F4", "negation_equivalence", "mixed_nested_de_morgan",
            NOT(OR(AND(atom("gte", "age", 20), atom("lte", "age", 40)), atom("in", "category5", 1, 2, 3))),
            AND(OR(atom("lt", "age", 20), atom("gt", "age", 40)),
                AND(NOT(atom("eq", "category5", 1)), NOT(atom("eq", "category5", 2)), NOT(atom("eq", "category5", 3)))),
            7, "mixed_negation", "age and category5 are non-null"),
    ])

    # F5: absorption with controlled redundant branch counts:
    # F OR (F AND G) = F and F AND (F OR G) = F.
    absorption_bases = (
        ("category_atom", atom("eq", "category", 1)),
        ("age_range", atom("between", "age", min(100, age_universe), min(300, age_universe))),
    )
    absorption_guards = (
        ("age_guard", atom("lte", "age", min(100, age_universe))),
        ("parity_guard", atom("eq", "parity", "even")),
        ("category5_guard", atom("in", "category5", 1, 2)),
    )
    for repeat in strengths:
        for base_label, base in absorption_bases:
            for guard_label, guard in absorption_guards:
                out.extend([
                    _ri("F5", "absorption", f"or_{base_label}_{guard_label}_{repeat}",
                        base, _tree("or", [base] + [AND(base, guard)] * repeat, "balanced"),
                        repeat, base_label,
                        "standard two-valued Boolean logic; referenced fields are non-null"),
                    _ri("F5", "absorption", f"and_{base_label}_{guard_label}_{repeat}",
                        base, _tree("and", [base] + [OR(base, guard)] * repeat, "balanced"),
                        repeat, base_label,
                        "standard two-valued Boolean logic; referenced fields are non-null"),
                ])

    # F6: factored form versus controlled DNF/CNF expansion.
    for branches in strengths:
        common = atom("in", "category5", 1, 2, 3)
        choices = [atom("eq", "age", 1 + index % age_universe) for index in range(branches)]
        out.extend([
            _ri("F6", "distribution", f"factor_to_dnf_{branches}", AND(common, _tree("or", choices, "balanced")),
                _tree("or", [AND(common, choice) for choice in choices], "balanced"), branches,
                "factored_dnf", "all fields are non-null"),
            _ri("F6", "distribution", f"factor_to_cnf_{branches}", OR(common, _tree("and", choices, "balanced")),
                _tree("and", [OR(common, choice) for choice in choices], "balanced"), branches,
                "factored_cnf", "all fields are non-null"),
        ])

    # F7: same leaves, only order and shape change.
    leaf_bank = [atom("gte", "age", 1), atom("lte", "age", age_universe),
                 atom("in", "category5", 1, 2, 3), atom("in", "parity", "even", "odd")]
    for count in sorted({4, 8, 16, 32, 64, 128}):
        leaves = [leaf_bank[index % len(leaf_bank)] for index in range(count)]
        original = _tree("and", leaves, "flat")
        for shape in ("left", "right", "balanced", "random"):
            out.append(_ri("F7", "structural_reordering", f"and_{shape}_{count}", original,
                           _tree("and", list(reversed(leaves)), shape, count), count,
                           "same_leaves", "all fields are non-null"))

    # F8: append predicates logically implied by the original predicate.
    for repeat in strengths:
        upper = atom("lt", "age", min(10, age_universe))
        weaker_upper = [atom("lt", "age", min(age_universe + 1, 20 + index * 5))
                        for index in range(repeat)]
        lower_cut = max(1, age_universe - 100)
        lower = atom("gt", "age", lower_cut)
        weaker_lower = [atom("gt", "age", max(0, lower_cut - (index + 1) * 5))
                        for index in range(repeat)]
        membership = atom("in", "category5", 1, 2)
        membership_superset = atom("in", "category5", 1, 2, 3, 4, 5)
        out.extend([
            _ri("F8", "implication_redundancy", f"upper_bounds_balanced_{repeat}", upper,
                _tree("and", [upper] + weaker_upper, "balanced"), repeat, "age_low",
                "the original upper bound implies every appended weaker upper bound", "domain-dependent"),
            _ri("F8", "implication_redundancy", f"lower_bounds_balanced_{repeat}", lower,
                _tree("and", [lower] + weaker_lower, "balanced"), repeat, "age_high",
                "the original lower bound implies every appended weaker lower bound", "domain-dependent"),
            _ri("F8", "implication_redundancy", f"membership_superset_{repeat}", membership,
                _tree("and", [membership] + [membership_superset] * repeat, "balanced"), repeat,
                "category5_1_2", "{1,2} is a subset of {1,2,3,4,5}", "domain-dependent"),
        ])

    # F9: decompose one discrete range into disjoint subranges or equalities.
    # Width, constants, part count, and AST representation vary by strength.
    for requested_parts in strengths:
        width = min(age_universe, max(16, requested_parts))
        low, high = 1, width
        segments = _disjoint_segments(low, high, requested_parts)
        out.append(_ri(
            "F9", "predicate_decomposition", f"range_{low}_{high}_segments_balanced_{len(segments)}",
            atom("between", "age", low, high), _tree("or", segments, "balanced"), len(segments),
            f"age_{low}_{high}", "age is a discrete non-null integer", "domain-dependent",
        ))

        shifted_width = min(age_universe, max(32, requested_parts * 2))
        shifted_low = max(1, (age_universe - shifted_width) // 2 + 1)
        shifted_high = shifted_low + shifted_width - 1
        shifted_segments = _disjoint_segments(shifted_low, shifted_high, requested_parts)
        out.append(_ri(
            "F9", "predicate_decomposition",
            f"range_{shifted_low}_{shifted_high}_segments_flat_{len(shifted_segments)}",
            atom("between", "age", shifted_low, shifted_high),
            _tree("or", shifted_segments, "flat"), len(shifted_segments),
            f"age_{shifted_low}_{shifted_high}", "age is a discrete non-null integer", "domain-dependent",
        ))

        equality_width = min(age_universe, requested_parts)
        equalities = [atom("eq", "age", value) for value in range(1, equality_width + 1)]
        out.append(_ri(
            "F9", "predicate_decomposition", f"range_1_{equality_width}_equalities_balanced",
            atom("between", "age", 1, equality_width), _tree("or", equalities, "balanced"), equality_width,
            f"age_1_{equality_width}", "age is a discrete non-null integer", "domain-dependent",
        ))

    # F10: finite universe complement at extreme and balanced selectivities.
    universe = list(range(1, age_universe + 1))
    for percentage in (1, 10, 50, 90, 99):
        size = max(1, min(age_universe - 1, round(age_universe * percentage / 100)))
        selected, complement = universe[:size], universe[size:]
        out.append(_ri("F10", "complement_equivalence", f"age_universe_{percentage}pct",
                       atom("in", "age", *selected), NOT(atom("in", "age", *complement)),
                       max(len(selected), len(complement)), f"age_{percentage}pct",
                       f"age is non-null and its universe is exactly [1,{age_universe}]", "domain-dependent"))
    selected, complement = universe[:max(1, age_universe // 10)], universe[max(1, age_universe // 10):]
    gate, left_branch, right_branch = atom("eq", "parity", "even"), atom("eq", "category5", 1), atom("eq", "category5", 2)
    out.append(_ri("F10", "complement_plus_distribution", "complement_combined_with_dnf",
                   AND(atom("in", "age", *selected), OR(left_branch, right_branch), gate),
                   OR(AND(NOT(atom("in", "age", *complement)), left_branch, gate),
                      AND(NOT(atom("in", "age", *complement)), right_branch, gate)),
                   max(len(selected), len(complement)), "complement_dnf",
                   f"age is non-null and its universe is exactly [1,{age_universe}]", "domain-dependent"))

    # F11: duplicate/reordered membership with controlled blow-up.
    for repeat in strengths:
        values = (1, 2, 3)
        duplicated = list(values) * repeat
        random.Random(repeat).shuffle(duplicated)
        out.append(_ri("F11", "set_idempotence", f"duplicate_in_shuffled_{repeat}",
                       atom("in", "category5", *values), atom("in", "category5", *duplicated), repeat,
                       "category5_1_3", "membership uses set semantics"))
    out.extend(build_arithmetic_filter_instances(requested_strengths))
    return out


def emit_filter_pairs_continuous(backend: str, strengths: list[int], age_universe: int = 1000) -> list[EmittedPair]:
    if backend in {"qdrant", "weaviate"}:
        unsupported = {"F12"}
    else:
        unsupported = set()
    renderer = render_qdrant if backend == "qdrant" else lambda value: render_text(
        value, "sql" if backend in {"pgvector", "seekdb"} else "milvus"
    )
    pairs = []
    for item in build_continuous_filter_instances(strengths, age_universe):
        if item.rule_id in unsupported:
            continue
        signature = _features(item.equivalent)
        signature.update({"transformation_depth": 1, "strength": item.strength})
        pairs.append(EmittedPair(item.name, renderer(item.original), renderer(item.equivalent), item.strength,
                                 item.selectivity_label, item.rule_id, item.case_name, item.layer,
                                 item.preconditions, item.equivalence, signature))
    return pairs
