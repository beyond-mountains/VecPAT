from __future__ import annotations

"""Expanded result_1002 generator: Rule x value x target x AST shape x strength."""

from dataclasses import replace

from .equivalent import (
    AND, NOT, OR, EmittedPair, Expr, RuleInstance, _features, atom,
    render_qdrant, render_text,
)


def _nary(op: str, xs: list[Expr]) -> Expr:
    return Expr(op, tuple(xs))


def _balanced(op: str, xs: list[Expr]) -> Expr:
    if len(xs) == 1:
        return xs[0]
    middle = len(xs) // 2
    return Expr(op, (_balanced(op, xs[:middle]), _balanced(op, xs[middle:])))


def _left_deep(op: str, xs: list[Expr]) -> Expr:
    node = xs[0]
    for item in xs[1:]:
        node = Expr(op, (node, item))
    return node


def _ri(rule: str, name: str, case: str, original: Expr, equivalent: Expr, strength: int,
        label: str, preconditions: str = "standard two-valued Boolean logic; referenced fields are non-null",
        equivalence: str = "guaranteed") -> RuleInstance:
    return RuleInstance(rule, "filter", name, case, original, equivalent, strength,
                        label, preconditions, equivalence)


def build_filter_instances_1002(strengths: list[int], age_universe: int = 1000) -> list[RuleInstance]:
    strengths = sorted(set(int(x) for x in strengths))
    out: list[RuleInstance] = []
    true_age = atom("between", "age", 1, age_universe)
    true_cat5 = atom("in", "category5", 1, 2, 3, 4, 5)
    false_category = atom("eq", "category", -1)
    false_age = atom("gt", "age", age_universe)

    # F1: numeric substitutions and two operator families.
    for low, high in ((1, 10), (7, 37), (101, 200), (401, 700)):
        out.append(_ri("F1", "operator_equivalence", f"age_{low}_{high}_between_vs_bounds",
                       atom("between", "age", low, high), AND(atom("gte", "age", low), atom("lte", "age", high)),
                       2, f"age_{low}_{high}", "age is a non-null integer"))
    for values in ((1,), (1, 3), (1, 3, 5), (2, 4, 5)):
        out.append(_ri("F1", "operator_equivalence", "category5_"+"_".join(map(str, values))+"_in_vs_or",
                       atom("in", "category5", *values), OR(*[atom("eq", "category5", x) for x in values]),
                       len(values), "category5_set", "category5 is non-null"))

    # F2: AND/OR idempotence on both atomic and grouped targets.
    targets = (("category_1", atom("eq", "category", 1)),
               ("age_lte_100", atom("lte", "age", 100)),
               ("category_pair", OR(atom("eq", "category5", 1), atom("eq", "category5", 2))),
               ("mixed_group", AND(atom("eq", "parity", "even"), atom("lte", "age", 500))))
    for n in strengths:
        for label, target in targets:
            for op in ("and", "or"):
                out.append(_ri("F2", "boolean_idempotence", f"{label}_{op}_repeat_{n}", target,
                               _nary(op, [target] * n), n, label))

    # F3: multiple TRUE/FALSE encodings and both identities.
    for n in strengths:
        for label, base in (("category_1", atom("eq", "category", 1)),
                            ("age_10_90", atom("between", "age", 10, 90))):
            out.extend([
                _ri("F3", "boolean_identity", f"{label}_and_true_age_{n}", base, AND(base, *([true_age] * n)), n, label,
                    f"age is constrained to [1,{age_universe}]", "domain-dependent"),
                _ri("F3", "boolean_identity", f"{label}_and_true_category5_{n}", base, AND(base, *([true_cat5] * n)), n, label,
                    "category5 universe is {1,2,3,4,5}", "domain-dependent"),
                _ri("F3", "boolean_identity", f"{label}_or_false_category_{n}", base, OR(base, *([false_category] * n)), n, label,
                    "category is always non-negative", "domain-dependent"),
                _ri("F3", "boolean_identity", f"{label}_or_false_age_{n}", base, OR(base, *([false_age] * n)), n, label,
                    f"age is constrained to [1,{age_universe}]", "domain-dependent"),
            ])

    # F4: double negation and both De Morgan directions with changed constants.
    for value in (1, 3, 5):
        p = atom("eq", "category5", value)
        out.append(_ri("F4", "negation_equivalence", f"double_not_category5_{value}", p, NOT(NOT(p)), 2, f"category5_{value}"))
    a, b = atom("eq", "category5", 4), atom("gt", "age", 750)
    out.extend([
        _ri("F4", "negation_equivalence", "de_morgan_not_or_mixed", NOT(OR(a, b)), AND(NOT(a), NOT(b)), 2, "not_or_mixed"),
        _ri("F4", "negation_equivalence", "de_morgan_not_and_mixed", NOT(AND(a, b)), OR(NOT(a), NOT(b)), 2, "not_and_mixed"),
    ])

    # F5: both absorption laws, multiple G predicates, controlled redundant branches.
    for n in strengths:
        for gname, g in (("age", atom("lte", "age", 100)), ("parity", atom("eq", "parity", "even"))):
            f = atom("eq", "category", 1)
            out.extend([
                _ri("F5", "absorption", f"or_absorption_{gname}_{n}", f, OR(f, *([AND(f, g)] * n)), n, "category_1"),
                _ri("F5", "absorption", f"and_absorption_{gname}_{n}", f, AND(f, *([OR(f, g)] * n)), n, "category_1"),
            ])

    # F6: both distribution directions and numerical substitutions.
    for idx, (a, b, c) in enumerate((
        (atom("eq", "parity", "even"), atom("eq", "category5", 1), atom("eq", "category5", 2)),
        (atom("lte", "age", 500), atom("eq", "category", 1), atom("eq", "category", 2)),
        (atom("gt", "age", 200), atom("eq", "category5", 3), atom("eq", "parity", "odd")),
    ), 1):
        out.extend([
            _ri("F6", "distribution", f"and_over_or_{idx}", AND(a, OR(b, c)), OR(AND(a, b), AND(a, c)), 2, f"distribution_{idx}"),
            _ri("F6", "distribution", f"or_over_and_{idx}", OR(a, AND(b, c)), AND(OR(a, b), OR(a, c)), 2, f"distribution_{idx}"),
        ])

    # F7: same leaves, different order and left/balanced AST shapes.
    leaves = [atom("gte", "age", 1), atom("lte", "age", 900), atom("in", "category5", 1, 2, 3),
              atom("in", "parity", "even", "odd")]
    for n in strengths:
        expanded = [leaves[i % len(leaves)] for i in range(n)]
        original = _nary("and", expanded)
        out.extend([
            _ri("F7", "structural_reordering", f"reverse_flat_{n}", original, _nary("and", list(reversed(expanded))), n, "structural_chain"),
            _ri("F7", "structural_reordering", f"balanced_reverse_{n}", original, _balanced("and", list(reversed(expanded))), n, "structural_chain"),
            _ri("F7", "structural_reordering", f"left_deep_rotate_{n}", original, _left_deep("and", expanded[1:]+expanded[:1]), n, "structural_chain"),
        ])

    # F8: weaker bounds, weaker supersets, and lower-bound implications.
    for n in strengths:
        out.extend([
            _ri("F8", "implication_redundancy", f"upper_bounds_{n}", atom("lt", "age", 10),
                AND(atom("lt", "age", 10), *[atom("lt", "age", 20+i*5) for i in range(n)]), n, "age_lt_10",
                "age < 10 implies every appended upper bound", "domain-dependent"),
            _ri("F8", "implication_redundancy", f"lower_bounds_{n}", atom("gt", "age", 900),
                AND(atom("gt", "age", 900), *[atom("gt", "age", max(0, 900-(i+1)*5)) for i in range(n)]), n, "age_gt_900",
                "age > 900 implies every appended lower bound", "domain-dependent"),
            _ri("F8", "implication_redundancy", f"membership_superset_{n}", atom("in", "category5", 1, 2),
                AND(atom("in", "category5", 1, 2), *([atom("in", "category5", 1, 2, 3, 4, 5)] * n)), n, "category5_1_2",
                "{1,2} is a subset of {1,2,3,4,5}", "domain-dependent"),
        ])

    # F9: value substitution plus equality and disjoint-range decompositions.
    for low, high, parts in ((1, 16, 2), (1, 64, 4), (101, 200, 4), (501, 700, 8)):
        base = atom("between", "age", low, high); width = (high-low+1)//parts
        segments=[]; cursor=low
        for i in range(parts):
            end = high if i == parts-1 else cursor+width-1; segments.append(atom("between", "age", cursor, end)); cursor=end+1
        out.append(_ri("F9", "predicate_decomposition", f"range_{low}_{high}_segments_{parts}", base, OR(*segments), parts,
                       f"age_{low}_{high}", "age is a discrete non-null integer", "domain-dependent"))
    for low, high in ((1, 8), (21, 36), (91, 122), (301, 364)):
        out.append(_ri("F9", "predicate_decomposition", f"range_{low}_{high}_equalities", atom("between", "age", low, high),
                       OR(*[atom("eq", "age", x) for x in range(low, high+1)]), high-low+1, f"age_{low}_{high}",
                       "age is a discrete non-null integer", "domain-dependent"))

    # F10: several subsets of the known category5 universe.
    universe = {1, 2, 3, 4, 5}
    for values in ((1,), (1, 2), (1, 3, 5), (2, 3, 4, 5)):
        complement = sorted(universe-set(values))
        out.append(_ri("F10", "complement_equivalence", "category5_"+"_".join(map(str, values)),
                       atom("in", "category5", *values), NOT(atom("in", "category5", *complement)), len(values), "category5_subset",
                       "category5 is non-null and its universe is exactly {1,2,3,4,5}", "domain-dependent"))

    # F11: duplicated and reordered lists with three base sets.
    for n in strengths:
        for values in ((1,), (1, 2, 3), (2, 4, 5)):
            repeated=list(values)*n; rotated=repeated[1:]+repeated[:1]
            out.append(_ri("F11", "set_idempotence", f"duplicate_{'_'.join(map(str,values))}_{n}", atom("in", "category5", *values),
                           atom("in", "category5", *rotated), n, "category5_set", "membership uses set semantics"))

    # Compound sequences: F1 -> F8 -> F2 -> F7, two AST layouts per strength.
    for n in strengths:
        base=atom("between", "age", 1, 100)
        core=AND(atom("gte", "age", 1), atom("lte", "age", 100), atom("lte", "age", 200))
        copies=[core]*n
        out.extend([
            _ri("C1", "compound_f1_f8_f2_f7", f"balanced_{n}", base, _balanced("and", list(reversed(copies))), n,
                "age_1_100", "age is a non-null integer; each step satisfies its rule precondition", "domain-dependent"),
            _ri("C1", "compound_f1_f8_f2_f7", f"left_deep_{n}", base, _left_deep("and", copies), n,
                "age_1_100", "age is a non-null integer; each step satisfies its rule precondition", "domain-dependent"),
        ])
    return out


def emit_filter_pairs_1002(backend: str, strengths: list[int], age_universe: int = 1000) -> list[EmittedPair]:
    renderer = render_qdrant if backend == "qdrant" else lambda x: render_text(x, "sql" if backend in {"pgvector", "seekdb"} else "milvus")
    pairs=[]
    for item in build_filter_instances_1002(strengths, age_universe):
        signature=_features(item.equivalent); signature.update({"transformation_depth":4 if item.rule_id=="C1" else 1,"strength":item.strength})
        pairs.append(EmittedPair(item.name,renderer(item.original),renderer(item.equivalent),item.strength,item.selectivity_label,
                                 item.rule_id,item.case_name,item.layer,item.preconditions,item.equivalence,signature))
    return pairs
