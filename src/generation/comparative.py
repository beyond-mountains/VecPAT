from __future__ import annotations

"""Portable parts of AMOEBA, SCor, and CODDTest for Filter-ANNS.

This is intentionally an *adaptation* baseline.  It does not claim to be the
original systems.  Every generated pair is represented with the project's
backend-neutral filter AST and is checked again by the eligible-ID oracle.
"""

from typing import Any

from .equivalent import (
    AND, OR, EmittedPair, Expr, RuleInstance, _features, arithmetic_compare,
    atom, render_qdrant, render_text, scalar_field, scalar_literal,
)

METHODS = ("amoeba", "scor", "coddtest")
ARITHMETIC_BACKENDS = frozenset({"milvus", "pgvector", "seekdb"})
SQL_BACKENDS = frozenset({"pgvector", "seekdb"})


def _rule(method: str, category: str, case_name: str, original: Expr,
          equivalent: Expr, strength: int, preconditions: str,
          equivalence: str = "guaranteed") -> RuleInstance:
    return RuleInstance(
        f"{method.upper()}_{category}", "filter", category,
        f"{method}_{case_name}", original, equivalent, strength,
        "comparative_baseline", preconditions, equivalence,
    )


def build_comparative_instances(method: str, strengths: list[int],
                                age_universe: int = 1000,
                                generation_round: int = 0) -> list[RuleInstance]:
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"unknown comparative method: {method}")
    scales = sorted({max(1, int(value)) for value in strengths})
    if generation_round > 0:
        return _build_round_instances(method, scales, age_universe, generation_round)
    f = AND(atom("gt", "age", 10), atom("in", "category5", 1, 2, 3))
    g = AND(atom("eq", "parity", "even"), atom("lt", "age", 900))
    true_domain = atom("between", "age", 1, age_universe)
    false_domain = atom("gt", "age", age_universe)
    out: list[RuleInstance] = []

    if method == "amoeba":
        # AMOEBA rules 53/54: reduce constant expressions in a Calc/filter.
        out.extend([
            _rule(method, "constant_expression_reduction", "constant_addition",
                  arithmetic_compare("gt", scalar_field("age"),
                                     scalar_literal(10, ("add", 5))),
                  atom("gt", "age", 15), 1,
                  "age is non-null numeric and arithmetic does not overflow"),
            _rule(method, "constant_expression_reduction", "nested_constant_expression",
                  arithmetic_compare("gt", scalar_field("age"),
                                     scalar_literal(2, ("mul", 5), ("add", 5))),
                  atom("gt", "age", 15), 2,
                  "age is non-null numeric and arithmetic does not overflow"),
            _rule(method, "constant_expression_reduction", "constant_subtraction",
                  arithmetic_compare("gt", scalar_field("age"), scalar_literal(20, ("sub", 5))),
                  atom("gt", "age", 15), 1,
                  "age is non-null numeric and arithmetic does not overflow"),
            _rule(method, "constant_expression_reduction", "constant_multiplication",
                  arithmetic_compare("gt", scalar_field("age"), scalar_literal(3, ("mul", 5))),
                  atom("gt", "age", 15), 1,
                  "age is non-null numeric and arithmetic does not overflow"),
            _rule(method, "constant_expression_reduction", "constant_division",
                  arithmetic_compare("gt", scalar_field("age"), scalar_literal(30, ("div", 2))),
                  atom("gt", "age", 15), 1,
                  "division has exact non-zero numeric semantics"),
            _rule(method, "constant_expression_reduction", "mixed_nested_constant",
                  arithmetic_compare("gte", scalar_field("age"), scalar_literal(2, ("add", 3), ("mul", 3))),
                  atom("gte", "age", 15), 3,
                  "age is non-null numeric and arithmetic does not overflow"),
            _rule(method, "constant_expression_reduction", "constant_less_than",
                  arithmetic_compare("lt", scalar_field("age"), scalar_literal(100, ("sub", 25))),
                  atom("lt", "age", 75), 1,
                  "age is non-null numeric and arithmetic does not overflow"),
        ])
        # Adapt relational Filter-Merge rules 14/18 to nested vs flat Filter ASTs.
        for scale in scales:
            leaves = [atom("gt", "age", 10), atom("lte", "age", age_universe),
                      atom("in", "category5", 1, 2, 3), atom("eq", "parity", "even")]
            leaves = [leaves[index % len(leaves)] for index in range(max(2, scale))]
            nested = leaves[0]
            for leaf in leaves[1:]:
                nested = AND(nested, leaf)
            out.append(_rule(method, "filter_merging", f"nested_to_flat_{scale}",
                             nested, Expr("and", tuple(leaves)), scale,
                             "all referenced metadata fields are non-null"))
            profiles = (
                ("selective", [atom("gt", "age", 900), atom("eq", "category5", 1),
                                atom("eq", "parity", "even")]),
                ("broad", [atom("gte", "age", 1), atom("lte", "age", age_universe),
                            atom("in", "category5", 1, 2, 3, 4)]),
            )
            for profile, bank in profiles:
                xs = [bank[index % len(bank)] for index in range(max(2, scale))]
                right = xs[-1]
                for leaf in reversed(xs[:-1]):
                    right = AND(leaf, right)
                out.append(_rule(method, "filter_merging", f"{profile}_right_nested_to_flat_{scale}",
                                 right, Expr("and", tuple(xs)), scale,
                                 "all referenced metadata fields are non-null"))

    elif method == "scor":
        # TRUE/FALSE are represented by predicates over the declared finite
        # metadata domain, because most vector DB filter DSLs have no literal
        # Boolean node.  This preserves SCor's short-circuit construction.
        for scale in scales:
            expensive = f
            for _ in range(max(1, scale)):
                expensive = OR(expensive, AND(f, g))
            out.extend([
                _rule(method, "and_false", f"and_false_{scale}", false_domain,
                      AND(expensive, false_domain), scale,
                      f"age is non-null and its universe is [1,{age_universe}]", "domain-dependent"),
                _rule(method, "or_true", f"or_true_{scale}", true_domain,
                      OR(expensive, true_domain), scale,
                      f"age is non-null and its universe is [1,{age_universe}]", "domain-dependent"),
                _rule(method, "and_true", f"and_true_{scale}", f,
                      AND(f, *([true_domain] * scale)), scale,
                      f"age is non-null and its universe is [1,{age_universe}]", "domain-dependent"),
                _rule(method, "or_false", f"or_false_{scale}", f,
                      OR(f, *([false_domain] * scale)), scale,
                      f"age is non-null and its universe is [1,{age_universe}]", "domain-dependent"),
                _rule(method, "absorption_or", f"absorption_or_{scale}", f,
                      OR(f, *([AND(f, g)] * scale)), scale,
                      "two-valued Boolean semantics; fields are non-null"),
                _rule(method, "absorption_and", f"absorption_and_{scale}", f,
                      AND(f, *([OR(f, g)] * scale)), scale,
                      "two-valued Boolean semantics; fields are non-null"),
            ])
            # A second selectivity/metadata profile prevents the baseline from
            # degenerating into repetitions of one predicate shape.
            f2 = OR(atom("between", "age", 400, 600), atom("eq", "category5", 5))
            g2 = AND(atom("gte", "age", 250), atom("eq", "parity", "odd"))
            expensive2 = f2
            for _ in range(max(1, scale)):
                expensive2 = AND(expensive2, OR(f2, g2))
            out.extend([
                _rule(method, "and_false", f"profile2_and_false_{scale}", false_domain,
                      AND(expensive2, false_domain), scale,
                      f"age is non-null and its universe is [1,{age_universe}]", "domain-dependent"),
                _rule(method, "or_true", f"profile2_or_true_{scale}", true_domain,
                      OR(expensive2, true_domain), scale,
                      f"age is non-null and its universe is [1,{age_universe}]", "domain-dependent"),
                _rule(method, "and_true", f"profile2_and_true_{scale}", f2,
                      AND(f2, *([true_domain] * scale)), scale,
                      f"age is non-null and its universe is [1,{age_universe}]", "domain-dependent"),
                _rule(method, "or_false", f"profile2_or_false_{scale}", f2,
                      OR(f2, *([false_domain] * scale)), scale,
                      f"age is non-null and its universe is [1,{age_universe}]", "domain-dependent"),
                _rule(method, "absorption_or", f"profile2_absorption_or_{scale}", f2,
                      OR(f2, *([AND(f2, g2)] * scale)), scale,
                      "two-valued Boolean semantics; fields are non-null"),
                _rule(method, "absorption_and", f"profile2_absorption_and_{scale}", f2,
                      AND(f2, *([OR(f2, g2)] * scale)), scale,
                      "two-valued Boolean semantics; fields are non-null"),
            ])

    else:  # CODDTest
        out.extend([
            _rule(method, "arithmetic_constant_folding", "constant_addition",
                  arithmetic_compare("gt", scalar_field("age"),
                                     scalar_literal(10, ("add", 5))),
                  atom("gt", "age", 15), 1,
                  "age is non-null numeric and arithmetic does not overflow"),
            _rule(method, "nested_expression_folding", "nested_arithmetic",
                  arithmetic_compare("gt", scalar_field("age"),
                                     scalar_literal(2, ("mul", 5), ("add", 5))),
                  atom("gt", "age", 15), 2,
                  "age is non-null numeric and arithmetic does not overflow"),
            _rule(method, "arithmetic_constant_folding", "constant_subtraction",
                  arithmetic_compare("gt", scalar_field("age"), scalar_literal(20, ("sub", 5))),
                  atom("gt", "age", 15), 1,
                  "age is non-null numeric and arithmetic does not overflow"),
            _rule(method, "arithmetic_constant_folding", "constant_multiplication",
                  arithmetic_compare("gt", scalar_field("age"), scalar_literal(3, ("mul", 5))),
                  atom("gt", "age", 15), 1,
                  "age is non-null numeric and arithmetic does not overflow"),
            _rule(method, "nested_expression_folding", "mixed_nested_constant",
                  arithmetic_compare("gte", scalar_field("age"), scalar_literal(2, ("add", 3), ("mul", 3))),
                  atom("gte", "age", 15), 3,
                  "age is non-null numeric and arithmetic does not overflow"),
        ])
        for scale in scales:
            out.append(_rule(method, "boolean_constant_folding",
                             f"known_true_domain_{scale}", f,
                             AND(f, *([true_domain] * scale)), scale,
                             f"fixed database state has non-null age in [1,{age_universe}]",
                             "database-state-dependent"))
            out.extend([
                _rule(method, "boolean_constant_folding", f"known_false_or_{scale}", f,
                      OR(f, *([false_domain] * scale)), scale,
                      f"fixed database state has non-null age in [1,{age_universe}]",
                      "database-state-dependent"),
                _rule(method, "boolean_constant_folding", f"range_known_true_{scale}",
                      atom("between", "age", 100, 300),
                      AND(atom("between", "age", 100, 300), *([true_domain] * scale)), scale,
                      f"fixed database state has non-null age in [1,{age_universe}]",
                      "database-state-dependent"),
                _rule(method, "boolean_constant_folding", f"range_known_false_{scale}",
                      atom("between", "age", 100, 300),
                      OR(atom("between", "age", 100, 300), *([false_domain] * scale)), scale,
                      f"fixed database state has non-null age in [1,{age_universe}]",
                      "database-state-dependent"),
            ])
    return out


def _build_round_instances(method: str, scales: list[int], age_universe: int,
                           generation_round: int) -> list[RuleInstance]:
    """Generate a fresh deterministic case family for one continuous round."""
    r = int(generation_round)
    low = 5 + (r * 37) % 700
    offset = 1 + (r * 13) % 97
    cat = 1 + r % 5
    parity = "even" if r % 2 == 0 else "odd"
    true_domain = atom("between", "age", 1, age_universe)
    false_domain = atom("gt", "age", age_universe)
    f = AND(atom("gt", "age", low), atom("eq", "category5", cat))
    g = OR(atom("eq", "parity", parity), atom("lt", "age", min(age_universe, low + 200)))
    out: list[RuleInstance] = []

    if method == "amoeba":
        out.append(_rule(method, "constant_expression_reduction", f"r{r}_constant_add_{low}_{offset}",
                         arithmetic_compare("gt", scalar_field("age"), scalar_literal(low, ("add", offset))),
                         atom("gt", "age", low + offset), offset,
                         "age is non-null numeric and arithmetic does not overflow"))
        for scale in scales:
            bank = [atom("gt", "age", low), atom("eq", "category5", cat),
                    atom("eq", "parity", parity), atom("lte", "age", age_universe)]
            xs = [bank[(index + r) % len(bank)] for index in range(max(2, scale))]
            left = xs[0]
            for leaf in xs[1:]: left = AND(left, leaf)
            right = xs[-1]
            for leaf in reversed(xs[:-1]): right = AND(leaf, right)
            flat = Expr("and", tuple(xs))
            out.extend([
                _rule(method, "filter_merging", f"r{r}_left_to_flat_{scale}", left, flat, scale,
                      "all referenced metadata fields are non-null"),
                _rule(method, "filter_merging", f"r{r}_right_to_flat_{scale}", right, flat, scale,
                      "all referenced metadata fields are non-null"),
            ])
    elif method == "scor":
        for scale in scales:
            expensive = f
            for _ in range(scale): expensive = OR(expensive, AND(f, g))
            prefix = f"r{r}_s{scale}"
            out.extend([
                _rule(method, "and_false", prefix + "_and_false", false_domain, AND(expensive, false_domain), scale,
                      "finite non-null age domain", "domain-dependent"),
                _rule(method, "or_true", prefix + "_or_true", true_domain, OR(expensive, true_domain), scale,
                      "finite non-null age domain", "domain-dependent"),
                _rule(method, "and_true", prefix + "_and_true", f, AND(f, *([true_domain] * scale)), scale,
                      "finite non-null age domain", "domain-dependent"),
                _rule(method, "or_false", prefix + "_or_false", f, OR(f, *([false_domain] * scale)), scale,
                      "finite non-null age domain", "domain-dependent"),
                _rule(method, "absorption_or", prefix + "_absorption_or", f,
                      OR(f, *([AND(f, g)] * scale)), scale, "two-valued non-null metadata"),
                _rule(method, "absorption_and", prefix + "_absorption_and", f,
                      AND(f, *([OR(f, g)] * scale)), scale, "two-valued non-null metadata"),
            ])
    else:
        out.append(_rule(method, "arithmetic_constant_folding", f"r{r}_constant_add_{low}_{offset}",
                         arithmetic_compare("gt", scalar_field("age"), scalar_literal(low, ("add", offset))),
                         atom("gt", "age", low + offset), offset,
                         "age is non-null numeric and arithmetic does not overflow"))
        for scale in scales:
            prefix = f"r{r}_s{scale}"
            out.extend([
                _rule(method, "boolean_constant_folding", prefix + "_and_true", f,
                      AND(f, *([true_domain] * scale)), scale, "fixed finite database state", "database-state-dependent"),
                _rule(method, "boolean_constant_folding", prefix + "_or_false", f,
                      OR(f, *([false_domain] * scale)), scale, "fixed finite database state", "database-state-dependent"),
            ])
    return out


def _renderer(backend: str):
    if backend == "qdrant":
        return render_qdrant
    dialect = "sql" if backend in SQL_BACKENDS else "milvus"
    return lambda value: render_text(value, dialect)


def emit_comparative_pairs(backend: str, method: str, strengths: list[int],
                           age_universe: int = 1000,
                           relation_name: str | None = None,
                           generation_round: int = 0) -> list[EmittedPair]:
    renderer = _renderer(backend)
    pairs: list[EmittedPair] = []
    for item in build_comparative_instances(method, strengths, age_universe, generation_round):
        if item.name in {"constant_expression_reduction",
                          "arithmetic_constant_folding",
                          "nested_expression_folding"} and backend not in ARITHMETIC_BACKENDS:
            continue
        signature = _features(item.equivalent)
        signature.update({"transformation_depth": 1, "strength": item.strength})
        pairs.append(EmittedPair(
            item.name, renderer(item.original), renderer(item.equivalent),
            item.strength, item.selectivity_label, item.rule_id,
            item.case_name, item.layer, item.preconditions, item.equivalence,
            signature,
        ))
    if method.lower() == "coddtest" and backend in SQL_BACKENDS and generation_round == 0:
        # SQL-only database-state constant propagation in CODDTest's style.
        # The generated dataset guarantees category5's observed maximum is 5.
        if not relation_name or not relation_name.replace("_", "").isalnum():
            raise ValueError("a safe SQL relation_name is required for CODDTest propagation")
        pairs.append(EmittedPair(
            "scalar_subquery_constant_propagation",
            f"age > (SELECT MAX(category5) FROM {relation_name})", "age > 5", 1,
            "comparative_baseline", "CODDTEST_SCALAR_SUBQUERY",
            "coddtest_scalar_subquery_to_constant", "filter",
            "the fixed database state has MAX(category5) = 5",
            "database-state-dependent",
            {"ast_nodes": 2, "filter_depth": 1, "predicate_count": 1,
             "and_count": 0, "or_count": 0, "not_count": 0,
             "in_size": 0, "range_count": 1, "arithmetic_count": 0,
             "transformation_depth": 1, "strength": 1},
        ))
    return pairs


def support_matrix() -> list[dict[str, Any]]:
    backends = ("milvus", "pgvector", "qdrant", "weaviate", "seekdb")
    rows: list[dict[str, Any]] = []

    def add(method: str, category: str, support: set[str], note: str) -> None:
        for backend in backends:
            rows.append({"method": method, "category": category,
                         "backend": backend,
                         "supported": backend in support,
                         "note": note if backend in support else "当前 metadata schema 或 Filter DSL 不支持"})

    all_backends = set(backends)
    arithmetic = set(ARITHMETIC_BACKENDS)
    sql = set(SQL_BACKENDS)
    add("AMOEBA", "Constant Expression Reduction", arithmetic,
        "字段与常量算术可在 Filter 中表达")
    add("AMOEBA", "Filter Merging", all_backends,
        "关系 Filter 合并适配为嵌套 AND 与扁平 AND")
    add("AMOEBA", "Date Range Rewriting", set(),
        "当前数据集没有日期字段，未强行迁移")
    add("AMOEBA", "NULL-safe Comparison Rewriting", set(),
        "当前 metadata 均非 NULL，且各 DSL 的 NULL 语义不统一")
    for category in ("AND-FALSE", "OR-TRUE", "AND-TRUE", "OR-FALSE",
                     "Absorption OR", "Absorption AND"):
        add("SCor", category, all_backends,
            "用声明的有限 metadata domain 构造恒真/恒假分支")
    add("CODDTest", "Arithmetic Constant Folding", arithmetic,
        "字段与常量算术可在 Filter 中表达")
    add("CODDTest", "Nested Expression Folding", arithmetic,
        "字段与嵌套常量算术可在 Filter 中表达")
    add("CODDTest", "Boolean Constant Folding", all_backends,
        "适配为固定数据库状态下的已知恒真 metadata 谓词")
    add("CODDTest", "Constant Propagation", sql,
        "SQL 标量子查询预求值；非 SQL Filter DSL 不支持子查询")
    return rows
