from __future__ import annotations

"""Rule-driven equivalent-query generator used by the result_1001 campaign.

The generator works on a small backend-neutral filter AST.  Rendering is a
separate step, so a transformation is defined once and emitted as Qdrant JSON,
Milvus expressions, or SQL predicates without string-level mutation.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ScalarExpr:
    """A field/literal followed by an ordered arithmetic operation chain."""

    value: Any
    is_field: bool = False
    operations: tuple[tuple[str, int | float], ...] = ()


def scalar_field(name: str, *operations: tuple[str, int | float]) -> ScalarExpr:
    return ScalarExpr(name, True, tuple(operations))


def scalar_literal(value: int | float, *operations: tuple[str, int | float]) -> ScalarExpr:
    return ScalarExpr(value, False, tuple(operations))


def arithmetic_compare(comparator: str, left: ScalarExpr, right: ScalarExpr) -> Expr:
    if comparator not in {"lt", "lte", "gt", "gte", "eq"}:
        raise ValueError(f"unsupported arithmetic comparator: {comparator}")
    return Expr("arith_cmp", (comparator, left, right))


def atom(op: str, field: str, *values: Any) -> Expr:
    return Expr(op, (field, *values))


def AND(*xs: Expr) -> Expr:
    return Expr("and", tuple(xs))


def OR(*xs: Expr) -> Expr:
    return Expr("or", tuple(xs))


def NOT(x: Expr) -> Expr:
    return Expr("not", (x,))


@dataclass(frozen=True)
class RuleInstance:
    rule_id: str
    layer: str
    name: str
    case_name: str
    original: Expr
    equivalent: Expr
    strength: int
    selectivity_label: str
    preconditions: str
    equivalence: str


@dataclass(frozen=True)
class EmittedPair:
    pattern: str
    base: Any
    variant: Any
    scale: int
    selectivity_label: str
    pattern_id: str
    case_name: str
    layer: str
    preconditions: str
    equivalence: str
    signature: dict[str, int]


def _repeat_tree(op: str, expression: Expr, count: int) -> Expr:
    return Expr(op, tuple(expression for _ in range(max(1, count))))


def build_arithmetic_filter_instances(strengths: list[int]) -> list[RuleInstance]:
    """Build F12 arithmetic rewrites over the non-null integer ``age`` field.

    The configured strengths are powers of two in the continuous campaign.
    That makes the division thresholds exactly representable and avoids an
    integer-division or floating-rounding counterexample at the boundary.
    """
    original = atom("gt", "age", 10)
    instances: list[RuleInstance] = []
    preconditions = (
        "age is a non-null numeric value; arithmetic does not overflow; "
        "division uses a positive floating-point divisor"
    )
    requested_strengths = sorted({max(1, int(value)) for value in strengths})
    if 1 in requested_strengths:
        instances.append(RuleInstance(
            "F12", "filter", "arithmetic_predicate_equivalence",
            "additive_identity_single", original,
            arithmetic_compare(
                "gt", scalar_field("age", ("add", 0)), scalar_literal(10)
            ),
            1, "age_gt_10", preconditions, "domain-dependent",
        ))
    for strength in (value for value in requested_strengths if value >= 2):
        offset = strength
        factor = strength
        divisor = float(strength)
        cases = (
            ("add_translation", scalar_field("age", ("add", offset)),
             scalar_literal(10 + offset), "gt"),
            ("subtract_translation", scalar_field("age", ("sub", offset)),
             scalar_literal(10 - offset), "gt"),
            ("positive_multiplication", scalar_field("age", ("mul", factor)),
             scalar_literal(10 * factor), "gt"),
            ("negative_multiplication", scalar_field("age", ("mul", -factor)),
             scalar_literal(-10 * factor), "lt"),
            ("positive_division", scalar_field("age", ("div", divisor)),
             scalar_literal(10 / divisor), "gt"),
            ("bilateral_addition", scalar_field("age", ("add", offset)),
             scalar_literal(10, ("add", offset)), "gt"),
            ("additive_identity", scalar_field("age", *([("add", 0)] * strength)),
             scalar_literal(10), "gt"),
            ("multiplicative_identity", scalar_field("age", *([("mul", 1)] * strength)),
             scalar_literal(10), "gt"),
            ("arithmetic_cancellation",
             scalar_field("age", *([("add", offset), ("sub", offset)] * strength)),
             scalar_literal(10), "gt"),
        )
        for case_name, left, right, comparator in cases:
            instances.append(RuleInstance(
                "F12", "filter", "arithmetic_predicate_equivalence",
                f"{case_name}_{strength}", original,
                arithmetic_compare(comparator, left, right), strength,
                "age_gt_10", preconditions, "domain-dependent",
            ))
    return instances


def build_filter_instances(strengths: list[int], age_universe: int = 1000) -> list[RuleInstance]:
    """Instantiate F1-F12 across controlled expansion strengths."""
    requested_strengths = sorted(set(max(1, int(x)) for x in strengths))
    strengths = sorted(set(max(2, int(x)) for x in strengths))
    category_1 = atom("eq", "category", 1)
    age_1_100 = atom("between", "age", 1, 100)
    true_expr = atom("between", "age", 1, age_universe)
    false_expr = atom("eq", "category", -1)
    a, b, c = atom("eq", "parity", "even"), atom("eq", "category5", 1), atom("eq", "category5", 2)
    out = [
        RuleInstance("F1", "filter", "operator_equivalence", "between_vs_bounds",
                     age_1_100, AND(atom("gte", "age", 1), atom("lte", "age", 100)), 2,
                     "age_1_100", "age is a non-null integer", "guaranteed"),
        RuleInstance("F4", "filter", "negation_equivalence", "de_morgan_not_or",
                     NOT(OR(atom("eq", "category5", 4), atom("eq", "category5", 5))),
                     AND(NOT(atom("eq", "category5", 4)), NOT(atom("eq", "category5", 5))), 2,
                     "category5_not_4_5", "category5 is non-null", "guaranteed"),
        RuleInstance("F6", "filter", "distribution", "and_over_or",
                     AND(a, OR(b, c)), OR(AND(a, b), AND(a, c)), 2,
                     "even_and_category5_1_2", "standard two-valued Boolean logic; fields are non-null", "guaranteed"),
        RuleInstance("F7", "filter", "structural_reordering", "commutative_associative_and",
                     AND(AND(a, b), atom("lte", "age", 500)),
                     AND(atom("lte", "age", 500), AND(b, a)), 3,
                     "reordered_conjunction", "standard two-valued Boolean logic; fields are non-null", "guaranteed"),
        RuleInstance("F9", "filter", "predicate_decomposition", "range_to_disjoint_segments",
                     age_1_100,
                     OR(atom("between", "age", 1, 25), atom("between", "age", 26, 50),
                        atom("between", "age", 51, 75), atom("between", "age", 76, 100)), 4,
                     "age_1_100", "age is a discrete, non-null integer", "domain-dependent"),
        RuleInstance("F10", "filter", "complement_equivalence", "finite_universe_complement",
                     atom("in", "category5", 1, 2, 3), NOT(atom("in", "category5", 4, 5)), 2,
                     "category5_1_3", "category5 is non-null and its universe is exactly {1,2,3,4,5}", "domain-dependent"),
    ]
    for n in strengths:
        out.extend([
            RuleInstance("F2", "filter", "boolean_idempotence", f"and_repeat_{n}",
                         category_1, _repeat_tree("and", category_1, n), n, "category_1",
                         "standard two-valued Boolean logic", "guaranteed"),
            RuleInstance("F3", "filter", "boolean_identity", f"and_true_repeat_{n}",
                         category_1, AND(category_1, *([true_expr] * n)), n, "category_1",
                         f"age is non-null and constrained to [1,{age_universe}]", "domain-dependent"),
            RuleInstance("F5", "filter", "absorption", f"or_absorption_{n}",
                         category_1, OR(category_1, *([AND(category_1, atom("lte", "age", 100))] * n)), n,
                         "category_1", "standard two-valued Boolean logic", "guaranteed"),
            RuleInstance("F8", "filter", "implication_redundancy", f"dominated_bounds_{n}",
                         atom("lt", "age", 10),
                         AND(atom("lt", "age", 10), *[atom("lt", "age", 20 + i * 10) for i in range(n)]),
                         n, "age_lt_10", "age < 10 implies every appended weaker upper bound", "domain-dependent"),
            RuleInstance("F11", "filter", "set_idempotence", f"duplicate_in_{n}",
                         atom("in", "category5", 1, 2, 3), atom("in", "category5", *([1, 2, 3] * n)), n,
                         "category5_1_3", "membership uses set semantics", "guaranteed"),
        ])
    out.extend(build_arithmetic_filter_instances(requested_strengths))
    return out


def _features(expr: Expr) -> dict[str, int]:
    result = {"ast_nodes": 1, "filter_depth": 1, "predicate_count": 0,
              "and_count": 0, "or_count": 0, "not_count": 0, "in_size": 0,
              "range_count": 0, "arithmetic_count": 0}
    if expr.op in {"and", "or", "not"}:
        result[f"{expr.op}_count"] += 1
        children = [_features(x) for x in expr.args]
        result["filter_depth"] = 1 + max((x["filter_depth"] for x in children), default=0)
        for child in children:
            for key in result:
                if key != "filter_depth":
                    result[key] += child[key]
    else:
        result["predicate_count"] = 1
        if expr.op == "arith_cmp":
            _, left, right = expr.args
            result["arithmetic_count"] = len(left.operations) + len(right.operations)
            result["range_count"] = 1
        if expr.op == "in":
            result["in_size"] = len(expr.args) - 1
        if expr.op in {"between", "lt", "lte", "gt", "gte"}:
            result["range_count"] = 1
    return result


def evaluate(expr: Expr, row: dict[str, Any]) -> bool:
    """Reference interpreter used before any backend query is executed."""
    op, args = expr.op, expr.args
    if op == "and": return all(evaluate(x, row) for x in args)
    if op == "or": return any(evaluate(x, row) for x in args)
    if op == "not": return not evaluate(args[0], row)
    if op == "arith_cmp":
        comparator, left_expr, right_expr = args
        left = _evaluate_scalar(left_expr, row)
        right = _evaluate_scalar(right_expr, row)
        if comparator == "eq": return left == right
        if comparator == "lt": return left < right
        if comparator == "lte": return left <= right
        if comparator == "gt": return left > right
        if comparator == "gte": return left >= right
        raise ValueError(comparator)
    value, operands = row[args[0]], args[1:]
    if op == "eq": return value == operands[0]
    if op == "in": return value in operands
    if op == "between": return operands[0] <= value <= operands[1]
    if op == "lt": return value < operands[0]
    if op == "lte": return value <= operands[0]
    if op == "gt": return value > operands[0]
    if op == "gte": return value >= operands[0]
    raise ValueError(op)


def _evaluate_scalar(expr: ScalarExpr, row: dict[str, Any]) -> int | float:
    value = row[expr.value] if expr.is_field else expr.value
    for operator, operand in expr.operations:
        if operator == "add": value = value + operand
        elif operator == "sub": value = value - operand
        elif operator == "mul": value = value * operand
        elif operator == "div": value = value / operand
        else: raise ValueError(operator)
    return value


def _qdrant_condition(expr: Expr) -> dict:
    op, args = expr.op, expr.args
    if op == "and": return {"must": [_qdrant_condition(x) for x in args]}
    if op == "or": return {"should": [_qdrant_condition(x) for x in args]}
    if op == "not": return {"must_not": [_qdrant_condition(args[0])]}
    if op == "arith_cmp":
        raise ValueError("Qdrant filters do not support field arithmetic (F12)")
    field, values = args[0], args[1:]
    if op == "eq": return {"key": field, "match": {"value": values[0]}}
    if op == "in": return {"key": field, "match": {"any": list(values)}}
    if op == "between": return {"key": field, "range": {"gte": values[0], "lte": values[1]}}
    return {"key": field, "range": {op: values[0]}}


def render_qdrant(expr: Expr) -> dict:
    rendered = _qdrant_condition(expr)
    return rendered if expr.op in {"and", "or", "not"} else {"must": [rendered]}


def render_text(expr: Expr, dialect: str) -> str:
    op, args = expr.op, expr.args
    if op in {"and", "or"}:
        token = f" {op.upper() if dialect == 'sql' else op} "
        return token.join(f"({render_text(x, dialect)})" for x in args)
    if op == "not": return f"{'NOT' if dialect == 'sql' else 'not'} ({render_text(args[0], dialect)})"
    if op == "arith_cmp":
        comparator, left, right = args
        token = {"eq": "=" if dialect == "sql" else "==", "lt": "<", "lte": "<=",
                 "gt": ">", "gte": ">="}[comparator]
        return f"{_render_scalar(left)} {token} {_render_scalar(right)}"
    field, values = args[0], args[1:]
    eq = "=" if dialect == "sql" else "=="
    if op == "eq":
        value = f"'{values[0]}'" if isinstance(values[0], str) else str(values[0])
        return f"{field} {eq} {value}"
    if op == "in":
        left, right = ("(", ")") if dialect == "sql" else ("[", "]")
        rendered_values = [f"'{value}'" if isinstance(value, str) else str(value) for value in values]
        return f"{field} {'IN' if dialect == 'sql' else 'in'} {left}{', '.join(rendered_values)}{right}"
    if op == "between":
        if dialect == "sql": return f"{field} BETWEEN {values[0]} AND {values[1]}"
        return f"({field} >= {values[0]}) and ({field} <= {values[1]})"
    token = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}[op]
    return f"{field} {token} {values[0]}"


def _render_scalar(expr: ScalarExpr) -> str:
    rendered = str(expr.value)
    tokens = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
    for operator, operand in expr.operations:
        rendered = f"({rendered} {tokens[operator]} {operand})"
    return rendered


def emit_filter_pairs(backend: str, strengths: list[int], age_universe: int = 1000) -> list[EmittedPair]:
    renderer = render_qdrant if backend == "qdrant" else lambda x: render_text(x, "sql" if backend in {"pgvector", "seekdb"} else "milvus")
    pairs = []
    for item in build_filter_instances(strengths, age_universe):
        signature = _features(item.equivalent)
        signature.update({"transformation_depth": 1, "strength": item.strength})
        pairs.append(EmittedPair(item.name, renderer(item.original), renderer(item.equivalent), item.strength,
                                 item.selectivity_label, item.rule_id, item.case_name, item.layer,
                                 item.preconditions, item.equivalence, signature))
    return pairs


ANN_RULES = [
    {"rule_id": "A1", "layer": "ann", "name": "isometric_joint_transformation", "case": "dimension_permutation",
     "preconditions": "the identical dimension permutation is applied to every data vector and query vector", "oracle": "exact ranking equality"},
    {"rule_id": "A2", "layer": "ann", "name": "order_preserving_joint_transformation", "case": "positive_global_scaling",
     "preconditions": "L2 metric and a shared positive scale factor", "oracle": "exact ranking equality"},
    {"rule_id": "A3", "layer": "ann", "name": "equivalent_metric_transformation", "case": "cosine_vs_inner_product_vs_l2",
     "preconditions": "all data vectors and query vectors have unit L2 norm", "oracle": "exact ranking equality"},
    {"rule_id": "A4", "layer": "ann", "name": "monotonic_distance_transformation", "case": "l2_vs_squared_l2",
     "preconditions": "distances are non-negative and squaring is strictly increasing", "oracle": "exact ranking equality"},
    {"rule_id": "FA1", "layer": "fanns", "name": "full_filter_degeneration", "case": "no_filter_vs_true_filter",
     "preconditions": "TRUE filter selects every row", "oracle": "exact top-k equality"},
    {"rule_id": "FA2", "layer": "fanns", "name": "prefilter_search_space_equivalence", "case": "filtered_full_vs_materialized_subset",
     "preconditions": "materialized subset contains exactly the rows selected by F", "oracle": "eligible-set and exact top-k equality"},
]


def rule_catalog(strengths: list[int]) -> list[dict[str, Any]]:
    filters = [{"rule_id": x.rule_id, "layer": x.layer, "name": x.name, "case": x.case_name,
                "strength": x.strength, "preconditions": x.preconditions, "equivalence": x.equivalence,
                "signature": _features(x.equivalent)} for x in build_filter_instances(strengths)]
    return filters + ANN_RULES
