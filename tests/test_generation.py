from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from src.core.confirmation import balanced_random_orders
from src.core.coverage import CoverageLedger
from src.generation.continuous import (
    build_continuous_filter_instances,
    emit_filter_pairs_continuous,
)
from src.generation.equivalent import evaluate


class GenerationTests(unittest.TestCase):
    def test_f4_double_not_replay_uses_generator_strength(self) -> None:
        instances = build_continuous_filter_instances([2])
        case = next(item for item in instances if item.case_name == "double_not_depth_4")
        self.assertEqual(case.strength, 4)

        emitted = emit_filter_pairs_continuous("qdrant", [2])
        names = {item.case_name for item in emitted}
        self.assertIn("double_not_depth_4", names)
        self.assertNotIn("double_not_depth_8", names)

    def test_f4_double_not_is_semantically_equal(self) -> None:
        case = next(
            item
            for item in build_continuous_filter_instances([2])
            if item.case_name == "double_not_depth_4"
        )
        for age in range(1, 1001):
            row = {
                "age": age,
                "category": age % 100 + 1,
                "category5": age % 5 + 1,
                "tenant": age % 100,
                "level": age % 32,
                "parity": "even" if age % 2 == 0 else "odd",
            }
            self.assertEqual(evaluate(case.original, row), evaluate(case.equivalent, row))

    def test_balanced_order_has_equal_first_positions(self) -> None:
        orders = balanced_random_orders(10, 20261011, "test")
        self.assertEqual(sum(order[0] == "base" for order in orders), 5)
        self.assertEqual(sum(order[0] == "variant" for order in orders), 5)

    def test_recall_gap_does_not_create_anomaly_candidate(self) -> None:
        config = {
            "backend": "qdrant",
            "layer": "filter",
            "rule_id": "F1",
            "case_name": "threshold_test",
            "transformation": {"strength": 2},
            "semantic": {"points": 100, "correlation": 0.0},
            "execution": {"dimensions": 8, "top_k": 10, "ef_search": 16},
        }
        with TemporaryDirectory() as directory:
            ledger = CoverageLedger(Path(directory) / "coverage.sqlite")
            identifier = ledger.add(config, 0)
            ledger.commit()
            ledger.finish(
                identifier,
                "test-run",
                "discovery",
                "result.jsonl",
                "executed",
                latency_ratio=1.1,
                recall_gap=1.0,
            )
            row = ledger.db.execute(
                "SELECT state, anomaly, recall_gap FROM configurations WHERE case_id=?",
                (identifier,),
            ).fetchone()
            self.assertEqual(row["state"], "complete")
            self.assertEqual(row["anomaly"], 0)
            self.assertEqual(row["recall_gap"], 1.0)
            ledger.db.close()


if __name__ == "__main__":
    unittest.main()

