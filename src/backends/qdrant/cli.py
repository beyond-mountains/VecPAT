from __future__ import annotations

import argparse
import os
from pathlib import Path

from .experiment import load_config, report, run, setup
from .client import Qdrant


def main() -> None:
    parser = argparse.ArgumentParser(description="Equivalent-filter anomaly experiments for Qdrant")
    parser.add_argument("command", choices=["setup", "run", "all", "report"])
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--output", default="results/latest.jsonl")
    args = parser.parse_args()
    cfg = load_config(args.config)
    q = Qdrant(os.environ.get("QDRANT_URL", "http://localhost:6333"))
    output = Path(args.output)
    if args.command in ("setup", "all"):
        setup(q, cfg)
    if args.command in ("run", "all"):
        run(q, cfg, output)
        report(output, output.with_suffix(""))
    elif args.command == "report":
        report(output, output.with_suffix(""))


if __name__ == "__main__":
    main()

