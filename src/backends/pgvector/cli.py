from __future__ import annotations

import argparse
import os
from pathlib import Path

from .client import PgvectorBackend
from ..sql.experiment import load_config, report, run, setup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["setup", "run", "all", "report"])
    parser.add_argument("--config", default="configs/pgvector_0924.yaml")
    parser.add_argument("--output", default="result_pgvector/pgvector_discovery.jsonl")
    args = parser.parse_args()
    cfg = load_config(args.config); output = Path(args.output)
    if args.command == "report":
        report(output, output.with_suffix(""), "pgvector"); return
    backend = PgvectorBackend(os.environ["PGVECTOR_DSN"], cfg["experiment"]["table"])
    if args.command in ("setup", "all"): setup(backend, cfg)
    if args.command in ("run", "all"):
        run(backend, cfg, output); report(output, output.with_suffix(""), "pgvector")


if __name__ == "__main__":
    main()
