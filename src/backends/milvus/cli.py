from __future__ import annotations

import argparse
import os
from pathlib import Path

from .client import MilvusBackend
from .analyze import analyze
from .experiment import load_config, report, run, setup


def main() -> None:
    parser = argparse.ArgumentParser(description="Milvus equivalent-filter anomaly experiments")
    parser.add_argument("command", choices=["setup", "run", "all", "report", "analyze"])
    parser.add_argument("--config", default="configs/milvus_smoke.yaml")
    parser.add_argument("--output", default="results/milvus_latest.jsonl")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(args.output)
    if args.command == "report":
        report(output, output.with_suffix(""))
        return
    if args.command == "analyze":
        analyze(output, output.with_name(output.stem + "_analysis"))
        return
    backend = MilvusBackend(os.environ.get("MILVUS_URI", "http://localhost:19530"))
    if args.command in ("setup", "all"):
        setup(backend, cfg)
    if args.command in ("run", "all"):
        run(backend, cfg, output)
        report(output, output.with_suffix(""))


if __name__ == "__main__":
    main()
