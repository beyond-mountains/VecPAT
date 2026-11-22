from __future__ import annotations

import argparse
import os
from pathlib import Path

from .client import SeekDBBackend
from ..sql.experiment import load_config, report, run, setup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["setup", "run", "all", "report"])
    parser.add_argument("--config", default="configs/seekdb_0924.yaml")
    parser.add_argument("--output", default="result_seekdb/seekdb_discovery.jsonl")
    args = parser.parse_args()
    cfg = load_config(args.config); output = Path(args.output)
    if args.command == "report":
        report(output, output.with_suffix(""), "seekdb"); return
    backend = SeekDBBackend(os.getenv("SEEKDB_HOST", "seekdb"), int(os.getenv("SEEKDB_PORT", "2881")),
                            os.getenv("SEEKDB_USER", "root"), os.environ["SEEKDB_PASSWORD"],
                            os.getenv("SEEKDB_DATABASE", "filter_anomaly"), cfg["experiment"]["table"])
    if args.command in ("setup", "all"): setup(backend, cfg)
    if args.command in ("run", "all"):
        run(backend, cfg, output); report(output, output.with_suffix(""), "seekdb")


if __name__ == "__main__":
    main()
