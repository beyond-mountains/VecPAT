from __future__ import annotations

import argparse
import os
from pathlib import Path

from .client import WeaviateBackend
from .experiment import load_config, report, run, setup


def main() -> None:
    parser = argparse.ArgumentParser(description="Equivalent-filter anomaly experiments for Weaviate")
    parser.add_argument("command", choices=["setup", "run", "confirm", "all", "report"])
    parser.add_argument("--config", default="configs/weaviate_1002.yaml")
    parser.add_argument("--output-dir", default="result_1002_confirm/weaviate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = WeaviateBackend(os.getenv("WEAVIATE_URL", "http://weaviate:8080"))
    backend.wait_until_ready()
    discovery = output_dir / "discovery.jsonl"
    confirmation = output_dir / "confirmation.jsonl"

    if args.command in {"setup", "all"}:
        setup(backend, cfg)
    if args.command in {"run", "all"}:
        run(backend, cfg, discovery, mode="discovery")
        report(discovery, output_dir / "discovery")
    if args.command in {"confirm", "all"}:
        run(
            backend,
            cfg,
            confirmation,
            mode="confirmation",
            candidate_source=discovery,
        )
        report(confirmation, output_dir / "confirmation")
    if args.command == "report":
        report(discovery, output_dir / "discovery")
        report(confirmation, output_dir / "confirmation")


if __name__ == "__main__":
    main()
