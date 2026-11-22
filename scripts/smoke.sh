#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RESULT_DIR="$(realpath -m "${1:-results/smoke}")"
export RESULT_DIR
mkdir -p "$RESULT_DIR"

cleanup() {
  docker compose -f docker-compose.yml down >/dev/null 2>&1 || true
}
trap cleanup EXIT

python3 -m src.campaigns.continuous \
  --ledger "$RESULT_DIR/coverage.sqlite" --pattern F1 \
  init --strengths 2 --backend qdrant > "$RESULT_DIR/init.json"
python3 -m src.campaigns.continuous \
  --ledger "$RESULT_DIR/coverage.sqlite" --pattern F1 \
  validate --strengths 2 --output "$RESULT_DIR/semantic_validation.jsonl" \
  > "$RESULT_DIR/semantic_validation_summary.json"

docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.yml up -d --wait qdrant
docker compose -f docker-compose.yml build runner
docker compose -f docker-compose.yml run --rm runner all \
  --config /app/configs/qdrant_smoke.yaml \
  --output /app/result/qdrant_smoke.jsonl

test -s "$RESULT_DIR/qdrant_smoke.jsonl"
grep -q '"semantic_equal": true' "$RESULT_DIR/qdrant_smoke.jsonl"
echo "VecPAT smoke test passed: $RESULT_DIR"

