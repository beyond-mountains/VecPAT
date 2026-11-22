#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "usage: $0 XHOURS_PER_BACKEND|until-exhausted [RESULT_DIR] [LEDGER] [--pattern RULE] [--backend BACKEND] [--strengths LIST]" >&2
  echo "       $0 until-exhausted --pattern F12 --backend milvus --strengths 2 [--result-dir DIR] [--ledger FILE]" >&2
  [[ $# -ge 1 ]] && exit 0
  exit 2
fi

XHOURS="$1"
shift
UNTIL_EXHAUSTED=false
if [[ "$XHOURS" == "until-exhausted" ]]; then
  UNTIL_EXHAUSTED=true
fi
RESULT_DIR=""
LEDGER=""
PATTERN=""
BACKEND_ONLY=""
STRENGTHS="1,2,8,32,128,512,1024"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pattern)
      [[ $# -ge 2 ]] || { echo "--pattern requires a rule ID" >&2; exit 2; }
      PATTERN="${2^^}"
      shift 2
      ;;
    --result-dir)
      [[ $# -ge 2 ]] || { echo "--result-dir requires a directory" >&2; exit 2; }
      RESULT_DIR="$2"
      shift 2
      ;;
    --ledger)
      [[ $# -ge 2 ]] || { echo "--ledger requires a file" >&2; exit 2; }
      LEDGER="$2"
      shift 2
      ;;
    --backend)
      [[ $# -ge 2 ]] || { echo "--backend requires a backend name" >&2; exit 2; }
      BACKEND_ONLY="${2,,}"
      shift 2
      ;;
    --strengths)
      [[ $# -ge 2 ]] || { echo "--strengths requires a comma-separated integer list" >&2; exit 2; }
      STRENGTHS="$2"
      shift 2
      ;;
    -h|--help)
      echo "usage: $0 XHOURS_PER_BACKEND|until-exhausted [RESULT_DIR] [LEDGER] [--pattern RULE] [--backend BACKEND] [--strengths LIST]"
      echo "       $0 until-exhausted --pattern F12 --backend milvus --strengths 2 [--result-dir DIR] [--ledger FILE]"
      exit 0
      ;;
    --*)
      echo "unknown option: $1" >&2
      exit 2
      ;;
    *)
      if [[ -z "$RESULT_DIR" ]]; then
        RESULT_DIR="$1"
      elif [[ -z "$LEDGER" ]]; then
        LEDGER="$1"
      else
        echo "unexpected argument: $1" >&2
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ -n "$PATTERN" && ! "$PATTERN" =~ ^[A-Z][A-Z0-9]*$ ]]; then
  echo "invalid pattern ID: $PATTERN" >&2
  exit 2
fi
if [[ -n "$BACKEND_ONLY" && ! " milvus pgvector qdrant seekdb weaviate " =~ " $BACKEND_ONLY " ]]; then
  echo "invalid backend: $BACKEND_ONLY" >&2
  exit 2
fi
if [[ ! "$STRENGTHS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
  echo "invalid --strengths value: $STRENGTHS (expected comma-separated positive integers)" >&2
  exit 2
fi
if [[ "$UNTIL_EXHAUSTED" == false && ! "$XHOURS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "invalid time budget: $XHOURS (expected a positive number or until-exhausted)" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d%H)"
BUDGET_TAG="${XHOURS}hours"
[[ "$UNTIL_EXHAUSTED" == true ]] && BUDGET_TAG="until_exhausted"
RESULT_DIR="${RESULT_DIR:-result_${STAMP}_${BUDGET_TAG}${PATTERN:+_${PATTERN}}}"
if [[ -n "$PATTERN" && "$(basename "$RESULT_DIR")" != *"$PATTERN"* ]]; then
  RESULT_DIR="${RESULT_DIR}_${PATTERN}"
fi
# Compose treats a bare value such as result_YYYYMMDDHH_1hours_F4 as a named
# volume.  Export an absolute host path so every runner receives a bind mount.
RESULT_DIR="$(realpath -m "$RESULT_DIR")"
LEDGER="${LEDGER:-continuous_state/coverage${PATTERN:+_${PATTERN}}.sqlite}"
RUN_ID="${STAMP}_$(date +%M%S)_${BUDGET_TAG}${PATTERN:+_${PATTERN}}"
PATTERN_ARGS=()
[[ -n "$PATTERN" ]] && PATTERN_ARGS=(--pattern "$PATTERN")
INIT_BACKEND_ARGS=()
[[ -n "$BACKEND_ONLY" ]] && INIT_BACKEND_ARGS=(--backend "$BACKEND_ONLY")
mkdir -p "$RESULT_DIR/control" "$RESULT_DIR/raw" "$RESULT_DIR/explain"
export RESULT_DIR

if [[ "$UNTIL_EXHAUSTED" == true ]]; then
  # The claim loop still receives a deadline for crash-safe metadata, but the
  # practical stopping condition is an empty backend queue.
  PER_BACKEND_SECONDS=$((10 * 365 * 24 * 3600))
else
  PER_BACKEND_SECONDS="$(python3 -c 'import sys; print(max(1, round(float(sys.argv[1])*3600)))' "$XHOURS")"
fi

python3 -m src.campaigns.continuous --ledger "$LEDGER" "${PATTERN_ARGS[@]}" init --strengths "$STRENGTHS" "${INIT_BACKEND_ARGS[@]}" > "$RESULT_DIR/control/init.json"
python3 -m src.campaigns.continuous --ledger "$LEDGER" "${PATTERN_ARGS[@]}" validate --strengths "$STRENGTHS" --output "$RESULT_DIR/semantic_validation.jsonl" \
  > "$RESULT_DIR/control/semantic_validation_summary.json"

stop_all() {
  docker compose -f docker-compose.milvus.yml stop milvus milvus-etcd milvus-minio >/dev/null 2>&1 || true
  docker compose -f docker-compose.pgvector.yml stop pgvector >/dev/null 2>&1 || true
  docker compose -f docker-compose.yml stop qdrant >/dev/null 2>&1 || true
  docker compose -f docker-compose.seekdb.yml stop seekdb >/dev/null 2>&1 || true
  docker compose -f docker-compose.weaviate.yml stop weaviate >/dev/null 2>&1 || true
}

start_backend() {
  case "$1" in
    milvus) docker compose -f docker-compose.milvus.yml up -d milvus && docker compose -f docker-compose.milvus.yml build milvus-runner ;;
    pgvector) docker compose -f docker-compose.pgvector.yml up -d pgvector && docker compose -f docker-compose.pgvector.yml build pgvector-runner ;;
    qdrant) docker compose -f docker-compose.yml up -d qdrant && docker compose -f docker-compose.yml build runner ;;
    seekdb)
      docker compose -f docker-compose.seekdb.yml up -d seekdb
      seekdb_deadline=$(( $(date +%s) + 600 ))
      until docker compose -f docker-compose.seekdb.yml exec -T seekdb \
        mysql -h127.0.0.1 -P2881 -uroot -panomaly-local-only -N -e "SELECT 1" >/dev/null 2>&1; do
        if [[ "$(date +%s)" -ge "$seekdb_deadline" ]]; then
          echo "seekDB readiness check timed out after 600 seconds" >&2
          return 1
        fi
        sleep 5
      done
      docker compose -f docker-compose.seekdb.yml build seekdb-runner
      ;;
    weaviate) docker compose -f docker-compose.weaviate.yml up -d weaviate && docker compose -f docker-compose.weaviate.yml --profile tools build weaviate-runner ;;
  esac
}

base_config() {
  case "$1" in
    milvus) echo configs/milvus.yaml ;;
    pgvector) echo configs/pgvector.yaml ;;
    qdrant) echo configs/qdrant.yaml ;;
    seekdb) echo configs/seekdb.yaml ;;
    weaviate) echo configs/weaviate.yaml ;;
  esac
}

run_filter() {
  local backend="$1" mode="$2" host_result="$3" needs_setup="$4"
  local container_result="/app/result/${host_result#"$RESULT_DIR"/}"
  local action="run"
  [[ "$needs_setup" == "yes" ]] && action="all"
  case "$backend" in
    milvus) docker compose -f docker-compose.milvus.yml run --rm milvus-runner "$action" --config /app/result/control/active.yaml --output "$container_result" ;;
    pgvector) docker compose -f docker-compose.pgvector.yml run --rm pgvector-runner "$action" --config /app/result/control/active.yaml --output "$container_result" ;;
    qdrant) docker compose -f docker-compose.yml run --rm runner "$action" --config /app/result/control/active.yaml --output "$container_result" ;;
    seekdb) docker compose -f docker-compose.seekdb.yml run --rm seekdb-runner "$action" --config /app/result/control/active.yaml --output "$container_result" ;;
    weaviate)
      local case_dir="/app/result/raw/weaviate/${CASE_ID}_${mode}"
      if [[ "$mode" == "confirmation" ]]; then
        [[ "$needs_setup" == "yes" ]] && docker compose -f docker-compose.weaviate.yml --profile tools run --rm weaviate-runner setup --config /app/result/control/active.yaml --output-dir "$case_dir"
        mkdir -p "$RESULT_DIR/raw/weaviate/${CASE_ID}_${mode}"
        cp "$RESULT_DIR/raw/weaviate/${CASE_ID}_discovery/discovery.jsonl" "$RESULT_DIR/raw/weaviate/${CASE_ID}_${mode}/discovery.jsonl"
        docker compose -f docker-compose.weaviate.yml --profile tools run --rm weaviate-runner confirm --config /app/result/control/active.yaml --output-dir "$case_dir"
      else
        [[ "$needs_setup" == "yes" ]] && docker compose -f docker-compose.weaviate.yml --profile tools run --rm weaviate-runner setup --config /app/result/control/active.yaml --output-dir "$case_dir"
        docker compose -f docker-compose.weaviate.yml --profile tools run --rm weaviate-runner run --config /app/result/control/active.yaml --output-dir "$case_dir"
      fi
      ;;
  esac
}

run_ann() {
  local backend="$1" host_result="$2"
  local container_result="/app/result/${host_result#"$RESULT_DIR"/}"
  case "$backend" in
    milvus) docker compose -f docker-compose.milvus.yml run --rm --entrypoint python milvus-runner -m src.campaigns.ann milvus --config /app/result/control/active.yaml --output "$container_result" ;;
    pgvector) docker compose -f docker-compose.pgvector.yml run --rm --entrypoint python pgvector-runner -m src.campaigns.ann pgvector --config /app/result/control/active.yaml --output "$container_result" ;;
    qdrant) docker compose -f docker-compose.yml run --rm --entrypoint python runner -m src.campaigns.ann qdrant --config /app/result/control/active.yaml --output "$container_result" ;;
    seekdb) docker compose -f docker-compose.seekdb.yml run --rm --entrypoint python seekdb-runner -m src.campaigns.ann seekdb --config /app/result/control/active.yaml --output "$container_result" ;;
    weaviate) docker compose -f docker-compose.weaviate.yml --profile tools run --rm --entrypoint python weaviate-runner -m src.campaigns.ann weaviate --config /app/result/control/active.yaml --output "$container_result" ;;
  esac
}

save_explain() {
  local backend="$1" host_result="$2" case_id="$3"
  local container_result="/app/result/${host_result#"$RESULT_DIR"/}"
  local container_output="/app/result/explain/${case_id}.json"
  case "$backend" in
    milvus) docker compose -f docker-compose.milvus.yml run --rm --entrypoint python milvus-runner -m src.analysis.explain_continuous milvus --config /app/result/control/active.yaml --result "$container_result" --output "$container_output" ;;
    pgvector) docker compose -f docker-compose.pgvector.yml run --rm --entrypoint python pgvector-runner -m src.analysis.explain_continuous pgvector --config /app/result/control/active.yaml --result "$container_result" --output "$container_output" ;;
    qdrant) docker compose -f docker-compose.yml run --rm --entrypoint python runner -m src.analysis.explain_continuous qdrant --config /app/result/control/active.yaml --result "$container_result" --output "$container_output" ;;
    seekdb) docker compose -f docker-compose.seekdb.yml run --rm --entrypoint python seekdb-runner -m src.analysis.explain_continuous seekdb --config /app/result/control/active.yaml --result "$container_result" --output "$container_output" ;;
    weaviate) docker compose -f docker-compose.weaviate.yml --profile tools run --rm --entrypoint python weaviate-runner -m src.analysis.explain_continuous weaviate --config /app/result/control/active.yaml --result "$container_result" --output "$container_output" ;;
  esac
}

stop_all
BACKENDS=(milvus pgvector qdrant seekdb weaviate)
[[ -n "$BACKEND_ONLY" ]] && BACKENDS=("$BACKEND_ONLY")
declare -A PREPARED_SIGNATURES=()
for index in "${!BACKENDS[@]}"; do
  backend="${BACKENDS[$index]}"
  stop_all
  start_backend "$backend"
  # xhours is a full, independent query budget for this backend. Container
  # startup and runner image build happen before its timer starts.
  BACKEND_STARTED_AT="$(date +%s)"
  BACKEND_DEADLINE=$((BACKEND_STARTED_AT + PER_BACKEND_SECONDS))
  printf '%s backend=%s started_at=%s deadline=%s budget=%s strengths=%s\n' \
    "$(date --iso-8601=seconds)" "$backend" "$BACKEND_STARTED_AT" "$BACKEND_DEADLINE" "$BUDGET_TAG" "$STRENGTHS" \
    | tee -a "$RESULT_DIR/run.log"
  printf '%s backend=%s pattern=%s ledger=%s result_dir=%s\n' \
    "$(date --iso-8601=seconds)" "$backend" "${PATTERN:-all}" "$LEDGER" "$RESULT_DIR" \
    | tee -a "$RESULT_DIR/run.log"
  mkdir -p "$RESULT_DIR/raw/$backend"
  while [[ "$(date +%s)" -lt "$BACKEND_DEADLINE" ]]; do
    set +e
    python3 -m src.campaigns.continuous --ledger "$LEDGER" "${PATTERN_ARGS[@]}" claim --backend "$backend" --run-id "$RUN_ID" \
      --deadline "$BACKEND_DEADLINE" --base-config "$(base_config "$backend")" --output "$RESULT_DIR/control/active.yaml" \
      > "$RESULT_DIR/control/current.json"
    claim_status=$?
    set -e
    [[ "$claim_status" -eq 3 ]] && break
    [[ "$claim_status" -ne 0 ]] && exit "$claim_status"
    CASE_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["case_id"])' "$RESULT_DIR/control/current.json")"
    mode="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["mode"])' "$RESULT_DIR/control/current.json")"
    layer="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["layer"])' "$RESULT_DIR/control/current.json")"
    signature="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_signature"])' "$RESULT_DIR/control/current.json")"
    setup_signature="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["setup_signature"])' "$RESULT_DIR/control/current.json")"
    prepared_key="${backend}:${setup_signature}"
    result="$RESULT_DIR/raw/$backend/${CASE_ID}_${mode}.jsonl"
    needs_setup=no
    [[ -z "${PREPARED_SIGNATURES[$prepared_key]+x}" ]] && needs_setup=yes
    printf '%s backend=%s case_id=%s dataset_signature=%s setup_signature=%s setup=%s\n' \
      "$(date --iso-8601=seconds)" "$backend" "$CASE_ID" "$signature" "$setup_signature" "$needs_setup" \
      | tee -a "$RESULT_DIR/run.log"
    case_log="$RESULT_DIR/control/${CASE_ID}_${mode}.log"
    set +e
    if [[ "$layer" == "filter" ]]; then
      run_filter "$backend" "$mode" "$result" "$needs_setup" 2>&1 \
        | tee -a "$RESULT_DIR/run.log" "$case_log"
      case_status=${PIPESTATUS[0]}
      [[ "$backend" == "weaviate" ]] && result="$RESULT_DIR/raw/weaviate/${CASE_ID}_${mode}/$([[ "$mode" == confirmation ]] && echo confirmation || echo discovery).jsonl"
    else
      run_ann "$backend" "$result" 2>&1 \
        | tee -a "$RESULT_DIR/run.log" "$case_log"
      case_status=${PIPESTATUS[0]}
    fi
    set -e
    if [[ "$case_status" -ne 0 ]]; then
      error_tail="$(awk 'NF {line=$0} END {print substr(line,1,1000)}' "$case_log" 2>/dev/null)"
      reason="case execution failed with exit status $case_status"
      [[ -n "$error_tail" ]] && reason="$reason: $error_tail"
      failure_status="failed"
      if grep -Eqi 'connection refused|connection reset|timed out|timeout|temporarily unavailable|service unavailable|server has gone away|lost connection' "$case_log" 2>/dev/null; then
        failure_status="retryable_failed"
      fi
      printf '%s backend=%s case_id=%s mode=%s status=%s reason=%s\n' \
        "$(date --iso-8601=seconds)" "$backend" "$CASE_ID" "$mode" "$failure_status" "$reason" \
        | tee -a "$RESULT_DIR/run.log"
      python3 -m src.campaigns.continuous --ledger "$LEDGER" "${PATTERN_ARGS[@]}" fail \
        --case-id "$CASE_ID" --run-id "$RUN_ID" --mode "$mode" --status "$failure_status" --reason "$reason"
      python3 -m src.campaigns.continuous --ledger "$LEDGER" "${PATTERN_ARGS[@]}" snapshot \
        --output "$RESULT_DIR/coverage.csv" --anomalies-output "$RESULT_DIR/anomalies.csv" >/dev/null
      continue
    fi
    [[ "$layer" == "filter" ]] && PREPARED_SIGNATURES["$prepared_key"]=1
    python3 -m src.campaigns.continuous --ledger "$LEDGER" "${PATTERN_ARGS[@]}" complete --case-id "$CASE_ID" --run-id "$RUN_ID" --mode "$mode" --result "$result"
    [[ "$mode" == "confirmation" ]] && save_explain "$backend" "$result" "$CASE_ID" 2>&1 | tee -a "$RESULT_DIR/run.log"
    python3 -m src.campaigns.continuous --ledger "$LEDGER" "${PATTERN_ARGS[@]}" snapshot --output "$RESULT_DIR/coverage.csv" --anomalies-output "$RESULT_DIR/anomalies.csv" >/dev/null
  done
  stop_all
done

python3 -m src.campaigns.continuous --ledger "$LEDGER" "${PATTERN_ARGS[@]}" snapshot --output "$RESULT_DIR/coverage.csv" --anomalies-output "$RESULT_DIR/anomalies.csv" > "$RESULT_DIR/summary.json"
stop_all
failed_case_count="$(python3 -c 'import json,sys; summary=json.load(open(sys.argv[1])); print(sum(int(row["n"]) for row in summary.get("coverage", []) if row.get("state") in {"failed", "retryable_failed"}))' "$RESULT_DIR/summary.json")"
if [[ "$failed_case_count" -gt 0 ]]; then
  echo "continuous campaign has $failed_case_count failed or retryable cases: $RESULT_DIR (pattern=${PATTERN:-all})" >&2
  exit 4
fi
echo "continuous campaign finished: $RESULT_DIR (pattern=${PATTERN:-all})"
