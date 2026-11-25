# VecPAT

## Supplementary Materials

- [Formalization of VecPAT Equivalence-Preserving Transformation Rules (PDF)](./vecpat_rules.pdf)

**Finding Performance Anomalies in Filtered Vector Search via Equivalent Query Transformations**

VecPAT is a rule-guided performance testing framework for filtered approximate nearest neighbor search (FANNS). It constructs semantically equivalent query pairs, translates them into backend-specific syntax, and validates their semantic equivalence. It then compares their execution latencies under controlled experimental conditions to identify potential performance anomalies.

The key idea is that semantically equivalent FANNS queries with identical filtering results and exact nearest-neighbor rankings may exhibit substantial latency differences.
Such differences, if reproducible, may reveal performance anomalies in filter optimization, FANNS planning, or vector search execution.

To date, VecPAT has uncovered 55 performance anomalies: 14 in seekDB, 10 in Milvus, 12 in pgvector, 7 in Qdrant, and 12 in Weaviate. Notably, all 14 anomalies in seekDB and all 10 anomalies in Milvus have been confirmed by their respective developers.

## Method

VecPAT runs each test through three stages:

1. **Query construction.** A backend-independent AST represents the original and equivalent predicates. It generates query pairs using equivalence-preserving transformation rules. Transformations vary constants, expansion strength, Boolean-tree shape, data distribution, selectivity, Top-K, and HNSW search parameters.
2. **Equivalence validation.** A reference interpreter checks the generated ASTs over a controlled metadata domain. The backend then verifies the complete eligible set. Exact vector search provides the Top-K ground truth.
3. **Performance anomaly detection.** Original and equivalent queries use the same dataset and query vector. Discovery screens candidates; confirmation repeats them with deterministic, fully balanced AB/BA ordering. VecPAT records client end-to-end latency, exact latency, Recall@K, raw samples, and execution metadata.

## Equivalence rules

### Filter rules

| ID | Transformation family | Representative equivalence |
|---|---|---|
| F1 | Operator equivalence | range, membership, bounds, and equality-OR representations |
| F2 | Boolean idempotence | `P = P AND ... AND P`; `P = P OR ... OR P` |
| F3 | Boolean identity | `F = F AND TRUE`; `F = F OR FALSE` |
| F4 | Negation equivalence | double negation and De Morgan transformations |
| F5 | Absorption | `F = F OR (F AND G)` and its dual |
| F6 | Distribution | factored predicates versus DNF/CNF expansion |
| F7 | Structural reordering | commutative/associative reorderings and alternative AST shapes |
| F8 | Implication redundancy | append predicates already implied by the original predicate |
| F9 | Predicate decomposition | a discrete range partitioned into disjoint ranges or equalities |
| F10 | Complement equivalence | membership in `S` versus negated membership in `U-S` |
| F11 | Set idempotence | duplicate and reorder membership values |
| F12 | Arithmetic predicate equivalence | translation, scaling, identity, and cancellation rewrites |

F12 is executed on Milvus, pgvector, and seekDB. Qdrant and Weaviate do not expose field arithmetic in their filter DSLs, so VecPAT does not generate F12 executions for those backends.

### Vector and filter–ANN rules

| ID | Transformation |
|---|---|
| A1 | Apply the same dimension permutation to all data and query vectors |
| A2 | Apply the same positive scaling factor under L2 distance |
| A3 | Compare cosine, inner product, and L2 rankings for unit vectors |
| A4 | Compare L2 and squared-L2 as a monotonic distance transformation; currently oracle-only |
| FA1 | No filter versus a filter that selects the full dataset |
| FA2 | Filter the full dataset versus search an exactly materialized eligible subset without a filter |

## Confirmed and reported anomalies

The following cases were reproduced by the balanced confirmation workflow and reported upstream. The links point to the public issue discussions.

### seekDB

| Rule | Issue |
|---|---|
| F1 | [oceanbase/seekdb#1501](https://github.com/oceanbase/seekdb/issues/1501) |
| F2 | [oceanbase/seekdb#1439](https://github.com/oceanbase/seekdb/issues/1439) |
| F3 | [oceanbase/seekdb#1489](https://github.com/oceanbase/seekdb/issues/1489) |
| F4 | [oceanbase/seekdb#1511](https://github.com/oceanbase/seekdb/issues/1511) |
| F5 | [oceanbase/seekdb#1535](https://github.com/oceanbase/seekdb/issues/1535) |
| F6 | [oceanbase/seekdb#2209](https://github.com/oceanbase/seekdb/issues/2209) |
| F8 | [oceanbase/seekdb#1534](https://github.com/oceanbase/seekdb/issues/1534) |
| F9 | [oceanbase/seekdb#1440](https://github.com/oceanbase/seekdb/issues/1440) |
| F10 | [oceanbase/seekdb#1513](https://github.com/oceanbase/seekdb/issues/1513) |
| F11 | [oceanbase/seekdb#1512](https://github.com/oceanbase/seekdb/issues/1512) |
| F12 | [oceanbase/seekdb#1538](https://github.com/oceanbase/seekdb/issues/1538) |
| A3 | [oceanbase/seekdb#1470](https://github.com/oceanbase/seekdb/issues/1470) |
| FA1 | [oceanbase/seekdb#1471](https://github.com/oceanbase/seekdb/issues/1471) |
| FA2 | [oceanbase/seekdb#1472](https://github.com/oceanbase/seekdb/issues/1472) |

### Milvus

| Rule | Issue |
|---|---|
| F1 | [milvus-io/milvus#54676](https://github.com/milvus-io/milvus/issues/54676) |
| F2 | [milvus-io/milvus#54043](https://github.com/milvus-io/milvus/issues/54043) |
| F3 | [milvus-io/milvus#54195](https://github.com/milvus-io/milvus/issues/54195) |
| F4 | [milvus-io/milvus#54592](https://github.com/milvus-io/milvus/issues/54592) |
| F5 | [milvus-io/milvus#54649](https://github.com/milvus-io/milvus/issues/54649) |
| F6 | [milvus-io/milvus#54692](https://github.com/milvus-io/milvus/issues/54692) |
| F8 | [milvus-io/milvus#54755](https://github.com/milvus-io/milvus/issues/54755) |
| F9 | [milvus-io/milvus#54693](https://github.com/milvus-io/milvus/issues/54693) |
| F10 | [milvus-io/milvus#54756](https://github.com/milvus-io/milvus/issues/54756) |
| F12 | [milvus-io/milvus#54757](https://github.com/milvus-io/milvus/issues/54757) |

## Tested database versions

The included Compose files pin the database versions used by this release:

| Backend | Version and deployment |
|---|---|
| Milvus | 3.0.0, standalone, Rocksmq, etcd 3.5.25, MinIO object storage |
| Qdrant | 1.19.0, single node |
| pgvector | 0.8.2 on PostgreSQL 15 (`pgvector/pgvector:pg15`) |
| seekDB | 1.4.0 (`1.4.0-100000172026082615`), single node |
| Weaviate | 1.39.7, single node, one shard, one replica, no vectorizer |

The paper experiments assigned each database container 8 physical CPU cores and 32 GiB of memory. `.env.example` contains the server20 CPU set (`13-20`). Change `VECPAT_CPUSET` to 8 valid CPUs on another host before running the pipeline.

## Requirements

- Linux with Bash
- Python 3.10 or later on the host
- Docker Engine with the Compose plugin
- At least 32 GiB of available memory for the default database limits
- Sufficient storage for persistent datasets and HNSW indexes

Create the local resource configuration:

```bash
cp .env.example .env
```

Run the generator and scheduling unit tests:

```bash
python3 -m unittest discover -s tests -v
```

## Quick validation

The smoke test validates the generator, deploys Qdrant, builds the runner, loads 2,000 32-dimensional vectors, and executes one equivalent-filter differential test:

```bash
chmod +x scripts/smoke.sh scripts/campaigns/run_continuous.sh
./scripts/smoke.sh
```

Successful output ends with `VecPAT smoke test passed`. Smoke artifacts are written to `results/smoke`.

## Running the continuous pipeline

Run every backend for one hour each:

```bash
./scripts/campaigns/run_continuous.sh 1
```

`XHOURS` is a separate budget for each selected vector database. With all five backends enabled, `1` therefore allocates up to five backend-hours, excluding container startup, data loading, and index construction.

Run one rule on one backend:

```bash
./scripts/campaigns/run_continuous.sh 1 --pattern F4 --backend qdrant
```

Run all pending F12 cases on seekDB:

```bash
./scripts/campaigns/run_continuous.sh until-exhausted \
  --pattern F12 --backend seekdb --strengths 1,2,8,32,128
```

Choose explicit result and ledger paths:

```bash
./scripts/campaigns/run_continuous.sh 2 \
  --pattern F6 \
  --result-dir results/f6_seekdb \
  --ledger continuous_state/f6.sqlite \
  --backend seekdb
```

Supported selectors are:

```text
--pattern F1..F12 | A1..A4 | FA1 | FA2
--backend milvus | pgvector | qdrant | seekdb | weaviate
--strengths comma-separated-positive-integers
```

The pipeline starts backends sequentially, groups cases by physical dataset signature, reuses compatible data and indexes, performs semantic validation before measurement, and resumes through a SQLite coverage ledger.

## Outputs

Each run directory contains:

```text
coverage.csv                 all scheduled configurations and states
anomalies.csv                cases exceeding an anomaly threshold
semantic_validation.jsonl   reference-interpreter results
run.log                      deployment and execution log
summary.json                 final coverage summary
raw/<backend>/               query-level JSONL and CSV measurements
explain/                     available execution-plan diagnostics
control/                     rendered per-case configs and case logs
```

The ledger states distinguish pending, running, candidate, complete, unsupported, semantic mismatch, failed, and retryable failure. A single case timeout is recorded without terminating the remaining campaign.

## Repository layout

```text
vecpat_1/
├── configs/                 backend base configurations and smoke config
├── scripts/
│   ├── smoke.sh             end-to-end Qdrant deployment test
│   └── campaigns/
│       └── run_continuous.sh
├── src/
│   ├── generation/          AST, rules, transformations, and renderers
│   ├── validation/          generator-side semantic validation
│   ├── core/                data generation, AB/BA ordering, coverage ledger
│   ├── backends/            five backend adapters and experiment runners
│   ├── campaigns/           continuous scheduler and ANN transformations
│   └── analysis/            execution-plan collection
├── docker-compose*.yml
├── Dockerfile*
└── requirements*.txt
```

