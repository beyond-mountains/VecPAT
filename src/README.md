# Source layout

`src` mirrors the method pipeline described in the project README:

1. `generation`: build equivalence-preserving AST pairs.
2. `validation`: verify rule preconditions and eligible-set equivalence.
3. `backends`: render and execute filters in each vector database.
4. `campaigns`: orchestrate cross-backend filter/ANN campaigns.
5. `analysis`: collect the execution-plan diagnostics exposed by a backend.

`core` contains shared dataset and balanced-order measurement primitives. Backend-neutral code must not import a concrete backend. Backend-specific implementations may import `core` and `generation`.

Run modules from the repository root with `python -m`, for example:

```bash
python3 -m src.validation.generator --help
python3 -m src.backends.qdrant.cli --help
python3 -m src.backends.milvus.cli --help
python3 -m src.backends.pgvector.cli --help
python3 -m src.backends.seekdb.cli --help
python3 -m src.backends.weaviate.cli --help
```

