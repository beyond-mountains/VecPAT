from __future__ import annotations

import numpy as np


def generate_dataset(cfg: dict):
    ds = cfg["dataset"]
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    n, dim, tenants = ds["points"], ds["dimensions"], ds["tenants"]
    tenant = rng.integers(0, tenants, size=n, dtype=np.int32)
    level = rng.integers(0, ds["levels"], size=n, dtype=np.int32)
    age = rng.integers(1, int(ds.get("age_universe", 1000)) + 1, size=n, dtype=np.int32)
    centers = rng.normal(size=(tenants, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    noise = rng.normal(size=(n, dim)).astype(np.float32)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    corr = float(ds["correlation"])
    vectors = corr * centers[tenant] + np.sqrt(max(0.0, 1.0 - corr * corr)) * noise
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    payloads = [
        {
            "tenant": int(tenant[index]), "level": int(level[index]),
            "category": int(tenant[index]) + 1, "category5": int(tenant[index] % 5) + 1,
            "age": int(age[index]), "parity": "even" if tenant[index] % 2 == 0 else "odd",
        }
        for index in range(n)
    ]
    return vectors, payloads, tenant, centers

