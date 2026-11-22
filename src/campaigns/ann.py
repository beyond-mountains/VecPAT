from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..generation.equivalent import ANN_RULES
from ..core.confirmation import ann_candidate_key, balanced_random_orders, load_candidate_keys, serialize_orders


def _vector_text(vector: np.ndarray) -> str:
    # Nine significant digits are sufficient for lossless float32 round trips.
    # This matters for equivalent-metric cases whose top-k boundary is narrow.
    return "[" + ",".join(f"{float(x):.9g}" for x in vector) + "]"


def _data(cfg: dict):
    ac = cfg["ann_campaign"]
    rng = np.random.default_rng(int(cfg["experiment"]["seed"]) + 1001)
    vectors = rng.normal(size=(int(ac["points"]), int(ac["dimensions"]))).astype(np.float32)
    distribution = str(ac.get("distribution", "gaussian"))
    if distribution in {"sparse", "sparse_continuous"}:
        # The legacy sparse profile zeroed 90% of coordinates and could create
        # large equal-score groups. Keep sparsity while retaining enough
        # continuously distributed coordinates to make rankings identifiable.
        vectors[rng.random(vectors.shape) < 0.7] = 0.0
    elif distribution == "high_variance":
        vectors *= np.geomspace(0.01, 100.0, vectors.shape[1], dtype=np.float32)
    elif distribution == "clustered":
        centers = rng.normal(size=(32, vectors.shape[1])).astype(np.float32)
        vectors = centers[np.arange(len(vectors)) % len(centers)] + 0.05 * vectors
    elif distribution in {"single_cluster", "near_tie"}:
        # A single dense cluster remains difficult for ANN, but 0.05 noise is
        # above float32 score resolution. "near_tie" is retained as a safe
        # legacy alias so resumed ledgers do not recreate the old 1e-4 data.
        center = rng.normal(size=(1, vectors.shape[1])).astype(np.float32)
        vectors = center + np.float32(0.05) * vectors
    elif distribution == "duplicates":
        # Legacy alias: preserve paired neighborhoods without exact duplicates.
        paired = vectors[::2][:len(vectors[1::2])]
        jitter = rng.normal(0, 0.01, size=paired.shape).astype(np.float32)
        vectors[1::2] = paired + jitter
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    zero_rows = np.flatnonzero(vector_norms[:, 0] == 0)
    if len(zero_rows):
        vectors[zero_rows, 0] = 1.0
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= vector_norms
    categories = (np.arange(len(vectors)) % 5 + 1).astype(np.int32)
    if distribution in {"clustered", "single_cluster", "near_tie"}:
        queries = vectors[np.linspace(0, len(vectors) - 1, int(ac["queries"]), dtype=int)].copy()
        query_noise = 0.005 if distribution in {"single_cluster", "near_tie"} else 0.001
        queries += rng.normal(0, query_noise, size=queries.shape).astype(np.float32)
    else:
        queries = rng.normal(size=(int(ac["queries"]), int(ac["dimensions"]))).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    dimensions = int(ac["dimensions"])
    permutation_kind = str(ac.get("permutation", "random"))
    if permutation_kind == "reverse":
        permutation = np.arange(dimensions - 1, -1, -1)
    elif permutation_kind.startswith("block"):
        block = max(1, int(permutation_kind.removeprefix("block")))
        permutation = np.concatenate([np.arange(start, min(start + block, dimensions))[::-1]
                                      for start in range(0, dimensions, block)])
    elif permutation_kind == "variance":
        permutation = np.argsort(np.var(vectors, axis=0), kind="stable")
    else:
        permutation = rng.permutation(dimensions)
    scale = float(ac["scale_factor"])
    return vectors, categories, queries, permutation, scale


def _exact(vectors: np.ndarray, query: np.ndarray, metric: str, ids: np.ndarray, k: int) -> list[int]:
    if metric in {"IP", "IP_DESC"}: scores = -(vectors @ query)
    elif metric == "COSINE": scores = -(vectors @ query)  # input is normalized
    else: scores = np.sum((vectors - query) ** 2, axis=1)
    order = np.argsort(scores, kind="stable")[:k]
    return [int(ids[i]) for i in order]


class QdrantAdapter:
    name = "qdrant"
    def __init__(self, cfg):
        from ..backends.qdrant.client import Qdrant
        self.q = Qdrant(os.getenv("QDRANT_URL", "http://qdrant:6333")); self.cfg = cfg
    def create(self, name, metric, vectors, ids, categories):
        q, ix = self.q, self.cfg["index"]
        try: q.request("DELETE", f"/collections/{name}")
        except Exception: pass
        q.request("PUT", f"/collections/{name}", {"vectors": {"size": vectors.shape[1], "distance": {"L2":"Euclid","COSINE":"Cosine","IP":"Dot"}[metric]},
                  "hnsw_config": {"m": ix["hnsw_m"], "ef_construct": ix["ef_construct"], "full_scan_threshold": 10},
                  "optimizers_config": {"indexing_threshold": 0, "default_segment_number": 1}})
        q.request("PUT", f"/collections/{name}/index?wait=true", {"field_name":"category5","field_schema":"integer"})
        for start in range(0, len(ids), 1000):
            sl = slice(start, min(start + 1000, len(ids)))
            q.upload(name, ids[sl].astype(int).tolist(), vectors[sl].tolist(), [{"category5":int(x)} for x in categories[sl]])
        q.enable_indexing(name, {**ix, "indexing_threshold": 1000}); q.wait_ready(name, len(ids))
    def search(self, name, metric, query, condition, k, ef):
        filt = None if condition is None else ({"must":[{"key":"category5","range":{"gte":1,"lte":5}}]} if condition == "true" else {"must":[{"key":"category5","match":{"value":int(condition)}}]})
        return self.q.search(name, query.tolist(), filt, k, ef)[0:2]


class MilvusAdapter:
    name = "milvus"
    def __init__(self, cfg):
        from pymilvus import DataType, MilvusClient
        self.DataType, self.Client = DataType, MilvusClient; self.client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://milvus:19530")); self.cfg = cfg
    def create(self, name, metric, vectors, ids, categories):
        c, DT = self.client, self.DataType
        if c.has_collection(name): c.drop_collection(name)
        schema = self.Client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DT.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DT.FLOAT_VECTOR, dim=vectors.shape[1])
        schema.add_field(field_name="category5", datatype=DT.INT64)
        c.create_collection(collection_name=name, schema=schema, consistency_level="Strong")
        for start in range(0, len(ids), 1000):
            end = min(start + 1000, len(ids)); c.insert(collection_name=name, data=[{"id":int(ids[i]),"vector":vectors[i].tolist(),"category5":int(categories[i])} for i in range(start,end)])
        c.flush(collection_name=name); params = c.prepare_index_params(); ix = self.cfg["index"]
        params.add_index(field_name="vector", index_name="vector_idx", index_type="HNSW", metric_type=metric,
                         params={"M":int(ix["hnsw_m"]),"efConstruction":int(ix["ef_construct"])})
        params.add_index(field_name="category5", index_name="category5_idx", index_type="INVERTED")
        c.create_index(collection_name=name, index_params=params, sync=True); c.load_collection(collection_name=name)
    def search(self, name, metric, query, condition, k, ef):
        expression = "" if condition is None else ("category5 >= 1 and category5 <= 5" if condition == "true" else f"category5 == {int(condition)}")
        metric = "IP" if metric == "IP_DESC" else metric
        started = time.perf_counter_ns(); result = self.client.search(collection_name=name, anns_field="vector", data=[query.tolist()], filter=expression,
            limit=k, search_params={"metric_type":metric,"params":{"ef":max(ef,k)}}, output_fields=[], consistency_level="Strong")
        return [int(x["id"]) for x in result[0]], {"client_wall_ms":(time.perf_counter_ns()-started)/1e6}


class PgvectorAdapter:
    name = "pgvector"
    def __init__(self, cfg):
        import psycopg2
        self.conn = psycopg2.connect(os.environ["PGVECTOR_DSN"]); self.conn.autocommit = True; self.cfg = cfg
    def create(self, name, metric, vectors, ids, categories):
        from psycopg2.extras import execute_values
        opclass = {"L2":"vector_l2_ops","COSINE":"vector_cosine_ops","IP":"vector_ip_ops"}[metric]
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {name}"); cur.execute(f"CREATE TABLE {name}(id BIGINT PRIMARY KEY, vector vector({vectors.shape[1]}) NOT NULL, category5 INT NOT NULL)")
            rows=[(int(ids[i]),_vector_text(vectors[i]),int(categories[i])) for i in range(len(ids))]
            execute_values(cur, f"INSERT INTO {name}(id,vector,category5) VALUES %s", rows, page_size=500)
            ix=self.cfg["index"]; cur.execute("SET maintenance_work_mem='2GB'")
            cur.execute(f"CREATE INDEX {name}_hnsw ON {name} USING hnsw(vector {opclass}) WITH(m={int(ix['hnsw_m'])},ef_construction={int(ix['ef_construct'])})")
            cur.execute(f"CREATE INDEX {name}_category5 ON {name}(category5)"); cur.execute(f"ANALYZE {name}")
    def search(self, name, metric, query, condition, k, ef):
        op={"L2":"<->","COSINE":"<=>","IP":"<#>","IP_DESC":"<#>"}[metric]; where="TRUE" if condition is None else ("category5 BETWEEN 1 AND 5" if condition=="true" else f"category5={int(condition)}")
        with self.conn.cursor() as cur:
            cur.execute("SET enable_seqscan=off; SET hnsw.ef_search=%s",(max(ef,k),)); started=time.perf_counter_ns()
            cur.execute(f"SELECT id FROM {name} WHERE {where} ORDER BY vector {op} %s::vector LIMIT %s",(_vector_text(query),k)); ids=[int(x[0]) for x in cur.fetchall()]
        return ids,{"client_wall_ms":(time.perf_counter_ns()-started)/1e6}


class SeekDBAdapter:
    name = "seekdb"
    def __init__(self, cfg):
        import pymysql
        host=os.getenv("SEEKDB_HOST","seekdb"); port=int(os.getenv("SEEKDB_PORT","2881")); user=os.getenv("SEEKDB_USER","root"); password=os.environ["SEEKDB_PASSWORD"]; db=os.getenv("SEEKDB_DATABASE","filter_anomaly")
        self.conn=pymysql.connect(host=host,port=port,user=user,password=password,database=db,autocommit=True,read_timeout=1800,write_timeout=1800); self.cfg=cfg
    def create(self, name, metric, vectors, ids, categories):
        distance={"L2":"l2","COSINE":"cosine","IP":"inner_product"}[metric]; ix=self.cfg["index"]
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {name}"); cur.execute(f"CREATE TABLE {name}(id BIGINT PRIMARY KEY, vector VECTOR({vectors.shape[1]}) NOT NULL, category5 INT NOT NULL, KEY {name}_cat(category5)) ORGANIZATION HEAP")
            sql=f"INSERT INTO {name}(id,vector,category5) VALUES(%s,%s,%s)"
            rows=[(int(ids[i]),_vector_text(vectors[i]),int(categories[i])) for i in range(len(ids))]
            for start in range(0,len(rows),200): cur.executemany(sql,rows[start:start+200])
            cur.execute(f"CREATE VECTOR INDEX {name}_hnsw ON {name}(vector) WITH(distance={distance},type=hnsw,lib=vsag,m={int(ix['hnsw_m'])},ef_construction={int(ix['ef_construct'])},sync_mode=immediate)")
            try: cur.execute("CALL DBMS_INDEX_MANAGER.REFRESH()")
            except Exception: pass
            cur.execute(f"ANALYZE TABLE {name}")
    def search(self, name, metric, query, condition, k, ef):
        function={"L2":"l2_distance","COSINE":"cosine_distance","IP":"negative_inner_product","IP_DESC":"inner_product"}[metric]; where="TRUE" if condition is None else ("category5 BETWEEN 1 AND 5" if condition=="true" else f"category5={int(condition)}")
        direction="DESC" if metric == "IP_DESC" else "ASC"
        with self.conn.cursor() as cur:
            cur.execute("SET ob_hnsw_ef_search=%s",(max(ef,k),)); started=time.perf_counter_ns()
            cur.execute(f"SELECT id FROM {name} WHERE {where} ORDER BY {function}(vector,%s) {direction} APPROXIMATE LIMIT {int(k)}",(_vector_text(query),)); ids=[int(x[0]) for x in cur.fetchall()]
        return ids,{"client_wall_ms":(time.perf_counter_ns()-started)/1e6}


class WeaviateAdapter:
    name = "weaviate"
    def __init__(self, cfg):
        from ..backends.weaviate.client import WeaviateBackend
        self.client = WeaviateBackend(os.getenv("WEAVIATE_URL", "http://weaviate:8080")); self.cfg = cfg
    def create(self, name, metric, vectors, ids, categories):
        distance = {"L2": "l2-squared", "COSINE": "cosine", "IP": "dot"}[metric]
        ix = self.cfg["index"]; ef = int(self.cfg["ann_campaign"]["ef_search"][0])
        self.client.recreate_collection(name, vectors.shape[1], ix, max(ef, int(self.cfg["ann_campaign"]["top_k"][0])), distance)
        payloads = [{"tenant": int(value) - 1, "level": 0, "category": int(value),
                     "category5": int(value), "age": int(index % 1000) + 1,
                     "parity": "even" if index % 2 == 0 else "odd"}
                    for index, value in enumerate(categories)]
        self.client.upload(name, ids, vectors, payloads, int(self.cfg["dataset"].get("batch_size", 400)))
        self.client.wait_ready(name, len(ids))
    def search(self, name, metric, query, condition, k, ef):
        if condition is None:
            where = None
        elif condition == "true":
            where = {"path": ["category5"], "operator": "GreaterThanEqual", "valueInt": 1}
        else:
            where = {"path": ["category5"], "operator": "Equal", "valueInt": int(condition)}
        return self.client.search(name, query.astype(float).tolist(), where, int(k))


def _ratio(a: float, b: float) -> float: return max(a,b)/max(min(a,b),1e-9)
def _recall(found, truth): return len(set(found)&set(truth))/max(len(truth),1)


def run(backend_name: str, cfg: dict, output: Path):
    adapter={"qdrant":QdrantAdapter,"milvus":MilvusAdapter,"pgvector":PgvectorAdapter,"seekdb":SeekDBAdapter,"weaviate":WeaviateAdapter}[backend_name](cfg)
    vectors,categories,queries,perm,scale=_data(cfg); ids=np.arange(len(vectors),dtype=np.int64); target=int(cfg["ann_campaign"]["subset_category5"]); mask=categories==target
    specs={"base_cos":("COSINE",vectors,ids,categories),"base_l2":("L2",vectors,ids,categories),"base_ip":("IP",vectors,ids,categories),
           "perm_l2":("L2",vectors[:,perm],ids,categories),"scale_l2":("L2",vectors*scale,ids,categories),"subset_cos":("COSINE",vectors[mask],ids[mask],categories[mask])}
    ac=cfg["ann_campaign"]
    selected=set(ac.get("cases",[]))
    required = {
      "dimension_permutation": {"base_l2", "perm_l2"},
      "positive_global_scaling": {"base_l2", "scale_l2"},
      "cosine_vs_inner_product": {"base_cos", "base_ip"},
      "cosine_vs_inner_product_desc": {"base_cos", "base_ip"},
      "cosine_vs_l2": {"base_cos", "base_l2"},
      "no_filter_vs_true_filter": {"base_cos"},
      "filtered_full_vs_materialized_subset": {"base_cos", "subset_cos"},
    }
    if selected:
        needed = set().union(*(required.get(case, set()) for case in selected))
        specs = {key: value for key, value in specs.items() if key in needed}
    if backend_name == "weaviate":
        token = str(cfg.get("continuous_case", {}).get("case_id", "1001"))[:10]
        prefix = f"EfaAnn{token}"
    else:
        prefix=f"{cfg['ann_campaign'].get('collection_prefix', 'efa1001')}_{backend_name}_"
    support={}
    build_seconds={}
    if cfg["ann_campaign"].get("reuse_existing"):
        support={key:"ready" for key in specs}; print("reusing existing auxiliary collections",flush=True)
    else:
        for key,(metric,vs,row_ids,cats) in specs.items():
            started=time.perf_counter()
            try: adapter.create(prefix+key,metric,vs,row_ids,cats); support[key]="ready"; build_seconds[key]=time.perf_counter()-started; print(f"created {key} ({len(row_ids)} rows)",flush=True)
            except Exception as exc: support[key]=f"unsupported: {type(exc).__name__}: {exc}"; print(f"{key}: {support[key]}",flush=True)
    cases=[
      ("A1","dimension_permutation","base_l2","perm_l2","L2","L2",lambda q:q,lambda q:q[perm],None,None,vectors,vectors[:,perm],ids,ids),
      ("A2","positive_global_scaling","base_l2","scale_l2","L2","L2",lambda q:q,lambda q:q*scale,None,None,vectors,vectors*scale,ids,ids),
      ("A3","cosine_vs_inner_product","base_cos","base_ip","COSINE","IP",lambda q:q,lambda q:q,None,None,vectors,vectors,ids,ids),
      ("A3","cosine_vs_inner_product_desc","base_cos","base_ip","COSINE","IP_DESC",lambda q:q,lambda q:q,None,None,vectors,vectors,ids,ids),
      ("A3","cosine_vs_l2","base_cos","base_l2","COSINE","L2",lambda q:q,lambda q:q,None,None,vectors,vectors,ids,ids),
      ("FA1","no_filter_vs_true_filter","base_cos","base_cos","COSINE","COSINE",lambda q:q,lambda q:q,None,"true",vectors,vectors,ids,ids),
      ("FA2","filtered_full_vs_materialized_subset","base_cos","subset_cos","COSINE","COSINE",lambda q:q,lambda q:q,target,None,vectors[mask],vectors[mask],ids[mask],ids[mask])]
    if selected: cases=[item for item in cases if item[1] in selected]
    manifest_enabled = bool(cfg.get("confirmation", {}).get("candidate_manifest"))
    candidate_keys = load_candidate_keys(cfg, backend_name, "ann") | load_candidate_keys(cfg, backend_name, "fanns")
    if manifest_enabled:
        candidate_rule_cases = {(key[0], key[1]) for key in candidate_keys}
        cases = [item for item in cases if (item[0], item[1]) in candidate_rule_cases]
    output.parent.mkdir(parents=True,exist_ok=True); rows=[]
    with output.open("w",encoding="utf-8") as stream:
      # A4 has a valid exact oracle, but none of these engines exposes squared-L2 as a separately selectable ANN metric.
      max_k=max(map(int,ac["top_k"])); a4_equal=all(
          np.array_equal(np.argsort(np.sqrt(np.sum((vectors-q)**2,axis=1)),kind="stable")[:max_k],
                         np.argsort(np.sum((vectors-q)**2,axis=1),kind="stable")[:max_k]) for q in queries)
      a4={"backend":backend_name,"rule_id":"A4","layer":"ann","case_name":"l2_vs_squared_l2","semantic_equal":bool(a4_equal),"native_variant_supported":False,
          "status":"oracle_only","preconditions":ANN_RULES[3]["preconditions"],"note":"Squared L2 is not exposed as a distinct ANN metric/index; exact rankings were compared in the generator oracle."}
      stream.write(json.dumps(a4)+"\n"); rows.append(a4)
      for rule,case,bkey,vkey,bmetric,vmetric,bq,vq,bcond,vcond,bvec,vvec,bids,vids in cases:
        catalog=next(x for x in ANN_RULES if x["rule_id"]==rule)
        if support.get(bkey)!="ready" or support.get(vkey)!="ready":
          row={"backend":backend_name,"rule_id":rule,"layer":catalog["layer"],"case_name":case,"status":"unsupported","semantic_equal":None,"native_variant_supported":False,"preconditions":catalog["preconditions"],"detail":f"base={support.get(bkey)}; variant={support.get(vkey)}"}; stream.write(json.dumps(row)+"\n"); rows.append(row); continue
        for k in ac["top_k"]:
          exact_ok=all(_exact(bvec,bq(q),bmetric,bids,int(k))==_exact(vvec,vq(q),vmetric,vids,int(k)) for q in queries)
          for ef in ac["ef_search"]:
            if manifest_enabled and ann_candidate_key(rule, case, int(k), int(ef)) not in candidate_keys:
              continue
            btimes=[]; vtimes=[]; brecall=[]; vrecall=[]
            query_details=[]
            for query_index,q in enumerate(queries):
              truth=_exact(bvec,bq(q),bmetric,bids,int(k))
              balanced_mode=bool(cfg.get("confirmation",{}).get("balanced_random_order"))
              order_seed=int(cfg.get("confirmation",{}).get("order_seed",cfg["experiment"]["seed"]))
              warmup_count=int(ac["warmups"])
              warmup_orders=(balanced_random_orders(warmup_count,order_seed,backend_name,case,k,ef,query_index,"ann_warmup",labels=("b","v"))
                             if balanced_mode else [("b","v")]*warmup_count)
              for order in warmup_orders:
                for side in order:
                  if side=="b": adapter.search(prefix+bkey,bmetric,bq(q),bcond,int(k),int(ef))
                  else: adapter.search(prefix+vkey,vmetric,vq(q),vcond,int(k),int(ef))
              bs=[]; vs=[]; bf=[]; vf=[]
              repetition_count=int(ac["repetitions"])
              measurement_orders=(balanced_random_orders(repetition_count,order_seed,backend_name,case,k,ef,query_index,"ann_measurement",labels=("b","v"))
                                  if balanced_mode else [("b","v") if rep%2==0 else ("v","b") for rep in range(repetition_count)])
              for order in measurement_orders:
                for side in order:
                  if side=="b": found,m=adapter.search(prefix+bkey,bmetric,bq(q),bcond,int(k),int(ef)); bs.append(m["client_wall_ms"]); bf.append(_recall(found,truth))
                  else: found,m=adapter.search(prefix+vkey,vmetric,vq(q),vcond,int(k),int(ef)); vs.append(m["client_wall_ms"]); vf.append(_recall(found,truth))
              btimes.append(statistics.median(bs)); vtimes.append(statistics.median(vs)); brecall.append(statistics.mean(bf)); vrecall.append(statistics.mean(vf))
              query_details.append({"query_index":query_index,"base_ms":statistics.median(bs),"variant_ms":statistics.median(vs),
                                    "base_samples":bs,"variant_samples":vs,"base_recall":statistics.mean(bf),"variant_recall":statistics.mean(vf),
                                    "warmup_orders":serialize_orders(warmup_orders),"measurement_orders":serialize_orders(measurement_orders)})
            bp, vp=statistics.median(btimes),statistics.median(vtimes); row={"backend":backend_name,"rule_id":rule,"layer":catalog["layer"],"case_name":case,"status":"executed","native_variant_supported":True,
              "campaign_parameters":{"dimensions":int(ac["dimensions"]),"points":int(ac["points"]),"scale_factor":scale,"permutation":str(ac.get("permutation","random")),"distribution":str(ac.get("distribution","gaussian"))},
              "base_index_build_seconds":build_seconds.get(bkey),"variant_index_build_seconds":build_seconds.get(vkey),
              "index_size_bytes":None,"index_size_note":"A portable per-collection index-size API is not available across all five adapters.",
              "preconditions":catalog["preconditions"],"semantic_equal":exact_ok,"base_metric":bmetric,"variant_metric":vmetric,"top_k":k,"ef_search":ef,"base_p50_ms":bp,"variant_p50_ms":vp,"latency_ratio":_ratio(bp,vp),"variant_over_base":vp/max(bp,1e-9),
              "base_recall":statistics.mean(brecall),"variant_recall":statistics.mean(vrecall),"recall_gap":abs(statistics.mean(brecall)-statistics.mean(vrecall)),"performance_anomaly":_ratio(bp,vp)>2.0,
              "order_protocol":"balanced_random_ab_ba" if balanced_mode else "alternating","query_details":query_details}
            stream.write(json.dumps(row)+"\n"); stream.flush(); rows.append(row); print(f"{rule}/{case} k={k} ef={ef} ratio={row['latency_ratio']:.2f}",flush=True)
    csv_path=output.with_suffix(".csv"); fields=sorted(set().union(*(r.keys() for r in rows)))
    with csv_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    output.with_name(output.stem+"_support.json").write_text(json.dumps({"backend":backend_name,"collections":support},indent=2),encoding="utf-8")


def main():
    p=argparse.ArgumentParser(); p.add_argument("backend",choices=["qdrant","milvus","pgvector","seekdb","weaviate"]); p.add_argument("--config",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    with open(a.config,encoding="utf-8") as f: cfg=yaml.safe_load(f)
    run(a.backend,cfg,Path(a.output))


if __name__=="__main__": main()
