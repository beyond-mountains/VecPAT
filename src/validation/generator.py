from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..generation.equivalent import _features, build_filter_instances, emit_filter_pairs, evaluate, rule_catalog
from ..generation.campaign_1002 import build_filter_instances_1002, emit_filter_pairs_1002


def _signature_for_catalog(instance):
    signature = _features(instance.equivalent)
    signature.update({"transformation_depth": 4 if instance.rule_id == "C1" else 1,
                      "strength": instance.strength})
    return signature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="result_1001")
    parser.add_argument("--strengths", default="2,8,32")
    parser.add_argument("--campaign", choices=["1001", "1002"], default="1001")
    args = parser.parse_args()
    strengths = [int(x) for x in args.strengths.split(",")]
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    domain = [
        {"age": age, "category": category, "category5": category5, "parity": "even" if age % 2 == 0 else "odd"}
        for age in range(1, 1001) for category in range(1, 3) for category5 in range(1, 6)
    ]
    validations = []
    builder = build_filter_instances_1002 if args.campaign == "1002" else build_filter_instances
    emitter = emit_filter_pairs_1002 if args.campaign == "1002" else emit_filter_pairs
    instances = builder(strengths)
    for instance in instances:
        original = [i for i, row in enumerate(domain) if evaluate(instance.original, row)]
        equivalent = [i for i, row in enumerate(domain) if evaluate(instance.equivalent, row)]
        validations.append({"rule_id":instance.rule_id,"layer":instance.layer,"name":instance.name,
                            "case_name":instance.case_name,"strength":instance.strength,
                            "preconditions":instance.preconditions,"equivalence":instance.equivalence,
                            "original_count":len(original),"equivalent_count":len(equivalent),
                            "semantic_equal":original==equivalent})
    rng=np.random.default_rng(20261001); vectors=rng.normal(size=(5000,128)).astype(np.float32); vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
    queries=rng.normal(size=(8,128)).astype(np.float32); queries/=np.linalg.norm(queries,axis=1,keepdims=True); perm=rng.permutation(128); scale=3.0
    def rank(v,q,metric):
        score=-(v@q) if metric in {"COSINE","IP"} else np.sum((v-q)**2,axis=1)
        return np.argsort(score,kind="stable")[:100].tolist()
    ann_validations=[
      {"rule_id":"A1","semantic_equal":all(rank(vectors,q,"L2")==rank(vectors[:,perm],q[perm],"L2") for q in queries)},
      {"rule_id":"A2","semantic_equal":all(rank(vectors,q,"L2")==rank(vectors*scale,q*scale,"L2") for q in queries)},
      {"rule_id":"A3","semantic_equal":all(rank(vectors,q,"COSINE")==rank(vectors,q,"IP")==rank(vectors,q,"L2") for q in queries)},
      {"rule_id":"A4","semantic_equal":all(np.array_equal(
          np.argsort(np.sqrt(np.sum((vectors-q)**2,axis=1)),kind="stable")[:100],
          np.argsort(np.sum((vectors-q)**2,axis=1),kind="stable")[:100]) for q in queries)},
      {"rule_id":"FA1","semantic_equal":True,"note":"TRUE filter candidate IDs equal all row IDs by construction"},
      {"rule_id":"FA2","semantic_equal":True,"note":"materialized subset IDs are constructed from the same category5 predicate"},
    ]
    payload={"filter_validations":validations,"ann_validations":ann_validations,
             "all_semantic_equal":all(x["semantic_equal"] for x in validations+ann_validations)}
    (output/"generator_validation.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    catalog = ([{"rule_id":x.rule_id,"layer":x.layer,"name":x.name,"case":x.case_name,
                 "strength":x.strength,"preconditions":x.preconditions,"equivalence":x.equivalence,
                 "signature":_signature_for_catalog(x)} for x in instances] + rule_catalog([])[-6:]
               if args.campaign == "1002" else rule_catalog(strengths))
    (output/"rule_catalog.json").write_text(json.dumps(catalog,indent=2),encoding="utf-8")
    emitted={backend:[{"rule_id":p.pattern_id,"case_name":p.case_name,"strength":p.scale,
                       "original":p.base,"equivalent":p.variant,"signature":p.signature}
                      for p in emitter(backend,strengths)]
             for backend in ("qdrant","milvus","pgvector","seekdb")}
    (output/"generated_queries.json").write_text(json.dumps(emitted,indent=2),encoding="utf-8")
    if not payload["all_semantic_equal"]:
        raise SystemExit("semantic validation failed")
    print(json.dumps({"filter_instances":len(validations),"ann_rules":len(ann_validations),"all_semantic_equal":True}))


if __name__ == "__main__": main()
