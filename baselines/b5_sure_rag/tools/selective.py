"""Selective-answering view of a B5 run (paper Eq. 1-2): coverage, risk (share of wrong answers among those given) and the
EM obtained when withheld answers count as wrong.
  python baselines/b5_sure_rag/tools/selective.py <results.jsonl> [--oof <aggregator>_oof_scores.json]
Without --oof the P(Safe) stored in the run is used (heuristic, or the aggregator the run was launched with)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from registerrag.metrics import exact_match

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--oof")
a = ap.parse_args()
oof = json.load(open(a.oof)) if a.oof else {}
S, C = [], []
for l in open(a.results, encoding="utf-8"):
    r = json.loads(l)
    if r.get("error") or "features" not in r.get("extra", {}):
        continue
    e = r["extra"]
    S.append(oof.get(r["qid"], e["p_safe"] - e["beta"] * e["u"]))
    C.append(int(exact_match(e["raw_answer"], r["gold_answers"])))
S, C = np.array(S), np.array(C)
order = np.argsort(-S, kind="stable")
print(f"n={len(C)}  base EM (answer everything) = {C.mean():.3f}")
print(f"{'coverage':>9} {'risk':>7} {'EM(withheld=wrong)':>19}")
for cov in (0.3, 0.5, 0.7, 0.9, 1.0):
    k = max(1, int(round(cov * len(C))))
    sel = order[:k]
    print(f"{cov:9.0%} {1 - C[sel].mean():7.3f} {C[sel].sum() / len(C):19.3f}")
