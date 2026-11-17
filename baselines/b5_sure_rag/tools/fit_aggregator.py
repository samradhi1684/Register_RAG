"""Fit the SURE-RAG answer-level aggregator g(z) (logistic regression, P(Safe)) from finished B5 runs.

  python baselines/b5_sure_rag/tools/fit_aggregator.py <results.jsonl> [more.jsonl ...] --out baselines/b5_sure_rag/aggregator/aggregator.json

Label = the base answer is correct (exact match, or F1 >= --f1 if given). Scores are cross-fitted (stratified K-fold) so
the printed numbers and *_oof_scores.json are out-of-fold; tau is chosen on the out-of-fold scores (max safe-F1). The paper
trains on a separate train split and fits tau/beta on a separate dev split; with only 100 questions per dataset
cross-fitting is the honest substitute. Use scripts/make_dev500.py for 500+ questions when possible."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from registerrag.metrics import exact_match, f1_score
from baselines.b5_sure_rag.sure_rag import FEATURES

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "aggregator" / "aggregator.json"))
ap.add_argument("--f1", type=float, default=None, help="label correct if F1 >= this (default: exact match)")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--C", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rows = []
for p in a.results:
    for l in open(p, encoding="utf-8"):
        r = json.loads(l)
        if r.get("error") or "features" not in r.get("extra", {}):
            continue
        ans = r["extra"]["raw_answer"]
        ok = (f1_score(ans, r["gold_answers"]) >= a.f1) if a.f1 is not None else bool(exact_match(ans, r["gold_answers"]))
        rows.append((r["qid"], [r["extra"]["features"][n] for n in FEATURES], int(ok)))
X, y = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
print(f"{len(y)} questions, {y.sum()} correct ({100 * y.mean():.0f}%)")
if len(y) < 2 * a.folds or y.min() == y.max():
    sys.exit("need both classes and at least 2 x folds questions")
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = (X - mu) / sd
oof = np.zeros(len(y))
for tr, te in StratifiedKFold(a.folds, shuffle=True, random_state=a.seed).split(Z, y):
    oof[te] = LogisticRegression(C=a.C, class_weight="balanced", max_iter=1000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
grid = np.linspace(0.05, 0.95, 91)


def f1_at(t):
    pred = oof >= t
    tp = (pred & (y == 1)).sum()
    return 0.0 if tp == 0 else 2 * tp / (pred.sum() + (y == 1).sum())


tau = float(max(grid, key=f1_at))
final = LogisticRegression(C=a.C, class_weight="balanced", max_iter=1000).fit(Z, y)
answered = oof >= tau
print(f"out-of-fold AUROC {roc_auc_score(y, oof):.3f}   safe-F1 {f1_at(tau):.3f} at tau={tau:.2f}   "
      f"(base accuracy {y.mean():.3f}; answered {100 * answered.mean():.0f}%, "
      f"accuracy when answered {y[answered].mean() if answered.any() else float('nan'):.3f})")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"features": FEATURES, "mean": mu.tolist(), "scale": sd.tolist(), "coef": final.coef_[0].tolist(),
           "intercept": float(final.intercept_[0]), "tau": tau, "beta": 0.0, "n_train": int(len(y)),
           "label": f"f1>={a.f1}" if a.f1 is not None else "exact_match", "folds": a.folds, "seed": a.seed,
           "source_runs": a.results}, open(a.out, "w"), indent=2)
json.dump(dict(zip([r[0] for r in rows], oof.tolist())), open(os.path.splitext(a.out)[0] + "_oof_scores.json", "w"))
print("wrote", a.out)
