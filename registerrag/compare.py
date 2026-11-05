"""Paired comparison of two runs on the same questions (needed because n=100 is too noisy for unpaired reads)."""
from __future__ import annotations

import json
import numpy as np
from math import comb


def load(path):
    return {r["qid"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def paired(a_path, b_path, metric="em", n_boot=5000, seed=0):
    A, B = load(a_path), load(b_path)
    qids = sorted(set(A) & set(B))
    d = [B[q][metric] - A[q][metric] for q in qids]
    m = len(d)
    arr = np.asarray(d, dtype=float)
    idx = np.random.default_rng(seed).integers(0, m, size=(n_boot, m))
    boots = np.sort(arr[idx].mean(axis=1))
    out = {"metric": metric, "n_paired": m, "mean_a": sum(A[q][metric] for q in qids) / m,
           "mean_b": sum(B[q][metric] for q in qids) / m, "diff_b_minus_a": sum(d) / m,
           "ci95": (float(boots[int(0.025 * n_boot)]), float(boots[int(0.975 * n_boot) - 1]))}
    if metric == "em":
        bw = sum(1 for q in qids if B[q]["em"] > A[q]["em"])
        aw = sum(1 for q in qids if A[q]["em"] > B[q]["em"])
        out.update(b_only_correct=bw, a_only_correct=aw, mcnemar_p=mcnemar_exact(bw, aw))
    return out
