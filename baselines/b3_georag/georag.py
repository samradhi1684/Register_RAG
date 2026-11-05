"""GeoRAG (Zhang, Jia, Zhu, 2026; arXiv:2606.29328) - training-free, retrieval-agnostic context selection by
information-demand coverage, re-implemented on the RegisterRAG harness. Follows the paper's Algorithms 1 and 2 and
Eqs. (6)-(13) exactly; all defaults are the paper's: K=200, k=5, n=20 sub-queries at T=0.7, m=10 kept by greedy max-min,
tau_RV=0.05, alpha=beta=1, epsilon=0.01, 100 log-domain Sinkhorn iterations.

  1. candidates C = top-K of the base retriever (here: our BM25+BGE hybrid, no reranker - GeoRAG replaces top-k truncation)
  2. demand proxy P_Q: LLM writes n sub-queries, greedy max-min cosine keeps m (Eq. 6); reverse validation weights each
     sub-query by Overlap x Novelty (Eq. 7), drops those < tau_RV, normalises (Eq. 8); candidate weights
     w_i ~ r_i^alpha * L_i^beta with r = softmax(retriever score), L_i = sum_j q~_j max(cos(c_i, q_j), 0) (Eqs. 9-11)
  3. greedy selection (Alg. 1): cold start = cosine top-1, then argmax (1 - max cos to S) * (W(P,E_S) - W(P,E_{S+c}))

NOT in the paper (ours, listed in README.md): the sub-query prompt, the softmax temperature of Eq. 9 (default 1.0), the
start point of the max-min filter (the sub-query closest to the question), the order of the k chunks in the prompt
(selection order), and numerics (marginal floor 1e-30).
--method-arg: pool_k (200), sub_n (20), sub_m (10), sub_temperature (0.7), tau_rv (0.05), alpha (1), beta (1),
epsilon (0.01), sinkhorn_iters (100), prior_temp (1.0), rv_k (20).
"""
from __future__ import annotations

import copy
import json
import re

import numpy as np

from registerrag.pipelines import BasePipeline, PipelineResult

PROMPT_VERSION = "georag-paper+subq-v1"

SUBQ_SYSTEM = (
    "You help a retrieval system cover everything a question needs. Write search sub-queries for the question. "
    "Each sub-query must be short, self-contained and aimed at a DIFFERENT piece of information or angle (entities, "
    "attributes, relations, intermediate facts) needed to answer it. "
    'Reply with JSON only: {"sub_queries": ["...", "..."]}')
SUBQ_USER = "Question: {question}\n\nWrite {n} diverse sub-queries."


# ------------------------------------------------------------------------------------------------ numerics
def _lse(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def sinkhorn_cost(M, p, e, eps=0.01, iters=100):
    """Algorithm 2, batched. M: (B,K,n) cost, p: (K,) demand marginal, e: (B,n) coverage marginal.
    Returns W_eps = sum_ij pi_ij M_ij for every batch element (log-domain, fixed number of iterations)."""
    M = np.asarray(M, np.float64)
    B, K, n = M.shape
    logp = np.log(np.maximum(p, 1e-30))[None, :]
    loge = np.log(np.maximum(e, 1e-30))
    f, g = np.zeros((B, K)), np.zeros((B, n))
    for _ in range(iters):
        f = eps * logp - eps * _lse((g[:, None, :] - M) / eps, axis=2)
        g = eps * loge - eps * _lse((f[:, :, None] - M) / eps, axis=1)
    pi = np.exp((f[:, :, None] + g[:, None, :] - M) / eps)
    return (pi * M).sum(axis=(1, 2))


def greedy_select(E, q_emb, P, k, eps=0.01, iters=100):
    """Algorithm 1. E: (K,d) unit-norm candidate embeddings, q_emb: (d,), P: (K,) demand distribution.
    Returns the selected candidate indices in selection order."""
    K = E.shape[0]
    k = min(k, K)
    if k == 0:
        return []
    cos = E @ E.T
    Mfull = 1.0 - cos                                   # ground metric M_ij = 1 - cos(c_i, c_j)
    S = [int(np.argmax(E @ q_emb))]                     # cold start
    while len(S) < k:
        rest = np.array([i for i in range(K) if i not in S])
        W_S = sinkhorn_cost(Mfull[:, S][None], P, np.full((1, len(S)), 1.0 / len(S)), eps, iters)[0]
        cols = np.concatenate([np.tile(np.array(S), (len(rest), 1)), rest[:, None]], axis=1)      # (B, |S|+1)
        Mb = Mfull[:, cols].transpose(1, 0, 2)                                                      # (B, K, |S|+1)
        W_new = sinkhorn_cost(Mb, P, np.full((len(rest), len(S) + 1), 1.0 / (len(S) + 1)), eps, iters)
        gate = 1.0 - cos[np.ix_(rest, S)].max(axis=1)   # NOT clipped, exactly as in Eq. 12
        score = gate * (W_S - W_new)
        S.append(int(rest[int(np.argmax(score))]))
    return S


def maxmin_filter(Q, q0, m):
    """Eq. 6: greedily keep the m sub-queries that are most dispersed (max-min cosine distance). Starts from the
    sub-query closest to the question q0 (the paper does not say where it starts)."""
    if len(Q) <= m:
        return list(range(len(Q)))
    sel = [int(np.argmax(Q @ q0))]
    while len(sel) < m:
        d = 1.0 - Q @ Q[sel].T
        mn = d.min(axis=1)
        mn[sel] = -np.inf
        sel.append(int(np.argmax(mn)))
    return sel


def demand_distribution(C_scores, E, Qs, quality, alpha=1.0, beta=1.0, prior_temp=1.0):
    """Eqs. 8-11. Returns (P_Q over the K candidates, q_tilde)."""
    qt = quality / quality.sum() if quality.sum() > 0 else np.full(len(quality), 1.0 / max(len(quality), 1))
    z = np.asarray(C_scores, np.float64) / prior_temp
    r = np.exp(z - z.max()); r /= r.sum()                                      # Eq. 9
    L = np.clip(E @ Qs.T, 0.0, 1.0) @ qt                                       # Eq. 10
    w = (r ** alpha) * (L ** beta)                                             # Eq. 11
    if w.sum() <= 0:
        w = r
    return w / w.sum(), qt


# ------------------------------------------------------------------------------------------------- pipeline
class GeoRAG(BasePipeline):
    name = "georag"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        m = self.margs = dict(self.cfg.method_args or {})
        self.K = int(m.get("pool_k", 200)); self.n = int(m.get("sub_n", 20)); self.m = int(m.get("sub_m", 10))
        self.tau = float(m.get("tau_rv", 0.05)); self.rv_k = int(m.get("rv_k", 20))
        self.alpha = float(m.get("alpha", 1.0)); self.beta = float(m.get("beta", 1.0))
        self.eps = float(m.get("epsilon", 0.01)); self.iters = int(m.get("sinkhorn_iters", 100))
        self.prior_temp = float(m.get("prior_temp", 1.0))
        if self.retriever.lam <= 0.0:
            raise ValueError("GeoRAG needs dense embeddings (lam > 0)")
        self.subq_gen = copy.copy(self.generator)                  # same model, sampling temperature 0.7 as in the paper
        if hasattr(self.subq_gen, "temperature"):
            self.subq_gen.temperature = float(m.get("sub_temperature", 0.7))

    def _subqueries(self, ex, ledger):
        try:
            r = self._lm("compiler", SUBQ_SYSTEM, SUBQ_USER.format(question=ex.question, n=self.n), ledger,
                         gen=self.subq_gen, json_mode=True, max_new_tokens=900)
            mm = re.search(r"\{.*\}", r.text, re.S)
            subs = [str(s).strip() for s in json.loads(mm.group(0))["sub_queries"] if str(s).strip()]
            subs = list(dict.fromkeys(subs))[: self.n]
            if subs:
                return subs, None
            return [], "empty sub-query list"
        except Exception as e:  # noqa: BLE001
            return [], str(e)

    def run(self, ex, ledger):
        allowed = self.store.allowed_for(ex.doc_ids) if self.cfg.corpus_scope == "per_question" else None
        hits = self.retriever.retrieve(ex.question, self.K, allowed)
        ledger.call("retriever")
        if not hits:
            return PipelineResult("", [], [], [], [], False, {"error": "no candidates"})
        units = [self.store.units[h.idx] for h in hits]
        E = self.retriever.emb[[h.idx for h in hits]].astype(np.float64)
        scores = np.array([h.score for h in hits])
        emb = self.retriever.embedder
        q0 = emb.encode_query(ex.question).astype(np.float64)
        subs, sub_err = self._subqueries(ex, ledger)
        info = {"sub_queries_generated": len(subs), "sub_error": sub_err}
        if subs:
            Qall = np.stack([emb.encode_query(s) for s in subs]).astype(np.float64)
            keep = maxmin_filter(Qall, q0, self.m)
            subs, Qs = [subs[i] for i in keep], Qall[keep]
            cand_ids = {h.idx for h in hits}
            R = []
            for s in subs:                                         # reverse validation (Eq. 7)
                R.append([h.idx for h in self.retriever.retrieve(s, self.rv_k, allowed)])
                ledger.call("retriever")
            quality = []
            for j, Rj in enumerate(R):
                others = set().union(*[set(R[k]) for k in range(len(R)) if k != j]) if len(R) > 1 else set()
                overlap = len(set(Rj) & cand_ids) / self.rv_k
                novelty = len(set(Rj) - others) / len(Rj) if Rj else 0.0
                quality.append(overlap * novelty)
            quality = np.array(quality)
            alive = quality >= self.tau
            info.update(sub_queries=subs, quality=[round(float(x), 4) for x in quality], n_alive=int(alive.sum()))
            if alive.any():
                Qs, quality = Qs[alive], quality[alive]
            else:
                info["rv_all_dropped_uniform_weights"] = True
                quality = np.ones(len(subs))
        else:
            Qs, quality = q0[None, :], np.ones(1)
            info["fallback_demand"] = "question only"
        P, qt = demand_distribution(scores, E, Qs, quality, self.alpha, self.beta, self.prior_temp)
        sel = greedy_select(E, q0, P, self.cfg.top_k, self.eps, self.iters)
        selected = [units[i] for i in sel]
        rest = [u for i, u in enumerate(units) if i not in set(sel)]
        ranking = selected + rest                                  # Recall@k therefore measures GeoRAG's selection
        pred, kept, trunc = self._answer(ex, selected, ledger)
        info.update(selected_ranks=[int(i) for i in sel], demand_entropy=float(-(P * np.log(np.maximum(P, 1e-30))).sum()),
                    prompt_version=PROMPT_VERSION, pool_size=len(units))
        return PipelineResult(pred, ranking, units, ranking, kept, trunc, info)
