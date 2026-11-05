"""SURE-RAG (Qiu, Han, Huang 2026; arXiv:2605.03534) - Sufficiency and Uncertainty-aware evidence verification for
selective RAG, re-implemented as an answer-gating layer over the B1 chunk-RAG pipeline.

Paper pipeline (Alg. 1): (q, a, E) -> claims C(a) -> shared pair verifier h(q, c_j, e_i) = [p_sup, p_ref, p_neu] ->
answer-level features z (coverage, relation strength, uncertainty) -> aggregator g(z) = P(Supported/Refuted/Insufficient)
-> selective score s = P(Safe) - beta * u -> answer iff argmax = Supported and s >= tau, else abstain.

What the paper does NOT release (so it cannot be reproduced byte-for-byte): the DeBERTa-v3-base pair verifier trained on
HotpotQA-RAG v3 construction metadata, the exact feature formulas, beta, tau, the logistic-regression training data.
This baseline therefore keeps the ARCHITECTURE and substitutes:
  * pair verifier h        -> public 3-way NLI cross-encoder cross-encoder/nli-deberta-v3-base (same backbone family, no
                              task-specific training); hypothesis = a declarative claim written by the generator from (q, a)
  * features z             -> the paper's Table I blocks with our own formulas (README.md)
  * aggregator g, tau, beta-> fitted offline by tools/fit_aggregator.py on labelled runs (label: answer correct vs not);
                              until a file is given, an un-trained heuristic is used and logged as such
  * claims                 -> one claim per answer (the paper's main experiments also use single-claim short answers)
--method-arg: gate (true), aggregator (path to aggregator.json), verifier (HF id | stub), tau, beta, claim_tokens (64),
premise (pairs = paper | concat = all passages joined into one premise).
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

from registerrag.pipelines import BasePipeline, PipelineResult

PROMPT_VERSION = "surerag-arch+claim-v1"
VERIFIER_DEFAULT = "cross-encoder/nli-deberta-v3-base"
FEATURES = ["max_sup", "max_ref", "mean_neu", "entropy", "disagreement", "conflict", "coverage_deficit",
            "frac_supported", "frac_refuted", "frac_insufficient"]

CLAIM_SYSTEM = ("Rewrite the question and its answer as ONE short declarative sentence that states the answer as a fact. "
                "Keep every name exactly as written. If the answer is yes or no, state the fact in the affirmative or "
                "negative accordingly. Reply with the sentence only.")
CLAIM_USER = "Question: {question}\nAnswer: {answer}\nStatement:"


# ------------------------------------------------------------------------------------------------ verifiers
class NLIVerifier:
    """h(q, c, e): P(entailment | support), P(contradiction | refute), P(neutral) of premise=passage, hypothesis=claim."""

    def __init__(self, name=VERIFIER_DEFAULT, max_length=512, batch_size=16):
        from sentence_transformers import CrossEncoder
        self.name, self.batch = name, batch_size
        self.model = CrossEncoder(name, max_length=max_length)
        id2label = {int(k): v.lower() for k, v in self.model.model.config.id2label.items()}
        inv = {v: k for k, v in id2label.items()}
        self.idx = [inv["entailment"], inv["contradiction"], inv["neutral"]]

    def probs(self, claim: str, passages: list) -> np.ndarray:
        """-> (len(passages), 3) = [sup, ref, neu]"""
        if not passages:
            return np.zeros((0, 3))
        lg = np.asarray(self.model.predict([(p, claim) for p in passages], batch_size=self.batch,
                                           show_progress_bar=False), dtype=np.float64)
        lg = lg[:, self.idx]
        e = np.exp(lg - lg.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


class OverlapVerifier:
    """TEST ONLY: lexical-overlap stand-in so the pipeline runs offline."""
    name = "overlap-stub"

    def probs(self, claim, passages):
        c = set(re.findall(r"\w+", claim.lower())) or {"_"}
        out = []
        for p in passages:
            o = len(c & set(re.findall(r"\w+", p.lower()))) / len(c)
            out.append([o * 0.9, 0.05, max(0.0, 0.95 - o * 0.9)])
        return np.asarray(out)


# ------------------------------------------------------------------------------------------------ features / aggregator
def features(P: np.ndarray) -> dict:
    """Answer-level feature vector from the (k,3) pair distributions (single claim). Names follow Table I."""
    if len(P) == 0:
        return dict(zip(FEATURES, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]))
    sup, ref, neu = P[:, 0], P[:, 1], P[:, 2]
    ent = -(P * np.log(np.maximum(P, 1e-12))).sum(axis=1) / math.log(3)
    mx_s, mx_r, mn = float(sup.max()), float(ref.max()), float(neu.mean())
    claim_label = int(np.argmax([mx_s, mx_r, mn]))                 # per-claim arg-max -> coverage fractions (m = 1)
    return {"max_sup": mx_s, "max_ref": mx_r, "mean_neu": mn, "entropy": float(ent.mean()),
            "disagreement": float(sup.std()), "conflict": min(mx_s, mx_r), "coverage_deficit": 1.0 - mx_s,
            "frac_supported": float(claim_label == 0), "frac_refuted": float(claim_label == 1),
            "frac_insufficient": float(claim_label == 2)}


class Aggregator:
    """Logistic regression g(z) -> P(Safe). Loaded from tools/fit_aggregator.py output; heuristic if no file."""

    def __init__(self, path=None, tau=None, beta=None):
        self.path, self.d = path, None
        if path:
            self.d = json.load(open(path, encoding="utf-8"))
        self.tau = float(tau if tau is not None else (self.d or {}).get("tau", 0.5))
        self.beta = float(beta if beta is not None else (self.d or {}).get("beta", 0.0))
        self.kind = "logreg" if self.d else "heuristic(max_sup*(1-max_ref))"

    def p_safe(self, f: dict) -> float:
        if self.d is None:
            return float(f["max_sup"] * (1.0 - f["max_ref"]))
        x = (np.array([f[n] for n in self.d["features"]]) - np.array(self.d["mean"])) / np.array(self.d["scale"])
        z = float(np.dot(x, self.d["coef"]) + self.d["intercept"])
        return 1.0 / (1.0 + math.exp(-z))

    def score(self, f: dict):
        p = self.p_safe(f)
        u = f["entropy"]                                            # uncertainty penalty u: mean normalised pair entropy
        return p, u, p - self.beta * u


# ------------------------------------------------------------------------------------------------- the pipeline
class SureRAG(BasePipeline):
    name = "surerag"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        m = self.margs = dict(self.cfg.method_args or {})
        self.gate = bool(m.get("gate", True))
        self.agg = Aggregator(m.get("aggregator"), m.get("tau"), m.get("beta"))
        self.claim_tokens = int(m.get("claim_tokens", 64))
        self.premise = str(m.get("premise", "pairs"))          # pairs (paper) | concat
        if self.premise not in ("pairs", "concat"):
            raise ValueError("premise must be pairs or concat")
        self._verifier = None

    @property
    def verifier(self):
        if self._verifier is None:
            name = str(self.margs.get("verifier", VERIFIER_DEFAULT))
            self._verifier = OverlapVerifier() if name == "stub" else NLIVerifier(name)
        return self._verifier

    @verifier.setter
    def verifier(self, v):
        self._verifier = v

    def run(self, ex, ledger):
        stage1, ranking = self._retrieve_rank(ex.question, ex, ledger)
        pool = ranking[: self.cfg.top_k]
        answer, kept, trunc = self._answer(ex, pool, ledger)
        claim, claim_err = answer, None
        try:
            r = self._lm("compiler", CLAIM_SYSTEM, CLAIM_USER.format(question=ex.question, answer=answer), ledger,
                         max_new_tokens=self.claim_tokens)
            claim = (r.text.strip().splitlines() or [answer])[0].strip() or answer
        except RuntimeError as e:
            claim_err = str(e)
        texts = [self.retriever.index_text(u) for u in kept]
        if self.premise == "concat":        # one pair over all passages joined (the paper's concat comparator, not SURE-RAG proper)
            texts = ["\n".join(texts)] if texts else []
        P = self.verifier.probs(claim, texts)
        ledger.call("verifier", len(texts))
        ledger.side_tokens(sum(self.counter.count(t) + self.counter.count(claim) for t in texts))
        f = features(P)
        p_safe, u, s = self.agg.score(f)
        # decision: answer iff the claim is Supported (P(Safe) is the Supported probability) and s >= tau
        decision = "answer" if s >= self.agg.tau else "abstain"
        pred = answer if (decision == "answer" or not self.gate) else ""
        extra = {"claim": claim, "claim_error": claim_err, "pair_probs": np.round(P, 4).tolist(), "features": f,
                 "p_safe": p_safe, "u": u, "s": s, "tau": self.agg.tau, "beta": self.agg.beta, "decision": decision,
                 "gate": self.gate, "raw_answer": answer, "aggregator": self.agg.kind,
                 "verifier": getattr(self.verifier, "name", "?"), "prompt_version": PROMPT_VERSION}
        return PipelineResult(pred, ranking, stage1, pool, kept, trunc, extra)
