"""
Fuzzy scope anchoring (dataset-agnostic).

A *scope* is what a question is about: a Wikipedia article, a policy
program.  Ingestion declares scopes (`<index>/scopes.json`, name → document
ids; default one scope per document title).  Here the question text is
matched against scope names with IDF-weighted token overlap, so paraphrased
or re-ordered names still match:

    "the Charge Ahead Program by Madison Gas & Electric"
        → "Madison Gas & Electric - Charge Ahead Program (Wisconsin)"

Score of scope s for text q (tokens lower-cased, stop words dropped, simple
plural folding, "&" → "and"):

    hit(s)   = Σ idf(w)  for w in tokens(s) ∩ tokens(q)
    total(s) = Σ idf(w)  for w in tokens(s)
    score    = hit / √total          (rewards matching rare name words, does
                                      not over-penalise long names)
    coverage = hit / total           (gate: ≥ min_coverage)

idf is computed over scope names, so words shared by many names ("program",
"rebate", a state) weigh little and distinctive name words weigh a lot.

Measured on RePolicyQA (500 questions, 481 program scopes): top-1 = gold
program for 428/500 vs 286/500 for exact substring matching; at
coverage ≥ 0.5 the top-1 is right 92% of the time.
"""
from __future__ import annotations
import json
import math
import os
import re
from collections import Counter

_STOP = frozenset(
    "a an the of to in for and or is are be on by with as at from that this it its "
    "i my we our me you your what which who whom how does do did can could will would "
    "should if under about any there their they them has have had was were".split())
_WORD = re.compile(r"[a-z0-9]+")


def name_tokens(text: str) -> set[str]:
    out = set()
    for w in _WORD.findall(text.lower().replace("&", " and ")):
        if w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return out


class ScopeMatcher:
    def __init__(self, scopes: dict[str, list[str]]):
        self.scopes = {n: list(ids) for n, ids in scopes.items() if n}
        self._toks = {n: name_tokens(n) for n in self.scopes}
        df = Counter(w for ws in self._toks.values() for w in ws)
        n = len(self._toks)
        self._idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}
        self._total = {s: sum(self._idf[w] for w in ws) for s, ws in self._toks.items()}

    @classmethod
    def for_index(cls, index_dir: str, units) -> "ScopeMatcher":
        """scopes.json if ingestion wrote one, else one scope per document title."""
        path = os.path.join(index_dir or "", "scopes.json")
        if index_dir and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return cls(json.load(f))
        scopes: dict[str, set] = {}
        for u in units:
            if u.document_title:
                scopes.setdefault(u.document_title, set()).add(u.document_id)
        return cls({k: sorted(v) for k, v in scopes.items()})

    def rank(self, text: str) -> list[tuple[float, float, str]]:
        """[(score, coverage, scope name)] best first; only scopes sharing a token."""
        q = name_tokens(text)
        out = []
        for s, ws in self._toks.items():
            common = ws & q
            if not common or not self._total[s]:
                continue
            hit = sum(self._idf[w] for w in common)
            out.append((hit / math.sqrt(self._total[s]), hit / self._total[s], s))
        out.sort(reverse=True)
        return out

    def match(self, text: str, max_scopes: int = 2, min_coverage: float = 0.5,
              rel: float = 0.8) -> list[str]:
        """
        Scopes named by `text`: the best one if its coverage ≥ min_coverage,
        plus runners-up (up to max_scopes) scoring ≥ rel × the best.
        """
        r = self.rank(text)
        if not r or r[0][1] < min_coverage:
            return []
        best = r[0][0]
        return [s for sc, cov, s in r[:max_scopes] if cov >= min_coverage and sc >= rel * best]
