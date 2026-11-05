"""Dataset loaders -> Example objects with gold evidence and draft gold requirement programs."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from .units import PysbdSplitter, doc_from_native_sentences, doc_from_text, norm_ws


@dataclass
class Requirement:
    rid: str
    text: str
    depends_on: list = field(default_factory=list)
    gold_doc_ids: list = field(default_factory=list)
    gold_unit_ids: Optional[list] = None     # sentence-level gold (None if dataset has paragraph gold only)
    gold_answer: Optional[str] = None        # intermediate answer when the dataset provides it
    heuristic: bool = False                  # True => auto-derived, needs manual review

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class Example:
    qid: str
    dataset: str
    question: str
    answers: list
    doc_ids: list                 # this question's own paragraphs (per_question corpus scope)
    gold_doc_ids: list
    gold_unit_ids: Optional[list]  # sentence-level gold; None for MuSiQue
    n_hops: int
    qtype: str = ""
    gold_requirements: list = field(default_factory=list)


def _norm_key(s: str) -> str:
    return norm_ws(s).lower()


def _wiki_requirements(rec, title2doc, sf_units):
    """Draft requirements from 2Wiki `evidences` triples. Triples are used ONLY to author the
    plain-English requirement text and hop order; they are not used as an evidence representation."""
    ev = rec.get("evidences") or []
    reqs = []
    for i, trip in enumerate(ev, 1):
        if len(trip) != 3:
            continue
        s, rel, _o = trip
        deps = [f"R{j+1}" for j in range(i - 1) if len(ev[j]) == 3 and _norm_key(ev[j][2]) == _norm_key(s)]
        text = (f"Find the {rel} of the entity identified in {deps[0]}." if deps
                else f"Find the {rel} of {s}.")
        doc = title2doc.get(_norm_key(s))
        gdocs = [doc] if doc else []
        gunits = sf_units.get(_norm_key(s), []) if doc else []
        reqs.append(Requirement(f"R{i}", text, deps, gdocs, gunits, None, heuristic=True))
    return reqs


def parse_native_sentence_record(rec: dict, dataset: str, docs: dict) -> Example:
    """2WikiMultiHopQA and HotpotQA: context = [[title, [sentences...]], ...]."""
    title2doc, doc_ids = {}, []
    ctx = rec["context"]
    if isinstance(ctx, dict):                       # HuggingFace columnar layout (HotpotQA)
        ctx = list(zip(ctx["title"], ctx["sentences"]))
    sfs = rec.get("supporting_facts", [])
    if isinstance(sfs, dict):
        sfs = list(zip(sfs["title"], sfs["sent_id"]))
    for title, sents in ctx:
        d = doc_from_native_sentences(title, sents)
        if d is None:
            continue
        docs.setdefault(d.doc_id, d)
        doc_ids.append(d.doc_id)
        title2doc.setdefault(_norm_key(title), d.doc_id)
    gold_units, sf_by_title, gold_docs = [], {}, []
    for title, idx in sfs:
        did = title2doc.get(_norm_key(title))
        if did is None:
            continue
        d = docs[did]
        if idx >= len(d.sentences):
            continue
        uid = f"{did}#s{idx}"
        gold_units.append(uid)
        sf_by_title.setdefault(_norm_key(title), []).append(uid)
        if did not in gold_docs:
            gold_docs.append(did)
    ev = rec.get("evidences") or []
    n_hops = len(ev) if ev else max(len(gold_docs), 1)
    reqs = _wiki_requirements(rec, title2doc, sf_by_title) if dataset == "2wiki" else []
    qid = str(rec.get("_id") or rec.get("id") or hashlib.sha1(rec["question"].encode()).hexdigest()[:10])
    return Example(qid, dataset, rec["question"], [rec["answer"]] + list(rec.get("answer_aliases", [])),
                   doc_ids, gold_docs, gold_units, n_hops, rec.get("type", ""), reqs)


def parse_musique_record(rec: dict, docs: dict, splitter) -> Example:
    idx2doc, doc_ids, gold_docs = {}, [], []
    for p in rec["paragraphs"]:
        d = doc_from_text(p.get("title", ""), p["paragraph_text"], splitter)
        if d is None:
            continue
        docs.setdefault(d.doc_id, d)
        idx2doc[p["idx"]] = d.doc_id
        doc_ids.append(d.doc_id)
        if p.get("is_supporting") and d.doc_id not in gold_docs:
            gold_docs.append(d.doc_id)
    reqs = []
    for i, step in enumerate(rec.get("question_decomposition", []), 1):
        q = step["question"]
        deps = [f"R{k}" for k in sorted({int(x) for x in re.findall(r"#(\d+)", q)})]
        text = re.sub(r"#(\d+)", lambda m: f"the answer to R{m.group(1)}", q)
        gd = idx2doc.get(step.get("paragraph_support_idx"))
        reqs.append(Requirement(f"R{i}", text, deps, [gd] if gd else [], None, step.get("answer")))
    answers = [rec["answer"]] + list(rec.get("answer_aliases", []))
    return Example(str(rec["id"]), "musique", rec["question"], answers, doc_ids, gold_docs, None,
                   len(reqs) or 2, "", reqs)


def load_examples(path: str, dataset: str, limit: Optional[int] = None, sample_seed: Optional[int] = None):
    """Returns (examples, docs, data_sha256). The 100-question files are already frozen samples;
    `sample_seed` is only for sub-sampling a larger file."""
    raw_bytes = open(path, "rb").read()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    recs = json.loads(raw_bytes)
    if dataset == "musique":
        recs = [r for r in recs if r.get("answerable", True)]
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(recs)
    if limit:
        recs = recs[:limit]
    docs, exs = {}, []
    splitter = PysbdSplitter() if dataset == "musique" else None
    for r in recs:
        if dataset in ("2wiki", "hotpot"):
            exs.append(parse_native_sentence_record(r, dataset, docs))
        elif dataset == "musique":
            exs.append(parse_musique_record(r, docs, splitter))
        else:
            raise ValueError(f"unknown dataset {dataset}")
    return exs, docs, sha
