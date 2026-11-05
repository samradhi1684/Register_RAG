"""Evidence units with provenance (unit_id, document, sentence index, character offsets).

Key correction over the old baselines: for datasets that ship sentence lists (2Wiki, HotpotQA)
the *native* sentence boundaries are kept, so `supporting_facts = [title, sent_idx]` maps
exactly onto unit ids. pysbd is only used for datasets without sentence annotations
(MuSiQue paragraphs, markdown policy documents).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np


def norm_ws(s: str) -> str:
    return " ".join(s.split())


def make_doc_id(title: str, text: str) -> str:
    return "D" + hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()[:12]


def has_alnum(s: str) -> bool:
    return any(ch.isalnum() for ch in s)


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str
    sentences: tuple          # one entry per ORIGINAL sentence (index == sent_idx)
    sent_offsets: tuple       # ((char_start, char_end), ...) into `text`
    native_sentences: bool    # True if boundaries come from the dataset, False if from a splitter


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str              # "<doc_id>#s<i>" (sentence) or "<doc_id>#c<j>" (chunk)
    doc_id: str
    title: str
    text: str
    unit_type: str            # "source_sentence" | "chunk"  (later: "atomic_fact", "obligation", ...)
    sent_idx: Optional[int]   # for sentences
    covered_sents: tuple      # sentence indices wholly contained in this unit
    char_start: int
    char_end: int
    section: Optional[str] = None

    def covered_ids(self) -> set:
        return {f"{self.doc_id}#s{i}" for i in self.covered_sents}


# --------------------------------------------------------------------------- documents
def doc_from_native_sentences(title: str, sentences: Iterable[str]) -> Optional[Doc]:
    sents = tuple(norm_ws(s) for s in sentences)
    if not any(has_alnum(s) for s in sents):
        return None
    text, offs, pos = " ".join(sents), [], 0
    for s in sents:
        offs.append((pos, pos + len(s)))
        pos += len(s) + 1
    title = norm_ws(title)
    return Doc(make_doc_id(title, text), title, text, sents, tuple(offs), True)


class PysbdSplitter:
    def __init__(self, language: str = "en"):
        import pysbd
        self.language, self.version = language, pysbd.__version__
        self._seg = pysbd.Segmenter(language=language, clean=False)

    def split(self, text: str) -> list:
        return [norm_ws(s) for s in self._seg.segment(text)]

    def describe(self) -> dict:
        return {"splitter": "pysbd", "splitter_version": self.version, "language": self.language}


def doc_from_text(title: str, text: str, splitter) -> Optional[Doc]:
    text, title = norm_ws(text), norm_ws(title)
    if not has_alnum(text):
        return None
    sents = [s for s in splitter.split(text) if s]
    offs, pos = [], 0
    for s in sents:
        i = text.find(s, pos)
        i = pos if i < 0 else i
        offs.append((i, i + len(s)))
        pos = i + len(s)
    return Doc(make_doc_id(title, text), title, text, tuple(sents), tuple(offs), False)


# --------------------------------------------------------------------------- unit builders
def build_sentence_units(docs: Iterable[Doc]) -> list:
    units = []
    for d in docs:
        for i, (s, (a, b)) in enumerate(zip(d.sentences, d.sent_offsets)):
            if has_alnum(s):
                units.append(EvidenceUnit(f"{d.doc_id}#s{i}", d.doc_id, d.title, s,
                                          "source_sentence", i, (i,), a, b))
    return units


def build_chunk_units(docs: Iterable[Doc], size: int = 256, overlap: int = 50) -> list:
    """Fixed-size word windows. `size`/`overlap` are in WORDS (a documented proxy for tokens).
    `covered_sents` lists the sentences wholly inside the window, so sentence-level gold recall
    is measurable for chunks too."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    step, units = size - overlap, []
    for d in docs:
        words = [(m.start(), m.end()) for m in re.finditer(r"\S+", d.text)]
        if not words:
            continue
        j, start = 0, 0
        while True:
            end = min(start + size, len(words))
            cs, ce = words[start][0], words[end - 1][1]
            cov = tuple(i for i, (a, b) in enumerate(d.sent_offsets) if a >= cs and b <= ce)
            units.append(EvidenceUnit(f"{d.doc_id}#c{j}", d.doc_id, d.title, d.text[cs:ce],
                                      "chunk", None, cov, cs, ce))
            j += 1
            if end >= len(words):
                break
            start += step
    return units


class UnitStore:
    def __init__(self, units: list, docs: dict, describe: dict):
        self.units, self.docs, self._describe = units, docs, describe
        self.by_id = {u.unit_id: u for u in units}
        d2i: dict = {}
        for i, u in enumerate(units):
            d2i.setdefault(u.doc_id, []).append(i)
        self.doc2idx = {k: np.asarray(v, dtype=np.int64) for k, v in d2i.items()}
        h = hashlib.sha256()
        for u in units:
            h.update(u.unit_id.encode()); h.update(b"\0"); h.update(u.text.encode()); h.update(b"\n")
        self.fingerprint = h.hexdigest()

    def allowed_for(self, doc_ids: Iterable[str]) -> np.ndarray:
        parts = [self.doc2idx[d] for d in doc_ids if d in self.doc2idx]
        return np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)

    def describe(self) -> dict:
        return {**self._describe, "n_units": len(self.units), "n_docs": len(self.docs),
                "units_fingerprint": self.fingerprint}


def make_store(docs: dict, mode: str, chunk_size: int = 256, chunk_overlap: int = 50) -> UnitStore:
    if mode == "sentence":
        units = build_sentence_units(docs.values())
        desc = {"unit_mode": "sentence",
                "native_sentence_docs": sum(d.native_sentences for d in docs.values())}
    elif mode == "chunk":
        units = build_chunk_units(docs.values(), chunk_size, chunk_overlap)
        desc = {"unit_mode": "chunk", "chunk_size_words": chunk_size, "chunk_overlap_words": chunk_overlap}
    else:
        raise ValueError(mode)
    return UnitStore(units, docs, desc)
