"""Hybrid retrieval: S = lam * dense + (1 - lam) * BM25, both min-max normalised per query."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np


class HashEmbedder:
    """Deterministic feature-hashing embedder. TEST/STUB ONLY - lets the pipeline run offline."""
    key = "hash-256"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_query(self, text):
        return self._vec(text)


class STEmbedder:
    """sentence-transformers embedder (e.g. BAAI/bge-small-en-v1.5). Normalised, so dot = cosine."""

    def __init__(self, model_name: str, device: Optional[str] = None, query_prefix: str = "", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.key, self.query_prefix, self.batch_size = model_name, query_prefix, batch_size

    def encode_docs(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 2000).astype(np.float32)

    def encode_query(self, text):
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0].astype(np.float32)


def _tok(text: str):
    return re.findall(r"\w+", text.lower())


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


@dataclass
class Hit:
    idx: int
    score: float
    dense: float
    bm25: float


class HybridRetriever:
    def __init__(self, units, embedder, lam: float = 0.5, prepend_title: bool = False,
                 cache_dir: Optional[str] = None, store_fingerprint: str = ""):
        from rank_bm25 import BM25Okapi
        self.units, self.embedder, self.lam, self.prepend_title = units, embedder, lam, prepend_title
        texts = [self.index_text(u) for u in units]
        self.emb = self._embeddings(texts, cache_dir, store_fingerprint)
        self.bm25 = BM25Okapi([_tok(t) or ["_"] for t in texts]) if lam < 1.0 else None

    def index_text(self, u) -> str:
        return f"{u.title}. {u.text}" if self.prepend_title else u.text

    def _embeddings(self, texts, cache_dir, fp):
        if self.lam <= 0.0:
            return np.zeros((len(texts), 1), np.float32)
        path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            k = hashlib.sha1(f"{self.embedder.key}|{self.prepend_title}|{fp}".encode()).hexdigest()[:16]
            path = os.path.join(cache_dir, f"emb_{k}.npy")
            if os.path.exists(path):
                return np.load(path)
        emb = self.embedder.encode_docs(texts)
        if path:
            np.save(path, emb)
        return emb

    def retrieve(self, query: str, k: int, allowed: Optional[np.ndarray] = None) -> list:
        idx = np.arange(len(self.units)) if allowed is None else allowed
        if idx.size == 0:
            return []
        d = (self.emb[idx] @ self.embedder.encode_query(query)) if self.lam > 0 else np.zeros(idx.size, np.float32)
        b = self.bm25.get_scores(_tok(query))[idx] if self.bm25 is not None else np.zeros(idx.size)
        s = self.lam * _minmax(np.asarray(d, np.float64)) + (1 - self.lam) * _minmax(np.asarray(b, np.float64))
        order = np.argsort(-s, kind="stable")[:k]
        return [Hit(int(idx[o]), float(s[o]), float(d[o]), float(b[o])) for o in order]
