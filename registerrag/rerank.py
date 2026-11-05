from __future__ import annotations

import re


class OverlapReranker:
    """Stub reranker (token overlap). TEST ONLY."""
    name = "overlap-stub"

    def rerank(self, query, units, index_text):
        q = set(re.findall(r"\w+", query.lower()))
        sc = [len(q & set(re.findall(r"\w+", index_text(u).lower()))) / (1 + len(q)) for u in units]
        return sorted(zip(units, sc), key=lambda t: -t[1])


class CrossEncoderReranker:
    """e.g. cross-encoder/ms-marco-MiniLM-L-6-v2."""

    def __init__(self, model_name: str, device=None, max_length: int = 512, batch_size: int = 32):
        from sentence_transformers import CrossEncoder
        self.name, self.batch_size = model_name, batch_size
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def rerank(self, query, units, index_text):
        if not units:
            return []
        sc = self.model.predict([(query, index_text(u)) for u in units], batch_size=self.batch_size,
                                show_progress_bar=False)
        return sorted(zip(units, [float(s) for s in sc]), key=lambda t: -t[1])
