from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from .prompts import PROMPT_VERSION


@dataclass
class RunConfig:
    system: str = "sentence"
    dataset: str = "2wiki"
    data_path: str = ""
    limit: Optional[int] = None
    sample_seed: Optional[int] = None
    corpus_scope: str = "pooled"          # pooled | per_question
    chunk_size_words: int = 256
    chunk_overlap_words: int = 50
    embedder: str = "BAAI/bge-small-en-v1.5"
    prepend_title: bool = False
    lam: float = 0.5
    retrieve_k: int = 50                  # stage-1 candidates
    top_k: int = 5                        # units handed to the generator (per sub-question for 'iterative')
    reranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    generator: str = "qwen2.5:14b"        # or openai:gpt-4o-mini
    temperature: float = 0.0
    max_new_tokens: int = 64
    seed: int = 0
    num_ctx: int = 8192
    budget_tokens: Optional[int] = None   # hard cap on the generator prompt; None = unlimited
    prompt_version: str = PROMPT_VERSION
    tokenizer: str = ""                   # filled at run time
    unit_mode: Optional[str] = None       # None = system default (chunk for 'chunk'/'pyrag', else sentence)
    method_args: dict = field(default_factory=dict)   # method-specific knobs (--method-arg k=v)

    def frozen(self) -> dict:
        d = asdict(self)
        if d.get("unit_mode") is None:    # omitted when unset so hashes of earlier runs stay valid
            d.pop("unit_mode", None)
        if not d.get("method_args"):
            d.pop("method_args", None)
        # NOTE: `limit` stays in the hash on purpose - with corpus_scope=pooled the corpus is built from the loaded
        # questions, so a different limit is a different corpus and must be a different run folder.
        return d

    def hash(self, data_sha: str = "") -> str:
        blob = json.dumps({**self.frozen(), "data_sha256": data_sha}, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]
