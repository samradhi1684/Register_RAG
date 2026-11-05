"""Per-query accounting: model calls by type, peak/total prompt tokens, latency, GPU memory."""
from __future__ import annotations

import time
from collections import Counter

KINDS = ("retriever", "reranker", "verifier", "compiler", "generator")


class Ledger:
    def __init__(self):
        self.calls = Counter({k: 0 for k in KINDS})
        self.gen_prompt_tokens = []      # one entry per SLM call -> peak = max
        self.tokens_processed = 0        # prompt+completion of every LM call + reranker/verifier inputs
        self.t0 = time.perf_counter()

    def call(self, kind: str, n: int = 1):
        self.calls[kind] += n

    def lm_call(self, kind: str, prompt_tokens: int, completion_tokens: int):
        self.calls[kind] += 1
        self.gen_prompt_tokens.append(prompt_tokens)
        self.tokens_processed += prompt_tokens + completion_tokens

    def side_tokens(self, n: int):
        self.tokens_processed += n

    @property
    def peak_prompt_tokens(self) -> int:
        return max(self.gen_prompt_tokens, default=0)

    def latency(self) -> float:
        return time.perf_counter() - self.t0


def gpu_peak_mb():
    try:
        import sys
        torch = sys.modules.get("torch")      # never import torch here: a first import mid-run can crash natively on Windows
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2**20
    except Exception:  # noqa: BLE001
        pass
    return None
