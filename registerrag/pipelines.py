"""Shared pipeline machinery (base class, result type) and the vanilla one-shot RAG used by B1 and B2.
Baseline-specific pipelines live in their own folder under baselines/."""
from __future__ import annotations

from dataclasses import dataclass, field

from .generate import build_prompt
from .prompts import QA_SYSTEMS


@dataclass
class PipelineResult:
    prediction: str
    ranking: list                 # final ordering used for Recall@k
    stage1: list                  # all first-stage candidates (before reranking)
    pool: list                    # everything offered to the prompt builder
    selected: list                # what actually fit into the prompt
    truncated: bool
    extra: dict = field(default_factory=dict)


class BasePipeline:
    name = "base"

    def __init__(self, store, retriever, reranker, generator, counter, cfg, aux=None):
        self.store, self.retriever, self.reranker = store, retriever, reranker
        self.generator, self.counter, self.cfg = generator, counter, cfg
        self.aux = aux or {}                  # extra generators for method-specific roles, e.g. {'plan': gen}

    # -- shared steps ---------------------------------------------------------
    def _retrieve_rank(self, query, ex, ledger):
        allowed = self.store.allowed_for(ex.doc_ids) if self.cfg.corpus_scope == "per_question" else None
        hits = self.retriever.retrieve(query, self.cfg.retrieve_k, allowed)
        ledger.call("retriever")
        stage1 = [self.store.units[h.idx] for h in hits]
        if self.reranker is None:
            return stage1, stage1
        pairs = self.reranker.rerank(query, stage1, self.retriever.index_text)
        ledger.call("reranker")
        ledger.side_tokens(sum(self.counter.count(query) + self.counter.count(self.retriever.index_text(u))
                               for u in stage1))
        return stage1, [u for u, _ in pairs]

    def _lm(self, kind, system, user, ledger, gen=None, **gen_kw):
        try:
            r = (gen or self.generator).generate(system, user, **gen_kw)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"[{kind}] {e}") from e
        # canonical token count = our counter (comparable across generators); reported count kept for audit
        ledger.lm_call(kind, self.counter.count(system) + self.counter.count(user), r.completion_tokens)
        return r

    def _answer(self, ex, pool, ledger):
        system = QA_SYSTEMS[self.cfg.prompt_version]
        user, kept, trunc = build_prompt(self.counter, system, ex.question, pool,
                                         self.retriever.index_text, self.cfg.budget_tokens)
        r = self._lm("generator", system, user, ledger)
        return r.text, kept, trunc

    def run(self, ex, ledger) -> PipelineResult:
        raise NotImplementedError


class OneShotRAG(BasePipeline):
    """B1 (chunk units) and B2 (sentence units): one retrieval, rerank, top-k, generate."""
    name = "oneshot"

    def run(self, ex, ledger):
        stage1, ranking = self._retrieve_rank(ex.question, ex, ledger)
        pool = ranking[: self.cfg.top_k]
        pred, kept, trunc = self._answer(ex, pool, ledger)
        return PipelineResult(pred, ranking, stage1, pool, kept, trunc)


def make_pipeline(system, *args, **kw):
    """Instantiate the pipeline of any registered baseline (see registerrag/registry.py)."""
    from .registry import pipeline_class
    return pipeline_class(system)(*args, **kw)
