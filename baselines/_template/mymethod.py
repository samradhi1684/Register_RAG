"""Pipeline skeleton. Use self._retrieve_rank(...) for retrieval and self._lm(kind, ...) for EVERY model call so that
tokens, calls and latency are counted like the other systems."""
from registerrag.pipelines import BasePipeline, PipelineResult


class MyMethod(BasePipeline):
    name = "mymethod"

    def run(self, ex, ledger):
        stage1, ranking = self._retrieve_rank(ex.question, ex, ledger)
        pred, kept, trunc = self._answer(ex, ranking[: self.cfg.top_k], ledger)
        return PipelineResult(pred, ranking, stage1, ranking[: self.cfg.top_k], kept, trunc, {})
