"""Reference floor: the SLM answers from the question alone (parametric knowledge + guessing)."""
from registerrag.pipelines import BasePipeline, PipelineResult


class NoRetrievalRAG(BasePipeline):
    """Floor: the SLM answers from the question alone (parametric knowledge + guessing)."""
    name = "none"

    def run(self, ex, ledger):
        pred, kept, trunc = self._answer(ex, [], ledger)
        return PipelineResult(pred, [], [], [], kept, trunc)
