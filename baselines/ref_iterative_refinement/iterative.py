"""Reference row: decompose -> retrieve per sub-question -> concatenate everything -> answer (no register, no verification)."""
from registerrag.generate import parse_requirements
from registerrag.pipelines import BasePipeline, PipelineResult
from registerrag.prompts import SYSTEM_DECOMPOSE


class IterativeRefinementRAG(BasePipeline):
    """Baseline 3: decompose -> retrieve per sub-question -> concatenate EVERYTHING -> answer.
    No register, no verification, no binding substitution, no eviction (the budget just truncates)."""
    name = "iterative"

    def run(self, ex, ledger):
        # JSON mode + a larger token cap: the decomposition is longer than a short answer and was truncated at 64 tokens.
        # If the JSON-constrained call crashes the server, retry without it; if that fails too, keep the question.
        reqs, ok, comp_err = None, False, None
        for kw in ({"json_mode": True, "max_new_tokens": 384}, {"max_new_tokens": 384}):
            try:
                r = self._lm("compiler", SYSTEM_DECOMPOSE, f"Question: {ex.question}", ledger, **kw)
                reqs, ok = parse_requirements(r.text, ex.question)
                break
            except Exception as e:  # noqa: BLE001
                comp_err = str(e)
        if reqs is None:
            reqs, ok = [{"id": "R1", "need": ex.question, "depends_on": []}], False
        per_req, stage1_all = [], []
        for q in reqs:
            s1, rk = self._retrieve_rank(q["need"], ex, ledger)
            per_req.append(rk)
            stage1_all += s1
        pool, seen = [], set()
        for rk in per_req:                                   # concatenation order = round order
            for u in rk[: self.cfg.top_k]:
                if u.unit_id not in seen:
                    seen.add(u.unit_id); pool.append(u)
        ranking, seen = [], set()                            # round-robin interleave for fair Recall@k
        for i in range(max(len(x) for x in per_req)):
            for rk in per_req:
                if i < len(rk) and rk[i].unit_id not in seen:
                    seen.add(rk[i].unit_id); ranking.append(rk[i])
        pred, kept, trunc = self._answer(ex, pool, ledger)
        return PipelineResult(pred, ranking, stage1_all, pool, kept, trunc,
                              {"requirements": reqs, "compiler_parse_ok": ok, "compiler_error": comp_err})
