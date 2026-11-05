"""B5 - SURE-RAG (Qiu, Han, Huang 2026, arXiv:2605.03534), sufficiency-aware answer gating. See README.md."""
SYSTEM = "surerag"
LABEL = "B5 SURE-RAG"
DEFAULT_UNIT = "chunk"               # verifies the answer of the B1 pipeline against its retrieved chunks


def pipeline_class():
    from .sure_rag import SureRAG
    return SureRAG


def variants(generator, coder=None, openai=False):
    # the aggregator file is produced by tools/fit_aggregator.py after the scoring pass (see README)
    return [("B5 SURE-RAG (per-passage pair verifier, scores only)", ["--method-arg", "gate=false"]),
            ("B5 SURE-RAG (concat premise comparator, scores only)",
             ["--method-arg", "gate=false", "--method-arg", "premise=concat"])]
