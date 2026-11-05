"""B2 - vanilla RAG over single sentences (same pipeline as B1, finer unit)."""
SYSTEM = "sentence"
LABEL = "B2 vanilla RAG (sentence)"
DEFAULT_UNIT = "sentence"


def pipeline_class():
    from registerrag.pipelines import OneShotRAG
    return OneShotRAG


def variants(generator, coder=None, openai=False):
    return [("B2 vanilla RAG, sentence", [])]
