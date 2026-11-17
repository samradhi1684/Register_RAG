"""B1 - vanilla RAG over fixed-size chunks (the standard deployed setup)."""
SYSTEM = "chunk"
LABEL = "B1 vanilla RAG (chunk)"
DEFAULT_UNIT = "chunk"


def pipeline_class():
    from registerrag.pipelines import OneShotRAG      # shared one-shot pipeline; B1 and B2 differ only in the unit
    return OneShotRAG


def variants(generator, coder=None, openai=False):
    v = [("B1 vanilla RAG, chunk", [])]
    if openai:
        v.append(("ref: GPT-4o-mini + chunk RAG", ["--generator", "openai:gpt-4o-mini"]))
    return v
