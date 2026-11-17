"""Reference floor - no retrieval at all."""
SYSTEM = "none"
LABEL = "ref: no retrieval (floor)"
DEFAULT_UNIT = "sentence"


def pipeline_class():
    from .none import NoRetrievalRAG
    return NoRetrievalRAG


def variants(generator, coder=None, openai=False):
    return [("ref: no retrieval", [])]
