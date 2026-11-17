"""Reference row - iterative refinement (decompose, retrieve per sub-question, concatenate everything)."""
SYSTEM = "iterative"
LABEL = "ref: iterative refinement"
DEFAULT_UNIT = "sentence"


def pipeline_class():
    from .iterative import IterativeRefinementRAG
    return IterativeRefinementRAG


def variants(generator, coder=None, openai=False):
    return [("ref: iterative refinement", [])]
