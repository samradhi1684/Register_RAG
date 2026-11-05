"""B8 - PyRAG (Sun et al. 2026, arXiv:2605.12975), training-free re-implementation. See README.md."""
SYSTEM = "pyrag"
LABEL = "B8 PyRAG"
DEFAULT_UNIT = "chunk"


def pipeline_class():
    from .pyrag import PyRAG
    return PyRAG


def variants(generator, coder=None, openai=False):
    v = [("B8 PyRAG (same model plans)", [])]
    if coder and coder != generator:
        v.append((f"B8 PyRAG (code model plans: {coder})", ["--method-arg", f"plan_generator={coder}"]))
    return v
