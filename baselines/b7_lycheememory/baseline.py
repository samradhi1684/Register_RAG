"""B7 - LycheeMemory V2 (Li et al. 2026, arXiv:2608.12990), training-free re-implementation. See README.md."""
SYSTEM = "lychee"
LABEL = "B7 LycheeMemory V2"
DEFAULT_UNIT = "sentence"            # raw turns = sentences


def pipeline_class():
    from .lychee import LycheeMemory
    return LycheeMemory


def variants(generator, coder=None, openai=False):
    return [("B7 LycheeMemory V2 (planner + 4 channels + RRF + MMR)", []),
            ("B7 LycheeMemory V2 (no planner: single query)", ["--method-arg", "plan=false"])]
