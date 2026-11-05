"""B6 - EviMem (Li et al. 2026, arXiv:2604.27695), training-free re-implementation. See README.md."""
SYSTEM = "evimem"
LABEL = "B6 EviMem"
DEFAULT_UNIT = "sentence"            # the Raw layer of LaceMem = verbatim sentences


def pipeline_class():
    from .evimem import EviMem
    return EviMem


def variants(generator, coder=None, openai=False):
    return [("B6 EviMem (answer from Index tuples, as in the authors' code)", []),
            ("B6 EviMem (answer grounded in linked Raw sentences, as the paper text says)",
             ["--method-arg", "answer_context=raw"])]
