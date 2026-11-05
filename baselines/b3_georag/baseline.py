"""B3 - GeoRAG (Zhang, Jia, Zhu 2026, "Covering the Unseen: Information Demand Coverage Optimization for RAG",
arXiv:2606.29328), training-free re-implementation. See README.md."""
SYSTEM = "georag"
LABEL = "B3 GeoRAG"
DEFAULT_UNIT = "chunk"


def pipeline_class():
    from .georag import GeoRAG
    return GeoRAG


def variants(generator, coder=None, openai=False):
    return [("B3 GeoRAG (chunk units, top-200 -> 5)", []),
            ("B3 GeoRAG (sentence units, top-200 -> 5)", ["--unit", "sentence"])]
