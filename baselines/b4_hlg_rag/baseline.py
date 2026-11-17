"""B4 - HLG-RAG (Ghassel et al., KDD 2025, arXiv:2506.08074), in-memory re-implementation. See README.md."""
SYSTEM = "hlg"
LABEL = "B4 HLG-RAG"
DEFAULT_UNIT = "sentence"            # the harness unit used only to map statements onto gold sentences for the metrics


def pipeline_class():
    from .hlg import HLGRAG
    return HLGRAG


def variants(generator, coder=None, openai=False):
    return [("B4 HLG-RAG (StatementGraphRAG)", []),
            ("B4 HLG-RAG (TopicGraphRAG)", ["--method-arg", "retriever=topic"])]
