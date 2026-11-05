import os
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[3])
DATA = {"musique": os.path.join(ROOT, "datasets", "musique_data", "musique_100.json")}

from baselines.ref_iterative_refinement.iterative import IterativeRefinementRAG
from registerrag.config import RunConfig
from registerrag.data import load_examples
from registerrag.generate import StubGenerator, TokenCounter
from registerrag.ledger import Ledger
from registerrag.retrieval import HashEmbedder, HybridRetriever
from registerrag.units import make_store


def test_compiler_call_uses_json_mode_and_large_token_cap():

    class Rec(StubGenerator):
        calls = []
        def generate(self, system, user, json_mode=False, max_new_tokens=None):
            Rec.calls.append((json_mode, max_new_tokens))
            return super().generate(system, user, json_mode, max_new_tokens)

    ex, docs, _ = load_examples(DATA["musique"], "musique", limit=3)
    st, cfg = make_store(docs, "sentence"), RunConfig(system="iterative", prepend_title=False)
    ret = HybridRetriever(st.units, HashEmbedder(), 0.5)
    IterativeRefinementRAG(st, ret, None, Rec(), TokenCounter(), cfg).run(ex[0], Ledger())
    assert Rec.calls[0] == (True, 384)        # compiler
    assert Rec.calls[-1] == (False, None)     # final answer keeps the short cap


def test_compiler_falls_back_when_json_mode_crashes():

    class Boom(StubGenerator):
        def generate(self, system, user, json_mode=False, max_new_tokens=None):
            if json_mode:
                raise RuntimeError("500 server error")
            return super().generate(system, user, json_mode, max_new_tokens)

    ex, docs, _ = load_examples(DATA["musique"], "musique", limit=3)
    st, cfg = make_store(docs, "sentence"), RunConfig(system="iterative")
    ret = HybridRetriever(st.units, HashEmbedder(), 0.5)
    res = IterativeRefinementRAG(st, ret, None, Boom(), TokenCounter(), cfg).run(ex[0], Ledger())
    assert res.prediction == "stub" and res.extra["compiler_parse_ok"] is True   # 2nd (plain) attempt succeeded
