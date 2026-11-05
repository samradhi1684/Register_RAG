import json, os, subprocess, sys
import numpy as np
import pytest

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
DATA = os.path.join(ROOT, "datasets", "wiki2_data", "2wiki_100.json")

from baselines.b4_hlg_rag import hlg as H
from registerrag.config import RunConfig
from registerrag.data import load_examples
from registerrag.generate import GenResult, TokenCounter
from registerrag.ledger import Ledger
from registerrag.rerank import OverlapReranker
from registerrag.retrieval import HashEmbedder, HybridRetriever
from registerrag.units import make_store


class Scripted:
    """Proposition call -> one proposition per sentence; topic call -> one topic, entities = capitalised spans."""
    name = "scripted"

    def __init__(self, broken=False):
        self.broken, self.log = broken, []

    def generate(self, system, user, json_mode=False, max_new_tokens=None):
        self.log.append((system, user))
        if self.broken and ("atomic statements" in system or "knowledge graph. (1)" in system):
            return GenResult("not json", 0, 1, 0.0)
        if "atomic statements" in system:
            text = user.split("Text:\n", 1)[1]
            props = [s for s in H.re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            return GenResult(json.dumps({"propositions": props}), 0, 1, 0.0)
        if "knowledge graph. (1)" in system:
            props = H.re.findall(r"^(\d+)\. (.*)$", user, H.re.M)
            ents = [{"name": sp, "classification": "X", "statement_ids": [int(n)]} for n, p in props for sp in H.cap_spans(p)[:3]]
            return GenResult(json.dumps({"topics": [{"name": "T", "statement_ids": [int(n) for n, _ in props]}],
                                         "entities": ents, "facts": []}), 0, 1, 0.0)
        if "Extract the keywords" in system:
            q = user.split("Query:", 1)[1]
            return GenResult(json.dumps({"keywords": H.cap_spans(q)[:3]}), 0, 1, 0.0)
        return GenResult("ANS", 0, 1, 0.0)


def make(gen, tmp_path, rerank=True, **margs):
    ex, docs, _ = load_examples(DATA, "2wiki", limit=5)
    st = make_store(docs, "sentence")
    cfg = RunConfig(system="hlg", prepend_title=True, method_args={"cache_dir": str(tmp_path), **margs})
    ret = HybridRetriever(st.units, HashEmbedder(), 0.5, prepend_title=True)
    return H.HLGRAG(st, ret, OverlapReranker() if rerank else None, gen, TokenCounter(), cfg), ex[0]


def test_graph_has_all_three_tiers(tmp_path):
    p, ex = make(Scripted(), tmp_path)
    stats = p.build_index()
    g = p.g
    assert stats["statements"] > 0 and stats["topics"] > 0 and stats["entities"] > 0 and stats["fallback_chunks"] == 0
    st = g.statements[0]
    assert st["title"] and st["doc_id"] in g.store.docs and st["topics"] and st["ents"]      # lineage + summarisation + entity
    assert g.S.shape[0] == len(g.statements) and g.T.shape[0] == len(g.topics)
    some_ent = next(iter(g.ent_index))
    assert all(some_ent in g.statements[s]["ents"] for s in g.ent_index[some_ent])


def test_broken_extraction_never_loses_text(tmp_path):
    p, ex = make(Scripted(broken=True), tmp_path); stats = p.build_index()
    assert stats["fallback_chunks"] == stats["chunks"] and stats["statements"] > 0 and stats["entities"] > 0


def test_extraction_cache_is_reused(tmp_path):
    g1 = Scripted(); p, _ = make(g1, tmp_path); p.build_index()
    g2 = Scripted(); p2, _ = make(g2, tmp_path); p2.build_index()
    assert g1.log and not g2.log


def test_statement_pipeline_end_to_end(tmp_path):
    g = Scripted(); p, ex = make(g, tmp_path, vss_k=20, kw_k=10, beam_width=10); led = Ledger()
    res = p.run(ex, led)
    assert res.prediction == "ANS" and res.extra["mode"] == "statement" and res.extra["statements_in_prompt"]
    assert len(res.extra["statements_in_prompt"]) <= 5 and res.extra["n_final"] >= res.extra["n_init"]
    assert led.calls["compiler"] == 1 and led.calls["generator"] == 1 and led.calls["reranker"] == 1
    assert res.ranking and res.selected


def test_topic_pipeline_end_to_end(tmp_path):
    g = Scripted(); p, ex = make(g, tmp_path, retriever="topic", topic_k=5, beam_width=5); res = p.run(ex, Ledger())
    assert res.prediction == "ANS" and res.extra["mode"] == "topic" and res.extra["n_topics"] == 5


def test_works_without_reranker_and_rejects_bad_mode(tmp_path):
    p, ex = make(Scripted(), tmp_path, rerank=False); assert p.run(ex, Ledger()).prediction == "ANS"
    with pytest.raises(ValueError):
        make(Scripted(), tmp_path, retriever="nope")


def test_beam_expands_only_through_shared_entities_and_only_when_improving(tmp_path):
    p, ex = make(Scripted(), tmp_path); p.build_index(); g = p.g
    qv = p.retriever.embedder.encode_query(ex.question).astype(np.float32)
    start = [0]
    found = p._beam(start, qv, None)
    for sid in found:
        if sid != 0:
            assert any(g.statements[sid]["ents"] & g.statements[o]["ents"] for o in found if o != sid)


def test_diversity_filter_drops_near_duplicates_keeps_higher_score():
    texts = ["Alice was born in Paris in 1900.", "Alice was born in Paris in 1900", "Bob directed Inception."]
    keep = H.diversity_filter([0, 1, 2], [0.2, 0.9, 0.5], texts, tau=0.005)
    assert 1 in keep and 0 not in keep and 2 in keep
    assert H.diversity_filter([0, 1], [1, 2], ["a b", "a b"], tau=0) == [0, 1]


def test_cli_smoke_with_stub(tmp_path):
    cmd = [sys.executable, "-m", "registerrag.cli", "run", "--system", "hlg", "--dataset", "2wiki", "--data", DATA,
           "--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "c"), "--limit", "5", "--stub",
           "--method-arg", f"cache_dir={tmp_path / 'g'}"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr[-2000:]
    run_dir = next((tmp_path / "runs").iterdir())
    recs = [json.loads(l) for l in open(run_dir / "results.jsonl", encoding="utf-8")]
    assert len(recs) == 5 and all(x["error"] is None for x in recs)
