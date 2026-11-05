import json, os, re, subprocess, sys
import numpy as np
import pytest

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
DATA = os.path.join(ROOT, "datasets", "wiki2_data", "2wiki_100.json")

from baselines.b7_lycheememory import lychee as Ly
from registerrag.config import RunConfig
from registerrag.data import load_examples
from registerrag.generate import GenResult, TokenCounter
from registerrag.ledger import Ledger
from registerrag.rerank import OverlapReranker
from registerrag.retrieval import HashEmbedder, HybridRetriever
from registerrag.units import make_store


def test_paper_constants():
    assert Ly.length_pressure(100) == -1.30 and Ly.length_pressure(900) == 3.00 and Ly.length_pressure(2000) == 3.00
    assert -1.30 < Ly.length_pressure(600) < 3.00
    assert [Ly.turn_signal(n) for n in (1, 2, 3)] == [-0.85, -0.15, 0.15]
    assert Ly.turn_signal(4) == pytest.approx(0.30) and Ly.turn_signal(6) == pytest.approx(0.60) and Ly.turn_signal(50) == 1.0


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_segmentation_cuts_on_topic_shift_and_on_capacity():
    A, B = unit([1, 0.02, 0, 0]), unit([0, 0, 1, 0.02])
    embs = [A, A, A, A, A, B, B, B, B, B]
    segs = Ly.segment(embs, [200] * 10, delta=0.5, tmin=300, tmax=900, max_ex=10)
    assert segs[0][0] == 0 and segs[-1][1] == 10 and len(segs) >= 2
    assert all(a < b for a, b in segs) and [s[1] for s in segs[:-1]] == [s[0] for s in segs[1:]]
    cut = [s[1] for s in segs][:-1]
    assert 5 in cut or any(c <= 5 for c in cut)                       # the shift A->B is a boundary (or earlier)
    segs = Ly.segment([A] * 12, [10] * 12, max_ex=4)
    assert all(b - a <= 4 for a, b in segs)
    assert Ly.segment([], []) == []


def test_short_document_stays_one_segment():
    embs = [unit([1, 0.1 * i, 0, 0]) for i in range(4)]
    assert Ly.segment(embs, [30] * 4) == [(0, 4)]


def test_rrf_and_mmr():
    sc = Ly.rrf([["a", "b"], ["b", "c"]], kappa=60)
    assert sc["b"] > sc["a"] and sc["b"] > sc["c"] and sc["a"] == pytest.approx(1 / 61)
    emb = np.stack([unit([1, 0]), unit([1, 0.01]), unit([0, 1])])
    sel = Ly.mmr_select([0, 1, 2], {0: 1.0, 1: 0.99, 2: 0.9}, emb, 2)
    assert sel[0] == 0 and sel[1] == 2                                # the near-duplicate is penalised


class Scripted:
    name = "scripted"

    def __init__(self, broken=False):
        self.broken, self.log = broken, []

    def generate(self, system, user, json_mode=False, max_new_tokens=None):
        self.log.append((system, user))
        if "memory encoder" in system:
            if self.broken:
                return GenResult("junk", 0, 1, 0.0)
            turns = re.findall(r"^\[(\d+)\] (.*)$", user, re.M)
            recs = [{"memory_type": "fact", "semantic_text": t, "entities": [], "tags": [], "temporal": {}, "evidence_turns": [int(n)]}
                    for n, t in turns]
            return GenResult(json.dumps({"records": recs, "disambiguation_context": "ctx"}), 0, 1, 0.0)
        if "JSON plan" in system:
            q = user.split("<USER_QUERY>")[1].split("</USER_QUERY>")[0]
            return GenResult(json.dumps({"semantic_queries": [q, q + " born", q + " director"],
                                         "evidence_routes": [{"queries": [q], "temporal_filter": ""},
                                                             {"queries": [q + " director"], "temporal_filter": ""}]}), 0, 1, 0.0)
        return GenResult("ANS", 0, 1, 0.0)


def make(gen, tmp_path, **margs):
    ex, docs, _ = load_examples(DATA, "2wiki", limit=5)
    st = make_store(docs, "sentence")
    cfg = RunConfig(system="lychee", prepend_title=True, method_args={"cache_dir": str(tmp_path), **margs})
    ret = HybridRetriever(st.units, HashEmbedder(), 0.5, prepend_title=True)
    return Ly.LycheeMemory(st, ret, OverlapReranker(), gen, TokenCounter(), cfg), ex[0]


def test_memory_records_nodes_and_cache(tmp_path):
    g = Scripted(); p, ex = make(g, tmp_path); stats = p.build_index()
    assert stats["records"] > 0 and stats["fallback_segments"] == 0 and stats["segments"] >= stats["docs"] - 1
    r = p.mem.records[0]
    assert r["type"] == "fact" and r["sent_idx"] and p.mem.nodes[f"frame:{r['doc_id']}:{r['seg']}"]
    assert p.mem.by_sent[(r["doc_id"], r["sent_idx"][0])]
    n_calls = len(g.log); assert n_calls == stats["segments"]          # ONE consolidation call per segment
    g2 = Scripted(); p2, _ = make(g2, tmp_path); p2.build_index(); assert not g2.log


def test_broken_encoder_falls_back_to_verbatim_records(tmp_path):
    p, ex = make(Scripted(broken=True), tmp_path); s = p.build_index()
    assert s["fallback_segments"] == s["segments"] and s["records"] > 0


def test_pipeline_end_to_end_and_call_budget(tmp_path):
    g = Scripted(); p, ex = make(g, tmp_path); p.build_index(); g.log.clear(); led = Ledger()
    res = p.run(ex, led)
    assert res.prediction == "ANS" and res.extra["plan_ok"] and res.extra["n_routes"] == 2 and res.extra["k"] == 15
    assert led.calls["compiler"] == 1 and led.calls["generator"] == 1                  # one planner call + one answer call
    assert 0 < len(res.extra["records_in_prompt"]) <= 15 and res.selected and res.ranking
    user = [u for s, u in g.log if "memory assistant" in s][0]
    assert "# Episodic/Semantic Memories" in user and "less than 5-6 words" in user


def test_no_planner_variant_and_unparseable_plan_fall_back_to_the_question(tmp_path):
    g = Scripted(); p, ex = make(g, tmp_path, plan=False); led = Ledger()
    res = p.run(ex, led)
    assert led.calls["compiler"] == 0 and res.extra["n_routes"] == 1 and res.prediction == "ANS"


def test_cli_smoke_with_stub(tmp_path):
    cmd = [sys.executable, "-m", "registerrag.cli", "run", "--system", "lychee", "--dataset", "2wiki", "--data", DATA,
           "--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "c"), "--limit", "5", "--stub",
           "--method-arg", f"cache_dir={tmp_path / 'm'}"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr[-2000:]
    run_dir = next((tmp_path / "runs").iterdir())
    recs = [json.loads(l) for l in open(run_dir / "results.jsonl", encoding="utf-8")]
    assert len(recs) == 5 and all(x["error"] is None for x in recs)
