import json, os, subprocess, sys
import numpy as np
import pytest

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
DATA = os.path.join(ROOT, "datasets", "wiki2_data", "2wiki_100.json")

from baselines.b3_georag import georag as G
from registerrag.config import RunConfig
from registerrag.data import load_examples
from registerrag.generate import GenResult, TokenCounter
from registerrag.ledger import Ledger
from registerrag.retrieval import HashEmbedder, HybridRetriever
from registerrag.units import make_store


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_sinkhorn_identical_measures_cost_zero_and_known_distance():
    E = np.stack([unit([1, 0]), unit([0, 1])])
    M = 1.0 - E @ E.T
    p = np.array([0.5, 0.5])
    same = G.sinkhorn_cost(M[None], p, np.array([[0.5, 0.5]]), eps=0.01, iters=100)[0]
    assert same < 1e-3
    moved = G.sinkhorn_cost(M[None][:, :, :1], p, np.array([[1.0]]), eps=0.01, iters=100)[0]
    assert moved == pytest.approx(0.5, abs=1e-3)      # half the mass sits at cos-distance 1 from the single support point


def test_sinkhorn_is_stable_with_zero_marginals_and_batches():
    M = np.random.default_rng(0).uniform(0, 2, (3, 6, 4))
    p = np.array([0.5, 0.5, 0, 0, 0, 0])
    out = G.sinkhorn_cost(M, p, np.full((3, 4), 0.25), 0.01, 100)
    assert np.all(np.isfinite(out)) and out.shape == (3,)


def test_greedy_prefers_covering_both_demand_peaks_over_a_near_duplicate():
    # two demand peaks A and B; candidates: a1 (top by cosine to Q), a2 ~ a1 (duplicate), b1 (second peak)
    E = np.stack([unit([1, 0.05]), unit([1, 0.06]), unit([0.05, 1]), unit([0.7, 0.7])])
    P = np.array([0.35, 0.15, 0.35, 0.15])
    q = unit([1, 0.2])
    sel = G.greedy_select(E, q, P, k=2)
    assert sel[0] in (0, 1) and sel[1] in (2, 3)          # one chunk per demand peak, not both near-duplicates


def test_greedy_k_larger_than_pool_and_empty():
    E = np.stack([unit([1, 0]), unit([0, 1])])
    assert sorted(G.greedy_select(E, unit([1, 1]), np.array([.5, .5]), k=5)) == [0, 1]
    assert G.greedy_select(E, unit([1, 1]), np.array([.5, .5]), k=0) == []


def test_maxmin_filter_keeps_dispersed_subqueries():
    Q = np.stack([unit([1, 0]), unit([1, 0.01]), unit([0, 1]), unit([-1, 0])])
    keep = G.maxmin_filter(Q, unit([1, 0]), 3)
    assert 1 not in keep and len(keep) == 3 and keep[0] == 0
    assert G.maxmin_filter(Q, unit([1, 0]), 10) == [0, 1, 2, 3]


def test_demand_distribution_is_a_distribution_and_uses_quality_weights():
    E = np.stack([unit([1, 0]), unit([0, 1]), unit([1, 1])])
    Qs = np.stack([unit([1, 0]), unit([0, 1])])
    P, qt = G.demand_distribution(np.zeros(3), E, Qs, np.array([3.0, 1.0]))
    assert P.sum() == pytest.approx(1.0) and qt.tolist() == [0.75, 0.25] and P[0] > P[1]


class Scripted:
    name = "scripted"
    temperature = 0.0

    def __init__(self, subs):
        self.subs, self.temps = subs, []

    def generate(self, system, user, json_mode=False, max_new_tokens=None):
        if "search sub-queries" in system:
            self.temps.append(self.temperature)
            return GenResult(json.dumps({"sub_queries": self.subs}), 0, 1, 0.0)
        return GenResult("ANS", 0, 1, 0.0)


def make(gen, **margs):
    ex, docs, _ = load_examples(DATA, "2wiki", limit=10)
    st = make_store(docs, "sentence")
    cfg = RunConfig(system="georag", method_args=margs)
    ret = HybridRetriever(st.units, HashEmbedder(), 0.5)
    return G.GeoRAG(st, ret, None, gen, TokenCounter(), cfg), ex[0]


def test_pipeline_selects_top_k_from_pool_and_uses_t07_for_subqueries():
    base = Scripted([f"sub {i} " + w for i, w in enumerate(["director", "born", "film", "award", "spouse"] * 4)])
    p, ex = make(base, pool_k=60, tau_rv=0.0)
    led = Ledger(); res = p.run(ex, led)
    assert res.prediction == "ANS" and len(res.selected) == 5 and len(set(u.unit_id for u in res.selected)) == 5
    assert base.temps == [0.7] and base.temperature == 0.0       # the sub-query copy samples at 0.7, the original is untouched ...
    assert p.subq_gen.temperature == 0.7 and p.subq_gen is not base
    assert res.extra["pool_size"] == 60 and len(res.extra["sub_queries"]) == 10
    assert [u.unit_id for u in res.ranking[:5]] == [u.unit_id for u in res.selected]
    assert led.calls["retriever"] == 1 + 10 and led.calls["compiler"] == 1 and led.calls["generator"] == 1


def test_failed_subquery_generation_falls_back_without_crashing():
    p, ex = make(Scripted([]), pool_k=30)
    res = p.run(ex, Ledger())
    assert res.extra["fallback_demand"] == "question only" and len(res.selected) == 5


def test_cli_smoke_with_stub(tmp_path):
    cmd = [sys.executable, "-m", "registerrag.cli", "run", "--system", "georag", "--dataset", "2wiki", "--data", DATA,
           "--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "c"), "--limit", "6", "--stub"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr[-2000:]
    run_dir = next((tmp_path / "runs").iterdir())
    recs = [json.loads(l) for l in open(run_dir / "results.jsonl", encoding="utf-8")]
    assert len(recs) == 6 and all(x["error"] is None for x in recs)
