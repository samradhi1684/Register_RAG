import json, os, subprocess, sys
import numpy as np
import pytest

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
DATA = os.path.join(ROOT, "datasets", "wiki2_data", "2wiki_100.json")

from baselines.b5_sure_rag import sure_rag as S
from registerrag.config import RunConfig
from registerrag.data import load_examples
from registerrag.generate import GenResult, TokenCounter
from registerrag.ledger import Ledger
from registerrag.retrieval import HashEmbedder, HybridRetriever
from registerrag.units import make_store


def test_features_follow_table_one_blocks():
    P = np.array([[0.8, 0.1, 0.1], [0.2, 0.1, 0.7]])
    f = S.features(P)
    assert f["max_sup"] == 0.8 and f["max_ref"] == pytest.approx(0.1) and f["mean_neu"] == pytest.approx(0.4)
    assert f["conflict"] == pytest.approx(0.1) and f["coverage_deficit"] == pytest.approx(0.2)
    assert f["disagreement"] == pytest.approx(0.3) and f["frac_supported"] == 1.0
    assert set(f) == set(S.FEATURES)
    empty = S.features(np.zeros((0, 3)))
    assert empty["frac_insufficient"] == 1.0 and empty["max_sup"] == 0.0


def test_conflict_is_high_when_support_and_refutation_both_present():
    f = S.features(np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]))
    assert f["conflict"] == pytest.approx(0.9)


def test_aggregator_heuristic_and_file(tmp_path):
    f = S.features(np.array([[0.9, 0.05, 0.05]]))
    p, u, s = S.Aggregator().score(f)
    assert p == pytest.approx(0.9 * 0.95) and s == p
    d = {"features": S.FEATURES, "mean": [0] * 10, "scale": [1] * 10, "coef": [4] + [0] * 9, "intercept": -2.0,
         "tau": 0.7, "beta": 0.5}
    path = tmp_path / "agg.json"
    path.write_text(json.dumps(d))
    agg = S.Aggregator(str(path))
    p, u, s = agg.score(f)
    assert p == pytest.approx(1 / (1 + np.exp(-(4 * 0.9 - 2.0)))) and agg.tau == 0.7 and s == pytest.approx(p - 0.5 * u)
    assert S.Aggregator(str(path), tau=0.1).tau == 0.1


class Gen:
    name = "gen"

    def generate(self, system, user, json_mode=False, max_new_tokens=None):
        if "declarative sentence" in system:
            return GenResult("The claim is true.", 0, 1, 0.0)
        return GenResult("ANS", 0, 1, 0.0)


class FixedVerifier:
    name = "fixed"

    def __init__(self, probs):
        self.p = probs

    def probs(self, claim, passages):
        return np.tile(np.array(self.p), (len(passages), 1))


def make(**margs):
    ex, docs, _ = load_examples(DATA, "2wiki", limit=4)
    st = make_store(docs, "chunk")
    cfg = RunConfig(system="surerag", method_args=margs)
    ret = HybridRetriever(st.units, HashEmbedder(), 0.5)
    return S.SureRAG(st, ret, None, Gen(), TokenCounter(), cfg), ex[0]


def test_supported_claim_is_answered_and_unsupported_is_withheld():
    p, ex = make()
    p.verifier = FixedVerifier([0.95, 0.02, 0.03])
    led = Ledger()
    res = p.run(ex, led)
    assert res.prediction == "ANS" and res.extra["decision"] == "answer" and len(res.extra["pair_probs"]) == 5
    assert led.calls["verifier"] == 5 and led.calls["compiler"] == 1 and led.calls["generator"] == 1
    p, ex = make()
    p.verifier = FixedVerifier([0.05, 0.05, 0.9])
    res = p.run(ex, Ledger())
    assert res.prediction == "" and res.extra["decision"] == "abstain" and res.extra["raw_answer"] == "ANS"


def test_gate_false_logs_scores_but_never_withholds():
    p, ex = make(gate=False)
    p.verifier = FixedVerifier([0.05, 0.05, 0.9])
    res = p.run(ex, Ledger())
    assert res.prediction == "ANS" and res.extra["decision"] == "abstain" and res.extra["gate"] is False


def test_cli_smoke_with_stub_and_selective_tool(tmp_path):
    cmd = [sys.executable, "-m", "registerrag.cli", "run", "--system", "surerag", "--dataset", "2wiki", "--data", DATA,
           "--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "c"), "--limit", "6", "--stub",
           "--method-arg", "verifier=stub"]
    env = {**os.environ, "PYTHONPATH": ROOT}
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    run_dir = next((tmp_path / "runs").iterdir())
    recs = [json.loads(l) for l in open(run_dir / "results.jsonl", encoding="utf-8")]
    assert len(recs) == 6 and all(x["error"] is None for x in recs) and all("p_safe" in x["extra"] for x in recs)
    tool = os.path.join(ROOT, "baselines", "b5_sure_rag", "tools", "selective.py")
    sel = subprocess.run([sys.executable, tool, str(run_dir / "results.jsonl")], cwd=ROOT, capture_output=True,
                         text=True, env=env)
    assert sel.returncode == 0 and "coverage" in sel.stdout, sel.stderr[-1500:]
