import json, os, sys

import pytest

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
DATA = os.path.join(ROOT, "datasets", "wiki2_data", "2wiki_100.json")
FAKE = os.path.join(os.path.dirname(__file__), "fake_worker.py")

from baselines.b7b_lychee_acl import lychee_acl as L
from registerrag.config import RunConfig
from registerrag.data import load_examples
from registerrag.generate import TokenCounter
from registerrag.ledger import Ledger
from registerrag.rerank import OverlapReranker
from registerrag.retrieval import HashEmbedder, HybridRetriever
from registerrag.units import make_store


def make(mode="ok", marker=""):
    ex, docs, _ = load_examples(DATA, "2wiki", limit=3)
    st = make_store(docs, "chunk")
    cmd = json.dumps([sys.executable, FAKE, mode, marker])
    cfg = RunConfig(system="lycheeacl", prepend_title=True, method_args={"worker_cmd": cmd})
    ret = HybridRetriever(st.units, HashEmbedder(), 0.5, prepend_title=True)
    L.LycheeACL._worker = None
    return L.LycheeACL(st, ret, OverlapReranker(), None, TokenCounter(), cfg), ex


def test_extract_answer():
    assert L.extract_answer("so...\n\\boxed{Paris}") == "Paris"
    assert L.extract_answer("\\boxed{a} then \\boxed{b {c}}") == "b {c}"
    assert L.extract_answer("Answer: Paris\nmore") == "Paris"
    assert L.extract_answer("  Paris  ") == "Paris" and L.extract_answer("") == ""


def test_context_layout_matches_authors_test_json():
    p, ex = make()
    docs = [p.store.docs[d] for d in ex[0].doc_ids]
    ctx = L.build_context(docs)
    assert ctx.startswith("Document 1:\n" + docs[0].title + "\n") and f"Document {len(docs)}:" in ctx


def test_pipeline_ok_and_noise_is_ignored():
    p, ex = make("ok"); led = Ledger(); res = p.run(ex[0], led)
    assert res.prediction == "Paris" and res.extra["raw_output"].startswith("Answer:")
    assert led.calls["generator"] == 1 and led.peak_prompt_tokens > 50 and res.extra["worker_peak_mem_mb"] == 1234
    assert res.ranking == [] and res.selected == []


def test_boxed_variant():
    p, ex = make("boxed"); assert p.run(ex[0], Ledger()).prediction == "Paris"


def test_worker_error_becomes_a_logged_question_error():
    p, ex = make("err")
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        p.run(ex[0], Ledger())


def test_dead_worker_is_restarted_once(tmp_path):
    p, ex = make("die_once", str(tmp_path / "m"))
    res = p.run(ex[0], Ledger())
    assert res.prediction == "Paris"


def test_cli_end_to_end_with_fake_worker(tmp_path):
    import subprocess
    cmd = [sys.executable, "-m", "registerrag.cli", "run", "--system", "lycheeacl", "--dataset", "2wiki", "--data", DATA,
           "--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "c"), "--limit", "3", "--stub",
           "--method-arg", "worker_cmd=" + json.dumps([sys.executable, FAKE, "ok"])]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr[-2000:]
    run_dir = next((tmp_path / "runs").iterdir())
    recs = [json.loads(l) for l in open(run_dir / "results.jsonl", encoding="utf-8")]
    assert len(recs) == 3 and all(x["error"] is None and x["prediction"] == "Paris" for x in recs)
