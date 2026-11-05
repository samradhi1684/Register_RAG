import json, os, subprocess, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
D = os.path.join(ROOT, "datasets")
DATA = {"2wiki": os.path.join(D, "wiki2_data/2wiki_100.json"),
        "hotpot": os.path.join(D, "hotpotqa_data/hotpot_100.json"),
        "musique": os.path.join(D, "musique_data/musique_100.json")}

from registerrag.config import RunConfig
from registerrag.data import load_examples
from registerrag.generate import TokenCounter, build_prompt
from registerrag.prompts import QA_SYSTEMS
from registerrag.retrieval import HashEmbedder, HybridRetriever
from registerrag.units import make_store, doc_from_native_sentences


@pytest.mark.parametrize("ds", ["2wiki", "hotpot"])
def test_native_gold_sentences_map_exactly(ds):
    """Fix for the pysbd re-split problem: every gold sentence must exist as a unit with the native index."""
    ex, docs, _ = load_examples(DATA[ds], ds)
    store = make_store(docs, "sentence")
    total = sum(len(e.gold_unit_ids) for e in ex)
    assert total > 200
    for e in ex:
        for uid in e.gold_unit_ids:
            assert uid in store.by_id, uid


def test_offsets_roundtrip():
    d = doc_from_native_sentences("T", [" Hello world. ", "Second one!", "Third."])
    for s, (a, b) in zip(d.sentences, d.sent_offsets):
        assert d.text[a:b] == s


def test_chunk_covers_sentences():
    ex, docs, _ = load_examples(DATA["2wiki"], "2wiki")
    st = make_store(docs, "chunk", 256, 50)
    gold = {u for e in ex for u in e.gold_unit_ids}
    cov = set().union(*[u.covered_ids() for u in st.units])
    assert len(gold & cov) / len(gold) > 0.95      # nearly every gold sentence lies wholly in some chunk


def test_musique_requirements_have_gold_paragraph():
    ex, _, _ = load_examples(DATA["musique"], "musique")
    reqs = [r for e in ex for r in e.gold_requirements]
    assert all(r.gold_doc_ids for r in reqs)
    assert any(r.depends_on for r in reqs)


def test_hybrid_lambda_extremes():
    ex, docs, _ = load_examples(DATA["2wiki"], "2wiki", limit=20)
    st = make_store(docs, "sentence")
    for lam in (0.0, 0.5, 1.0):
        r = HybridRetriever(st.units, HashEmbedder(), lam)
        hits = r.retrieve(ex[0].question, 5)
        assert len(hits) == 5 and hits[0].score >= hits[-1].score
    scoped = r.retrieve(ex[0].question, 5, st.allowed_for(ex[0].doc_ids))
    assert all(st.units[h.idx].doc_id in set(ex[0].doc_ids) for h in scoped)


def test_budget_is_hard():
    ex, docs, _ = load_examples(DATA["2wiki"], "2wiki", limit=5)
    st, c = make_store(docs, "sentence"), TokenCounter()
    from registerrag.prompts import SYSTEM_QA
    for b in (512, 1024):
        user, kept, trunc = build_prompt(c, SYSTEM_QA, ex[0].question, st.units[:400], lambda u: u.text, b)
        assert c.count(SYSTEM_QA) + c.count(user) <= b + 8 * len(kept) and trunc and kept


@pytest.mark.parametrize("system,ds", [("sentence", "2wiki"), ("chunk", "2wiki"), ("iterative", "musique"),
                                       ("sentence", "hotpot"), ("none", "2wiki")])
def test_smoke_stub_end_to_end(tmp_path, system, ds):
    out = tmp_path / "runs"
    cmd = [sys.executable, "-m", "registerrag.cli", "run", "--system", system, "--dataset", ds, "--data",
           DATA[ds], "--out", str(out), "--cache-dir", str(tmp_path / "c"), "--limit", "10", "--stub",
           "--budget", "1024"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr
    run_dir = next(out.iterdir())
    recs = [json.loads(l) for l in open(run_dir / "results.jsonl")]
    assert len(recs) == 10
    need = {"qid", "em", "f1", "prediction", "peak_prompt_tokens", "total_tokens_processed", "calls",
            "latency_s", "gpu_peak_mb", "failure_category", "retrieval"}
    for rec in recs:
        assert need <= rec.keys()
        assert {"retriever", "reranker", "verifier", "compiler", "generator"} <= rec["calls"].keys()
        assert rec["peak_prompt_tokens"] <= 1024 + 40
    if system == "iterative":
        assert recs[0]["calls"]["compiler"] == 1
    s = json.load(open(run_dir / "summary.json"))
    assert s["n"] == 10 and "config_hash" in s


def test_config_hash_stable_and_sensitive():
    a, b = RunConfig(), RunConfig()
    assert a.hash("x") == b.hash("x")
    b.lam = 0.7
    assert a.hash("x") != b.hash("x")


def test_limit_changes_run_hash():
    a, b = RunConfig(limit=20), RunConfig(limit=100)
    assert a.hash("x") != b.hash("x")


def test_prompt_versions_differ_and_change_hash():
    assert "unanswerable" in QA_SYSTEMS["v1"] and "unanswerable" not in QA_SYSTEMS["v2"]
    assert RunConfig(prompt_version="v1").hash("x") != RunConfig(prompt_version="v2").hash("x")


def test_failures_file_is_utf8(tmp_path):
    import json
    rec = {"qid": "1", "question": "Q \u0159", "gold_answers": ["a"], "prediction": "b", "failure_category": "X",
           "selected": [{"unit_id": "u", "text": "Ond\u0159ej"}]}
    p = tmp_path / "r.jsonl"; p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    out = tmp_path / "f.md"
    subprocess.run([sys.executable, "-m", "registerrag.cli", "sample-failures", str(p), "--out", str(out)],
                   cwd=ROOT, check=True, env={**os.environ, "PYTHONPATH": ROOT, "PYTHONUTF8": "0"})
    assert "Ond\u0159ej" in out.read_text(encoding="utf-8")


def test_resume_survives_corrupt_trailing_line(tmp_path):
    cmd = [sys.executable, "-m", "registerrag.cli", "run", "--system", "sentence", "--dataset", "2wiki", "--data",
           DATA["2wiki"], "--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "c"), "--limit", "5", "--stub"]
    env = {**os.environ, "PYTHONPATH": ROOT}
    assert subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True).returncode == 0
    f = next((tmp_path / "runs").iterdir()) / "results.jsonl"
    with open(f, "a", encoding="utf-8") as fh:
        fh.write('{"qid": "half-written", "em": ')              # simulate a crash in the middle of a write
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len([l for l in open(f, encoding="utf-8") if l.strip()]) == 5
    assert "env" in json.load(open(f.parent / "summary.json", encoding="utf-8"))
