import json, os, re, subprocess, sys
import numpy as np
import pytest

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
DATA = os.path.join(ROOT, "datasets", "wiki2_data", "2wiki_100.json")

from baselines.b6_evimem import evimem as E
from registerrag.config import RunConfig
from registerrag.data import load_examples
from registerrag.generate import GenResult, TokenCounter
from registerrag.ledger import Ledger
from registerrag.retrieval import HashEmbedder, HybridRetriever
from registerrag.units import make_store

EXACT = "EXACT: yes\nINFERRABLE: no\nPARTIAL: no\nCONFIDENCE: 0.9\nMISSING: none"
PARTIAL = "EXACT: no\nINFERRABLE: no\nPARTIAL: yes\nCONFIDENCE: 0.4\nMISSING: the birth year of the director"


class Scripted:
    name = "scripted"

    def __init__(self, judges, extract_ok=True):
        self.judges, self.extract_ok, self.n_judge, self.log = judges, extract_ok, 0, []

    def generate(self, system, user, json_mode=False, max_new_tokens=None):
        self.log.append((system, user))
        if "factorize text" in system:
            if not self.extract_ok:
                return GenResult("garbage", 0, 1, 0.0)
            sents = re.findall(r"^(\d+)\. (.*)$", user, re.M)
            facts = [{"sent": int(n), "subject": "Ent" + n, "predicate": "is", "object": " ".join(s.split()[:3]), "time": ""}
                     for n, s in sents]
            return GenResult(json.dumps({"facts": facts}), 0, 1, 0.0)
        if "evaluates information sufficiency" in system:
            out = self.judges[min(self.n_judge, len(self.judges) - 1)]; self.n_judge += 1
            return GenResult(out, 0, 1, 0.0)
        if "refines search queries" in system:
            return GenResult('"improved query"', 0, 1, 0.0)
        if "analyzes reasoning requirements" in system:
            return GenResult("TYPE: DIRECT", 0, 1, 0.0)
        return GenResult("ANS", 0, 1, 0.0)


def make(gen, tmp_path, scope="pooled", **margs):
    ex, docs, _ = load_examples(DATA, "2wiki", limit=4)
    st = make_store(docs, "sentence")
    cfg = RunConfig(system="evimem", method_args={"cache_dir": str(tmp_path), **margs}, corpus_scope=scope)
    ret = HybridRetriever(st.units, HashEmbedder(), 0.5)
    return E.EviMem(st, ret, None, gen, TokenCounter(), cfg), ex[0], st


# ---------------------------------------------------------------------------------------------------- pure helpers
def test_parse_sufficiency_handles_both_label_styles_and_percent_confidence():
    v = E.parse_sufficiency(EXACT)
    assert v["exact"] and not v["inferrable"] and v["confidence"] == 0.9 and v["missing"] == "" and v["parse_ok"]
    v = E.parse_sufficiency("EXACT_MATCH: no\nINFERRABLE: YES\nPARTIAL: no\nCONFIDENCE: 80\nMISSING: Birth date of X")
    assert v["inferrable"] and v["confidence"] == 0.8 and v["missing"] == "Birth date of X"
    assert not E.parse_sufficiency("nonsense")["parse_ok"]


def test_temporal_detection_and_calibration():
    assert E.is_temporal("When was he born?") and E.is_temporal("What year did it open?")
    assert not E.is_temporal("Who directed the film?")
    v = {"exact": True, "inferrable": False, "partial": False, "confidence": 0.6, "missing": ""}
    assert E.calibrate(v, True)["confidence"] == 0.85 and E.calibrate(v, False)["confidence"] == 0.6
    assert E.calibrate({**v, "exact": False, "inferrable": True, "confidence": 0.9}, True)["confidence"] == 0.75
    assert E.calibrate({**v, "exact": False, "partial": True, "confidence": 0.9}, True)["confidence"] == 0.5


def test_stopping_rule_matches_the_released_code():
    ex_ = {"exact": True, "inferrable": False, "partial": False, "confidence": 0.7}
    assert E.should_stop(ex_, temporal=False) and not E.should_stop(ex_, temporal=True)         # 0.7 vs 0.85
    inf = {"exact": False, "inferrable": True, "partial": False, "confidence": 0.7}
    assert E.should_stop(inf, temporal=True) and not E.should_stop({**inf, "confidence": 0.69}, False)


def test_answer_instruction_tiers():
    v = {"exact": False, "inferrable": False, "partial": False, "confidence": 0.0}
    assert E.answer_instruction({**v, "exact": True}, False) == E.INSTR_CONFIDENT
    assert E.answer_instruction({**v, "inferrable": True, "confidence": 0.8}, False) == E.INSTR_CONFIDENT
    assert E.answer_instruction({**v, "inferrable": True, "confidence": 0.6}, False) == E.INSTR_INFER
    assert E.answer_instruction(v, False) == E.INSTR_ELSE
    assert E.answer_instruction({**v, "exact": True}, True) == E.INSTR_TEMPORAL_EXACT
    assert E.answer_instruction(v, True) == E.INSTR_TEMPORAL_ELSE


def test_not_mentioned_rule_is_logged_logic_only():
    v = {"exact": False, "inferrable": False, "partial": False, "confidence": 0.5}
    assert E.would_say_not_mentioned(v, 2, 10, 3) and not E.would_say_not_mentioned(v, 1, 10, 3)
    assert E.would_say_not_mentioned({**v, "partial": True, "confidence": 0.1}, 3, 2, 3)


def test_fact_string():
    f = {"subject": "A", "predicate": "born in", "object": "Paris", "time": "1900", "fallback": False}
    assert E.fact_string(f) == "A born in Paris (time: 1900)"
    assert E.fact_string({"fallback": True, "text": "T: raw"}) == "T: raw"


# ------------------------------------------------------------------------------------------------------- memory
def test_index_edges_and_cache_resume(tmp_path):
    g = Scripted([EXACT]); p, ex, st = make(g, tmp_path)
    stats = p.build_index()
    assert stats["tuples"] > 0 and stats["fallback_sentences"] == 0 and stats["docs"] == len(st.docs)
    mem = p.mem
    # every tuple of one sentence is linked with weight 1.0 (same_raw)
    by = {}
    for i, f in enumerate(mem.facts):
        by.setdefault(f["unit_id"], []).append(i)
    multi = next((ids for ids in by.values() if len(ids) > 1), None)
    if multi:
        assert mem.adj[multi[0]][multi[1]] == 1.0
    calls = len([1 for s, _ in g.log if "factorize" in s])
    g2 = Scripted([EXACT]); p2, _, _ = make(g2, tmp_path); p2.build_index()
    assert not [1 for s, _ in g2.log if "factorize" in s] and calls > 0     # second build is served from the cache


def test_failed_extraction_never_loses_a_sentence(tmp_path):
    g = Scripted([EXACT], extract_ok=False); p, ex, st = make(g, tmp_path)
    stats = p.build_index()
    assert stats["tuples"] == 0 and stats["fallback_sentences"] == len(st.units) and stats["failed_calls"] > 0


def test_search_falls_back_from_embedding_to_bm25(tmp_path):
    g = Scripted([EXACT]); p, ex, st = make(g, tmp_path); p.build_index()
    assert p.mem.search(ex.question, 5, thr=-1.0)                      # embedding path
    assert len(p.mem.search(ex.question, 5, thr=2.0)) > 0              # threshold unreachable -> BM25 path


# ------------------------------------------------------------------------------------------------------ pipeline
def test_stops_at_first_iteration_when_sufficient(tmp_path):
    g = Scripted([EXACT]); p, ex, st = make(g, tmp_path); led = Ledger()
    res = p.run(ex, led)
    assert res.prediction == "ANS" and res.extra["stopped_at"] == 1 and len(res.extra["iterations"]) == 1
    assert led.calls["verifier"] == 1 and led.calls["generator"] == 1 and led.calls["compiler"] == 0
    assert res.selected and res.ranking


def test_insufficient_triggers_refinement_up_to_three_iterations_then_answers(tmp_path):
    g = Scripted([PARTIAL]); p, ex, st = make(g, tmp_path); led = Ledger()
    res = p.run(ex, led)
    assert res.extra["stopped_at"] == 3 and led.calls["verifier"] == 3
    assert [t["query"] for t in res.extra["iterations"]][1:] == ["improved query", "improved query"]
    refine_users = [u for s, u in g.log if "refines search queries" in s]
    assert len(refine_users) == 2 and "Missing information: the birth year of the director" in refine_users[0]
    assert "Iteration: 1/3" in refine_users[0] and "Strategy: Focus on specific keywords" in refine_users[0]
    assert res.extra["would_abstain"] is False and res.prediction == "ANS"        # forced answer policy


def test_iteration_budgets_follow_the_code(tmp_path):
    g = Scripted([PARTIAL]); p, ex, st = make(g, tmp_path, embedding_threshold=-1.0); res = p.run(ex, Ledger())
    its = res.extra["iterations"]
    assert [t["n_refined"] for t in its] == [10, 13, 16] and [t["n_anchor"] for t in its] == [5, 6, 8]


def test_judge_sees_at_most_20_facts_and_answer_30(tmp_path):
    g = Scripted([PARTIAL]); p, ex, st = make(g, tmp_path); res = p.run(ex, Ledger())
    judge_user = next(u for s, u in g.log if "evaluates information sufficiency" in s)
    assert judge_user.count("\n- ") <= 20 and len(res.extra["answer_facts"]) <= 30


def test_raw_variant_grounds_answer_in_sentences(tmp_path):
    g = Scripted([EXACT]); p, ex, st = make(g, tmp_path, answer_context="raw"); res = p.run(ex, Ledger())
    ans_user = next(u for s, u in g.log if "answers questions based on provided facts" in s)
    assert res.extra["answer_context"] == "raw" and "Ent" not in ans_user.split("Relevant Facts:")[1]


def test_short_answer_note_is_appended_by_default_and_can_be_disabled(tmp_path):
    g = Scripted([EXACT]); p, ex, st = make(g, tmp_path); p.run(ex, Ledger())
    assert "shortest possible answer" in next(u for s, u in g.log if "answers questions" in s)
    g = Scripted([EXACT]); p, ex, st = make(g, tmp_path / "b", short_answer=False); p.run(ex, Ledger())
    assert "shortest possible answer" not in next(u for s, u in g.log if "answers questions" in s)


def test_budget_truncates_facts(tmp_path):
    g = Scripted([PARTIAL]); p, _, _ = make(g, tmp_path)
    ex, docs, _ = load_examples(DATA, "2wiki", limit=4)
    p.cfg.budget_tokens = 330
    res = p.run(ex[0], Ledger())
    assert res.truncated and len(p.counter.count(g.log[-1][1]) * [0]) <= 330 + 400


def test_cli_smoke_with_stub(tmp_path):
    cmd = [sys.executable, "-m", "registerrag.cli", "run", "--system", "evimem", "--dataset", "2wiki", "--data", DATA,
           "--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "c"), "--limit", "5", "--stub",
           "--method-arg", f"cache_dir={tmp_path / 'idx'}"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr[-2000:]
    run_dir = next((tmp_path / "runs").iterdir())
    recs = [json.loads(l) for l in open(run_dir / "results.jsonl", encoding="utf-8")]
    assert len(recs) == 5 and all(x["error"] is None for x in recs) and all(x["extra"]["iterations"] for x in recs)
