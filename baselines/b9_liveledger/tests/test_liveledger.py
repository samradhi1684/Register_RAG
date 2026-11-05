import json, os, re, subprocess, sys
import pytest

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
DATA = os.path.join(ROOT, "datasets", "wiki2_data", "2wiki_100.json")

from baselines.b9_liveledger import liveledger as L
from registerrag.config import RunConfig
from registerrag.data import load_examples
from registerrag.generate import GenResult, TokenCounter
from registerrag.ledger import Ledger
from registerrag.retrieval import HashEmbedder, HybridRetriever
from registerrag.units import make_store

CONS = json.dumps({"constraint_1": {"constraint": "A must be X"}, "constraint_2": {"constraint": "A must be Y"}})


class Scripted:
    name = "scripted"

    def __init__(self, actions, fake_quote=False):
        self.actions, self.i, self.fake_quote, self.log = actions, 0, fake_quote, []

    def generate(self, system, user, json_mode=False, max_new_tokens=None):
        self.log.append((system, user))
        if "extract constraints" in system:
            return GenResult(CONS, 0, 1, 0.0)
        if "answer a question by searching" in system:
            a = self.actions[min(self.i, len(self.actions) - 1)]; self.i += 1
            return GenResult(json.dumps(a), 0, 1, 0.0)
        if "updates an epistemic ledger" in system:
            res = user.rsplit("**Search Results:**", 1)[1].split("### Output", 1)[0]
            first = re.search(r"\[1\] (.{20})", res)
            quote = "this quote is not in the results" if self.fake_quote else (first.group(1) if first else "x")
            return GenResult(json.dumps({"entries": [
                {"candidate": "Cand A", "constraint": "C1", "obj": True, "obj_evidence": quote},
                {"candidate": "Cand A", "constraint": "C9", "obj": True, "obj_evidence": quote}]}), 0, 1, 0.0)
        return GenResult("FORCED", 0, 1, 0.0)


SEARCH = {"thought": "need A", "action": "search", "query": "who is A"}
ANSWER = {"thought": "done", "action": "answer", "answer": "Cand A"}


def make(gen, **margs):
    ex, docs, _ = load_examples(DATA, "2wiki", limit=6)
    st = make_store(docs, "sentence")
    cfg = RunConfig(system="liveledger", method_args=margs)
    ret = HybridRetriever(st.units, HashEmbedder(), 0.5)
    return L.LiveLedgerAgent(st, ret, None, gen, TokenCounter(), cfg), ex[0]


def test_prompts_are_the_papers():
    assert "externally verifiable constraints" in L.CONSTRAINT_PROMPT and "Call `update_ledger`" in L.UPDATE_PROMPT
    assert "Always include a verbatim quote" in L.UPDATE_PROMPT
    L.CONSTRAINT_PROMPT.format(question="q")                         # brace-escaping is valid
    L.UPDATE_PROMPT.format(question="q", constraints="c", ledger="l", thinking="t", search_query="s", retrieval_results="r")


def test_search_update_answer_flow_and_lag():
    g = Scripted([SEARCH, SEARCH, ANSWER]); p, ex = make(g); led = Ledger()
    res = p.run(ex, led)
    assert res.prediction == "Cand A" and res.extra["n_searches"] == 2 and not res.extra["fallback"]
    assert res.extra["constraints"] == {"C1": "A must be X", "C2": "A must be Y"}
    assert led.calls["compiler"] == 1 and led.calls["verifier"] == 2 and led.calls["generator"] == 3
    agent_users = [u for s, u in g.log if "answer a question by searching" in s]
    assert "(no candidate yet)" in agent_users[1]                    # step 2 sees the ledger through step 0 only (lag)
    assert "Cand A: C1=Satisfied" in agent_users[2]                  # step 3 sees the ledger updated through step 1
    fl = res.extra["final_ledger"]["Cand A"]
    assert fl["C1"]["obj"] is True and fl["C2"]["obj"] is None and "C9" not in fl
    assert res.extra["final_candidate"] == "Cand A" and res.extra["fully_verified"] is False


def test_no_lag_shows_the_latest_ledger():
    g = Scripted([SEARCH, ANSWER]); p, ex = make(g, lag=False); p.run(ex, Ledger())
    assert "Cand A: C1=Satisfied" in [u for s, u in g.log if "answer a question by searching" in s][1]


def test_quote_guard_refuses_unverbatim_evidence():
    g = Scripted([SEARCH, ANSWER], fake_quote=True); p, ex = make(g); res = p.run(ex, Ledger())
    assert res.extra["n_quote_guard_rejected"] >= 1 and res.extra["final_ledger"]["Cand A"]["C1"]["obj"] is None
    g = Scripted([SEARCH, ANSWER], fake_quote=True); p, ex = make(g, quote_guard=False); res = p.run(ex, Ledger())
    assert res.extra["final_ledger"]["Cand A"]["C1"]["obj"] is True


def test_ledger_off_is_a_plain_search_agent():
    g = Scripted([SEARCH, ANSWER]); p, ex = make(g, ledger=False); led = Ledger()
    res = p.run(ex, led)
    assert res.prediction == "Cand A" and led.calls["verifier"] == 0 and led.calls["compiler"] == 0
    assert not any("Constraint ledger" in u for s, u in g.log) and "final_ledger" not in res.extra


def test_step_budget_exhaustion_uses_the_forced_answer():
    g = Scripted([SEARCH]); p, ex = make(g, max_steps=2); res = p.run(ex, Ledger())
    assert res.extra["fallback"] is True and res.prediction == "FORCED" and res.extra["n_searches"] == 2 and res.selected


def test_unparseable_agent_output_falls_back_without_crashing():
    class Bad(Scripted):
        def generate(self, system, user, json_mode=False, max_new_tokens=None):
            if "answer a question by searching" in system:
                return GenResult("not json", 0, 1, 0.0)
            return super().generate(system, user, json_mode, max_new_tokens)
    p, ex = make(Bad([SEARCH])); res = p.run(ex, Ledger())
    assert res.extra["fallback"] is True and res.extra["agent_parse_failures"] == 2 and res.prediction == "FORCED"


def test_parsers():
    assert L.parse_constraints(CONS, "q") == ({"C1": "A must be X", "C2": "A must be Y"}, True)
    assert L.parse_constraints("junk", "my q") == ({"C1": "my q"}, False)
    assert L.parse_json_obj('x {"a": 1} y') == {"a": 1} and L.parse_json_obj("nope") is None


def test_cli_smoke_with_stub(tmp_path):
    cmd = [sys.executable, "-m", "registerrag.cli", "run", "--system", "liveledger", "--dataset", "2wiki", "--data", DATA,
           "--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "c"), "--limit", "5", "--stub"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr[-2000:]
    run_dir = next((tmp_path / "runs").iterdir())
    recs = [json.loads(l) for l in open(run_dir / "results.jsonl", encoding="utf-8")]
    assert len(recs) == 5 and all(x["error"] is None for x in recs)
