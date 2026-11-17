import json, os, re, subprocess, sys
import pytest

ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[3])
DATA = os.path.join(ROOT, "datasets", "wiki2_data", "2wiki_100.json")

from baselines.b8_pyrag.pyrag import (PlanRejected, PyRAG, extract_answer_tag, extract_code, extract_json_list,
                                      is_insufficient, run_limited, validate)
from registerrag.config import RunConfig
from registerrag.data import load_examples
from registerrag.generate import GenResult, TokenCounter
from registerrag.ledger import Ledger
from registerrag.retrieval import HashEmbedder, HybridRetriever
from registerrag.units import make_store

GOOD = ('d1 = retrieve("who is it")\n'
        'a1 = answer("who is it", d1)\n'
        'final_answer = answer("Given: " + a1 + ". Answer the question: Who is it?")')
DOCS_SYS, NODOCS_SYS = "You are given a question and retrieved documents", "There are NO retrieved documents"


class Scripted:
    """Role-aware fake LLM keyed on the authors' real prompts. plans = programs returned on successive plan/fix calls."""
    name = "scripted"

    def __init__(self, plans, docs_answer="Alice", need_docs=0, synth="Alice", decompose=None):
        self.plans, self.docs_answer, self.need_docs, self.synth = plans, docs_answer, need_docs, synth
        self.decompose = decompose if decompose is not None else '["sub one", "sub two"]'
        self.plan_calls, self.log = 0, []

    def generate(self, system, user, json_mode=False, max_new_tokens=None):
        self.log.append((system, user))
        if "question decomposition agent" in system:
            out = self.decompose
        elif "planning agent that writes Python code" in system:
            out = "```python\n" + self.plans[min(self.plan_calls, len(self.plans) - 1)] + "\n```"
            self.plan_calls += 1
        elif DOCS_SYS in system:
            n = len(re.findall(r"^\[Doc \d+\]", user, re.M))
            out = f"<thinking>t</thinking><answer>{self.docs_answer if n >= self.need_docs else 'unknown'}</answer>"
        elif NODOCS_SYS in system:
            out = f"<thinking>t</thinking><answer>{self.synth}</answer>"
        else:
            out = "FALLBACK"
        return GenResult(out, 0, 1, 0.0)


def make(gen, budget=None, **margs):
    ex, docs, _ = load_examples(DATA, "2wiki", limit=5)
    st = make_store(docs, "chunk")
    cfg = RunConfig(system="pyrag", method_args=margs, budget_tokens=budget)
    ret = HybridRetriever(st.units, HashEmbedder(), 0.5)
    return PyRAG(st, ret, None, gen, TokenCounter(), cfg), ex[0]


def doc_counts(g):
    return [len(re.findall(r"^\[Doc \d+\]", u, re.M)) for s, u in g.log if DOCS_SYS in s]


def test_happy_path_counts_calls_by_role():
    g = Scripted([GOOD]); p, ex = make(g); led = Ledger()
    res = p.run(ex, led)
    assert res.prediction == "Alice" and res.extra["repairs"] == [] and not res.extra["fallback"]
    assert res.extra["n_tool_calls"] == 3                       # retrieve, answer(docs), answer(synth)
    assert led.calls["retriever"] == 1 and led.calls["generator"] == 2 and led.calls["compiler"] == 2
    assert res.pool and res.selected and res.ranking and res.extra["sub_queries"] == ["sub one", "sub two"]


def test_prompts_are_the_authors_prompts():
    g = Scripted([GOOD]); p, ex = make(g); p.run(ex, Ledger())
    sys_plan = next(s for s, u in g.log if "planning agent" in s)
    assert "Do NOT pass a topk argument" in sys_plan and "TWO-PART format" in sys_plan
    user_plan = next(u for s, u in g.log if "planning agent" in s)
    assert "Sub-queries to resolve:\n['sub one', 'sub two']" in user_plan and "Reference example" in user_plan
    user_ans = next(u for s, u in g.log if DOCS_SYS in s)
    assert user_ans.startswith("=== QUESTION ===") and "=== RETRIEVED DOCUMENTS ===\n[Doc 1]\n" in user_ans


def test_self_repair_feeds_the_error_back_to_the_plan_agent():
    g = Scripted(['final_answer = answr("x")', GOOD]); p, ex = make(g)
    res = p.run(ex, Ledger())
    assert res.prediction == "Alice" and g.plan_calls == 2
    assert res.extra["repairs"][0].startswith("NameError")
    fix = [u for s, u in g.log if "planning agent" in s and "runtime error" in u]
    assert fix and "NameError" in fix[0] and "=== Failing code ===" in fix[0]


def test_syntax_error_is_fed_back_within_plan_generation():
    g = Scripted(["final_answer = (", GOOD]); p, ex = make(g)
    res = p.run(ex, Ledger())
    assert res.prediction == "Alice" and res.extra["repairs"] == [] and g.plan_calls == 2
    assert any("NOT valid Python" in u for s, u in g.log if "planning agent" in s)


def test_adaptive_retrieval_reruns_the_whole_program_with_topk_10_for_the_implicated_retrieve():
    g = Scripted([GOOD], need_docs=10, docs_answer="Bob", synth="Bob"); p, ex = make(g)
    res = p.run(ex, Ledger())
    assert res.extra["retried_with_topk"] is True and res.prediction == "Bob"
    assert doc_counts(g) == [5, 10]                               # first top-5, then the boosted top-10
    assert g.plan_calls == 1                                      # the program itself is NOT regenerated


def test_adaptive_retrieval_can_be_switched_off():
    g = Scripted([GOOD], need_docs=10, docs_answer="Bob", synth="Z"); p, ex = make(g, adaptive=False)
    assert p.run(ex, Ledger()).extra["retried_with_topk"] is False


def test_only_one_adaptive_retry():
    g = Scripted([GOOD], need_docs=99, synth="unknown"); p, ex = make(g)
    res = p.run(ex, Ledger())
    assert doc_counts(g) == [5, 10] and res.extra["fallback"] is True


def test_boost_map_rules():
    log = [{"type": "retrieve", "idx": 0, "topk": 5}, {"type": "answer", "out": "Unknown"},
           {"type": "retrieve", "idx": 1, "topk": 5}, {"type": "answer", "out": "Paris"}]
    assert PyRAG._boost_map(log, "Paris", 10) == {0: 10}
    assert PyRAG._boost_map(log[2:], "cannot answer", 10) == {1: 10}      # only the final answer is insufficient
    assert PyRAG._boost_map(log[:1], "Paris", 10) == {}


@pytest.mark.parametrize("bad", ["import os", "x = ().__class__", "f = open('a')", "eval('1')",
                                 "s = '{0.__class__}'.format(1)", "class A: pass", "y = getattr(1, 'real')"])
def test_sandbox_rejects_dangerous_programs(bad):
    with pytest.raises(PlanRejected):
        validate(bad)


def test_sandbox_allows_the_intended_surface_and_syntax_errors_propagate():
    validate('a = retrieve("q")\nb = answer(f"x {a[0]}", a)\nfinal_answer = b.strip().lower()')
    with pytest.raises(SyntaxError):
        validate("final_answer = (")


def test_infinite_loop_is_stopped():
    with pytest.raises(TimeoutError):
        run_limited("while True:\n    pass", {"__builtins__": {}}, max_lines=200)


def test_rejected_program_is_repaired_not_executed():
    g = Scripted(["import os\nfinal_answer = os.getcwd()", GOOD]); p, ex = make(g)
    res = p.run(ex, Ledger())
    assert res.prediction == "Alice" and res.extra["repairs"][0].startswith("PlanRejected")


def test_final_answer_never_assigned_falls_back_without_repair():
    g = Scripted(["x = 1"]); p, ex = make(g)
    res = p.run(ex, Ledger())
    assert res.extra["fallback"] is True and res.prediction == "FALLBACK"
    assert res.extra["program_error"] == "final_answer was never assigned" and res.selected


def test_repeated_runtime_errors_stop_after_three_fix_rounds_then_fall_back():
    g = Scripted(["final_answer = nope"]); p, ex = make(g)
    res = p.run(ex, Ledger())
    assert len(res.extra["repairs"]) == 4 and g.plan_calls == 4     # 1 plan + 3 fixes, 4 executions
    assert res.extra["fallback"] is True and res.prediction == "FALLBACK"


def test_token_budget_caps_every_answer_prompt():
    prog = 'd = retrieve("q", 20)\nfinal_answer = answer("q", d)'
    g = Scripted([prog]); p, ex = make(g, budget=1100); c = TokenCounter()
    res = p.run(ex, Ledger())
    assert res.truncated
    for s, u in g.log:
        if DOCS_SYS in s:
            assert c.count(s) + c.count(u) <= 1100 + 8 * 20


def test_helpers():
    assert extract_code("text\n```python\nx=1\n```\nmore") == "x=1"
    assert extract_code("```python\nx=1") == "x=1"                  # truncated output without closing fence
    assert is_insufficient("Unknown.") and is_insufficient("") and not is_insufficient("Paris")
    assert extract_answer_tag("<thinking>x</thinking><answer> Ted </answer>") == "Ted"
    assert extract_answer_tag("no tags here") == "no tags here" and extract_answer_tag("<answer> cut off") == "cut off"
    assert extract_json_list('Sure: ["a", "b"]') == ["a", "b"] and extract_json_list("nope") is None


def test_old_config_hashes_are_unchanged_by_the_new_fields():
    assert RunConfig().hash("x") == "bc89222cce"
    assert RunConfig(system="chunk", dataset="2wiki", prepend_title=True, prompt_version="v2").hash("abc") == "53ced8c4e4"
    assert RunConfig(method_args={"plan_generator": "m"}).hash("x") != RunConfig().hash("x")
    assert RunConfig(unit_mode="sentence").hash("x") != RunConfig().hash("x")


def test_cli_smoke_pyrag_with_stub_and_method_args(tmp_path):
    cmd = [sys.executable, "-m", "registerrag.cli", "run", "--system", "pyrag", "--dataset", "2wiki", "--data", DATA,
           "--out", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "c"), "--limit", "6", "--stub",
           "--method-arg", "max_repairs=1", "--method-arg", "plan_generator=stub-coder"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ROOT})
    assert r.returncode == 0, r.stderr
    run_dir = next((tmp_path / "runs").iterdir())
    recs = [json.loads(l) for l in open(run_dir / "results.jsonl", encoding="utf-8")]
    assert len(recs) == 6 and all(x["extra"]["code"] for x in recs) and all(x["error"] is None for x in recs)
    assert recs[0]["calls"]["compiler"] == 2 and recs[0]["calls"]["generator"] == 1
    cfg = json.load(open(run_dir / "config.json", encoding="utf-8"))
    assert cfg["method_args"] == {"max_repairs": 1, "plan_generator": "stub-coder"}
