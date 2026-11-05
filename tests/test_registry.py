import json, os, subprocess, sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parents[1])
DATA = os.path.join(ROOT, "datasets", "wiki2_data", "2wiki_100.json")
ENV = {**os.environ, "PYTHONPATH": ROOT, "PYTHONUTF8": "1"}

from registerrag import registry
from registerrag.paths import run_folders


def test_registry_finds_every_baseline_folder():
    impl = {b.system: b.folder for b in registry.implemented()}
    assert impl == {"chunk": "b1_vanilla_chunk", "sentence": "b2_vanilla_sentence", "georag": "b3_georag",
                    "hlg": "b4_hlg_rag", "surerag": "b5_sure_rag", "evimem": "b6_evimem", "lychee": "b7_lycheememory", "lycheeacl": "b7b_lychee_acl",
                    "pyrag": "b8_pyrag", "liveledger": "b9_liveledger", "none": "ref_no_retrieval",
                    "iterative": "ref_iterative_refinement"}
    assert not [b for b in registry.discover() if b.status == "planned"]
    assert not [b for b in registry.discover() if b.status == "broken"]


def test_every_baseline_folder_is_self_contained():
    for b in registry.discover():
        assert os.path.exists(os.path.join(b.path, "README.md")), b.folder
        assert os.path.isdir(b.results_dir), b.folder
    for b in registry.implemented():
        assert os.path.exists(os.path.join(b.path, "requirements.txt")), b.folder


def test_resolve_accepts_system_folder_and_short_id():
    assert registry.resolve("pyrag").folder == registry.resolve("b8_pyrag").folder == registry.resolve("b8").folder
    assert registry.resolve("b1").system == "chunk"
    with pytest.raises(ValueError):
        registry.resolve("nonsense")


def test_defaults_and_variants():
    assert registry.default_unit("pyrag") == "chunk" and registry.default_unit("sentence") == "sentence"
    assert [v[0] for v in registry.get("pyrag").variants("qwen2.5:3b", None)] == ["B8 PyRAG (same model plans)"]
    two = registry.get("pyrag").variants("qwen2.5:3b", "qwen2.5-coder:3b")
    assert two[1][1] == ["--method-arg", "plan_generator=qwen2.5-coder:3b"]
    assert len(registry.get("chunk").variants("m", None, openai=True)) == 2
    assert registry.results_dir("pyrag").endswith(os.path.join("baselines", "b8_pyrag", "results"))


def test_run_folders_finds_nested_runs_and_skips_stability(tmp_path):
    for rel in ["b1/results/run_a", "b2/results/run_b", "b2/results/stability/rep1/run_b", "legacy/run_c"]:
        d = tmp_path / rel; d.mkdir(parents=True); (d / "results.jsonl").write_text("{}\n")
    names = lambda fs: sorted(os.path.relpath(f, tmp_path).replace(os.sep, "/") for f in fs)
    assert names(run_folders(str(tmp_path))) == ["b1/results/run_a", "b2/results/run_b", "legacy/run_c"]
    assert "b2/results/stability/rep1/run_b" in names(run_folders(str(tmp_path), include_stability=True))
    assert names(run_folders(str(tmp_path / "b1" / "results" / "run_a"))) == ["b1/results/run_a"]


def test_cli_default_output_goes_into_the_baseline_folder(tmp_path, monkeypatch):
    from registerrag import cli
    monkeypatch.setattr(registry, "BASELINES_DIR", str(tmp_path))
    cli.main(["run", "--system", "sentence", "--dataset", "2wiki", "--data", DATA, "--limit", "4", "--stub",
              "--cache-dir", str(tmp_path / "c")])
    runs = list((tmp_path / "b2_vanilla_sentence" / "results").glob("sentence_2wiki_*"))
    assert len(runs) == 1 and (runs[0] / "summary.json").exists()


def test_migration_is_dry_run_by_default_and_moves_with_apply(tmp_path):
    src, stab, dest = tmp_path / "runs", tmp_path / "runs_stab", tmp_path / "baselines"
    def mk(base, name, system=None):
        d = base / name; d.mkdir(parents=True); (d / "results.jsonl").write_text("{}\n")
        if system:
            (d / "config.json").write_text(json.dumps({"system": system}))
    mk(src, "chunk_2wiki_aaa", "chunk"); mk(src, "pyrag_musique_bbb")          # second one: no config.json (unfinished)
    mk(src, "weird_x_ccc", "weird"); mk(stab / "rep1", "sentence_2wiki_ddd", "sentence")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "migrate_runs.py"), "--src", str(src), "--stab", str(stab), "--dest", str(dest)]
    r = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0 and "would move" in r.stdout and (src / "chunk_2wiki_aaa").exists()
    r = subprocess.run(cmd + ["--apply"], env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (dest / "b1_vanilla_chunk" / "results" / "chunk_2wiki_aaa" / "results.jsonl").exists()
    assert (dest / "b8_pyrag" / "results" / "pyrag_musique_bbb").exists()
    assert (dest / "b2_vanilla_sentence" / "results" / "stability" / "rep1" / "sentence_2wiki_ddd").exists()
    assert (src / "weird_x_ccc").exists() and "SKIPPED" in r.stdout


def test_run_baselines_script_end_to_end_with_stub(tmp_path):
    out = tmp_path / "out"
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_baselines.py"), "--stub", "--limit", "4", "--datasets", "2wiki",
           "--methods", "b1,pyrag", "--out", str(out)]
    r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    names = sorted(p.name.split("_")[0] for p in out.iterdir())
    assert names == ["chunk", "pyrag", "pyrag"]                         # B1 + both PyRAG planner variants
    table = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "summarize_runs.py"), str(out)],
                           cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert table.returncode == 0 and "pyrag" in table.stdout


def test_cli_baselines_overview_lists_everything():
    r = subprocess.run([sys.executable, "-m", "registerrag.cli", "baselines"], cwd=ROOT, env=ENV, capture_output=True, text=True)
    assert r.returncode == 0
    for word in ("b1_vanilla_chunk", "b8_pyrag", "b3_georag", "b9_liveledger", "implemented"):
        assert word in r.stdout
