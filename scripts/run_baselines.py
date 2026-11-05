"""Run baselines with the FROZEN settings (title on, forced-answer prompt v2, top-k 5). Results go to
baselines/<id>/results/ of each baseline. Finished runs are skipped; a crashed or interrupted run is retried
(--attempts) and RESUMES where it stopped. A configuration that still fails does not stop the others.

  python scripts/run_baselines.py                                          floor, B1, B2, iterative reference
  python scripts/run_baselines.py --generator qwen2.5:7b --methods none,chunk,sentence
  python scripts/run_baselines.py --methods pyrag                          B8, same-model and code-model planners
  python scripts/run_baselines.py --methods all --datasets musique --limit 10 --out runs_smoke
  python scripts/run_baselines.py --methods chunk --rep 2                  stability repetition -> results/stability/rep2
  python scripts/run_baselines.py --suffix 500                             after scripts/make_dev500.py
  --methods accepts --system names (chunk), folder ids (b8_pyrag), short ids (b8), or 'all'."""
import argparse, os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from registerrag import registry

INDEXED = {"hlg", "evimem", "lychee"}      # baselines with an offline index (scripts/build_index.py)
DATA = {"2wiki": "datasets/wiki2_data/2wiki_{s}.json", "musique": "datasets/musique_data/musique_{s}.json"}

ap = argparse.ArgumentParser()
ap.add_argument("--generator", default="qwen2.5:3b")
ap.add_argument("--datasets", default="2wiki,musique")
ap.add_argument("--methods", default="none,chunk,sentence,iterative")
ap.add_argument("--coder", default=None, help="code model for planners (default: qwen2.5-coder:<same size>)")
ap.add_argument("--suffix", default="100")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--attempts", type=int, default=3)
ap.add_argument("--rep", type=int, default=0, help="stability repetition number -> baselines/<id>/results/stability/rep<N>")
ap.add_argument("--out", default=None, help="override the output directory for every run")
ap.add_argument("--openai", action="store_true", help="also run the GPT-4o-mini reference row (needs OPENAI_API_KEY)")
ap.add_argument("--stub", action="store_true", help="offline stub models (smoke test)")
ap.add_argument("--no-prebuild", action="store_true", help="do not pre-build offline indexes (B4/B6/B7) before the runs")
a = ap.parse_args()

methods = [b.system for b in registry.implemented()] if a.methods == "all" else [m for m in a.methods.split(",") if m]
baselines = [registry.resolve(m) for m in methods]
coder = a.coder or re.sub(r"^qwen2\.5:", "qwen2.5-coder:", a.generator)
coder = None if coder == a.generator else coder
env = {**os.environ, "PYTHONPATH": ROOT, "PYTHONUTF8": "1"}
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # fewer native crashes on this Windows machine
    env.setdefault(k, "1")
failed = []
for ds in [d for d in a.datasets.split(",") if d]:
    data = os.path.join(ROOT, DATA[ds].format(s=a.suffix))
    if not os.path.exists(data):
        print(f"missing {data} - skipping {ds}"); continue
    for b in baselines:
        if os.path.isdir(b.path) and b.status == "implemented" and not a.no_prebuild:
            pre = [sys.executable, os.path.join(ROOT, "scripts", "build_index.py"), "--system", b.system, "--dataset", ds,
                   "--data", data, "--generator", a.generator] + (["--limit", str(a.limit)] if a.limit else []) + (["--stub"] if a.stub else [])
            if b.system in INDEXED:
                print(f"\n=== {ds} | prebuilding the {b.folder} index (cached, resumable) ===", flush=True)
                for attempt in range(1, a.attempts + 1):
                    if subprocess.run(pre, cwd=ROOT, env=env).returncode == 0:
                        break
                    print(f"index build attempt {attempt} failed; finished documents are cached, retrying...", flush=True)
                    time.sleep(3)
        for label, extra in b.variants(a.generator, coder, a.openai):
            out = a.out or (os.path.join(b.results_dir, "stability", f"rep{a.rep}") if a.rep else b.results_dir)
            cmd = [sys.executable, "-m", "registerrag.cli", "run", "--system", b.system, "--dataset", ds, "--data", data,
                   "--out", out, "--prepend-title", "--prompt", "v2", "--top-k", "5", "--generator", a.generator]
            cmd += (["--limit", str(a.limit)] if a.limit else []) + (["--stub"] if a.stub else []) + extra
            print(f"\n=== {ds} | {label}  [{a.generator}] ===", flush=True)
            for attempt in range(1, a.attempts + 1):
                if subprocess.run(cmd, cwd=ROOT, env=env).returncode == 0:
                    break
                print(f"attempt {attempt} of {a.attempts} failed; finished questions are kept, retrying...", flush=True)
                time.sleep(3)
            else:
                failed.append(f"{ds} | {label}")
if not a.rep:
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "summarize_runs.py"), "--hops"], cwd=ROOT, env=env)
if failed:
    print("\nSTILL FAILING after", a.attempts, "attempts:\n  " + "\n  ".join(failed))
    sys.exit(1)
