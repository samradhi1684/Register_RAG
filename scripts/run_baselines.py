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
import argparse, os, re, signal, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from registerrag import registry


def _newest_mtime(dirs):
    """Newest modification time of any file below the given directories (progress signal: results are fsynced per question)."""
    best = 0.0
    for d in dirs:
        for r, _, fs in os.walk(d):
            for f in fs:
                try:
                    best = max(best, os.path.getmtime(os.path.join(r, f)))
                except OSError:
                    pass
    return best


def _kill_tree(p):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        p.kill()


def run_watched(cmd, cwd, env, stall_min, watch_dirs, what=""):
    """subprocess.run with a no-progress watchdog. The child's output is relayed live. It counts as progress when the child
    prints a line OR a file below watch_dirs changes. If neither happens for stall_min minutes the whole process tree is
    killed and 124 is returned, so the caller's retry loop resumes the run (finished questions are kept). stall_min<=0 disables it."""
    kw = {"start_new_session": True} if os.name != "nt" else {}
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1, **kw)
    last = [time.time()]

    def pump():
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); last[0] = time.time()

    th = threading.Thread(target=pump, daemon=True); th.start()
    next_scan = 0.0
    while p.poll() is None:
        time.sleep(2)
        if stall_min <= 0 or time.time() < next_scan:
            continue
        next_scan = time.time() + 30
        activity = max(last[0], _newest_mtime(watch_dirs))
        if time.time() - activity > stall_min * 60:
            print(f"\nWATCHDOG: {what or 'child'} made no progress for {stall_min} min (no output, no file change); killing it so it can be retried.", flush=True)
            _kill_tree(p)
            p.wait(); th.join(timeout=5)
            return 124
    th.join(timeout=5)
    return p.returncode

INDEXED = {"hlg", "evimem", "lychee"}      # baselines with an offline index (scripts/build_index.py)
DATA = {"hotpot": "datasets/hotpotqa_data/hotpot_{s}.json", "2wiki": "datasets/wiki2_data/2wiki_{s}.json", "musique": "datasets/musique_data/musique_{s}.json", "dsire": "datasets/dsire_data/repolicyqa_{s}.json"}

ap = argparse.ArgumentParser()
ap.add_argument("--generator", default="qwen2.5:3b")
ap.add_argument("--datasets", default="2wiki,musique,dsire")
ap.add_argument("--methods", default="none,chunk,sentence,iterative")
ap.add_argument("--coder", default=None, help="code model for planners (default: qwen2.5-coder:<same size>)")
ap.add_argument("--suffix", default="100")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--attempts", type=int, default=3)
ap.add_argument("--rep", type=int, default=0, help="stability repetition number -> baselines/<id>/results/stability/rep<N>")
ap.add_argument("--out", default=None, help="override the output directory for every run")
ap.add_argument("--openai", action="store_true", help="also run the GPT-4o-mini reference row (needs OPENAI_API_KEY)")
ap.add_argument("--stub", action="store_true", help="offline stub models (smoke test)")
ap.add_argument("--stall-min", type=float, default=45.0, help="watchdog: kill and retry a run/index build with no output and no file change for this many minutes (0 = off)")
ap.add_argument("--stall-min-b7b", type=float, default=240.0, help="same, for B7b (one question can take an hour)")
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
                    if run_watched(pre, ROOT, env, max(a.stall_min, 60.0) if a.stall_min > 0 else 0,
                                   [os.path.join(b.path, "index")], f"{b.system} index build") == 0:
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
                _stall = a.stall_min_b7b if b.system == "lycheeacl" else a.stall_min
                if run_watched(cmd, ROOT, env, _stall, [out], f"{b.system} on {ds}") == 0:
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
