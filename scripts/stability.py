"""Step 3 stability check: same config run several times -> EM range must be < 0.5 points.
Usage: python scripts/stability.py runs_stab/rep1/chunk_2wiki_XXXX runs_stab/rep2/chunk_2wiki_XXXX ...
   or: python scripts/stability.py --baseline b1_vanilla_chunk   (reads baselines/<id>/results/stability/rep*/<run>)
   or: python scripts/stability.py --root DIR                    (DIR/rep*/<run>/results.jsonl, grouped by run name)"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict

def load(folder):
    return {r["qid"]: r for r in map(json.loads, open(os.path.join(folder, "results.jsonl"), encoding="utf-8"))}

args = sys.argv[1:]
groups = defaultdict(list)
if args and args[0] == "--baseline":
    from registerrag import registry
    args = ["--root", os.path.join(registry.resolve(args[1]).results_dir, "stability")]
if args and args[0] == "--root":
    for p in glob.glob(os.path.join(args[1], "*", "*", "results.jsonl")):
        d = os.path.dirname(p); groups[os.path.basename(d)].append(d)
else:
    groups["given"] = args
print(f"{'config folder':34s} {'reps':>4s} {'EM per rep':>30s} {'range (pts)':>11s} {'q w/ different answer':>22s}  verdict")
for name, folders in sorted(groups.items()):
    if len(folders) < 2:
        print(f"{name:34s} {len(folders):4d}  (need >= 2 repetitions)"); continue
    runs = [load(f) for f in sorted(folders)]
    ems = [sum(r["em"] for r in run.values()) / len(run) for run in runs]
    common = set.intersection(*[set(r) for r in runs])
    diff = sum(1 for q in common if len({run[q]["prediction"].strip().lower() for run in runs}) > 1)
    rng = 100 * (max(ems) - min(ems))
    mean = sum(ems) / len(ems); sd = (sum((e - mean) ** 2 for e in ems) / (len(ems) - 1)) ** 0.5
    print(f"{name:34s} {len(folders):4d} {', '.join(f'{e:.3f}' for e in ems):>30s} {rng:11.2f} {diff:22d}  "
          f"{'OK' if rng < 0.5 else 'UNSTABLE (>=0.5 pt)'}   mean {mean:.3f} sd {sd:.3f}")
