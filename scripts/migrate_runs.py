"""Move existing run folders into the per-baseline layout. DRY RUN unless --apply.
  runs/<system>_<dataset>_<hash>/            ->  baselines/<id>/results/<same name>/
  runs_stab/rep<N>/<system>_..._<hash>/      ->  baselines/<id>/results/stability/rep<N>/<same name>/
Folder names are hashes of the config, so a moved run still resumes and still matches new runs of the same config.
Usage: python scripts/migrate_runs.py [--src runs] [--stab runs_stab] [--dest baselines] [--apply]"""
import argparse, json, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from registerrag import registry

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(ROOT, "runs"))
ap.add_argument("--stab", default=os.path.join(ROOT, "runs_stab"))
ap.add_argument("--dest", default=registry.BASELINES_DIR)
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()
by_system = {b.system: b for b in registry.implemented()}


def system_of(folder):
    cfg = os.path.join(folder, "config.json")
    if os.path.exists(cfg):
        try:
            return json.load(open(cfg, encoding="utf-8")).get("system")
        except ValueError:
            pass
    return os.path.basename(folder).split("_")[0]          # unfinished runs have no config.json: use the name prefix


moves, problems = [], []
def plan(src_dir, sub=""):
    if not os.path.isdir(src_dir):
        return
    for name in sorted(os.listdir(src_dir)):
        d = os.path.join(src_dir, name)
        if not os.path.exists(os.path.join(d, "results.jsonl")):
            continue
        b = by_system.get(system_of(d))
        if b is None:
            problems.append(f"{d}: unknown system '{system_of(d)}'"); continue
        target = os.path.join(a.dest, b.folder, "results", sub, name)
        if os.path.exists(target):
            problems.append(f"{d}: target already exists ({target})"); continue
        moves.append((d, target))

plan(a.src)
if os.path.isdir(a.stab):
    for rep in sorted(os.listdir(a.stab)):
        plan(os.path.join(a.stab, rep), os.path.join("stability", rep))
for s, t in moves:
    print(("MOVE " if a.apply else "would move ") + os.path.relpath(s, ROOT) + "  ->  " + os.path.relpath(t, ROOT))
    if a.apply:
        os.makedirs(os.path.dirname(t), exist_ok=True)
        shutil.move(s, t)
for p in problems:
    print("SKIPPED", p)
print(f"\n{len(moves)} run folder(s) {'moved' if a.apply else 'would be moved'}; {len(problems)} skipped."
      + ("" if a.apply else "  Re-run with --apply to move them."))
