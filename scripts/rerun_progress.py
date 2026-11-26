"""How much of a rerun is done and what remains.
Counts every expected run (method x variant x dataset x generator) against the run folders found, and the questions answered so far.
Usage: python scripts/rerun_progress.py [runs_dir] [--suffix 100] [--generators qwen2.5:3b,qwen2.5:7b] [--methods ...] [--datasets ...]
  runs_dir default: runs_rerun   (use runs_progressive --suffix prog500 for the progressive study)
Cell = finished runs / expected runs, plus 'p' for partial ones with question counts, e.g. '2/3  (+1 partial 40/100)'. Read-only."""
import argparse, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from registerrag import registry
from registerrag.paths import run_folders

ap = argparse.ArgumentParser()
ap.add_argument("dir", nargs="?", default="runs_rerun")
ap.add_argument("--suffix", default="100")
ap.add_argument("--generators", default="qwen2.5:3b,qwen2.5:7b")
ap.add_argument("--datasets", default="2wiki,musique,dsire")
ap.add_argument("--methods", default="none,chunk,sentence,iterative,pyrag,evimem,georag,surerag,liveledger,hlg,lychee,lycheeacl")
a = ap.parse_args()
DATA = {"hotpot": "datasets/hotpotqa_data/hotpot_{s}.json", "2wiki": "datasets/wiki2_data/2wiki_{s}.json", "musique": "datasets/musique_data/musique_{s}.json", "dsire": "datasets/dsire_data/repolicyqa_{s}.json"}
gens, dss = a.generators.split(","), a.datasets.split(",")
bs = [registry.resolve(m) for m in a.methods.split(",") if m]

N = {}
for ds in dss:
    p = os.path.join(ROOT, DATA[ds].format(s=a.suffix))
    if os.path.exists(p):
        recs = json.load(open(p, encoding="utf-8"))
        N[ds] = len([r for r in recs if r.get("answerable", True)]) if ds == "musique" else len(recs)
    else:
        N[ds] = 0

# what exists: (system, dataset, generator, variant-key) -> (answered, finished)
found = {}
for d in run_folders(a.dir):
    cp, rp = os.path.join(d, "config.json"), os.path.join(d, "results.jsonl")
    if not (os.path.exists(cp) and os.path.exists(rp)):
        continue
    c = json.load(open(cp, encoding="utf-8"))
    if c.get("limit"):
        continue
    if a.suffix not in os.path.basename(c.get("data_path", "")):
        continue
    ok = set()
    for line in open(rp, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                if not r.get("error"):
                    ok.add(r["qid"])
            except ValueError:
                pass
    key = (c["system"], c["dataset"], c["generator"],
           json.dumps([c.get("unit_mode"), c.get("method_args") or {}], sort_keys=True))
    n_done = len(ok)
    fin = os.path.exists(os.path.join(d, "summary.json")) and n_done >= N.get(c["dataset"], 10 ** 9)
    if key not in found or n_done > found[key][0]:
        found[key] = (n_done, fin)

rows, tot_runs, done_runs, tot_q, done_q = [], 0, 0, 0, 0
hdr = ["system"] + [f"{ds} {g.split(':')[1]}" for g in gens for ds in dss]
for b in bs:
    cells = []
    for gi, g in enumerate(gens):
        coder = re.sub(r"^qwen2\.5:", "qwen2.5-coder:", g)
        coder = None if coder == g else coder
        nvar = len(b.variants(g, coder, False))
        for ds in dss:
            if b.system == "lycheeacl" and gi > 0:          # generator-independent: counted once, under the first generator
                cells.append("(see first)"); continue
            mine = [v for k, v in found.items() if k[0] == b.system and k[1] == ds and (k[2] == g or b.system == "lycheeacl")]
            fin = sum(1 for v in mine if v[1]); part = [v for v in mine if not v[1]]
            fin = min(fin, nvar)
            tot_runs += nvar; done_runs += fin
            tot_q += nvar * N[ds]
            done_q += fin * N[ds] + sum(min(v[0], N[ds]) for v in part)
            s = f"{fin}/{nvar}"
            if part and fin < nvar:
                s += " (+" + ",".join(f"{v[0]}/{N[ds]}" for v in part[:2]) + ")"
            cells.append(("ok " if fin >= nvar else "") + s)
    rows.append([b.system] + cells)

w = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
line = lambda r: "  ".join(str(x).ljust(w[i]) for i, x in enumerate(r))
print(f"{a.dir}  (suffix {a.suffix}; questions per dataset: " + ", ".join(f"{k} {v}" for k, v in N.items()) + ")\n")
print(line(hdr)); print(line(["-" * x for x in w]))
for r in rows:
    print(line(r))
print(f"\nruns finished : {done_runs} / {tot_runs}  ({100 * done_runs / max(tot_runs, 1):.0f} %)")
print(f"questions done: {done_q} / {tot_q}  ({100 * done_q / max(tot_q, 1):.0f} %)")
print(f"remaining     : {tot_runs - done_runs} runs, about {tot_q - done_q} questions")
print("(cell format: finished/expected; '(+n/N)' = a partial run with n of N questions answered; 'ok' = complete)")
