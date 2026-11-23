"""Progressive-n report: how every baseline's score changes as the number of questions grows (100, 200, ... , all).
One full run per configuration is enough: questions are answered in data-file order, so the first N questions of the run
ARE the N-question experiment (nested, paired across systems). Works on partial runs too (a step is shown once all its
questions are answered).
Usage: python scripts/progressive_report.py [runs_dir] [--md out.md] [--csv out.csv] [--step 100] [--generator qwen2.5:3b]
Defaults: runs_dir=runs_progressive, md=<runs_dir>/progressive_report.md, csv=<runs_dir>/progressive_report.csv"""
import argparse, csv, json, os, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from registerrag import registry
from registerrag.data import load_examples
from registerrag.paths import run_folders

ap = argparse.ArgumentParser()
ap.add_argument("dir", nargs="?", default="runs_progressive")
ap.add_argument("--md"); ap.add_argument("--csv"); ap.add_argument("--step", type=int, default=100)
ap.add_argument("--generator", default=None)
a = ap.parse_args()
md_path = a.md or os.path.join(a.dir, "progressive_report.md")
csv_path = a.csv or os.path.join(a.dir, "progressive_report.csv")
LABEL = {b.system: b.label for b in registry.implemented()}
ORDER = list(LABEL)
PRIMARY = {"dsire": "f1", "2wiki": "em", "musique": "em"}
rng = np.random.default_rng(0)


def find_data(cfg):
    p = cfg["data_path"]
    if os.path.exists(p):
        return p
    base = os.path.basename(p)
    for r, _, fs in os.walk(os.path.join(ROOT, "datasets")):
        if base in fs:
            return os.path.join(r, base)
    return None


_order_cache = {}


def qid_order(cfg):
    dp = find_data(cfg)
    if dp is None:
        return None
    if dp not in _order_cache:
        _order_cache[dp] = [e.qid for e in load_examples(dp, cfg["dataset"])[0]]
    return _order_cache[dp]


def boot(x, n=1000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


runs = {}   # (dataset, generator) -> {(order_idx, variant, system): (qid -> row)}
meta = {}
for d in run_folders(a.dir):
    cp, rp = os.path.join(d, "config.json"), os.path.join(d, "results.jsonl")
    if not (os.path.exists(cp) and os.path.exists(rp)):
        continue
    c = json.load(open(cp, encoding="utf-8"))
    if not (c.get("prepend_title") and c.get("prompt_version") == "v2" and c.get("budget_tokens") is None and c.get("top_k") == 5
            and c["system"] in LABEL and not c.get("limit")) or (a.generator and c["generator"] != a.generator):
        continue
    rows = {}
    for line in open(rp, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line); rows[r["qid"]] = r
    parts = [f"unit={c['unit_mode']}"] if c.get("unit_mode") else []
    parts += [f"{k}={v}" for k, v in sorted((c.get("method_args") or {}).items())]
    variant = ", ".join(parts) or "-"
    order = qid_order(c) or list(rows)
    runs.setdefault((c["dataset"], c["generator"]), {})[(ORDER.index(c["system"]), variant, c["system"])] = (rows, order)

if not runs:
    sys.exit(f"no runs with the frozen settings found under {a.dir}")

f3 = lambda x: "-" if x is None or x != x else f"{x:.3f}"
out = ["# Progressive results: score vs number of questions", "",
       "One full run per configuration; the first N questions of that run form the N-question experiment (nested sets, so "
       "systems are compared on the same questions). Steps appear once all N questions are answered. "
       "Primary metric: **EM** for 2Wiki/MuSiQue, **token-F1** for DSIRE (free-text answers). Cells: `value [95% bootstrap CI]`.", "",
       "*Caveat:* the corpus is pooled over all questions of the 600-question file, so retrieval difficulty is the same at every N "
       "(the curve shows sampling stability, not a growing corpus). The first 100 questions are the set used in the 100-question "
       "table, but its numbers are not identical to it (bigger pool, fresh run).", ""]
long_rows = []
for (ds, gen) in sorted(runs):
    cfgs = sorted(runs[(ds, gen)])
    total = max(len(o) for _, (_, o) in runs[(ds, gen)].items())
    steps = list(range(a.step, total + 1, a.step))
    if not steps or steps[-1] != total:
        steps.append(total)
    pm = PRIMARY[ds]
    out += [f"## {ds} | {gen} | primary metric: {pm.upper()} | {total} questions in the set", ""]
    tabs = {"primary": [], "other": [], "delta": [], "recall": []}
    base = runs[(ds, gen)].get(next((k for k in cfgs if k[2] == "chunk"), None))
    for key in cfgs:
        rows, order = runs[(ds, gen)][key]
        name = LABEL[key[2]] + ("" if key[1] == "-" else f" [{key[1]}]")
        cells = {t: [name] for t in tabs}
        for N in steps:
            ids = order[:N]
            if len(ids) < N or any(q not in rows for q in ids):
                got = sum(q in rows for q in ids)
                for t in cells:
                    cells[t].append("-")
                continue
            sel = [rows[q] for q in ids]
            ym = [r[pm] for r in sel]; yo = [r["f1" if pm == "em" else "em"] for r in sel]
            m, lo, hi = boot(ym); mo = float(np.mean(yo))
            cells["primary"].append(f"{f3(m)} [{f3(lo)}, {f3(hi)}]")
            cells["other"].append(f3(mo))
            rec = [r["retrieval"]["doc_recall@10"] for r in sel if r.get("retrieval") and "doc_recall@10" in r["retrieval"]]
            cells["recall"].append(f3(float(np.mean(rec))) if rec else "-")
            dtxt, dm = "-", None
            if base is not None and key[2] != "chunk" and all(q in base[0] for q in ids):
                diff = np.array(ym) - np.array([base[0][q][pm] for q in ids])
                dm, dlo, dhi = boot(diff)
                dtxt = f"{dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}]" + ("*" if dlo > 0 or dhi < 0 else "")
            cells["delta"].append(dtxt)
            long_rows.append([ds, gen, key[2], key[1], N, f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", pm, f"{mo:.4f}",
                              f"{np.mean(rec):.4f}" if rec else "", f"{dm:.4f}" if dm is not None else "",
                              sum(1 for r in sel if r.get("error"))])
        for t in tabs:
            tabs[t].append(cells[t])
    hdr = "| system | " + " | ".join(str(N) for N in steps) + " |\n|---|" + "---|" * len(steps)
    for t, title in (("primary", f"{pm.upper()} (primary)"), ("delta", f"paired difference vs B1 chunk RAG in {pm.upper()} (* = interval excludes 0)"),
                     ("other", "F1" if pm == "em" else "EM"), ("recall", "doc recall@10")):
        out += [f"**{title}**", "", hdr] + ["| " + " | ".join(r) + " |" for r in tabs[t]] + [""]
out += ["Notes: B7b (LycheeMemory ACL) answers in sentences, so its strict EM/F1 are about 0 by construction; B5 SURE-RAG has its gate off "
        "(B1's answers). 'doc recall@10' is empty for systems without retrieval. See `baselines/FIDELITY_AUDIT.md`."]
os.makedirs(os.path.dirname(os.path.abspath(md_path)), exist_ok=True)
open(md_path, "w", encoding="utf-8").write("\n".join(out) + "\n")
with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "generator", "system", "variant", "n", "primary", "ci_lo", "ci_hi", "metric", "other_metric", "doc_recall@10", "delta_vs_b1", "errors"])
    w.writerows(long_rows)
print("\n".join(out)); print(f"\nwritten to {md_path} and {csv_path}")
