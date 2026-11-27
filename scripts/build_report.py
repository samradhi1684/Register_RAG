"""Build the shareable analysis page (HTML): RegisterRAG vs every baseline.

Reference system: RegisterRAG (register_rag_results/*.csv, qwen2.5:7b). Baselines: runs_progressive/*/results.jsonl.
RegisterRAG predictions are re-scored with the harness scorer against the baselines' gold answers (with aliases), and
every comparison is paired on the questions both sides answered. Every number on the page is computed here.
Styles live in scripts/report_style.css. Usage: venv/bin/python scripts/build_report.py [--out path.html]
"""
import argparse, collections, csv, glob, html, json, math, os, random, sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registerrag.metrics import exact_match, f1_score, normalize_answer as N  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs_progressive")
ap.add_argument("--rr", default="register_rag_results")
ap.add_argument("--data", default="datasets/dsire_data/repolicyqa_prog500.json")
ap.add_argument("--out", default="runs_progressive/report/baseline_report.html")
a = ap.parse_args()

G7, G3 = "qwen2.5:7b", "qwen2.5:3b"
GENS = [G3, G7]
DSETS = ["2wiki", "hotpot", "musique", "dsire"]
DNAME = {"2wiki": "2WikiMultiHopQA", "hotpot": "HotpotQA", "musique": "MuSiQue", "dsire": "DSIRE (RePolicyQA)"}
DSHORT = {"2wiki": "2Wiki", "hotpot": "HotpotQA", "musique": "MuSiQue", "dsire": "DSIRE"}
RRFILE = {"2wiki": "results_2wiki_500.csv", "hotpot": "results_hotpotqa_500.csv", "musique": "results_musique_500.csv",
          "dsire": "results_repolicyqa_500.csv"}
PRIMARY = {"2wiki": "f1", "hotpot": "f1", "musique": "f1", "dsire": "f1"}  # F1 is the headline metric; EM is reported alongside
MNAME = {"em": "EM", "f1": "F1"}
RR = ("registerrag", "-")
SHORT = {
    RR: "RegisterRAG",
    ("chunk", "-"): "B1 Chunk RAG", ("sentence", "-"): "B2 Sentence RAG",
    ("georag", "-"): "B3 GeoRAG", ("georag", "unit=sentence"): "B3 GeoRAG, sentence units",
    ("hlg", "-"): "B4 HLG-RAG", ("hlg", "retriever=topic"): "B4 HLG-RAG, topic retriever",
    ("surerag", "gate=False"): "B5 SURE-RAG, gate off",
    ("evimem", "-"): "B6 EviMem", ("evimem", "answer_context=raw"): "B6 EviMem, raw context",
    ("lychee", "-"): "B7 LycheeMem V2", ("lychee", "plan=False"): "B7 LycheeMem V2, no plan",
    ("pyrag", "-"): "B8 PyRAG", ("pyrag", "coder"): "B8 PyRAG, coder planner",
    ("liveledger", "-"): "B9 LiveLedger", ("liveledger", "ledger=False"): "B9 LiveLedger, no ledger",
    ("iterative", "-"): "Ref: iterative decomposition", ("none", "-"): "Ref: no retrieval",
    ("lycheeacl", "-"): "B7b LycheeMem ACL checkpoint",
}
ORDER = list(SHORT)
BASE = [k for k in ORDER if k not in (RR, ("none", "-"), ("lycheeacl", "-"))]


def var_key(system, var):
    return "coder" if system == "pyrag" and "plan_generator" in var else var


# ---------------------------------------------------------------- load baselines
runs, incomplete = {}, []
for f in sorted(glob.glob(os.path.join(a.dir, "*", "results.jsonl"))):
    d = os.path.dirname(f)
    cfg = json.load(open(os.path.join(d, "config.json"), encoding="utf-8"))
    if not os.path.exists(os.path.join(d, "summary.json")):
        incomplete.append(os.path.basename(d))
        continue
    ma = dict(cfg.get("method_args") or {})
    if cfg.get("unit_mode"):
        ma["unit"] = cfg["unit_mode"]
    var = ",".join(f"{k}={v}" for k, v in ma.items()) or "-"
    if var == "gate=False,premise=concat":
        continue  # behaves identically to gate=False; reported once
    runs[(cfg["dataset"], cfg["generator"], cfg["system"], var_key(cfg["system"], var))] = {
        r["qid"]: r for r in map(json.loads, open(f, encoding="utf-8"))}

# ---------------------------------------------------------------- load RegisterRAG, map onto baseline qids, re-score
rr_csv_score, rr_unmatched = {}, {}
for ds in DSETS:
    B = runs[(ds, G7, "chunk", "-")]
    byq = {r["question"].strip(): q for q, r in B.items()}
    rows = list(csv.DictReader(open(os.path.join(a.rr, RRFILE[ds]), encoding="utf-8-sig")))
    out, csv_sc = {}, []
    for r in rows:
        q = r["question_id"] if r["question_id"] in B else byq.get(r["question"].strip())
        if not q:
            continue
        gold = B[q]["gold_answers"]
        csv_sc.append(float(r[PRIMARY[ds]]))
        out[q] = {"qid": q, "qtype": B[q]["qtype"], "question": B[q]["question"], "gold_answers": gold,
                  "prediction": r["predicted_answer"], "em": exact_match(r["predicted_answer"], gold),
                  "f1": f1_score(r["predicted_answer"], gold), "n_hops": B[q]["n_hops"],
                  "latency_s": float(r["latency_s"]), "total_tokens_processed": float(r["total_tokens"]),
                  "calls": {"generator": float(r["llm_calls"])}, "judge": float(r["llm_judge"] or 0),
                  "req": int(r["requirements_count"]), "sup": int(r["supported_requirements"]),
                  "rr_failure": r["failure_type"], "retrieval": {"doc_recall@10": float(r["recall_at_10"] or 0)}}
    rr_csv_score[ds] = mean(csv_sc)
    rr_unmatched[ds] = len(rows) - len(out)
    runs[(ds, G7, *RR)] = out
data = json.load(open(a.data, encoding="utf-8"))


# ---------------------------------------------------------------- helpers
def m(xs):
    xs = list(xs)
    return mean(xs) if xs else float("nan")


def qt(r):
    return r.get("qtype") or r["qid"].split("__")[0]


def boot(xs, ys, n=2000, seed=0):
    rng = random.Random(seed)
    dl = [x - y for x, y in zip(xs, ys)]
    bs = sorted(mean(rng.choices(dl, k=len(dl))) for _ in range(n))
    return mean(dl), bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def first(p):
    w = N(p).split()
    return w[0] if w and w[0] in ("yes", "no") else "other"


def get(ds, g, s, v="-"):
    return runs.get((ds, g, s, v))


def label(s, v):
    return SHORT.get((s, v), f"{s} {v}")


def esc(s):
    return html.escape(str(s))


def pct(x):
    return f"{100 * x:.0f}%"


def fs(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def rrset(ds):
    return runs[(ds, G7, *RR)]


def vs_rr(ds, k, ids=None, met=None):
    """RegisterRAG minus system k (7B), paired on shared questions. Positive = RegisterRAG better."""
    R, X = rrset(ds), get(ds, G7, *k)
    met = met or PRIMARY[ds]
    ids = [i for i in (ids if ids is not None else R) if i in X and i in R]
    return boot([R[i][met] for i in ids], [X[i][met] for i in ids]) + (len(ids),)


def score(ds, k, ids=None, g=G7):
    X = get(ds, g, *k)
    ids = [i for i in (ids if ids is not None else rrset(ds)) if i in X]
    return m(X[i][PRIMARY[ds]] for i in ids)


def emscore(ds, k, ids=None, g=G7):
    X = get(ds, g, *k)
    ids = [i for i in (ids if ids is not None else rrset(ds)) if i in X]
    return m(X[i]["em"] for i in ids)


# ---------------------------------------------------------------- core comparison
SIG = {(ds, k): vs_rr(ds, k) for ds in DSETS for k in BASE + [("none", "-")] if get(ds, G7, *k)}
SIG_EM = {(ds, k): vs_rr(ds, k, met="em") for (ds, k) in SIG}
rr_better = [(d, k) for (d, k), v in SIG.items() if v[1] > 0 and k != ("none", "-")]
rr_worse = [(d, k) for (d, k), v in SIG.items() if v[2] < 0]
rr_tie = [(d, k) for (d, k), v in SIG.items() if v[1] <= 0 <= v[2]]
n_cmp = sum(1 for (d, k) in SIG if k != ("none", "-"))


def best_base(ds, ids):
    best = None
    for k in BASE:
        X = get(ds, G7, *k)
        if not X:
            continue
        v = m(X[i][PRIMARY[ds]] for i in ids if i in X)
        if best is None or v > best[0]:
            best = (v, k)
    return best


# ---------------------------------------------------------------- chart: forest of RegisterRAG advantage per dataset
def forest(ds):
    rows = [k for k in BASE if (ds, k) in SIG]
    L, W, top, rh = 196, 520, 34, 22
    H = top + rh * len(rows) + 30
    lo_d, hi_d = -0.2, 0.4
    X = lambda v: L + (min(max(v, lo_d), hi_d) - lo_d) / (hi_d - lo_d) * (W - L - 14)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="RegisterRAG minus each baseline on {esc(DNAME[ds])}">']
    for t in [-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]:
        x = X(t)
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{H - 26}" class="{"zero" if t == 0 else "grid"}"/>'
                 f'<text x="{x:.1f}" y="{H - 10}" class="tick" text-anchor="middle">{"0" if t == 0 else f"{t:+.1f}"}</text>')
    o.append(f'<text x="{X(0) - 6:.1f}" y="16" class="axlab" text-anchor="end">← baseline better</text>')
    o.append(f'<text x="{X(0) + 6:.1f}" y="16" class="axlab">RegisterRAG better →</text>')
    for i, k in enumerate(rows):
        y = top + rh * i + rh / 2
        d, lo, hi, n = SIG[(ds, k)]
        if i % 2 == 0:
            o.append(f'<rect x="0" y="{y - rh / 2:.1f}" width="{W}" height="{rh}" class="band"/>')
        o.append(f'<text x="{L - 10}" y="{y + 4:.1f}" class="rowlab" text-anchor="end">{esc(label(*k))}</text>')
        cls = "pos" if lo > 0 else "neg" if hi < 0 else "nul"
        e = SIG_EM[(ds, k)]
        tip = f"RegisterRAG − {label(*k)}: F1 {d:+.3f} [{lo:+.3f}, {hi:+.3f}]; EM {e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}]; n = {n} shared questions"
        o.append(f'<g class="pt"><title>{esc(tip)}</title>'
                 f'<rect x="{L}" y="{y - rh / 2:.1f}" width="{W - L}" height="{rh}" fill="transparent"/>'
                 f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="ci {cls}"/>'
                 f'<circle cx="{X(d):.1f}" cy="{y:.1f}" r="4.5" class="dot {cls}"/></g>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- chart: RegisterRAG score by register support
def support_chart():
    W, H, L, top, bot = 520, 250, 44, 26, 46
    gw = (W - L - 16) / len(DSETS)
    Y = lambda v: top + (1 - v / 0.7) * (H - top - bot)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="RegisterRAG score when its register is fully vs partly supported">']
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        o.append(f'<line x1="{L}" x2="{W - 10}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="{"zero" if t == 0 else "grid"}"/>'
                 f'<text x="{L - 8}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')
    for i, ds in enumerate(DSETS):
        R = rrset(ds)
        full = [r for r in R.values() if r["req"] == r["sup"]]
        part = [r for r in R.values() if r["req"] != r["sup"]]
        x0 = L + i * gw + gw * 0.14
        bw = gw * 0.34
        for j, (grp, cls, nm) in enumerate([(full, "bar7", "fully supported"), (part, "bar3", "partly supported")]):
            v = m(r[PRIMARY[ds]] for r in grp)
            x = x0 + j * (bw + 2)
            o.append(f'<g class="pt"><title>{DSHORT[ds]}, register {nm}: {MNAME[PRIMARY[ds]]} {v:.3f} on {len(grp)} questions ({pct(len(grp) / len(R))})</title>'
                     f'<path d="M{x:.1f},{Y(0):.1f} V{Y(v) + 4:.1f} q0,-4 4,-4 H{x + bw - 4:.1f} q4,0 4,4 V{Y(0):.1f} Z" class="{cls}"/>'
                     f'<text x="{x + bw / 2:.1f}" y="{Y(v) - 6:.1f}" class="vlab" text-anchor="middle">{v:.2f}</text></g>')
        o.append(f'<text x="{x0 + bw + 1:.1f}" y="{H - 26}" class="tick" text-anchor="middle">{DSHORT[ds]} ({MNAME[PRIMARY[ds]]})</text>'
                 f'<text x="{x0 + bw + 1:.1f}" y="{H - 11}" class="tick dim" text-anchor="middle">{pct(len(full) / len(R))} fully supported</text>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- chart: yes/no polarity dumbbell (DSIRE, 7B, shared questions)
def yesno():
    ids = [i for i, r in rrset("dsire").items() if r["qtype"] in ("Yes-No", "Yes-No-cond") and first(r["gold_answers"][0]) != "other"]
    out = {}
    for k in [RR] + BASE:
        X = get("dsire", G7, *k)
        if not X:
            continue
        gy = [X[i] for i in ids if i in X and first(X[i]["gold_answers"][0]) == "yes"]
        gn = [X[i] for i in ids if i in X and first(X[i]["gold_answers"][0]) == "no"]
        out[k] = dict(ny=len(gy), nn=len(gn), yes=m(first(r["prediction"]) == "yes" for r in gy),
                      no=m(first(r["prediction"]) == "no" for r in gn),
                      acc=m(first(r["prediction"]) == first(r["gold_answers"][0]) for r in gy + gn))
    return out


def dumbbell(yn):
    rows = list(yn)
    L, W, top, rh = 196, 520, 30, 22
    H = top + rh * len(rows) + 30
    X = lambda v: L + v * (W - L - 16)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Yes/no accuracy split by gold polarity">']
    for t in [0, 0.25, 0.5, 0.75, 1]:
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 6}" y2="{H - 26}" class="grid"/>'
                 f'<text x="{X(t):.1f}" y="{H - 10}" class="tick" text-anchor="middle">{int(t * 100)}%</text>')
    for i, k in enumerate(rows):
        y = top + rh * i + rh / 2
        d = yn[k]
        if i % 2 == 0:
            o.append(f'<rect x="0" y="{y - rh / 2:.1f}" width="{W}" height="{rh}" class="band"/>')
        o.append(f'<text x="{L - 10}" y="{y + 4:.1f}" class="rowlab{" hl" if k == RR else ""}" text-anchor="end">{esc(label(*k))}</text>')
        tip = f"{label(*k)}: correct on gold-yes {pct(d['yes'])} (n={d['ny']}), on gold-no {pct(d['no'])} (n={d['nn']})"
        o.append(f'<g class="pt"><title>{esc(tip)}</title><rect x="{L}" y="{y - rh / 2:.1f}" width="{W - L}" height="{rh}" fill="transparent"/>'
                 f'<line x1="{X(d["yes"]):.1f}" x2="{X(d["no"]):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="link"/>'
                 f'<circle cx="{X(d["yes"]):.1f}" cy="{y:.1f}" r="4.5" class="dot s7"/>'
                 f'<rect x="{X(d["no"]) - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" rx="1.5" class="dot s3"/></g>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- chart: DSIRE long-form F1 vs answer length (all baseline runs)
def length_chart():
    bins = collections.OrderedDict((b, []) for b in ["1–3", "4–10", "11–25", "26+"])
    for k, R in runs.items():
        if k[0] != "dsire" or k[2] in ("none", "registerrag"):
            continue
        for r in R.values():
            if r["qtype"] in ("Descriptive", "Legal-Obligation"):
                n = len(r["prediction"].split())
                bins["1–3" if n <= 3 else "4–10" if n <= 10 else "11–25" if n <= 25 else "26+"].append(r["f1"])
    W, H, L, top, bot = 520, 230, 44, 18, 40
    bw = (W - L - 20) / len(bins)
    Y = lambda v: top + (1 - v / 0.5) * (H - top - bot)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Token F1 by answer length">']
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        o.append(f'<line x1="{L}" x2="{W - 10}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="{"zero" if t == 0 else "grid"}"/>'
                 f'<text x="{L - 8}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')
    for i, (b, v) in enumerate(bins.items()):
        x, w, val = L + i * bw + bw * 0.22, bw * 0.56, m(v)
        o.append(f'<g class="pt"><title>{b} words: mean F1 {val:.3f} over {len(v)} answers</title>'
                 f'<path d="M{x:.1f},{Y(0):.1f} V{Y(val) + 4:.1f} q0,-4 4,-4 H{x + w - 4:.1f} q4,0 4,4 V{Y(0):.1f} Z" class="bar7"/>'
                 f'<text x="{x + w / 2:.1f}" y="{Y(val) - 6:.1f}" class="vlab" text-anchor="middle">{val:.2f}</text>'
                 f'<text x="{x + w / 2:.1f}" y="{H - 22}" class="tick" text-anchor="middle">{b} words</text>'
                 f'<text x="{x + w / 2:.1f}" y="{H - 8}" class="tick dim" text-anchor="middle">n = {len(v):,}</text></g>')
    o.append("</svg>")
    return "".join(o)


# ---------------------------------------------------------------- tables
def cell_vs(t, em=False):
    d, lo, hi = t[:3]
    cls = ("rr-up" if lo > 0 else "rr-down" if hi < 0 else "") + (" emcell" if em else "")
    return f'<td class="num {cls}" title="95% CI [{lo:+.3f}, {hi:+.3f}]">{d:+.3f}</td>'


def qtype_table(ds):
    R = rrset(ds)
    met = PRIMARY[ds]
    rows = []
    for q in sorted({qt(r) for r in R.values()}) + ["All"]:
        ids = [i for i, r in R.items() if q == "All" or qt(r) == q]
        bb = best_base(ds, ids)
        rows.append(f"<tr{' class=total' if q == 'All' else ''}><th scope='row'>{esc(q)}</th><td class='num'>{len(ids)}</td>"
                    f"<td class='num'>{score(ds, ('none', '-'), ids):.3f}</td><td class='num'>{score(ds, ('chunk', '-'), ids):.3f} <span class='em'>{emscore(ds, ('chunk', '-'), ids):.3f}</span></td>"
                    f"<td class='num'>{bb[0]:.3f}</td><td>{esc(label(*bb[1]))}</td><td class='num rrcol'>{m(R[i][met] for i in ids):.3f} <span class='em'>{m(R[i]['em'] for i in ids):.3f}</span></td>"
                    f"{cell_vs(vs_rr(ds, bb[1], ids))}{cell_vs(vs_rr(ds, ('chunk', '-'), ids))}{cell_vs(vs_rr(ds, ('chunk', '-'), ids, met='em'), em=True)}</tr>")
    M = MNAME[met]
    return (f'<div class="tablewrap"><table class="data"><thead><tr><th>Question type</th><th>Shared n</th><th>No retrieval {M}</th>'
            f'<th>B1 F1 <span class="sub">EM</span></th><th>Best baseline F1</th><th>Best baseline</th><th>RegisterRAG F1 <span class="sub">EM</span></th>'
            f'<th>RegisterRAG − best<br><span class="sub">F1</span></th><th>RegisterRAG − B1<br><span class="sub">F1</span></th><th>RegisterRAG − B1<br><span class="sub">EM (secondary)</span></th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


QT = ["Yes-No", "Yes-No-cond", "Factual", "Descriptive", "Legal-Obligation"]


def dsire_type_table():
    R = rrset("dsire")
    head = "".join(f"<th>{q}</th>" for q in QT) + "<th>All</th><th>Long-form answer words</th>"
    body = []
    for k in [RR] + BASE:
        X = get("dsire", G7, *k)
        if not X:
            continue
        cells = []
        for q in QT + ["All"]:
            ids = [i for i in R if (q == "All" or R[i]["qtype"] == q) and i in X]
            f1 = m(X[i]["f1"] for i in ids)
            if k == RR:
                cells.append(f'<td class="num rrcol">{f1:.3f}</td>')
                continue
            d, lo, hi = boot([X[i]["f1"] for i in ids], [R[i]["f1"] for i in ids])
            cls = "up" if lo > 0 else "down" if hi < 0 else ""
            mk = "▲" if lo > 0 else "▼" if hi < 0 else ""
            cells.append(f'<td class="num {cls}" title="{esc(label(*k))} − RegisterRAG: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]">{f1:.3f}<span class="mk">{mk}</span></td>')
        words = m(len(X[i]["prediction"].split()) for i in R if i in X and R[i]["qtype"] in ("Descriptive", "Legal-Obligation"))
        body.append(f'<tr{" class=rrrow" if k == RR else ""}><th scope="row">{esc(label(*k))}</th>{"".join(cells)}<td class="num">{words:.1f}</td></tr>')
    return f'<div class="tablewrap"><table class="data"><thead><tr><th>System (qwen2.5:7b)</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def stats(R, ids):
    X = [R[i] for i in ids if i in R]
    lat = sorted(r["latency_s"] for r in X)
    return dict(lat=lat[len(lat) // 2], calls=m(r["calls"].get("generator", 0) + r["calls"].get("compiler", 0) for r in X))


def eff_table():
    body = []
    for k in [RR] + BASE:
        cells, lat, calls = [], [], []
        for ds in DSETS:
            X = get(ds, G7, *k)
            if not X:
                cells.append('<td class="num dim">not run</td>')
                continue
            st = stats(X, rrset(ds))
            lat.append(st["lat"]); calls.append(st["calls"])
            if k == RR:
                cells.append(f'<td class="num rrcol">{score(ds, RR):.3f} <span class="em">{emscore(ds, RR):.3f}</span></td>')
            else:
                d, lo, hi, _ = SIG[(ds, k)]
                e = SIG_EM[(ds, k)]
                cls = "rr-up" if lo > 0 else "rr-down" if hi < 0 else ""
                cells.append(f'<td class="num {cls}" title="F1 95% CI [{lo:+.3f}, {hi:+.3f}]; EM {e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}]">{d:+.3f} <span class="em">{e[0]:+.3f}</span></td>')
        body.append(f'<tr{" class=rrrow" if k == RR else ""}><th scope="row">{esc(label(*k))}</th>{"".join(cells)}'
                    f'<td class="num">{m(calls):.1f}</td><td class="num">{m(lat):.2f}s</td></tr>')
    hd = "".join(f"<th>{DSHORT[d]}<br><span class='sub'>F1 · EM</span></th>" for d in DSETS)
    return (f'<div class="tablewrap"><table class="data"><thead><tr><th>System (qwen2.5:7b)</th>{hd}'
            f'<th>LLM calls / q<br><span class="sub">answer + planning</span></th><th>Median latency</th></tr></thead><tbody>{"".join(body)}</tbody></table></div>')


# ---------------------------------------------------------------- numbers for the prose
def tok_pr(pred, gold):
    p, g = N(pred).split(), N(gold).split()
    c = sum((collections.Counter(p) & collections.Counter(g)).values())
    return (c / len(p) if p else 0, c / len(g) if g else 0)


Rd, Bd = rrset("dsire"), get("dsire", G7, "chunk")
prr = {}
for q in ("Factual", "Descriptive", "Legal-Obligation"):
    ids = [i for i in Rd if Rd[i]["qtype"] == q]
    a1 = [tok_pr(Rd[i]["prediction"], Rd[i]["gold_answers"][0]) for i in ids]
    b1 = [tok_pr(Bd[i]["prediction"], Bd[i]["gold_answers"][0]) for i in ids]
    prr[q] = dict(rp=m(x for x, _ in a1), rr=m(y for _, y in a1), bp=m(x for x, _ in b1), br=m(y for _, y in b1),
                  rw=m(len(Rd[i]["prediction"].split()) for i in ids), bw=m(len(Bd[i]["prediction"].split()) for i in ids),
                  judge=m(Rd[i]["judge"] for i in ids))
yn = yesno()
gy_ids = [i for i in Rd if Rd[i]["qtype"] in ("Yes-No", "Yes-No-cond") and first(Rd[i]["gold_answers"][0]) == "yes"]
rr_yes_wrong = [i for i in gy_ids if first(Rd[i]["prediction"]) != "yes"]
rr_yes_right = [i for i in gy_ids if i not in rr_yes_wrong]
part_share_wrong = m(Rd[i]["req"] != Rd[i]["sup"] for i in rr_yes_wrong)
part_share_right = m(Rd[i]["req"] != Rd[i]["sup"] for i in rr_yes_right)
b1_right_on_rr_wrong = m(first(Bd[i]["prediction"]) == "yes" for i in rr_yes_wrong)
sup = {ds: (m(r[PRIMARY[ds]] for r in rrset(ds).values() if r["req"] == r["sup"]),
            m(r[PRIMARY[ds]] for r in rrset(ds).values() if r["req"] != r["sup"]),
            m(float(r["req"] == r["sup"]) for r in rrset(ds).values())) for ds in DSETS}
gold_len = {q: sorted(len(x["answer"].split()) for x in data if x["question_type"] == q) for q in QT}
med = lambda L: L[len(L) // 2]
noise = m(len(x["context"].split()) < 6 or x["context"].lstrip("- ").startswith("#") for x in data)
docs = collections.Counter(x["document_id"] for x in data)
judge_em0 = m(r["judge"] for r in Rd.values() if r["em"] == 0)
rr_lat = {ds: stats(rrset(ds), rrset(ds))["lat"] for ds in DSETS}
cond = {}
for ds in DSETS:
    for g in GENS:
        a_, b_ = [], []
        for (d2, gg, s, v), R in runs.items():
            if d2 == ds and gg == g and s not in ("none", "lycheeacl", "registerrag"):
                for r in R.values():
                    (a_ if r.get("retrieval", {}).get("ctx_all_gold") else b_).append(r[PRIMARY[ds]])
        cond[(ds, g)] = (m(a_), m(b_))


def p(ds, g, s, v="-"):
    X = get(ds, g, s, v)
    return m(r[PRIMARY[ds]] for r in X.values())


def S(ds, s, v="-"):
    return SIG[(ds, (s, v))]


def typ(ds, q, k=("chunk", "-"), met=None):
    ids = [i for i, r in rrset(ds).items() if qt(r) == q]
    return vs_rr(ds, k, ids, met=met)


mus4_ids = [i for i, r in rrset("musique").items() if qt(r).startswith("4hop")]
mus4_rr = m(rrset("musique")[i]["f1"] for i in mus4_ids)
mus4_b1 = score("musique", ("chunk", "-"), mus4_ids)
mus4_ll = score("musique", ("liveledger", "ledger=False"), mus4_ids)
mus4_d_b1 = vs_rr("musique", ("chunk", "-"), mus4_ids)
mus4_d_ll = vs_rr("musique", ("liveledger", "ledger=False"), mus4_ids)
hp_bridge = typ("hotpot", "bridge")


def confusion(types):
    """Gold yes/no vs predicted yes/no/other for RegisterRAG and B1 on shared DSIRE questions with a yes/no gold."""
    ids = [i for i in Rd if Rd[i]["qtype"] in types and first(Rd[i]["gold_answers"][0]) != "other" and i in Bd]
    out = {}
    for nm, X in (("RegisterRAG", Rd), ("B1 Chunk RAG", Bd)):
        c = collections.Counter((first(X[i]["gold_answers"][0]), first(X[i]["prediction"])) for i in ids)
        out[nm] = c
    agree = collections.Counter((first(Rd[i]["prediction"]) == first(Rd[i]["gold_answers"][0]),
                                 first(Bd[i]["prediction"]) == first(Bd[i]["gold_answers"][0])) for i in ids)
    return ids, out, agree


def cm_table(title, types):
    ids, out, agree = confusion(types)
    head = "".join(f'<th colspan="3" class="grp">{nm}</th>' for nm in out)
    sub = "".join("<th>yes</th><th>no</th><th>other</th>" for _ in out)
    rows = []
    for gl in ("yes", "no"):
        n = sum(out["RegisterRAG"][(gl, x)] for x in ("yes", "no", "other"))
        cells = ""
        for nm, c in out.items():
            for pr in ("yes", "no", "other"):
                v = c[(gl, pr)]
                cls = "cm-ok" if pr == gl else ("cm-bad" if v else "cm-zero")
                cells += f'<td class="num {cls}">{v}</td>'
        rows.append(f"<tr><th scope='row'>Gold {gl} <span class='sub'>({n})</span></th>{cells}</tr>")
    acc = "".join(f'<td colspan="3" class="num"><b>{pct(sum(c[(x, x)] for x in ("yes", "no")) / len(ids))}</b></td>' for c in out.values())
    rows.append(f"<tr class='total'><th scope='row'>Accuracy</th>{acc}</tr>")
    return (f'<figure class="cm"><figcaption>{title}<span class="sub">n = {len(ids)} shared questions with a yes/no gold answer. Rows: gold. Columns: predicted. "Other" = neither yes nor no.</span></figcaption>'
            f'<div class="tablewrap" style="margin:6px 0 0"><table class="data cmt"><thead><tr><th rowspan="2">Gold ↓ / Predicted →</th>{head}</tr><tr>{sub}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></figure>'), out, agree, ids


cm_all, cm_out, cm_agree, cm_ids = cm_table("Yes-No + Yes-No-cond", ("Yes-No", "Yes-No-cond"))
cm_yn = cm_table("Yes-No only", ("Yes-No",))
cm_ync = cm_table("Yes-No-cond only", ("Yes-No-cond",))
_c_rr, _c_b1 = cm_out["RegisterRAG"], cm_out["B1 Chunk RAG"]
rr_no_said, rr_no_right = _c_rr[("yes", "no")] + _c_rr[("no", "no")], _c_rr[("no", "no")]
b1_no_said, b1_no_right = _c_b1[("yes", "no")] + _c_b1[("no", "no")], _c_b1[("no", "no")]
rr_err = len(cm_ids) - _c_rr[("yes", "yes")] - _c_rr[("no", "no")]
other_preds = [Bd[i]["prediction"][:40] for i in cm_ids if first(Bd[i]["prediction"]) == "other"]

# ---------------------------------------------------------------- RegisterRAG vs the strongest baseline on each dataset
def mcnemar(b, c):
    """Exact two-sided McNemar test on the discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(min(b, c) + 1)) / 2 ** n)


def kind(s):
    w = N(s).split()
    return w[0] if w and w[0] in ("yes", "no") and len(w) <= 3 else "other"


def pfmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


BEST = {ds: max((k for k in BASE if get(ds, G7, *k)), key=lambda k: score(ds, k)) for ds in DSETS}
B_HOPS = ("2wiki", "hotpot", "musique")
BS = {}
for ds in DSETS:
    R, X, B1x = rrset(ds), get(ds, G7, *BEST[ds]), get(ds, G7, "chunk")
    ids = [i for i in R if i in X and i in B1x]
    k = BEST[ds]
    s = dict(k=k, ids=ids, n=len(ids))
    s["rr"], s["bb"], s["b1"] = (m(Y[i]["f1"] for i in ids) for Y in (R, X, B1x))
    s["rr_em"], s["bb_em"] = (m(Y[i]["em"] for i in ids) for Y in (R, X))
    s["d"], s["d_em"] = vs_rr(ds, k, ids), vs_rr(ds, k, ids, met="em")
    s["wtl"] = [sum(1 for i in ids if c(R[i]["f1"], X[i]["f1"])) for c in (lambda a, b: a > b, lambda a, b: a == b, lambda a, b: a < b)]
    ag = collections.Counter((R[i]["f1"] >= 0.5, X[i]["f1"] >= 0.5) for i in ids)
    s["ag"], s["mcn"] = ag, mcnemar(ag[(True, False)], ag[(False, True)])
    s["calls_rr"] = m(R[i]["calls"]["generator"] for i in ids)
    s["calls_bb"] = m(X[i]["calls"].get("generator", 0) + X[i]["calls"].get("compiler", 0) for i in ids)
    s["rec_rr"] = m(R[i]["retrieval"]["doc_recall@10"] for i in ids)
    s["rec_bb"] = m(X[i]["retrieval"].get("doc_recall@10", 0) for i in ids)
    s["full_rr"] = m(R[i]["retrieval"]["doc_recall@10"] >= 1 for i in ids)
    s["full_bb"] = m(X[i]["retrieval"].get("doc_recall@10", 0) >= 1 for i in ids)
    both = [i for i in ids if R[i]["retrieval"]["doc_recall@10"] >= 1 and X[i]["retrieval"].get("doc_recall@10", 0) >= 1]
    s["both_full"] = vs_rr(ds, k, both)
    s["both_f1"] = (m(R[i]["f1"] for i in both), m(X[i]["f1"] for i in both))
    if ds in B_HOPS:
        buckets = (("fewer", lambda i: R[i]["req"] < R[i]["n_hops"]), ("equal", lambda i: R[i]["req"] == R[i]["n_hops"]),
                   ("more", lambda i: R[i]["req"] > R[i]["n_hops"]))
        s["dec"] = {}
        for nm, f in buckets:
            qi = [i for i in ids if f(i)]
            s["dec"][nm] = (vs_rr(ds, k, qi), m(R[i]["f1"] for i in qi), m(X[i]["f1"] for i in qi),
                            collections.Counter(qt(R[i]) for i in qi).most_common(1)[0] if qi else ("-", 0))
    full = lambda i: R[i]["req"] == R[i]["sup"]
    s["oracle"] = m(max(R[i]["f1"], X[i]["f1"]) for i in ids)
    s["route_bb"] = boot([R[i]["f1"] if full(i) else X[i]["f1"] for i in ids], [X[i]["f1"] for i in ids])
    s["route_bb_v"] = m(R[i]["f1"] if full(i) else X[i]["f1"] for i in ids)
    s["route_b1_v"] = m(R[i]["f1"] if full(i) else B1x[i]["f1"] for i in ids)
    s["route_b1"] = boot([R[i]["f1"] if full(i) else B1x[i]["f1"] for i in ids], [X[i]["f1"] for i in ids])
    s["full_share"] = m(full(i) for i in ids)
    s["sup_split"] = (m(R[i]["f1"] for i in ids if full(i)), m(X[i]["f1"] for i in ids if full(i)),
                      m(R[i]["f1"] for i in ids if not full(i)), m(X[i]["f1"] for i in ids if not full(i)))
    s["sup_d"] = (vs_rr(ds, k, [i for i in ids if full(i)]), vs_rr(ds, k, [i for i in ids if not full(i)]))
    s["kind"] = {nm: collections.Counter((kind(Y[i]["gold_answers"][0]), kind(Y[i]["prediction"])) for i in ids) for nm, Y in (("rr", R), ("bb", X))}
    pr = [tok_pr(R[i]["prediction"], R[i]["gold_answers"][0]) for i in ids]
    px = [tok_pr(X[i]["prediction"], X[i]["gold_answers"][0]) for i in ids]
    s["pr"] = (m(a_ for a_, _ in pr), m(b_ for _, b_ in pr), m(a_ for a_, _ in px), m(b_ for _, b_ in px))
    s["words"] = (m(len(R[i]["prediction"].split()) for i in ids), m(len(X[i]["prediction"].split()) for i in ids))
    loss = [i for i in ids if R[i]["f1"] < X[i]["f1"]]
    s["loss_fail"] = collections.Counter(R[i]["rr_failure"] for i in loss)
    s["nloss"] = len(loss)
    BS[ds] = s


def bshort(ds):
    return label(*BEST[ds])


def ci_plot(rows, lo_d, hi_d, aria, ticks, left="← strongest baseline better", right="RegisterRAG better →", L=210):
    """Generic paired-difference plot. rows: ('hdr', text) or (label, (d, lo, hi, n), tooltip)."""
    W, top, rh = 560, 34, 21
    H = top + rh * len(rows) + 30
    X = lambda v: L + (min(max(v, lo_d), hi_d) - lo_d) / (hi_d - lo_d) * (W - L - 14)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{esc(aria)}">']
    for t in ticks:
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 6}" y2="{H - 26}" class="{"zero" if t == 0 else "grid"}"/>'
                 f'<text x="{X(t):.1f}" y="{H - 10}" class="tick" text-anchor="middle">{"0" if t == 0 else f"{t:+.1f}"}</text>')
    o.append(f'<text x="{X(0) - 6:.1f}" y="16" class="axlab" text-anchor="end">{left}</text><text x="{X(0) + 6:.1f}" y="16" class="axlab">{right}</text>')
    for i, r in enumerate(rows):
        y = top + rh * i + rh / 2
        if r[0] == "hdr":
            o.append(f'<text x="8" y="{y + 4:.1f}" class="rowlab hl">{esc(r[1])}</text>')
            continue
        lab, (d, lo, hi, n), tip = r
        o.append(f'<rect x="0" y="{y - rh / 2:.1f}" width="{W}" height="{rh}" class="band"/>' if i % 2 else "")
        o.append(f'<text x="{L - 10}" y="{y + 4:.1f}" class="rowlab" text-anchor="end">{esc(lab)}</text>')
        cls = "pos" if lo > 0 else "neg" if hi < 0 else "nul"
        o.append(f'<g class="pt"><title>{esc(tip)}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}], n = {n}</title>'
                 f'<rect x="{L}" y="{y - rh / 2:.1f}" width="{W - L}" height="{rh}" fill="transparent"/>'
                 f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="ci {cls}"/>'
                 f'<circle cx="{X(d):.1f}" cy="{y:.1f}" r="4.5" class="dot {cls}"/></g>')
    o.append("</svg>")
    return "".join(o)


def outcome_chart():
    """Per-question F1 outcome vs the strongest baseline: 100% stacked bars."""
    W, L, top, rh, bh = 560, 200, 10, 46, 24
    H = top + rh * len(DSETS) + 8
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Per-question wins, ties and losses against the strongest baseline">']
    for j, ds in enumerate(DSETS):
        s = BS[ds]
        y = top + j * rh
        o.append(f'<text x="{L - 10}" y="{y + 11}" class="rowlab hl" text-anchor="end">{DSHORT[ds]}</text>'
                 f'<text x="{L - 10}" y="{y + 25}" class="tick" text-anchor="end">vs {esc(bshort(ds))}</text>')
        x = L
        for cnt, cls, nm in zip(s["wtl"], ("seg pos", "seg nul", "seg neg"), ("RegisterRAG higher F1", "same F1", "strongest baseline higher F1")):
            w = cnt / s["n"] * (W - L - 10)
            o.append(f'<g class="pt"><title>{DSHORT[ds]}: {nm} on {cnt} of {s["n"]} questions ({pct(cnt / s["n"])})</title>'
                     f'<rect x="{x:.1f}" y="{y}" width="{max(w - 2, 0):.1f}" height="{bh}" rx="3" class="{cls}"/>'
                     + (f'<text x="{x + w / 2 - 1:.1f}" y="{y + 16}" class="vlab" text-anchor="middle">{cnt} · {pct(cnt / s["n"])}</text>' if w > 62 else "") + "</g>")
            x += w
    o.append("</svg>")
    return "".join(o)


def hop_chart(ds, ks):
    """F1 by gold hop count (hop groups with at least 20 shared questions)."""
    s = BS[ds]
    R = rrset(ds)
    hops = sorted(h for h, c in collections.Counter(R[i]["n_hops"] for i in s["ids"]).items() if c >= 20)
    W, H, L, top, bot = 270, 210, 36, 16, 40
    Xp = lambda j: L + 22 + j * (W - L - 70) / max(len(hops) - 1, 1)
    Y = lambda v: top + (1 - v / 0.8) * (H - top - bot)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="F1 by number of hops on {DSHORT[ds]}">']
    for t in (0, 0.2, 0.4, 0.6, 0.8):
        o.append(f'<line x1="{L}" x2="{W - 8}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="{"zero" if t == 0 else "grid"}"/><text x="{L - 6}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')
    for j, h in enumerate(hops):
        nh = sum(1 for i in s["ids"] if R[i]["n_hops"] == h)
        o.append(f'<text x="{Xp(j):.1f}" y="{H - 22}" class="tick" text-anchor="middle">{h} hops</text><text x="{Xp(j):.1f}" y="{H - 9}" class="tick dim" text-anchor="middle">n = {nh}</text>')
    ends = []
    for k, cls, nm in ks:
        Xk = get(ds, G7, *k)
        vals = [m(Xk[i]["f1"] for i in s["ids"] if R[i]["n_hops"] == h) for h in hops]
        pts = " ".join(f"{Xp(j):.1f},{Y(v):.1f}" for j, v in enumerate(vals))
        o.append(f'<polyline points="{pts}" fill="none" class="ln {cls}"/>')
        for j, v in enumerate(vals):
            o.append(f'<g class="pt"><title>{esc(label(*k))}, {hops[j]} hops: F1 {v:.3f}</title><circle cx="{Xp(j):.1f}" cy="{Y(v):.1f}" r="4" class="dot {cls}"/></g>')
        ends.append([Y(vals[-1]) + 4, nm])
    ends.sort()
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 11)
    o.extend(f'<text x="{Xp(len(hops) - 1) + 8:.1f}" y="{y_:.1f}" class="tick">{nm}</text>' for y_, nm in ends)
    o.append("</svg>")
    return "".join(o)


def cost_chart(ds):
    """F1 vs LLM calls per question for every 7B system on the shared questions."""
    R = rrset(ds)
    pts = []
    for k in [RR] + BASE:
        X = get(ds, G7, *k)
        if not X:
            continue
        ids = [i for i in R if i in X]
        c = m(X[i]["calls"].get("generator", 0) + X[i]["calls"].get("compiler", 0) for i in ids)
        pts.append((k, c, m(X[i]["f1"] for i in ids)))
    W, H, L, top, bot = 270, 210, 36, 14, 36
    cmax = max(6.0, max(c for _, c, _ in pts) + 0.5)
    fmin = 0.0 if ds == "musique" else 0.2
    fmax = max(v for *_, v in pts) + 0.06
    Xc = lambda c: L + c / cmax * (W - L - 12)
    Y = lambda v: top + (1 - (v - fmin) / (fmax - fmin)) * (H - top - bot)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="F1 against LLM calls per question on {DSHORT[ds]}">']
    for t in [x / 10 for x in range(0, 10)]:
        if fmin <= t <= fmax:
            o.append(f'<line x1="{L}" x2="{W - 8}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/><text x="{L - 6}" y="{Y(t) + 4:.1f}" class="tick" text-anchor="end">{t:.1f}</text>')
    for c in range(0, int(cmax) + 1, 2):
        o.append(f'<text x="{Xc(c):.1f}" y="{H - 20}" class="tick" text-anchor="middle">{c}</text>')
    o.append(f'<text x="{(L + W) / 2:.1f}" y="{H - 5}" class="axlab" text-anchor="middle">LLM calls per question</text>')
    for k, c, v in sorted(pts, key=lambda p: p[0] in (RR, BEST[ds], ("chunk", "-"))):
        cls = "s7" if k == RR else "s3" if k == BEST[ds] else "g" if k == ("chunk", "-") else "nul"
        o.append(f'<g class="pt"><title>{esc(label(*k))}: F1 {v:.3f}, {c:.1f} LLM calls per question</title><circle cx="{Xc(c):.1f}" cy="{Y(v):.1f}" r="{5 if cls != "nul" else 3.5}" class="dot {cls}"/></g>')
        if cls != "nul":
            nm = {"s7": "RegisterRAG", "s3": "strongest", "g": "B1"}[cls]
            o.append(f'<text x="{Xc(c) + 8:.1f}" y="{Y(v) + 4:.1f}" class="tick">{nm}</text>')
    o.append("</svg>")
    return "".join(o)


def agree_table(ds):
    s, ag = BS[ds], BS[ds]["ag"]
    c = lambda a_, b_: f'<td class="num {"cm-ok" if a_ == b_ else "cm-bad"}">{ag[(a_, b_)]}</td>'
    return (f'<figure class="cm keep"><figcaption>{DSHORT[ds]}: RegisterRAG vs {esc(bshort(ds))}<span class="sub">n = {s["n"]}. A question counts as correct when F1 ≥ 0.5. '
            f'Exact McNemar test on the off-diagonal cells: p {pfmt(s["mcn"])}.</span></figcaption>'
            f'<div class="tablewrap" style="margin:6px 0 0"><table class="data cmt"><thead><tr><th>RegisterRAG ↓ / strongest →</th><th>correct</th><th>wrong</th></tr></thead><tbody>'
            f'<tr><th scope="row">correct</th>{c(True, True)}{c(True, False)}</tr><tr><th scope="row">wrong</th>{c(False, True)}{c(False, False)}</tr></tbody></table></div></figure>')


def kind_cm(ds):
    s = BS[ds]
    golds = [g for g in ("yes", "no", "other") if sum(s["kind"]["rr"][(g, p_)] for p_ in ("yes", "no", "other"))]
    head = f'<th colspan="3" class="grp">RegisterRAG</th><th colspan="3" class="grp">{esc(bshort(ds))}</th>'
    sub = "<th>yes</th><th>no</th><th>other</th>" * 2
    rows = []
    for g in golds:
        n = sum(s["kind"]["rr"][(g, p_)] for p_ in ("yes", "no", "other"))
        cells = ""
        for nm in ("rr", "bb"):
            for p_ in ("yes", "no", "other"):
                v = s["kind"][nm][(g, p_)]
                cls = ("cm-ok" if g != "other" else "") if p_ == g else ("cm-bad" if v else "cm-zero")
                cells += f'<td class="num {cls}">{v}</td>'
        rows.append(f"<tr><th scope='row'>Gold {g} <span class='sub'>({n})</span></th>{cells}</tr>")
    return (f'<figure class="cm keep"><figcaption>{DSHORT[ds]}<span class="sub">All {s["n"]} shared questions. Rows: gold answer type; columns: predicted type. '
            f'"yes"/"no" = an answer of at most three words starting with yes or no; "other" = any other answer.</span></figcaption>'
            f'<div class="tablewrap" style="margin:6px 0 0"><table class="data cmt"><thead><tr><th rowspan="2">Gold ↓ / Predicted →</th>{head}</tr><tr>{sub}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></figure>')


def examples_table(ds, n_loss=2, n_win=1, seed=7):
    """Deterministic sample of clear losses (RR F1 0, strongest 1) and clear wins (the reverse)."""
    s, R, X = BS[ds], rrset(ds), get(ds, G7, *BEST[ds])
    rng = random.Random(seed)
    loss = [i for i in s["ids"] if R[i]["f1"] == 0 and X[i]["f1"] == 1]
    win = [i for i in s["ids"] if R[i]["f1"] == 1 and X[i]["f1"] == 0]
    pick = [(i, "loss") for i in rng.sample(loss, min(n_loss, len(loss)))] + [(i, "win") for i in rng.sample(win, min(n_win, len(win)))]
    cut = lambda t, n=110: esc(t if len(t) <= n else t[:n - 1] + "…")
    return "".join(f'<tr><td>{DSHORT[ds]}</td><td><span class="verdict {"v-drop" if o_ == "loss" else "v-keep"}">{"RR loses" if o_ == "loss" else "RR wins"}</span></td>'
                   f'<td class="wrapcell">{cut(R[i]["question"])}</td><td class="wrapcell">{cut(R[i]["gold_answers"][0], 40)}</td><td class="wrapcell">{cut(R[i]["prediction"], 50)}</td>'
                   f'<td class="wrapcell">{cut(X[i]["prediction"], 50)}</td><td class="num">{R[i]["sup"]}/{R[i]["req"]}</td><td>{esc(R[i]["rr_failure"])}</td></tr>' for i, o_ in pick), (len(loss), len(win))


def main_table():
    body = []
    for ds in DSETS:
        s = BS[ds]
        w, t, l_ = s["wtl"]
        body.append(f"<tr><th scope='row'>{DSHORT[ds]}<br><span class='sub'>vs {esc(bshort(ds))}</span></th><td class='num'>{s['n']}</td>"
                    f"<td class='num rrcol'>{s['rr']:.3f}<span class='em'>{s['rr_em']:.3f}</span></td><td class='num'>{s['bb']:.3f}<span class='em'>{s['bb_em']:.3f}</span></td>"
                    f"{cell_vs(s['d'])}{cell_vs(s['d_em'], em=True)}<td class='num'>{w} / {t} / {l_}</td><td class='num'>{pfmt(s['mcn'])}</td>"
                    f"<td class='num'>{s['rec_rr']:.2f} / {s['rec_bb']:.2f}</td><td class='num'>{s['calls_rr']:.1f} / {s['calls_bb']:.1f}</td></tr>")
    return (f'<div class="tablewrap"><table class="data"><thead><tr><th>Dataset<br><span class="sub">vs strongest baseline (by F1)</span></th><th>n</th>'
            f'<th>RegisterRAG F1<br><span class="sub">EM</span></th><th>Strongest F1<br><span class="sub">EM</span></th><th>Δ F1<br><span class="sub">RR − strongest</span></th><th>Δ EM<br><span class="sub">secondary</span></th>'
            f'<th>Per question<br><span class="sub">RR higher / same / lower F1</span></th><th>McNemar p<br><span class="sub">F1 ≥ 0.5</span></th><th>Doc recall@10<br><span class="sub">RR / strongest</span></th><th>LLM calls / q<br><span class="sub">RR / strongest</span></th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def route_table():
    body = []
    for ds in DSETS:
        s = BS[ds]
        body.append(f"<tr><th scope='row'>{DSHORT[ds]}</th><td class='num'>{pct(s['full_share'])}</td><td class='num rrcol'>{s['rr']:.3f}</td><td class='num'>{s['bb']:.3f}</td>"
                    f"<td class='num'>{s['route_bb_v']:.3f}</td>{cell_vs(s['route_bb'])}<td class='num'>{s['route_b1_v']:.3f}</td>{cell_vs(s['route_b1'])}<td class='num'>{s['oracle']:.3f}</td></tr>")
    return (f'<div class="tablewrap"><table class="data"><thead><tr><th>Dataset</th><th>RR fully supported</th><th>RegisterRAG</th><th>Strongest baseline</th>'
            f'<th>Route → strongest<br><span class="sub">F1</span></th><th>Δ vs strongest</th><th>Route → B1<br><span class="sub">F1</span></th><th>Δ vs strongest</th><th>Oracle<br><span class="sub">per-question max</span></th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


type_rows = []
for ds in DSETS:
    type_rows.append(("hdr", f"{DSHORT[ds]} — vs {bshort(ds)}"))
    for q in sorted({qt(rrset(ds)[i]) for i in BS[ds]["ids"]}):
        qi = [i for i in BS[ds]["ids"] if qt(rrset(ds)[i]) == q]
        type_rows.append((f"{q} (n = {len(qi)})", vs_rr(ds, BEST[ds], qi), f"{DSHORT[ds]} {q}: RegisterRAG − {bshort(ds)}"))
dec_rows = []
for ds in B_HOPS:
    dec_rows.append(("hdr", f"{DSHORT[ds]} — vs {bshort(ds)}"))
    for nm, lab in (("fewer", "fewer requirements than hops"), ("equal", "requirements = hops"), ("more", "more requirements than hops")):
        t = BS[ds]["dec"][nm][0]
        if t[3]:
            dec_rows.append((f"{lab} (n = {t[3]})", t, f"{DSHORT[ds]}, {lab}"))
ret_rows = []
for ds in DSETS:
    ret_rows.append(("hdr", f"{DSHORT[ds]} — vs {bshort(ds)}"))
    ret_rows.append(("all shared questions", BS[ds]["d"], f"{DSHORT[ds]}, all"))
    ret_rows.append(("both retrieved every gold doc", BS[ds]["both_full"], f"{DSHORT[ds]}, both with doc recall@10 = 1"))
sup_rows = []
for ds in DSETS:
    sup_rows.append(("hdr", f"{DSHORT[ds]} — vs {bshort(ds)}"))
    sup_rows.append((f"register fully supported ({pct(BS[ds]['full_share'])})", BS[ds]["sup_d"][0], f"{DSHORT[ds]}, fully supported"))
    sup_rows.append((f"some requirement unsupported", BS[ds]["sup_d"][1], f"{DSHORT[ds]}, partly supported"))
type_reliable = []
for ds in DSETS:
    for q in sorted({qt(rrset(ds)[i]) for i in BS[ds]["ids"]}):
        t = vs_rr(ds, BEST[ds], [i for i in BS[ds]["ids"] if qt(rrset(ds)[i]) == q])
        if t[1] > 0 or t[2] < 0:
            type_reliable.append((q, t, None, DSHORT[ds]))
ex_rows, ex_counts = zip(*(examples_table(ds) for ds in B_HOPS))
yes_no_bias = {ds: (BS[ds]["kind"]["rr"][("yes", "no")], BS[ds]["kind"]["bb"][("yes", "no")],
                    sum(BS[ds]["kind"]["rr"][("yes", p_)] for p_ in ("yes", "no", "other"))) for ds in DSETS if ds != "musique"}
yn_other = {ds: (sum(BS[ds]["kind"]["rr"][("other", p_)] for p_ in ("yes", "no")), sum(BS[ds]["kind"]["bb"][("other", p_)] for p_ in ("yes", "no"))) for ds in DSETS}

mus_h = {h: vs_rr("musique", BEST["musique"], [i for i in BS["musique"]["ids"] if rrset("musique")[i]["n_hops"] == h]) for h in (2, 3, 4)}


# ---------------------------------------------------------------- page
CSS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_style.css"), encoding="utf-8").read()

feat_rows = [
    ("Requirement register with per-requirement evidence", "RegisterRAG",
     f"The source of its 2Wiki lead: compositional {fs(typ('2wiki','compositional'))} and inference {fs(typ('2wiki','inference'))} F1 over B1. When every requirement is supported, RegisterRAG scores higher on all four datasets (DSIRE F1 {sup['dsire'][0]:.3f} vs {sup['dsire'][1]:.3f}). The same mechanism hurts on DSIRE yes/no questions, where an unsupported requirement tends to become \"No\".",
     "keep", "Core strength; fix how missing support is read"),
    ("Iterative search loop (agent picks the next query)", "B9 LiveLedger, no-ledger ablation",
     f"The only baseline family that keeps up with RegisterRAG on 2Wiki and MuSiQue. LiveLedger is {fs(S('2wiki','liveledger'))} behind on 2Wiki (within noise), and the no-ledger loop is level on MuSiQue ({fs(S('musique','liveledger','ledger=False'))}). Both variants beat RegisterRAG on HotpotQA ({fs(S('hotpot','liveledger'))}).",
     "keep", "Worth borrowing: adaptive re-query"),
    ("Constraint ledger", "B9 full vs no-ledger",
     f"Helps on 2Wiki, where the ledger version is {S('2wiki','liveledger','ledger=False')[0] - S('2wiki','liveledger')[0]:.3f} F1 closer to RegisterRAG than the no-ledger loop, and hurts on MuSiQue. It is a register-like structure, and it helps on the same dataset where RegisterRAG's register helps most.",
     "mixed", "Comparison questions only"),
    ("One-shot chunk retrieval with whole-paragraph context", "B1, B5 gate off",
     f"Beats RegisterRAG on DSIRE (RegisterRAG {fs(S('dsire','chunk'))} F1 vs B1) and is level on HotpotQA ({fs(S('hotpot','chunk'))}). On policy yes/no questions it keeps the eligibility paragraph intact and never turns missing support into \"No\".",
     "keep", "Default for single-document policy QA"),
    ("Program-of-retrieval planning", "B8 PyRAG, coder planner",
     f"Best of the one-pass planners on 2Wiki but still well behind RegisterRAG ({fs(S('2wiki','pyrag','coder'))}); level on HotpotQA and DSIRE; behind on MuSiQue ({fs(S('musique','pyrag'))}).",
     "drop", "Superseded by RegisterRAG"),
    ("Sentence units", "B2",
     f"Level with RegisterRAG on HotpotQA ({fs(S('hotpot','sentence'))}) and DSIRE ({fs(S('dsire','sentence'))} F1) with one LLM call; far behind on 2Wiki and MuSiQue.",
     "keep", "Cheap control for retrieval-saturated sets"),
    ("Decompose once, retrieve per sub-question", "Ref: iterative decomposition",
     f"The closest one-shot analogue of RegisterRAG's decomposition, without the register. It is {fs(S('2wiki','iterative'))} behind on 2Wiki and {fs(S('musique','iterative'))} on MuSiQue, which bounds how much the register and its evidence checks add beyond decomposition alone.",
     "mixed", "Ablation reference"),
    ("Tuple memory with gap-driven re-query", "B6 EviMem",
     f"Behind RegisterRAG on 2Wiki and MuSiQue; level on HotpotQA and on DSIRE with raw context ({fs(S('dsire','evimem','answer_context=raw'))}).", "drop", "Superseded"),
    ("Coverage selection, lexical graph, segment memory", "B3 GeoRAG, B4 HLG-RAG, B7 LycheeMem",
     "Significantly behind RegisterRAG on 2Wiki and MuSiQue. GeoRAG and HLG are level with it on DSIRE only because RegisterRAG's own DSIRE score is below B1.",
     "drop", "Superseded"),
]
vcls = {"keep": "v-keep", "drop": "v-drop", "mixed": "v-mixed"}
feat_html = "".join(f'<tr><td><b>{esc(a1)}</b><br><span class="dim">{esc(b1)}</span></td><td><span class="verdict {vcls[v]}">{esc(t)}</span></td><td>{c}</td></tr>'
                    for a1, b1, c, v, t in feat_rows)

forest_figs = "".join(
    f'<figure><figcaption>{DNAME[ds]}<span class="sub">RegisterRAG F1 {score(ds, RR):.3f} (EM {emscore(ds, RR):.3f}) on {len(rrset(ds))} shared questions. Each mark: RegisterRAG minus that baseline in F1 (qwen2.5:7b), paired, 95% bootstrap CI. Hover for the EM difference.</span></figcaption>{forest(ds)}</figure>'
    for ds in DSETS)
legend = ('<div class="legend"><span><i class="pos"></i>RegisterRAG reliably better</span><span><i class="nul"></i>within noise (CI crosses 0)</span>'
          '<span><i class="neg"></i>baseline reliably better</span></div>')
n_better = len(rr_better)
by_ds = {ds: (sum(1 for d, k in rr_better if d == ds), sum(1 for d, k in rr_worse if d == ds),
              sum(1 for d, k in SIG if d == ds and k != ("none", "-"))) for ds in DSETS}
worse_list = "".join(f"<li><b>{esc(label(*k))}</b> on {DSHORT[d]}: RegisterRAG {fs(SIG[(d, k)][:3])}</li>" for d, k in rr_worse)
tie_close = sorted([(d, k) for d, k in rr_tie if k != ("none", "-")], key=lambda dk: SIG[dk][0])
tie_list = "".join(f"<li><b>{esc(label(*k))}</b> on {DSHORT[d]}: {fs(SIG[(d, k)][:3])}</li>" for d, k in tie_close)

page = f"""<title>RegisterRAG vs the Baselines</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Public+Sans:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header class="top">
  <div class="eyebrow">RegisterRAG · 9 baselines + references · qwen2.5:7b</div>
  <h1>RegisterRAG vs the Baselines</h1>
  <p class="lede">Where RegisterRAG beats each published baseline and where it doesn't, which question types decide it, and a close look at DSIRE, the one dataset where plain chunk retrieval still wins.</p>
  <div class="meta"><span>4 datasets, {sum(len(rrset(d)) for d in DSETS):,} shared questions</span><span>{len(BASE)} baseline configurations at 7B</span><span>paired bootstrap, 2,000 resamples</span><span>RegisterRAG re-scored with the harness scorer</span></div>
  <nav class="toc" aria-label="Sections"><a href="#summary">Summary</a><a href="#where">Head to head</a><a href="#types">By question type</a><a href="#best">vs strongest baseline</a><a href="#why">Why RegisterRAG wins</a><a href="#features">Baseline features</a><a href="#cost">Cost</a><a href="#dsire">DSIRE deep dive</a><a href="#size">3B vs 7B</a><a href="#reliability">Reliability</a><a href="#next">Improvements</a></nav>
</header>

<section id="summary">
  <h2>Summary</h2>
  <div class="col"><p>RegisterRAG (qwen2.5:7b) was compared with each baseline's qwen2.5:7b run on the questions both answered: {", ".join(f"{len(rrset(d))} on {DSHORT[d]}" for d in DSETS)}. <b>Token F1 is the headline metric everywhere on this page</b>: it gives partial credit for answers that differ from the gold only in wording, and works for both short and long-form answers. EM is shown beside it as a stricter, secondary check. A difference counts only when its 95% paired bootstrap interval excludes zero. On F1, RegisterRAG is reliably better in <b>{n_better} of {n_cmp}</b> comparisons, reliably worse in <b>{len(rr_worse)}</b>, and within noise in the rest.</p></div>
  <ul class="findings">
    <li><b>Against the strongest baseline: level or behind</b><span>Compared with each dataset's single best configuration, RegisterRAG is level on 2Wiki ({fs(BS['2wiki']['d'])}) and MuSiQue ({fs(BS['musique']['d'])}) with about half the LLM calls, and behind on HotpotQA and DSIRE. Where it loses, it is the reading of evidence, not the retrieval (see <a href="#best">vs strongest baseline</a>).</span></li>
    <li><b>2Wiki: a large, clear win</b><span>RegisterRAG scores {score('2wiki',RR):.3f} F1 ({emscore('2wiki',RR):.3f} EM) and beats {by_ds['2wiki'][0]} of {by_ds['2wiki'][2]} baselines, by {S('2wiki','chunk')[0]:+.3f} over B1. Only LiveLedger stays within noise ({fs(S('2wiki','liveledger'))}).</span></li>
    <li><b>MuSiQue: ahead of everything except the search loop</b><span>{score('musique',RR):.3f} F1 ({emscore('musique',RR):.3f} EM), {S('musique','chunk')[0]:+.3f} F1 over B1. Both LiveLedger variants are level with it ({fs(S('musique','liveledger','ledger=False'))} against the no-ledger loop). On 4-hop questions RegisterRAG leads B1 by {fs(mus4_d_b1)} F1 and is level with the no-ledger loop ({mus4_rr:.3f} vs {mus4_ll:.3f}).</span></li>
    <li><b>HotpotQA: no advantage</b><span>Retrieval is already saturated. RegisterRAG ({score('hotpot',RR):.3f}) is level with B1 ({fs(S('hotpot','chunk'))}) and reliably behind both LiveLedger variants ({fs(S('hotpot','liveledger'))}).</span></li>
    <li><b>DSIRE: B1 is reliably better</b><span>RegisterRAG's F1 ({score('dsire',RR):.3f}) is {S('dsire','chunk')[0]:+.3f} against B1. It loses most on yes/no, where an unsupported requirement becomes "No": it misses {pct(1-yn[RR]['yes'])} of gold-"yes" questions against {pct(1-yn[('chunk','-')]['yes'])} for B1.</span></li>
    <li><b>The register does the work</b><span>When all of RegisterRAG's requirements are supported by evidence, it scores higher on every dataset: {sup['musique'][0]:.3f} vs {sup['musique'][1]:.3f} F1 on MuSiQue, {sup['2wiki'][0]:.3f} vs {sup['2wiki'][1]:.3f} on 2Wiki. Support coverage is a usable confidence signal.</span></li>
    <li><b>What to borrow from the baselines</b><span>LiveLedger's adaptive re-query is the only baseline feature that matches or beats RegisterRAG on multi-hop data. B1's whole-paragraph context is what wins DSIRE yes/no.</span></li>
  </ul>
</section>

<section id="where">
  <h2>Head to head, per dataset</h2>
  <div class="col"><p>Each row is one baseline. The mark is RegisterRAG's score minus that baseline's on the same questions, so right of zero means RegisterRAG is better. Hover a mark for its interval and question count.</p></div>
  {legend}
  <div class="figs">{forest_figs}</div>
  <div class="figs">
    <div class="col"><h3>Where a baseline is reliably better than RegisterRAG</h3><ul class="tight">{worse_list}</ul></div>
    <div class="col"><h3>Within noise (closest first)</h3><ul class="tight">{tie_list}</ul></div>
  </div>
</section>

<section id="types">
  <h2>By question type</h2>
  <div class="col"><p>Shared questions only. "Best baseline" is the strongest of the {len(BASE)} baseline configurations on that question type. The Δ columns are RegisterRAG minus the baseline (F1 first, EM last); green and red cells have a 95% interval that excludes zero. Small grey figures under each score are EM.</p></div>
  <h3>2WikiMultiHopQA</h3>{qtype_table('2wiki')}
  <div class="col"><p>RegisterRAG leads B1 on every 2Wiki question type, by {min(typ('2wiki',q)[0] for q in ('comparison','bridge_comparison','compositional','inference')):+.2f} to {max(typ('2wiki',q)[0] for q in ('comparison','bridge_comparison','compositional','inference')):+.2f} F1. The largest gain is on <b>compositional</b> questions ({fs(typ('2wiki','compositional'))}), which need a bridge entity found first and then a relation applied to it, exactly what an explicit requirement list enforces. On <b>inference</b> questions (kinship chains such as "paternal grandfather") the F1 gain ({typ('2wiki','inference')[0]:+.3f}) is smaller than the EM gain ({typ('2wiki','inference',met='em')[0]:+.3f}): B1 often names a related person, which earns partial F1 but no EM.</p></div>
  <h3>HotpotQA</h3>{qtype_table('hotpot')}
  <div class="col"><p>Every system retrieves HotpotQA's two paragraphs (recall@10 ≥ 0.92), so decomposition has little to add. On bridge questions RegisterRAG is {fs(hp_bridge)} vs B1. Splitting a bridge question into requirements may lose the phrasing a one-shot reader uses directly.</p></div>
  <h3>MuSiQue</h3>{qtype_table('musique')}
  <div class="col"><p>MuSiQue was built so that no hop can be skipped. With top-5 units, B1 never has all four documents of a 4-hop question, while RegisterRAG collects evidence requirement by requirement. On 4-hop questions ({len(mus4_ids)} shared) RegisterRAG reaches {mus4_rr:.3f} F1 against {mus4_b1:.3f} for B1 ({fs(mus4_d_b1)}). LiveLedger without its ledger reaches the same level ({mus4_ll:.3f}; RegisterRAG minus it {fs(mus4_d_ll)}) with no requirement list at all. RegisterRAG's biggest MuSiQue lead over B1 is on 2-hop questions ({fs(typ('musique','2hop'))} F1). 4hop2 has only {len([i for i, r in rrset('musique').items() if qt(r) == '4hop2'])} shared questions; read it with care.</p></div>
  <h3>DSIRE</h3>{qtype_table('dsire')}
</section>

<section id="best">
  <h2>RegisterRAG vs the strongest baseline on each dataset</h2>
  <div class="col"><p>The sections above compare RegisterRAG with every baseline. This one asks a harder question: how does it do against the <b>single strongest</b> baseline on each dataset? "Strongest" means the 7B configuration with the highest F1 on the shared questions. It is picked after seeing the results, which favours the baseline, so this is a conservative test. All differences are RegisterRAG minus the strongest baseline in token F1, paired, with 95% bootstrap intervals. The agreement matrices and McNemar tests treat F1 ≥ 0.5 as correct.</p></div>

  <figure class="tbl keep"><figcaption>Table A. Main results against the strongest baseline (qwen2.5:7b)<span class="sub">Shared questions answered by RegisterRAG, B1 and the strongest baseline. Grey figures under each score are EM. Green/red Δ cells have a 95% interval excluding zero. LLM calls count answer and planning calls.</span></figcaption>{main_table()}</figure>

  <h3>Key findings</h3>
  <ul class="findings">
    <li><b>1 · Level on 2Wiki and MuSiQue, behind on HotpotQA and DSIRE</b><span>RegisterRAG minus the strongest: 2Wiki {fs(BS['2wiki']['d'])}, MuSiQue {fs(BS['musique']['d'])}, HotpotQA {fs(BS['hotpot']['d'])}, DSIRE {fs(BS['dsire']['d'])}. McNemar agrees: p {pfmt(BS['hotpot']['mcn'])} on HotpotQA and {pfmt(BS['dsire']['mcn'])} on DSIRE; not significant on the other two.</span></li>
    <li><b>2 · The bar is an iterative search loop, at about twice the calls</b><span>A LiveLedger variant is the strongest baseline on 2Wiki, HotpotQA and MuSiQue; on DSIRE it is one-shot B1. RegisterRAG matches LiveLedger on 2Wiki with {BS['2wiki']['calls_rr']:.1f} LLM calls per question against {BS['2wiki']['calls_bb']:.1f}, and on MuSiQue with {BS['musique']['calls_rr']:.1f} against {BS['musique']['calls_bb']:.1f}.</span></li>
    <li><b>3 · Retrieval is not the gap; reading is</b><span>RegisterRAG finds every gold document more often than the strongest baseline on HotpotQA ({pct(BS['hotpot']['full_rr'])} vs {pct(BS['hotpot']['full_bb'])}) and MuSiQue ({pct(BS['musique']['full_rr'])} vs {pct(BS['musique']['full_bb'])}). On the questions where <i>both</i> retrieved every gold document it still trails: HotpotQA {fs(BS['hotpot']['both_full'])}, DSIRE {fs(BS['dsire']['both_full'])}, MuSiQue {fs(BS['musique']['both_full'])}.</span></li>
    <li><b>4 · Decomposition fidelity decides the outcome</b><span>When RegisterRAG writes as many requirements as the question has gold hops, it beats LiveLedger on 2Wiki ({fs(BS['2wiki']['dec']['equal'][0])}). Too few requirements costs {BS['2wiki']['dec']['fewer'][0][0]:+.3f} on 2Wiki; too many costs {BS['hotpot']['dec']['more'][0][0]:+.3f} on HotpotQA and {BS['musique']['dec']['more'][0][0]:+.3f} on MuSiQue.</span></li>
    <li><b>5 · The "No" bias is not a DSIRE quirk</b><span>Gold "yes" answered "No": 2Wiki {yes_no_bias['2wiki'][0]} of {yes_no_bias['2wiki'][2]} (strongest {yes_no_bias['2wiki'][1]}), HotpotQA {yes_no_bias['hotpot'][0]} of {yes_no_bias['hotpot'][2]} ({yes_no_bias['hotpot'][1]}), DSIRE {yes_no_bias['dsire'][0]} of {yes_no_bias['dsire'][2]} ({yes_no_bias['dsire'][1]}). It is a property of how RegisterRAG answers, not of policy text.</span></li>
    <li><b>6 · The errors are complementary, and the register says when to defer</b><span>A per-question oracle of the two systems reaches {BS['2wiki']['oracle']:.3f} F1 on 2Wiki and {BS['hotpot']['oracle']:.3f} on HotpotQA. Sending only RegisterRAG's partly supported questions to the strongest baseline gives {BS['2wiki']['route_bb_v']:.3f} on 2Wiki ({fs(BS['2wiki']['route_bb'])} over the strongest alone) and halves the HotpotQA gap.</span></li>
  </ul>

  <h3>Per-question outcomes and paired agreement</h3>
  <div class="figs">
    <figure class="keep"><figcaption>Figure A. Who scores higher, question by question<span class="sub">Share of shared questions where RegisterRAG's F1 is higher than, equal to, or lower than the strongest baseline's. Hover for counts.</span></figcaption>
      <div class="legend"><span><i class="sq segpos"></i>RegisterRAG higher</span><span><i class="sq segnul"></i>same F1</span><span><i class="sq segneg"></i>strongest higher</span></div>{outcome_chart()}</figure>
    <div class="col" style="align-self:center">
      <p>Most questions are decided the same way by both systems: {", ".join(f"{pct(BS[d]['wtl'][1] / BS[d]['n'])} on {DSHORT[d]}" for d in DSETS)} have identical F1, usually both right or both wrong. The comparison rests on the remaining questions.</p>
      <p>Among those, RegisterRAG wins more than it loses only on 2Wiki ({BS['2wiki']['wtl'][0]} vs {BS['2wiki']['wtl'][2]}). On HotpotQA it loses {BS['hotpot']['wtl'][2]} and wins {BS['hotpot']['wtl'][0]}. On MuSiQue and DSIRE wins and losses are close in number, but the losses are larger.</p>
      <p>The agreement matrices below show the same picture with binary correctness. Large off-diagonal cells on every dataset mean the two systems fail on different questions, which is what makes routing worthwhile (finding 6).</p>
    </div>
  </div>
  <div class="figs4">{"".join(agree_table(d) for d in DSETS)}</div>

  <h3>By question type</h3>
  <div class="figs">
    <figure class="keep"><figcaption>Figure B. RegisterRAG minus the strongest baseline, per question type<span class="sub">F1, paired on shared questions, 95% bootstrap CI.</span></figcaption>{legend}{ci_plot(type_rows, -0.4, 0.3, "RegisterRAG minus the strongest baseline per question type", [-0.4, -0.2, 0, 0.2])}</figure>
    <div class="col" style="align-self:center">
      <p><b>Only {len(type_reliable)} of {len(type_rows) - len(DSETS)} type-level differences {"is" if len(type_reliable) == 1 else "are"} reliable:</b> {", ".join(f"{esc(r[0].split(' (')[0])} on {r[3]} ({fs(r[1])})" for r in type_reliable)}, in the baseline's favour. Two more come close: MuSiQue 3hop1 ({fs(typ('musique', '3hop1', BEST['musique']))}) and DSIRE Yes-No ({fs(typ('dsire', 'Yes-No', BEST['dsire']))}). Per type, RegisterRAG and the strongest baseline are otherwise statistically tied; the per-type samples ({min(r[1][3] for r in type_rows if r[0] != 'hdr')}–{max(r[1][3] for r in type_rows if r[0] != 'hdr')} questions) are too small to separate differences under about 0.1.</p>
      <p><b>RegisterRAG leans ahead on 2Wiki comparison and inference questions</b> (comparison {typ('2wiki', 'comparison', BEST['2wiki'])[0]:+.3f}, inference {typ('2wiki', 'inference', BEST['2wiki'])[0]:+.3f}), where a fixed list of attributes or relations is the right plan. It leans behind on questions whose structure is open-ended: bridge questions and 3-hop chains, where the iterative loop can follow whatever the first hop returns.</p>
    </div>
  </div>

  <h3>Retrieval is not the gap: reading is</h3>
  <div class="figs">
    <figure class="keep"><figcaption>Figure C. The gap with retrieval held fixed<span class="sub">RegisterRAG minus the strongest baseline on all shared questions, and on the subset where both systems had every gold document in their top 10.</span></figcaption>{legend}{ci_plot(ret_rows, -0.3, 0.2, "Gap with retrieval held fixed", [-0.3, -0.2, -0.1, 0, 0.1, 0.2])}</figure>
    <div class="col" style="align-self:center">
      <p>If RegisterRAG lost because it found less evidence, the gap would close when both systems have all the gold documents. It doesn't. On HotpotQA it stays at {BS['hotpot']['both_full'][0]:+.3f}, on DSIRE at {BS['dsire']['both_full'][0]:+.3f}, and on MuSiQue it grows to {BS['musique']['both_full'][0]:+.3f} ({BS['musique']['both_f1'][0]:.3f} vs {BS['musique']['both_f1'][1]:.3f}, n = {BS['musique']['both_full'][3]}).</p>
      <p>So with the same evidence available, the strongest baseline turns it into the right answer more often. RegisterRAG's own failure labels point the same way: on the MuSiQue questions it loses, {BS['musique']['loss_fail']['evidence_selection']} of {BS['musique']['nloss']} are <code>evidence_selection</code> (the evidence was in the pool but not attached to the requirement) and only {BS['musique']['loss_fail']['partial_retrieval'] + BS['musique']['loss_fail']['retrieval']} are retrieval failures.</p>
      <p>2Wiki is the exception. There, with all gold documents present, RegisterRAG leans ahead ({BS['2wiki']['both_f1'][0]:.3f} vs {BS['2wiki']['both_f1'][1]:.3f}; {fs(BS['2wiki']['both_full'])}), so on comparison-structured questions the register does help reading.</p>
    </div>
  </div>

  <h3>Decomposition fidelity</h3>
  <div class="figs">
    <figure class="keep"><figcaption>Figure D. The gap by how well the register matches the question's hop structure<span class="sub">Requirement count vs the dataset's gold hop count. HotpotQA questions are all 2-hop. DSIRE is single-document and left out.</span></figcaption>{legend}{ci_plot(dec_rows, -0.45, 0.25, "Gap by decomposition fidelity", [-0.4, -0.2, 0, 0.2])}</figure>
    <div class="col" style="align-self:center">
      <p><b>A correctly sized register is RegisterRAG's real advantage.</b> On 2Wiki, {BS['2wiki']['dec']['equal'][0][3]} of {BS['2wiki']['n']} questions get one requirement per hop, and on those RegisterRAG beats LiveLedger by {fs(BS['2wiki']['dec']['equal'][0])} ({BS['2wiki']['dec']['equal'][1]:.3f} vs {BS['2wiki']['dec']['equal'][2]:.3f}).</p>
      <p><b>Too few requirements</b> leaves a hop unplanned. On 2Wiki this happens on {BS['2wiki']['dec']['fewer'][0][3]} questions, {BS['2wiki']['dec']['fewer'][3][1]} of them {BS['2wiki']['dec']['fewer'][3][0]} (4-hop), and costs {fs(BS['2wiki']['dec']['fewer'][0])}. On MuSiQue it is common ({BS['musique']['dec']['fewer'][0][3]} questions) but costs little ({BS['musique']['dec']['fewer'][0][0]:+.3f}), because the loop also struggles there.</p>
      <p><b>Too many requirements</b> is rarer but worse: HotpotQA {fs(BS['hotpot']['dec']['more'][0])} on {BS['hotpot']['dec']['more'][0][3]} questions, MuSiQue {fs(BS['musique']['dec']['more'][0])} on {BS['musique']['dec']['more'][0][3]}. Each extra requirement is one more chance for a support check to fail or for the composer to pick up the wrong span.</p>
      <p>The planner's hop estimate is available before answering, so a cheap check (does the plan have more steps than the question has entities?) could catch over-decomposition at inference time.</p>
    </div>
  </div>

  <h3>Answer-type confusion matrices</h3>
  <div class="col"><p>These extend the DSIRE yes/no matrices to every dataset and add the "other" class: a question whose gold answer is an entity but is answered "yes" or "no" (or the reverse) is a form error that no amount of retrieval fixes. MuSiQue has no yes/no gold answers, so only its "other" row appears.</p></div>
  <div class="figs">{"".join(kind_cm(d) for d in DSETS)}</div>
  <div class="col"><ul class="tight">
    <li><b>Gold "yes" → "No" is RegisterRAG's dominant yes/no error on all three datasets that have yes/no questions,</b> and it makes it more often than the strongest baseline each time ({", ".join(f"{DSHORT[d]} {yes_no_bias[d][0]} vs {yes_no_bias[d][1]}" for d in yes_no_bias)}). On 2Wiki this is half of all gold-"yes" comparison questions. Both systems almost never say "yes" to a gold "no".</li>
    <li><b>Answering an entity question with yes/no</b> (the "Gold other" row) is a HotpotQA-specific RegisterRAG problem: {yn_other['hotpot'][0]} times against {yn_other['hotpot'][1]} for the strongest baseline. Questions such as "Which of these men…?" get "No". On 2Wiki and MuSiQue it is the iterative loop that does this more ({yn_other['2wiki'][1]} and {yn_other['musique'][1]} times, against {yn_other['2wiki'][0]} and {yn_other['musique'][0]} for RegisterRAG). On MuSiQue LiveLedger answers "no" to questions that never have a yes/no answer.</li>
    <li>A simple answer-type guard (yes/no only if the question starts with an auxiliary verb) should remove most of the "Gold other" errors for both systems.</li>
  </ul></div>

  <h3>Register support as a diagnostic, and a router</h3>
  <div class="figs">
    <figure class="keep"><figcaption>Figure E. The gap when RegisterRAG's register is fully vs partly supported<span class="sub">RegisterRAG minus the strongest baseline, split by RegisterRAG's own support signal.</span></figcaption>{legend}{ci_plot(sup_rows, -0.35, 0.2, "Gap by register support", [-0.3, -0.2, -0.1, 0, 0.1, 0.2], L=230)}</figure>
    <div class="col" style="align-self:center">
      <p><b>On 2Wiki the support signal separates RegisterRAG's wins from its losses.</b> On fully supported questions it beats LiveLedger ({fs(BS['2wiki']['sup_d'][0])}). On the {BS['2wiki']['sup_d'][1][3]} partly supported ones it loses badly ({fs(BS['2wiki']['sup_d'][1])}). HotpotQA and MuSiQue show the same direction, with the partly supported gap 2–3× larger.</p>
      <p><b>On DSIRE the signal carries no information</b> (the gap is {BS['dsire']['sup_d'][0][0]:+.3f} supported, {BS['dsire']['sup_d'][1][0]:+.3f} unsupported). That fits the deep dive: there the support checker fails on evidence that is present, so "unsupported" doesn't mean "the evidence is missing".</p>
      <p>The signal is available at inference time, so the router below is deployable. It is one fixed rule, not tuned on these results: keep RegisterRAG's answer when every requirement is supported, otherwise use the fallback's.</p>
    </div>
  </div>
  <figure class="tbl keep"><figcaption>Table B. Routing on RegisterRAG's support signal (F1)<span class="sub">Δ is the router minus the strongest baseline alone, paired. The oracle picks the better of RegisterRAG and the strongest baseline per question and is an upper bound, not a system.</span></figcaption>{route_table()}</figure>
  <div class="col"><p>The router beats both of its parts on 2Wiki and closes most of the gap on MuSiQue ({BS['musique']['route_bb'][0]:+.3f}, within noise). Falling back to cheap B1 instead keeps the 2Wiki result level with LiveLedger ({fs(BS['2wiki']['route_b1'])}) at about {BS['2wiki']['calls_rr'] + (1 - BS['2wiki']['full_share']):.1f} calls per question instead of {BS['2wiki']['calls_bb']:.1f}, but loses on MuSiQue, where B1 is weak. The oracle column shows the headroom: it is {min(BS[d]['oracle'] - max(BS[d]['rr'], BS[d]['bb']) for d in DSETS):.2f}–{max(BS[d]['oracle'] - max(BS[d]['rr'], BS[d]['bb']) for d in DSETS):.2f} F1 above the better of the two systems on every dataset.</p></div>

  <h3>Accuracy per LLM call</h3>
  <div class="figs4">{"".join(f'<figure class="keep"><figcaption>{DSHORT[d]}<span class="sub">Every 7B system on the shared questions. Grey: other baselines.</span></figcaption>{cost_chart(d)}</figure>' for d in DSETS)}</div>
  <div class="col"><p>On 2Wiki and MuSiQue RegisterRAG sits on the accuracy/cost frontier: no baseline with fewer calls comes close to it. The only systems level with it are LiveLedger variants using {BS['2wiki']['calls_bb'] / BS['2wiki']['calls_rr']:.1f}× (2Wiki) and {BS['musique']['calls_bb'] / BS['musique']['calls_rr']:.1f}× (MuSiQue) as many calls. On HotpotQA it is off the frontier: the no-ledger loop is more accurate at the same number of calls, and one-shot B1 ({BS['hotpot']['b1']:.3f}) is more accurate with one call. On DSIRE B1 is also more accurate with one call, so RegisterRAG's extra calls buy nothing there.</p></div>

  <h3>Hop depth</h3>
  <div class="figs4">
    <figure class="keep"><figcaption>2Wiki: F1 by gold hops<span class="sub">Hop groups with ≥ 20 shared questions.</span></figcaption>{hop_chart('2wiki', [(RR, 's7', 'RR'), (BEST['2wiki'], 's3', 'strongest'), (('chunk', '-'), 'g', 'B1')])}</figure>
    <figure class="keep"><figcaption>MuSiQue: F1 by gold hops<span class="sub">Hop groups with ≥ 20 shared questions.</span></figcaption>{hop_chart('musique', [(RR, 's7', 'RR'), (BEST['musique'], 's3', 'strongest'), (('chunk', '-'), 'g', 'B1')])}</figure>
    <div class="col" style="align-self:center;grid-column:span 2"><p>On MuSiQue all three systems lose accuracy with depth, but differently. B1 collapses at 4 hops; RegisterRAG and the loop do not. RegisterRAG's deficit against the loop sits at 3 hops ({fs(mus_h[3])}); at 4 hops the two are level ({fs(mus_h[4])}). On 2Wiki the 4-hop questions are comparisons of two 2-hop chains, which are no harder than 2-hop questions for either multi-call system.</p></div>
  </div>

  <h3>Key insights, dataset by dataset</h3>
  <div class="figs">
    <div class="col"><h3 style="margin-top:0">2WikiMultiHopQA · vs {esc(bshort('2wiki'))}</h3><ul class="tight">
      <li><b>Statistically level overall ({fs(BS['2wiki']['d'])}), at {BS['2wiki']['calls_rr']:.1f} vs {BS['2wiki']['calls_bb']:.1f} LLM calls.</b> This is the only dataset where RegisterRAG is the most accurate system in the study.</li>
      <li>Its edge comes from questions with a correctly sized register ({pct(BS['2wiki']['dec']['equal'][0][3] / BS['2wiki']['n'])} of questions, {fs(BS['2wiki']['dec']['equal'][0])}) and from fully supported registers ({fs(BS['2wiki']['sup_d'][0])}).</li>
      <li>Its weakest subsets are 4-hop bridge-comparison questions planned with too few requirements ({BS['2wiki']['dec']['fewer'][0][0]:+.3f}), partly supported registers ({BS['2wiki']['sup_d'][1][0]:+.3f}), and "No" answers to gold "yes" ({yes_no_bias['2wiki'][0]} of {yes_no_bias['2wiki'][2]}).</li>
      <li>Routing partly supported questions to LiveLedger gives {BS['2wiki']['route_bb_v']:.3f} F1, the best 2Wiki result available from these runs.</li></ul></div>
    <div class="col"><h3 style="margin-top:0">HotpotQA · vs {esc(bshort('hotpot'))}</h3><ul class="tight">
      <li><b>Reliably behind ({fs(BS['hotpot']['d'])}; McNemar p {pfmt(BS['hotpot']['mcn'])}) at the same cost</b> ({BS['hotpot']['calls_rr']:.1f} vs {BS['hotpot']['calls_bb']:.1f} calls). The loss is on bridge questions; comparison is level.</li>
      <li>It is not a retrieval loss: RegisterRAG has all gold documents more often ({pct(BS['hotpot']['full_rr'])} vs {pct(BS['hotpot']['full_bb'])}) and still trails when both do.</li>
      <li>Three causes, each measured: over-decomposition ({BS['hotpot']['dec']['more'][0][3]} questions, {BS['hotpot']['dec']['more'][0][0]:+.3f}); yes/no answers to entity questions ({yn_other['hotpot'][0]} vs {yn_other['hotpot'][1]}); and longer, less precise answers ({BS['hotpot']['words'][0]:.1f} vs {BS['hotpot']['words'][1]:.1f} words, token precision {BS['hotpot']['pr'][0]:.2f} vs {BS['hotpot']['pr'][2]:.2f}).</li>
      <li>Even on fully supported questions it trails ({fs(BS['hotpot']['sup_d'][0])}), so better support checks alone won't close this gap; the answer composer needs work.</li></ul></div>
    <div class="col"><h3 style="margin-top:0">MuSiQue · vs {esc(bshort('musique'))}</h3><ul class="tight">
      <li><b>Level overall ({fs(BS['musique']['d'])}) at less than half the calls</b> ({BS['musique']['calls_rr']:.1f} vs {BS['musique']['calls_bb']:.1f}).</li>
      <li>The deficit is on 3-hop questions ({fs(mus_h[3])}); 2- and 4-hop are level.</li>
      <li>RegisterRAG retrieves better (all gold documents on {pct(BS['musique']['full_rr'])} vs {pct(BS['musique']['full_bb'])} of questions) but uses the evidence worse: {BS['musique']['both_f1'][0]:.3f} vs {BS['musique']['both_f1'][1]:.3f} F1 when both have it. Most of its losses are <code>evidence_selection</code> failures ({BS['musique']['loss_fail']['evidence_selection']} of {BS['musique']['nloss']}).</li>
      <li>The rare over-decomposed questions are the worst case in the study ({fs(BS['musique']['dec']['more'][0])}).</li></ul></div>
    <div class="col"><h3 style="margin-top:0">DSIRE · vs {esc(bshort('dsire'))}</h3><ul class="tight">
      <li><b>Reliably behind ({fs(BS['dsire']['d'])}) with three times the calls.</b> See the <a href="#dsire">DSIRE deep dive</a>.</li>
      <li>Retrieval is identical (doc recall@10 {BS['dsire']['rec_rr']:.2f} vs {BS['dsire']['rec_bb']:.2f}), so the whole gap is answer composition: yes/no polarity and long answers ({BS['dsire']['words'][0]:.1f} vs {BS['dsire']['words'][1]:.1f} words, precision {BS['dsire']['pr'][0]:.2f} vs {BS['dsire']['pr'][2]:.2f}, recall {BS['dsire']['pr'][1]:.2f} vs {BS['dsire']['pr'][3]:.2f}).</li>
      <li>This is the only dataset where the support signal is uninformative; it is the support checker, not the evidence, that fails here.</li></ul></div>
  </div>

  <h3>Error analysis: examples</h3>
  <div class="col"><p>A fixed-seed sample of clear cases: two where RegisterRAG scores F1 0 and the strongest baseline F1 1, and one the other way round, per dataset. Clear losses outnumber clear wins on HotpotQA ({ex_counts[1][0]} vs {ex_counts[1][1]}) and MuSiQue ({ex_counts[2][0]} vs {ex_counts[2][1]}), and are about even on 2Wiki ({ex_counts[0][0]} vs {ex_counts[0][1]}). "Register" is supported / total requirements; "RR label" is RegisterRAG's own failure label.</p></div>
  <div class="tablewrap"><table class="data ex"><thead><tr><th>Dataset</th><th>Case</th><th>Question</th><th>Gold</th><th>RegisterRAG</th><th>Strongest baseline</th><th>Register</th><th>RR label</th></tr></thead><tbody>{"".join(ex_rows)}</tbody></table></div>

  <div class="callout"><p><b>Threats to validity for this section.</b> The strongest baseline is chosen on the same questions it is evaluated on, which inflates it slightly (winner's curse). This works against RegisterRAG, so "level" results are, if anything, conservative. The F1 ≥ 0.5 threshold used for agreement and McNemar is a convention; the F1 bootstrap intervals don't depend on it. RegisterRAG's recall comes from its own logs and may be computed over a different corpus pool. The routing result is a post-hoc analysis of one fixed rule; it should be confirmed on a held-out split before it is claimed as a system result.</p></div>
</section>

<section id="why">
  <h2>Why RegisterRAG wins where it wins</h2>
  <div class="figs">
    <figure><figcaption>RegisterRAG's score by register support<span class="sub">Questions where every requirement was supported by evidence vs those where at least one was not. Shared questions, primary metric.</span></figcaption>
      <div class="legend"><span><i class="sq b7"></i>all requirements supported</span><span><i class="sq b3"></i>some unsupported</span></div>{support_chart()}</figure>
    <div class="col" style="align-self:center">
      <p><b>The register separates confident answers from unconfident ones.</b> On every dataset, fully supported questions score higher. The gap is largest on MuSiQue ({sup['musique'][0]:.3f} vs {sup['musique'][1]:.3f}), where missing a hop is the main failure. The baselines have nothing like this: no record of which part of the question their evidence covers.</p>
      <p><b>Coverage differs a lot by dataset:</b> {", ".join(f"{pct(sup[d][2])} fully supported on {DSHORT[d]}" for d in DSETS)}. DSIRE's low coverage is surprising, since its retrieval is near perfect. RegisterRAG's requirement check fails to confirm evidence that is in the policy page, and that is where its DSIRE losses come from (see the DSIRE deep dive).</p>
    </div>
  </div>
</section>

<section id="features">
  <h2>What each baseline feature contributes, measured against RegisterRAG</h2>
  <div class="tablewrap"><table class="data feat"><thead><tr><th>Feature</th><th>Verdict</th><th>Evidence (RegisterRAG − baseline, 95% CI)</th></tr></thead><tbody>{feat_html}</tbody></table></div>
</section>

<section id="cost">
  <h2>Accuracy and cost</h2>
  <div class="col"><p>Dataset cells show RegisterRAG minus the baseline; the RegisterRAG row shows its own score. LLM calls and median latency are averaged over the shared questions of the datasets each system ran on.</p></div>
  {eff_table()}
  <div class="callout"><p><b>Latency and tokens are not directly comparable.</b> RegisterRAG's traces use Windows paths (<code>runs\\run2_500ctx\\…</code>), so it most likely ran on a different machine, with a median of {min(rr_lat.values()):.1f}–{max(rr_lat.values()):.1f} s per question. Its token count covers LLM prompt and completion only, while the baselines' count also includes reranker inputs, so token totals are left out of this table. LLM calls are comparable: RegisterRAG makes about 3 per question (2–5). That is fewer than PyRAG and LiveLedger, about one more than the decompose-once baselines, and two more than one-shot B1. Baseline counts are answer plus planning calls; verifier calls (EviMem about 2 per question, SURE-RAG 5) are not counted because they may run on a separate model.</p></div>
</section>

<section id="dsire">
  <h2>DSIRE deep dive</h2>
  <div class="col"><p>DSIRE (RePolicyQA) has {len(data)} questions over {len(docs)} renewable-energy program pages. Each question names its program and state, so retrieval is close to solved (doc recall@10: B1 {m(Bd[i]['retrieval']['doc_recall@10'] for i in Rd):.2f}, RegisterRAG {m(r['retrieval']['doc_recall@10'] for r in Rd.values()):.2f}). The scores are decided by how the answer is phrased and how eligibility evidence is read. The two sides share {len(Rd)} questions. RegisterRAG's set has 100 questions per type, so the shared set's type mix differs slightly from the baselines' 500.</p></div>

  <h3>1. Per type: RegisterRAG trails B1 on all five</h3>
  <div class="tablewrap"><table class="data"><thead><tr><th>Type</th><th>Shared n</th><th>Gold words<br><span class="sub">median</span></th><th>RegisterRAG F1</th><th>B1 F1</th><th>RegisterRAG − B1</th><th>RegisterRAG answer words</th><th>B1 answer words</th><th>RegisterRAG LLM judge</th></tr></thead><tbody>
  {"".join(f"<tr><th scope='row'>{q}</th><td class='num'>{len([i for i in Rd if Rd[i]['qtype']==q])}</td><td class='num'>{med(gold_len[q])}</td><td class='num rrcol'>{m(Rd[i]['f1'] for i in Rd if Rd[i]['qtype']==q):.3f}</td><td class='num'>{m(Bd[i]['f1'] for i in Rd if Rd[i]['qtype']==q):.3f}</td>{cell_vs(typ('dsire', q))}<td class='num'>{m(len(Rd[i]['prediction'].split()) for i in Rd if Rd[i]['qtype']==q):.1f}</td><td class='num'>{m(len(Bd[i]['prediction'].split()) for i in Rd if Rd[i]['qtype']==q):.1f}</td><td class='num'>{m(Rd[i]['judge'] for i in Rd if Rd[i]['qtype']==q):.3f}</td></tr>" for q in QT)}
  </tbody></table></div>

  <h3>2. Yes/no: unsupported requirements become "No"</h3>
  <div class="figs">
    <figure><figcaption>Accuracy on gold-"yes" vs gold-"no" (qwen2.5:7b)<span class="sub">DSIRE Yes-No and Yes-No-cond with a yes/no gold answer, shared questions ({yn[RR]['ny']} yes, {yn[RR]['nn']} no).</span></figcaption>
      <div class="legend"><span><i class="b7"></i>correct when gold is "yes"</span><span><i class="sq b3"></i>correct when gold is "no"</span></div>{dumbbell(yn)}</figure>
    <div class="col" style="align-self:center">
      <p>{pct(yn[RR]['ny']/(yn[RR]['ny']+yn[RR]['nn']))} of these gold answers are "yes". RegisterRAG gets {pct(yn[RR]['yes'])} of them right against {pct(yn[('chunk','-')]['yes'])} for B1, while its accuracy on gold-"no" is high ({pct(yn[RR]['no'])}). Its errors run one way: "No" when the answer is "yes".</p>
      <p><b>The register explains it.</b> Of the {len(rr_yes_wrong)} gold-"yes" questions RegisterRAG got wrong, {pct(part_share_wrong)} had at least one unsupported requirement, against {pct(part_share_right)} of the ones it got right. B1 answered {pct(b1_right_on_rr_wrong)} of those same questions correctly, so the evidence was retrievable. RegisterRAG treated an unconfirmed condition as a negative answer. For eligibility questions ("Can schools receive up to $1,550…?"), missing confirmation is not evidence of "no".</p>
    </div>
  </div>

  <h3>Confusion matrices: RegisterRAG vs B1 on yes/no questions</h3>
  <div class="figs">{cm_all[0]}
    <div class="col" style="align-self:center">
      <p><b>RegisterRAG's errors run one way.</b> {_c_rr[('yes','no')]} of its {rr_err} errors are gold "yes" answered "No"; it says "yes" to a gold "no" only {_c_rr[('no','yes')]} times. B1 makes the same mistake less often ({_c_b1[('yes','no')]} times).</p>
      <p><b>RegisterRAG's "No" is unreliable.</b> It says "No" {rr_no_said} times and is right {rr_no_right} times ({pct(rr_no_right/rr_no_said)}). B1 says "No" {b1_no_said} times and is right {b1_no_right} ({pct(b1_no_right/b1_no_said)}). Both are almost always right when they say "yes" (RegisterRAG {_c_rr[('yes','yes')]} of {_c_rr[('yes','yes')]+_c_rr[('no','yes')]}, B1 {_c_b1[('yes','yes')]} of {_c_b1[('yes','yes')]+_c_b1[('no','yes')]}).</p>
      <p><b>Question by question:</b> {cm_agree[(True,True)]} both correct, {cm_agree[(False,True)]} only B1 correct, {cm_agree[(True,False)]} only RegisterRAG correct, {cm_agree[(False,False)]} both wrong. B1 recovers about twice as many of RegisterRAG's errors as the reverse.</p>
      <p>The pattern is the same with and without explicit conditions (below), so it is not specific to Yes-No-cond. B1's {len(other_preds)} "other" answers ({", ".join(f'"{esc(x)}"' for x in other_preds)}) answered a gold-"yes" question with a fact from the evidence.</p>
    </div>
  </div>
  <div class="figs">{cm_yn[0]}{cm_ync[0]}</div>

  <h3>3. Long-form answers: longer, but less precise</h3>
  <div class="tablewrap"><table class="data"><thead><tr><th>Type</th><th>Answer words<br><span class="sub">RegisterRAG / B1</span></th><th>Token precision<br><span class="sub">RegisterRAG / B1</span></th><th>Token recall<br><span class="sub">RegisterRAG / B1</span></th><th>RegisterRAG LLM judge</th></tr></thead><tbody>
  {"".join(f"<tr><th scope='row'>{q}</th><td class='num'>{prr[q]['rw']:.1f} / {prr[q]['bw']:.1f}</td><td class='num'>{prr[q]['rp']:.2f} / {prr[q]['bp']:.2f}</td><td class='num'>{prr[q]['rr']:.2f} / {prr[q]['br']:.2f}</td><td class='num'>{prr[q]['judge']:.3f}</td></tr>" for q in ('Factual','Descriptive','Legal-Obligation'))}
  </tbody></table></div>
  <div class="figs">
    <figure><figcaption>Across the baseline runs, longer long-form answers score higher<span class="sub">Mean token F1 by prediction length, Descriptive + Legal-Obligation, every DSIRE baseline run pooled (both generators).</span></figcaption>{length_chart()}</figure>
    <div class="col" style="align-self:center">
      <p>The baselines' shared prompt asks for "the shortest possible answer", which caps their recall on long-form gold (median {med(gold_len['Descriptive'])} words for Descriptive). RegisterRAG writes 2–4× longer answers and recovers more of the gold ({prr['Descriptive']['rr']:.2f} vs {prr['Descriptive']['br']:.2f} recall on Descriptive). Its precision drops more than its recall rises ({prr['Descriptive']['rp']:.2f} vs {prr['Descriptive']['bp']:.2f}), so its F1 ends up slightly below B1's.</p>
      <p>The extra words come from combining several requirement answers rather than the one gold sentence. Longer answers help only when the added words come from the right span.</p>
      <p><b>F1 likely underrates RegisterRAG's long-form answers.</b> Its LLM-judge score is {m(r['judge'] for r in Rd.values()):.3f} against an F1 of {score('dsire',RR):.3f}, and {judge_em0:.2f} on answers with EM = 0. The baselines have no judge scores, so this can't be used for a comparison yet. Scoring the baselines with the same judge is the most useful next measurement on DSIRE.</p>
    </div>
  </div>

  <h3>4. Every baseline vs RegisterRAG, per type (F1)</h3>
  <div class="col"><p>Green ▲: the baseline is reliably better than RegisterRAG on that type. Red ▼: reliably worse. Hover a cell for its interval.</p></div>
  {dsire_type_table()}

  <h3>5. Dataset caveats that affect every DSIRE number</h3>
  <div class="col"><ul class="tight">
    <li><b>Label noise of about {pct(noise)}.</b> That share of gold evidence spans are page headings or under six words (three read "Duplicate Entry"). Differences under about 0.03 F1 are not meaningful.</li>
    <li><b>EM doesn't work for long-form types.</b> No system reaches 0.08 EM on Descriptive or Legal-Obligation. Use F1 or the judge.</li>
    <li><b>Type labels overlap.</b> About half of "Yes-No" questions also carry conditions, so Yes-No vs Yes-No-cond is not a clean split.</li>
  </ul></div>
  <div class="callout"><p><b>Bottom line for DSIRE.</b> RegisterRAG's machinery targets missing evidence, a problem DSIRE doesn't have. It loses to B1 for two fixable reasons: it reads an unconfirmed eligibility condition as "No", and its multi-requirement answers add words outside the gold span. Two changes are likely to close most of the gap:</p>
  <ul class="tight"><li>For yes/no questions with an unsupported requirement, answer from the full retrieved paragraph instead of defaulting to "No".</li><li>For long-form types, answer from the single best-supported span.</li></ul>
  <p>Then compare every system with the same LLM judge.</p></div>
</section>

<section id="size">
  <h2>Context: what the model size changes for the baselines</h2>
  <div class="col"><p>RegisterRAG ran only at 7B, so this section uses the baselines alone. It explains why the comparison is made at 7B.</p></div>
  <div class="tablewrap"><table class="data"><thead><tr><th>Dataset</th><th>Baseline score with all gold in context<br><span class="sub">3B → 7B</span></th><th>With gold missing<br><span class="sub">3B → 7B</span></th><th>B1 3B → 7B</th><th>LiveLedger 3B → 7B</th></tr></thead><tbody>
  {"".join(f"<tr><th scope='row'>{DNAME[ds]}</th><td class='num'>{cond[(ds,G3)][0]:.3f} → {cond[(ds,G7)][0]:.3f}</td><td class='num'>{cond[(ds,G3)][1]:.3f} → {cond[(ds,G7)][1]:.3f}</td><td class='num'>{p(ds,G3,'chunk'):.3f} → {p(ds,G7,'chunk'):.3f}</td><td class='num'>{p(ds,G3,'liveledger'):.3f} → {p(ds,G7,'liveledger'):.3f}</td></tr>" for ds in DSETS)}
  </tbody></table></div>
  <div class="col"><p>The bigger model mainly reads evidence better. Its gains come when the gold evidence is in context, and they are largest for multi-call pipelines: LiveLedger gains {p('2wiki',G7,'liveledger')-p('2wiki',G3,'liveledger'):+.3f} F1 on 2Wiki against {p('2wiki',G7,'chunk')-p('2wiki',G3,'chunk'):+.3f} for B1. RegisterRAG is also a multi-call pipeline, so it may degrade more than B1 at 3B; a 3B run would show it.</p></div>
</section>

<section id="reliability">
  <h2>Reliability of these comparisons</h2>
  <div class="col"><ul class="tight">
    <li><b>Partial overlap.</b> RegisterRAG's 500-question sets are type-balanced samples. {", ".join(f"{len(rrset(d))} of its {DSHORT[d]} questions" for d in DSETS)} are also in the baseline runs. All comparisons use only those; the other {sum(rr_unmatched.values())} RegisterRAG questions are left out.</li>
    <li><b>Why F1 is the headline metric.</b> EM gives zero credit to "Antoine" for "Anthony, Duke of Brabant" and to every long policy answer that is not word-for-word identical. On 2Wiki, HotpotQA and MuSiQue the EM and F1 verdicts are identical for every comparison. On DSIRE they disagree in 7 of 16 comparisons, because EM cannot score the long-form answer types at all. So F1 is the reference and EM a secondary check.</li>
    <li><b>Same scorer for both sides.</b> RegisterRAG's answers were re-scored with the harness's EM/F1 against the baselines' gold answers, aliases included. MuSiQue has answer aliases, so this moves RegisterRAG's MuSiQue EM from {rr_csv_score['musique']:.3f} (CSV) to {score('musique',RR):.3f}. The other datasets are unchanged.</li>
    <li><b>Generator.</b> The CSVs don't record the model; qwen2.5:7b is taken from the project owner. The retrieval corpus and machine may differ from the baseline runs (pooled corpus, Windows paths), which can shift retrieval difficulty slightly.</li>
    <li><b>Noise floor.</b> Each system is a single run. Repeated baseline runs (SURE-RAG with the gate off is effectively B1) differ by about ±0.01, and a typical paired interval here is ±0.04–0.05.</li>
    <li><b>Interpretations.</b> "An unsupported requirement becomes No" and the bridge-question phrasing loss on HotpotQA are readings of the per-question records. They fit the counts but haven't been tested by an intervention.</li>
  </ul></div>
</section>

<section id="next">
  <h2>What to improve in RegisterRAG</h2>
  <div class="col"><p>These are the changes the analysis points to, in rough order of expected payoff. Each comes with the evidence from this report that motivates it.</p>
  <ol class="improve">
    <li><b>Stop reading "unsupported" as "No" on yes/no questions.</b> On DSIRE, RegisterRAG misses {pct(1-yn[RR]['yes'])} of questions whose answer is "yes" (B1 misses {pct(1-yn[('chunk','-')]['yes'])}). {pct(part_share_wrong)} of those misses had a requirement the evidence didn't confirm, and B1 got {pct(b1_right_on_rr_wrong)} of the same questions right, so the evidence was retrievable. When a requirement stays unsupported, answer from the full retrieved paragraph instead of defaulting to "No".</li>
    <li><b>Re-query when a requirement stays unsupported.</b> LiveLedger's open-ended search loop is the only baseline that matches or beats RegisterRAG on multi-hop data:
      <ul class="tight"><li>it beats RegisterRAG on HotpotQA ({S('hotpot','liveledger')[0]:+.3f} and {S('hotpot','liveledger','ledger=False')[0]:+.3f} F1, RegisterRAG minus LiveLedger);</li><li>it ties it on MuSiQue;</li><li>without its ledger it matches RegisterRAG on MuSiQue 4-hop questions ({mus4_ll:.3f} vs {mus4_rr:.3f} F1) with no requirement list at all.</li></ul>
      Re-querying for a requirement that stays unsupported would let RegisterRAG combine both strengths. The HotpotQA result suggests the loop finds evidence a fixed requirement list misses.</li>
    <li><b>Answer long-form questions from the best-supported span.</b> On DSIRE's descriptive questions, RegisterRAG's answers are {prr['Descriptive']['rw']/prr['Descriptive']['bw']:.1f}× longer than B1's. Recall goes up ({prr['Descriptive']['rr']:.2f} vs {prr['Descriptive']['br']:.2f}) but precision drops more ({prr['Descriptive']['rp']:.2f} vs {prr['Descriptive']['bp']:.2f}), because the extra words come from combining several requirement answers rather than the gold sentence.</li>
    <li><b>Don't over-decompose questions whose evidence is already easy to find.</b> On HotpotQA and DSIRE, retrieval is already near perfect, so splitting the question adds nothing and can lose phrasing: HotpotQA bridge questions are {hp_bridge[0]:+.3f} F1 vs B1. Consider a one-shot path, or a single requirement, when the first retrieval already covers the question.</li>
    <li><b>Improve how requirements are confirmed as supported.</b> Support coverage is a strong confidence signal: fully supported questions score higher on every dataset (MuSiQue {sup['musique'][0]:.2f} vs {sup['musique'][1]:.2f} F1). But only {pct(sup['dsire'][2])} of DSIRE questions are fully supported, even though doc recall@10 there is {m(r['retrieval']['doc_recall@10'] for r in Rd.values()):.2f}. The requirement checker is failing to confirm evidence that is in the page.</li>
    <li><b>Use support coverage for selective answering or fallback.</b> Partly supported answers are reliably worse, so hand those questions to a fallback (a B1-style full-context answer or another retrieval round), or flag them. Sending them to the strongest baseline gives {BS['2wiki']['route_bb_v']:.3f} F1 on 2Wiki ({fs(BS['2wiki']['route_bb'])} over LiveLedger alone) and closes the MuSiQue gap to {BS['musique']['route_bb'][0]:+.3f} (Table B). Confirm this on a held-out split before reporting it as a system result.</li>
    <li><b>Size the register to the question.</b> One requirement per gold hop is where RegisterRAG beats the strongest baseline ({fs(BS['2wiki']['dec']['equal'][0])} on 2Wiki). Over-decomposition costs {BS['hotpot']['dec']['more'][0][0]:+.3f} on HotpotQA and {BS['musique']['dec']['more'][0][0]:+.3f} on MuSiQue; under-decomposition of 4-hop 2Wiki questions costs {BS['2wiki']['dec']['fewer'][0][0]:+.3f} (Figure D). Check the plan length against the number of entities and relations in the question before retrieving.</li>
    <li><b>Add an answer-type guard.</b> RegisterRAG answers yes/no to {yn_other['hotpot'][0]} HotpotQA entity questions (strongest baseline {yn_other['hotpot'][1]}). Allow a yes/no answer only when the question is phrased as a yes/no question, and otherwise compose an entity answer from the best-supported requirement.</li>
    <li><b>Fix answer composition, not just retrieval.</b> With every gold document retrieved by both systems, RegisterRAG still trails the strongest baseline on HotpotQA ({fs(BS['hotpot']['both_full'])}) and DSIRE ({fs(BS['dsire']['both_full'])}) (Figure C), and its token precision is lower on HotpotQA, MuSiQue and DSIRE. The final composer should answer from the requirement that resolves the question, not combine every requirement's answer.</li>
    <li><b>Measure fairly before tuning further.</b>
      <ul class="tight"><li>Score all systems with the same LLM judge; RegisterRAG's DSIRE judge score is {m(r['judge'] for r in Rd.values()):.3f} against {score('dsire',RR):.3f} F1.</li><li>Rerun on the baselines' exact 500 questions (and at 3B), on the same machine, so every question is covered and latency is comparable.</li></ul></li>
  </ol></div>
</section>
</div>
"""
os.makedirs(os.path.dirname(a.out), exist_ok=True)
open(a.out, "w", encoding="utf-8").write(page)
print("wrote", a.out, f"{len(page)//1024} KB; RR better {n_better}/{n_cmp}, worse {len(rr_worse)}; by dataset {by_ds}; unmatched {rr_unmatched}; incomplete {incomplete}")
