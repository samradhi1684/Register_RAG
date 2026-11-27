"""
Plots for a run folder holding <dataset>/<name>/ results (report.py + failure_analysis.py outputs).

  python tools/plot_metrics.py [RUN_ROOT] [NAME]      default: runs/run2_500ctx all500
Writes RUN_ROOT/plots/*.png.  If RUN_ROOT/2wiki/<NAME>_3b exists, also plots 3b vs 7b.
"""
from __future__ import annotations
import csv
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "run2_500ctx")
NAME = sys.argv[2] if len(sys.argv) > 2 else "all500"
OUT = os.path.join(RUN, "plots")
os.makedirs(OUT, exist_ok=True)

DATASETS = [("2wiki", "2Wiki"), ("hotpotqa", "HotpotQA"), ("musique", "MuSiQue"), ("repolicyqa", "RePolicyQA")]
# validated categorical slots (fixed order), surface and ink
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CLASSES = [("retrieval", "Retrieval"), ("reranking", "Reranking"),
           ("step_not_found", "Step not found"), ("wrong_value", "Wrong value")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def rows(ds, name=NAME):
    p = os.path.join(RUN, ds, name, "failures.csv")
    return list(csv.DictReader(open(p, encoding="utf-8-sig"))) if os.path.exists(p) else []


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def judged(r):
    return num(r["judge"])


def label_bars(ax, bars, fmt="{:.2f}", inside=False):
    for b in bars:
        h = b.get_width() if inside else b.get_height()
        if inside:
            if h >= 0.04:
                ax.text(b.get_x() + h / 2, b.get_y() + b.get_height() / 2, fmt.format(h),
                        ha="center", va="center", fontsize=8, color=INK)
        else:
            ax.text(b.get_x() + b.get_width() / 2, h + 0.01, fmt.format(h),
                    ha="center", va="bottom", fontsize=8, color=INK2)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


data = {ds: rows(ds) for ds, _ in DATASETS}
data = {k: v for k, v in data.items() if v}
names = [lbl for ds, lbl in DATASETS if ds in data]
keys = [ds for ds, _ in DATASETS if ds in data]
n_q = {ds: len(data[ds]) for ds in keys}

# 1. answer quality: EM / F1 / judge per dataset
fig, ax = plt.subplots(figsize=(8, 4))
metrics = [("EM", lambda r: num(r["em"])), ("F1", lambda r: num(r["f1"])), ("LLM judge", judged)]
w = 0.26
for i, (m, fn) in enumerate(metrics):
    vals = [mean(fn(r) for r in data[ds]) for ds in keys]
    bars = ax.bar([x + (i - 1) * (w + 0.01) for x in range(len(keys))], vals, w, color=C[i], label=m)
    label_bars(ax, bars)
ax.set_xticks(range(len(keys)), [f"{n}\n(n={n_q[k]})" for n, k in zip(names, keys)])
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Answer quality per dataset (qwen2.5:7b, 500 questions each)")
ax.legend(ncol=3, loc="upper right")
ax.grid(axis="x", visible=False)
save(fig, "1_answer_quality.png")

# 2. where questions are lost: gold in pool -> gold in top-10 -> answered correctly
fig, ax = plt.subplots(figsize=(8, 4))
stages = [("All gold evidence retrieved (pool)", lambda r: r["category"] not in ("retrieval",)),
          ("All gold evidence in top-10", lambda r: r["category"] not in ("retrieval", "reranking")),
          ("Answer correct (F1≥0.5 or judge=1)", lambda r: r["category"] in ("correct", "form"))]
for i, (s, fn) in enumerate(stages):
    vals = [sum(1 for r in data[ds] if fn(r)) / n_q[ds] for ds in keys]
    bars = ax.bar([x + (i - 1) * (w + 0.01) for x in range(len(keys))], vals, w, color=C[i], label=s)
    label_bars(ax, bars)
ax.set_xticks(range(len(keys)), names)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Share of questions")
ax.set_title("Where questions are lost along the pipeline")
ax.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.0))
ax.grid(axis="x", visible=False)
save(fig, "2_pipeline_funnel.png")

# 3. failure classes, share of failures (stacked horizontal, 2px surface gaps)
fig, ax = plt.subplots(figsize=(8, 3.4))
for y, ds in enumerate(keys):
    fails = [r for r in data[ds] if r["category"] not in ("correct", "form")]
    left = 0.0
    for i, (c, lbl) in enumerate(CLASSES):
        share = sum(1 for r in fails if r["category"] == c) / max(1, len(fails))
        b = ax.barh(y, share, left=left, color=C[i], edgecolor=SURFACE, linewidth=2,
                    label=lbl if y == 0 else None, height=0.6)
        label_bars(ax, b, fmt="{:.0%}", inside=True)
        left += share
ax.set_yticks(range(len(keys)),
              [f"{n} ({sum(1 for r in data[k] if r['category'] not in ('correct', 'form'))} failures)"
               for n, k in zip(names, keys)])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_title("Failure classes (share of wrong answers)")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.38))
ax.grid(axis="y", visible=False)
save(fig, "3_failure_classes.png")

# 4. F1, judge and recall@10 by question type (small multiples)
fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.8), sharey=True)
for ax, ds, n in zip(axes, keys, names):
    types = sorted({r["type"] for r in data[ds]})
    for i, (m, fn) in enumerate([("F1", lambda r: num(r["f1"])), ("Judge", judged),
                                 ("Recall@10", lambda r: num(r["recall_at_10"]))]):
        vals = [mean(fn(r) for r in data[ds] if r["type"] == t) for t in types]
        bars = ax.bar([x + (i - 1) * 0.27 for x in range(len(types))], vals, 0.26, color=C[i],
                      label=m if ds == keys[0] else None)
        label_bars(ax, bars, fmt="{:.2f}")
    ax.set_xticks(range(len(types)), [t.replace("_", "\n").replace("-", "-\n", 1) if len(t) > 9 else t
                                      for t in types], fontsize=8)
    ax.set_title(n)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Score")
fig.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.suptitle("By question type", y=1.07, fontweight="bold")
save(fig, "4_by_question_type.png")

# 5. failure sub-causes (from failure_analysis.py)
sub = {}
for ds in keys:
    p = os.path.join(RUN, ds, NAME, "failure_subcauses.json")
    if os.path.exists(p):
        sub[ds] = json.load(open(p))
if sub:
    fig, axes = plt.subplots(len(sub), 1, figsize=(8, 2.2 * len(sub) + 0.6))
    axes = axes if len(sub) > 1 else [axes]
    cls_color = {c: C[i] for i, (c, _) in enumerate(CLASSES)}
    for ax, ds in zip(axes, sub):
        items = sorted(sub[ds]["subcauses"].items(), key=lambda kv: -kv[1])
        lbls = [k.split(": ", 1)[1] for k, _ in items]
        cols = [cls_color[k.split(": ")[0]] for k, _ in items]
        bars = ax.barh(range(len(items)), [v for _, v in items], color=cols, height=0.65)
        for b, (_, v) in zip(bars, items):
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(items)), lbls, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{dict(DATASETS)[ds]}: {sub[ds]['failures']} failures", loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(v for _, v in items) * 1.15)
    handles = [matplotlib.patches.Patch(color=C[i], label=l) for i, (_, l) in enumerate(CLASSES)]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Failure sub-causes (number of questions)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "5_failure_subcauses.png"), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(OUT, "5_failure_subcauses.png"))

# 6. cost: latency, tokens and LLM requests per question (separate panels, no dual axis)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (t, fn, fmt) in zip(axes, [("Latency per question (s)", lambda r: num(r["latency_s"]), "{:.1f}"),
                                   ("Tokens per question", lambda r: num(r["total_tokens"]), "{:,.0f}"),
                                   ("LLM requests per question", lambda r: num(r["llm_requests"]), "{:.1f}")]):
    vals = [mean(fn(r) for r in data[ds]) for ds in keys]
    bars = ax.bar(range(len(keys)), vals, 0.55, color=C[0])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.01, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(keys)), names, fontsize=8)
    ax.set_title(t)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(vals) * 1.15)
save(fig, "6_cost.png")

# 7. MuSiQue / any typed set: recall@10 vs F1 by hop is in plot 4; here F1 vs number of planner steps
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, ds in enumerate(keys):
    by = {}
    for r in data[ds]:
        by.setdefault(int(r["steps"]), []).append(num(r["f1"]))
    xs = sorted(k for k in by if len(by[k]) >= 10)
    ax.plot(xs, [mean(by[k]) for k in xs], marker="o", markersize=7, linewidth=2, color=C[i], label=names[i])
ax.set_xlabel("Number of planner steps (groups with ≥10 questions)")
ax.set_ylabel("Mean F1")
ax.set_ylim(0, 1)
ax.set_title("Mean F1 by number of planner steps")
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
ax.legend(ncol=4, loc="upper right")
save(fig, "7_f1_by_steps.png")

# 8. model size per dataset (if <NAME>_3b / <NAME>_14b runs exist)
for DS, DSN in DATASETS:
    variants = [(m, f"{NAME}_{m}") for m in ("3b", "14b") if rows(DS, f"{NAME}_{m}")]
    if variants:
        models = [("7b", NAME)] + variants
        models.sort(key=lambda x: float(x[0][:-1]))
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for i, (m, fn) in enumerate([("EM", lambda r: num(r["em"])), ("F1", lambda r: num(r["f1"])),
                                     ("LLM judge", judged), ("Recall@10", lambda r: num(r["recall_at_10"]))]):
            vals = [mean(fn(r) for r in rows(DS, nm)) for _, nm in models]
            bars = axes[0].bar([x + (i - 1.5) * 0.2 for x in range(len(models))], vals, 0.19, color=C[i], label=m)
            label_bars(axes[0], bars)
        axes[0].set_xticks(range(len(models)), [f"qwen2.5:{m}" for m, _ in models])
        axes[0].set_ylim(0, 1.1)
        axes[0].set_title(f"{DSN}: quality by model size")
        axes[0].legend(ncol=4, fontsize=8, loc="upper left")
        axes[0].grid(axis="x", visible=False)
        for y, (m, nm) in enumerate(models):
            rr = rows(DS, nm)
            fails = [r for r in rr if r["category"] not in ("correct", "form")]
            left = 0
            for i, (c, lbl) in enumerate(CLASSES):
                v = sum(1 for r in fails if r["category"] == c)
                b = axes[1].barh(y, v, left=left, color=C[i], edgecolor=SURFACE, linewidth=2, height=0.6,
                                 label=lbl if y == 0 else None)
                if v >= 8:
                    axes[1].text(left + v / 2, y, str(v), ha="center", va="center", fontsize=8)
                left += v
        axes[1].set_yticks(range(len(models)), [f"qwen2.5:{m}" for m, _ in models])
        axes[1].invert_yaxis()
        axes[1].set_title(f"{DSN}: failures by class (questions)")
        axes[1].legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
        axes[1].grid(axis="y", visible=False)
        save(fig, f"8_model_size_{DS}.png")
