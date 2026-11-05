"""Runs one system over a testbed and writes results.jsonl + summary.json (named by config hash)."""
from __future__ import annotations

import json
import os
import platform
import sys
import numpy as np
import statistics
import subprocess
import time
from collections import Counter, defaultdict

from .ledger import Ledger, gpu_peak_mb
from .metrics import classify_failure, exact_match, f1_score, retrieval_report


def git_state():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).strip())
        return {"git_sha": sha, "git_dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"git_sha": None, "git_dirty": None}


def env_info(generator=None):
    """Versions that can change results: record them so another machine can tell why numbers differ."""
    from importlib.metadata import PackageNotFoundError, version
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("numpy", "torch", "transformers", "sentence-transformers", "rank-bm25", "pysbd", "tiktoken"):
        try:
            info[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    host = getattr(generator, "host", None)
    if host:
        try:
            import requests
            info["ollama_version"] = requests.get(f"{host}/api/version", timeout=3).json().get("version")
        except Exception:  # noqa: BLE001
            pass
        info["OLLAMA_FLASH_ATTENTION(client env)"] = os.environ.get("OLLAMA_FLASH_ATTENTION")
    return info


def bootstrap_ci(xs, n=2000, seed=0, alpha=0.05):
    if not xs:
        return (None, None)
    a = np.asarray(xs, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    means = np.sort(a[idx].mean(axis=1))
    return (float(means[int(alpha / 2 * n)]), float(means[int((1 - alpha / 2) * n) - 1]))


def run_system(cfg, examples, pipeline, out_dir, data_sha="", resume=True, log=print):
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.jsonl")
    json.dump({**cfg.frozen(), "data_sha256": data_sha}, open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8"), indent=2)
    done = set()
    if resume and os.path.exists(res_path):
        keep, bad = [], 0
        for l in open(res_path, encoding="utf-8"):
            try:
                if json.loads(l).get("error") is None:
                    keep.append(l if l.endswith("\n") else l + "\n")
            except ValueError:                                           # line cut off by a crash mid-write
                bad += 1
        if bad:
            log(f"  dropped {bad} corrupt line(s) left by an interrupted write")
        open(res_path, "w", encoding="utf-8").writelines(keep)          # failed queries are retried, never silently kept
        done = {json.loads(l)["qid"] for l in keep}
    elif os.path.exists(res_path):
        os.remove(res_path)
    consecutive_errors = 0
    with open(res_path, "a", encoding="utf-8") as fout:
        for i, ex in enumerate(examples, 1):
            if ex.qid in done:
                continue
            led, err, res = Ledger(), None, None
            try:
                res = pipeline.run(ex, led)
            except Exception as e:  # noqa: BLE001
                err = repr(e)
            pred = res.prediction if res else ""
            em, f1 = exact_match(pred, ex.answers), f1_score(pred, ex.answers)
            rec = {"qid": ex.qid, "dataset": ex.dataset, "system": cfg.system, "question": ex.question,
                   "gold_answers": ex.answers, "prediction": pred, "em": em, "f1": f1,
                   "n_hops": ex.n_hops, "qtype": ex.qtype, "error": err,
                   "peak_prompt_tokens": led.peak_prompt_tokens, "total_tokens_processed": led.tokens_processed,
                   "calls": dict(led.calls), "latency_s": led.latency(), "gpu_peak_mb": gpu_peak_mb()}
            if res:
                rec["retrieval"] = retrieval_report(res.ranking, res.selected, ex)
                rec["truncated"] = res.truncated
                rec["selected"] = [{"unit_id": u.unit_id, "title": u.title, "text": u.text} for u in res.selected]
                rec["extra"] = res.extra
            rec["failure_category"] = (classify_failure(em, err, ex, res.selected, res.pool, res.stage1,
                                                        res.truncated, pipeline.reranker is not None)
                                       if res else "SYSTEM_ERROR")
            if cfg.system == "none" and em < 1.0 and not err:
                rec["failure_category"] = "NO_RETRIEVAL"
            rec["failure_category_auto"] = bool(rec["failure_category"])
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            try:
                os.fsync(fout.fileno())                                  # on disk before the next question: survives a power loss / BSOD
            except OSError:
                pass
            consecutive_errors = consecutive_errors + 1 if err else 0
            if err:
                log(f"  [{i}/{len(examples)}] ERROR on {ex.qid}: {err[:300]}")
            if consecutive_errors >= 3:
                raise RuntimeError(f"3 consecutive errors, aborting. Last: {err}")
            if i % 10 == 0 or i == len(examples):
                log(f"  [{i}/{len(examples)}] last EM={em:.0f}  latency={led.latency():.1f}s")
    summary = summarize(res_path)
    summary["config"] = cfg.frozen()
    summary["config_hash"] = cfg.hash(data_sha)
    summary["data_sha256"] = data_sha
    summary.update(git_state())
    summary["env"] = env_info(pipeline.generator)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"), indent=2)
    json.dump({**cfg.frozen(), "data_sha256": data_sha}, open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8"), indent=2)
    return summary


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(int(p / 100 * len(xs)), len(xs) - 1)] if xs else None


def summarize(res_path):
    recs = [json.loads(l) for l in open(res_path, encoding="utf-8")]
    ems, f1s = [r["em"] for r in recs], [r["f1"] for r in recs]
    S = {"n": len(recs), "em": _mean(ems), "em_ci95": bootstrap_ci(ems), "f1": _mean(f1s),
         "f1_ci95": bootstrap_ci(f1s)}
    for key in ("doc_recall@5", "doc_recall@10", "doc_recall@20", "sent_recall@5", "sent_recall@10",
                "sent_recall@20", "unit_precision@5", "unit_precision@10", "ctx_doc_recall", "ctx_sent_recall",
                "ctx_unit_precision"):
        S[key] = _mean([r.get("retrieval", {}).get(key) for r in recs])
    S["ctx_all_gold_rate"] = _mean([float(r["retrieval"]["ctx_all_gold"]) for r in recs if "retrieval" in r])
    lat = [r["latency_s"] for r in recs]
    S.update(latency_p50=_pct(lat, 50), latency_p95=_pct(lat, 95),
             peak_prompt_tokens_mean=_mean([r["peak_prompt_tokens"] for r in recs]),
             peak_prompt_tokens_max=max((r["peak_prompt_tokens"] for r in recs), default=0),
             total_tokens_mean=_mean([r["total_tokens_processed"] for r in recs]),
             truncated_rate=_mean([float(r.get("truncated", False)) for r in recs]),
             gpu_peak_mb_max=max((r["gpu_peak_mb"] for r in recs if r["gpu_peak_mb"]), default=None))
    calls = defaultdict(list)
    for r in recs:
        for k, v in r["calls"].items():
            calls[k].append(v)
    S["calls_mean"] = {k: _mean(v) for k, v in calls.items()}
    S["failure_counts"] = dict(Counter(r["failure_category"] or "CORRECT" for r in recs))
    by_hop = defaultdict(list)
    for r in recs:
        by_hop[r["n_hops"]].append(r)
    S["by_hops"] = {str(h): {"n": len(v), "em": _mean([x["em"] for x in v]), "f1": _mean([x["f1"] for x in v])}
                    for h, v in sorted(by_hop.items())}
    hop_rec = defaultdict(list)
    for r in recs:
        for h in r.get("retrieval", {}).get("per_hop", []):
            if h["recall@10"] is not None:
                hop_rec[h["hop"]].append(h["recall@10"])
    S["per_hop_recall@10"] = {str(k): {"n": len(v), "recall": _mean(v)} for k, v in sorted(hop_rec.items())}
    return S
