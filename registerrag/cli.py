from __future__ import annotations

import argparse
import json
import os
import random
import sys

from . import compare as cmp
from . import gold_programs
from . import registry
from .config import RunConfig
from .data import load_examples
from .generate import HFGenerator, OllamaGenerator, OpenAIGenerator, StubGenerator, TokenCounter
from .pipelines import make_pipeline
from .rerank import CrossEncoderReranker, OverlapReranker
from .retrieval import HashEmbedder, HybridRetriever, STEmbedder
from .runner import run_system
from .units import make_store


def make_generator(spec, cfg, stub):
    if stub:
        return StubGenerator()
    if spec.startswith("hf:"):
        return HFGenerator(spec.split(":", 1)[1], cfg.temperature, cfg.seed, cfg.max_new_tokens)
    if spec.startswith("openai:"):
        return OpenAIGenerator(spec.split(":", 1)[1], cfg.temperature, cfg.seed, cfg.max_new_tokens)
    return OllamaGenerator(spec, temperature=cfg.temperature, seed=cfg.seed, num_ctx=cfg.num_ctx,
                           max_new_tokens=cfg.max_new_tokens)


def parse_method_args(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if v.lower() in ("true", "false"):
            val = v.lower() == "true"
        else:
            try:
                val = int(v)
            except ValueError:
                try:
                    val = float(v)
                except ValueError:
                    val = v
        out[k.strip()] = val
    return out


def build_run(cfg: RunConfig, stub: bool, cache_dir: str):
    examples, docs, sha = load_examples(cfg.data_path, cfg.dataset, cfg.limit, cfg.sample_seed)
    mode = cfg.unit_mode or registry.default_unit(cfg.system)
    store = make_store(docs, mode, cfg.chunk_size_words, cfg.chunk_overlap_words)
    counter = TokenCounter()
    cfg.tokenizer = counter.name
    if stub:
        emb, rer = HashEmbedder(), OverlapReranker()
        cfg.embedder, cfg.reranker, cfg.generator = "STUB", "STUB", "STUB"
    else:
        emb = STEmbedder(cfg.embedder, query_prefix="")
        rer = CrossEncoderReranker(cfg.reranker) if cfg.reranker and cfg.reranker.lower() != "none" else None
    gen = make_generator(cfg.generator, cfg, stub)
    if hasattr(gen, "counter"):
        counter = gen.counter                          # budgets measured with the model's own tokenizer
        cfg.tokenizer = counter.name
    aux = {}
    if cfg.method_args.get("plan_generator"):
        aux["plan"] = make_generator(str(cfg.method_args["plan_generator"]), cfg, stub)
    ret = HybridRetriever(store.units, emb, cfg.lam, cfg.prepend_title, cache_dir, store.fingerprint)
    return examples, make_pipeline(cfg.system, store, ret, rer, gen, counter, cfg, aux=aux), sha, store


def cmd_run(a):
    cfg = RunConfig(system=a.system, dataset=a.dataset, data_path=a.data, limit=a.limit, sample_seed=a.sample_seed,
                    corpus_scope=a.corpus_scope, chunk_size_words=a.chunk_size, chunk_overlap_words=a.chunk_overlap,
                    embedder=a.embedder, prepend_title=a.prepend_title, lam=a.lam, retrieve_k=a.retrieve_k,
                    top_k=a.top_k, reranker=a.reranker, generator=a.generator, seed=a.seed, num_ctx=a.num_ctx,
                    budget_tokens=a.budget, prompt_version=a.prompt,
                    unit_mode=a.unit, method_args=parse_method_args(a.method_arg))
    examples, pipe, sha, store = build_run(cfg, a.stub, a.cache_dir)
    for role, gen in [("generator", pipe.generator)] + list(pipe.aux.items()):
        if hasattr(gen, "ping"):
            print(f"preflight[{role}]:", gen.ping())
    out = os.path.join(a.out or registry.results_dir(cfg.system), f"{cfg.system}_{cfg.dataset}_{cfg.hash(sha)}")
    print(f"run dir: {out}\nunits: {store.describe()}")
    S = run_system(cfg, examples, pipe, out, sha, resume=not a.no_resume)
    keys = ["n", "em", "em_ci95", "f1", "sent_recall@10", "doc_recall@10", "ctx_all_gold_rate",
            "peak_prompt_tokens_max", "total_tokens_mean", "latency_p50", "latency_p95", "failure_counts"]
    print(json.dumps({k: S[k] for k in keys}, indent=2))


def cmd_baselines(a):
    """Overview of every baseline folder: status, system name, number of finished runs."""
    from .paths import run_folders
    print(f"{'folder':26s} {'status':12s} {'--system':10s} {'unit':9s} {'runs':>5s}  label")
    for b in registry.discover():
        n = len([d for d in run_folders(b.results_dir) if os.path.exists(os.path.join(d, "summary.json"))]) if os.path.isdir(b.results_dir) else 0
        print(f"{b.folder:26s} {b.status:12s} {str(b.system or '-'):10s} {b.default_unit if b.system else '-':9s} {n:5d}  {b.label if b.system else ''}")


def cmd_compare(a):
    for m in ("em", "f1"):
        print(json.dumps(cmp.paired(a.a, a.b, m), indent=2))


def cmd_gold(a):
    ex, _, _ = load_examples(a.data, a.dataset, a.limit)
    print(json.dumps(gold_programs.export(ex, a.out), indent=2))


def cmd_sample_sentences(a):
    """Step 2: manual check of splitting quality (30 random units)."""
    ex, docs, _ = load_examples(a.data, a.dataset)
    store = make_store(docs, "sentence")
    rng = random.Random(a.seed)
    for u in rng.sample(store.units, min(a.n, len(store.units))):
        print(f"- [{u.unit_id}] ({u.title}) {u.text}")


def cmd_sample_failures(a):
    """Step 3: dump N failures of a run for manual labelling."""
    recs = [json.loads(l) for l in open(a.results, encoding="utf-8")]
    bad = [r for r in recs if r["failure_category"]]
    random.Random(a.seed).shuffle(bad)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in bad[: a.n]:
            f.write(f"## {r['qid']}  auto={r['failure_category']}\nQ: {r['question']}\nGold: {r['gold_answers']}"
                    f"\nPred: {r['prediction']}\nEvidence shown:\n")
            for s in r.get("selected", []):
                f.write(f"  - [{s['unit_id']}] {s['text'][:200]}\n")
            f.write("Manual label (RETRIEVAL / RERANKING / GENERATION / other): \n\n")
    print(f"wrote {min(a.n, len(bad))} of {len(bad)} failures to {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser("registerrag")
    sp = p.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--system", choices=registry.systems(), required=True)
    r.add_argument("--dataset", choices=["2wiki", "musique", "hotpot"], required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--out", default=None, help="default: baselines/<id>/results/ of the chosen system")
    r.add_argument("--cache-dir", default=".cache")
    r.add_argument("--limit", type=int)
    r.add_argument("--sample-seed", type=int)
    r.add_argument("--corpus-scope", choices=["pooled", "per_question"], default="pooled")
    r.add_argument("--chunk-size", type=int, default=256)
    r.add_argument("--chunk-overlap", type=int, default=50)
    r.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    r.add_argument("--prepend-title", action="store_true")
    r.add_argument("--lam", type=float, default=0.5)
    r.add_argument("--retrieve-k", type=int, default=50)
    r.add_argument("--top-k", type=int, default=5)
    r.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    r.add_argument("--generator", default="qwen2.5:14b", help="ollama model, hf:<hub id>, or openai:<model>")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--num-ctx", type=int, default=8192)
    r.add_argument("--budget", type=int, help="hard prompt-token budget (512/1024/2048/4096)")
    r.add_argument("--prompt", choices=["v1", "v2"], default="v1", help="v1 may abstain; v2 forces an answer")
    r.add_argument("--unit", choices=["chunk", "sentence"], help="override the retrieval unit (default depends on system)")
    r.add_argument("--method-arg", action="append", default=[], metavar="K=V", help="method-specific setting, repeatable")
    r.add_argument("--stub", action="store_true", help="offline stub models (smoke test only)")
    r.add_argument("--no-resume", action="store_true")
    r.set_defaults(fn=cmd_run)
    sp.add_parser("baselines").set_defaults(fn=cmd_baselines)
    c = sp.add_parser("compare"); c.add_argument("a"); c.add_argument("b"); c.set_defaults(fn=cmd_compare)
    g = sp.add_parser("gold-programs")
    g.add_argument("--dataset", choices=["2wiki", "musique"], required=True)
    g.add_argument("--data", required=True); g.add_argument("--out", required=True); g.add_argument("--limit", type=int)
    g.set_defaults(fn=cmd_gold)
    s = sp.add_parser("sample-sentences")
    s.add_argument("--dataset", choices=["2wiki", "musique", "hotpot"], required=True)
    s.add_argument("--data", required=True); s.add_argument("--n", type=int, default=30); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sample_sentences)
    f = sp.add_parser("sample-failures")
    f.add_argument("results"); f.add_argument("--out", default="failures_to_label.md")
    f.add_argument("--n", type=int, default=50); f.add_argument("--seed", type=int, default=0)
    f.set_defaults(fn=cmd_sample_failures)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
