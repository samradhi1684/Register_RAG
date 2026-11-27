"""
RegisterRAG Stage 1 — entry point.

Every parameter is controllable from the CLI.
No code edits needed to swap models, datasets, top-k, or budgets.

Sub-commands
────────────
  build   Build corpus indexes + graph (one-time, slow)
  eval    Run evaluation on a dataset
  single  Run a single question by ID

Quick start (2WikiMultiHopQA + Ollama qwen2.5:7b, 5 questions)
──────────────────────────────────────────────────────────────
  # 1. Pull model once
  ollama pull qwen2.5:7b

  # 2. Build index
  python run.py build \\
      --data train.json --format 2wiki \\
      --index-dir indexes/2wiki

  # 3. Run 5 questions
  python run.py eval \\
      --data train.json --format 2wiki \\
      --provider ollama --model qwen2.5:7b \\
      --max-questions 5 --run-name smoke_test \\
      --index-dir indexes/2wiki

  # 4. Inspect one trace
  python run.py single \\
      --data train.json --format 2wiki \\
      --provider ollama --model qwen2.5:7b \\
      --question-id 6bde832c0bdb11eba7f7acde48001122 \\
      --index-dir indexes/2wiki
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("registerrag")


# ─────────────────────────────────────────────────────────────────────────────
# Sub-commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_build(args, config):
    """Build corpus indexes and evidence graph from a dataset."""
    from registerrag.data_loaders import (
        load_hotpotqa, load_2wiki, load_generic_factual, load_policy_docs
    )
    from registerrag.corpus.processor import DocumentProcessor
    from registerrag.corpus.indexer import HybridIndex, embed_units
    from registerrag.corpus.graph import EvidenceGraph
    import numpy as np

    logger.info("=== BUILD PHASE ===")
    logger.info(config.summary())

    # ── Load ──────────────────────────────────────────────────────────────────
    if args.format == "hotpotqa":
        _, documents = load_hotpotqa(args.data)
    elif args.format == "2wiki":
        _, documents = load_2wiki(args.data)
    elif args.format == "musique":
        from registerrag.data_loaders import load_musique
        _, documents = load_musique(args.data, sentence_units=config.retrieval_unit == "sentence")
    elif args.format == "generic":
        _, documents = load_generic_factual(args.data)
    elif args.format == "policy":
        documents = load_policy_docs(args.data)
    elif args.format == "repolicyqa":
        from registerrag.data_loaders import load_repolicyqa
        _, documents = load_repolicyqa(args.data, getattr(args, "corpus_dir", None))
    else:
        raise ValueError(f"Unknown format: {args.format}")

    # ── Process → EvidenceUnits ───────────────────────────────────────────────
    processor = DocumentProcessor(config.spacy_model)
    doc_type = {"policy": "policy", "repolicyqa": "structured"}.get(args.format, "hotpotqa")
    if config.retrieval_unit not in ("sentence", "passage"):
        raise ValueError(f"retrieval_unit must be 'sentence' or 'passage', got {config.retrieval_unit!r}")
    logger.info("Retrieval unit: %s", config.retrieval_unit)
    units = processor.process_all(documents, doc_type=doc_type,
                                  sentence_units=config.retrieval_unit == "sentence")

    # ── Embed ─────────────────────────────────────────────────────────────────
    units = embed_units(units, config)

    # ── Build hybrid index ────────────────────────────────────────────────────
    index = HybridIndex(units, config)
    index.save(config.index_dir)

    # ── Build graph ───────────────────────────────────────────────────────────
    embeddings = index.get_embeddings()
    graph = EvidenceGraph.build(units, embeddings, config)
    graph.save(os.path.join(config.index_dir, "graph.pkl"))

    from registerrag.data_loaders import write_scopes
    logger.info("Scopes: %s", write_scopes(documents, config.index_dir))
    logger.info("Build complete. Index + graph saved to: %s", config.index_dir)


def cmd_eval(args, config):
    """Evaluate Stage 1 or Stage 2 pipeline on a dataset."""
    import spacy
    from registerrag.data_loaders import (
        load_hotpotqa, load_2wiki, load_generic_factual
    )
    from registerrag.corpus.indexer import HybridIndex
    from registerrag.corpus.graph import EvidenceGraph
    from registerrag.pipeline.verifier import CrossEncoderVerifier
    from registerrag.pipeline.stage1 import Stage1Pipeline
    from registerrag.pipeline.stage2 import Stage2Pipeline
    from registerrag.evaluation.metrics import aggregate_results
    from registerrag.evaluation.failure import failure_analysis

    stage = getattr(args, "stage", 1)
    logger.info("=== EVAL PHASE (Stage %d) ===", stage)
    logger.info(config.summary())
    config.validate()

    # ── Load artefacts ────────────────────────────────────────────────────────
    index = HybridIndex.load(config.index_dir, config)
    graph = EvidenceGraph.load(os.path.join(config.index_dir, "graph.pkl"))

    nlp = _load_spacy(config.spacy_model)
    verifier = CrossEncoderVerifier(
        config.verifier_model, config.embedding_device, config.verifier_batch_size
    )

    # ── Run directory ─────────────────────────────────────────────────────────
    run_name = args.run_name or f"run_{int(time.time())}"
    runs_dir = os.path.join(config.runs_dir, run_name)
    os.makedirs(runs_dir, exist_ok=True)
    logger.info("Run directory: %s", runs_dir)

    # Save config snapshot alongside results
    _save_config_snapshot(config, runs_dir)

    # ── Load questions ────────────────────────────────────────────────────────
    questions = _load_questions(args.data, args.format, getattr(args, "corpus_dir", None))
    questions = _filter_questions(questions, args)

    # ── Resume: load already-completed results ────────────────────────────────
    results_path = os.path.join(runs_dir, "results.jsonl")
    all_results = []
    done_ids: set = set()
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    all_results.append(r)
                    done_ids.add(r.get("question_id", ""))
        if done_ids:
            logger.info("Resuming: %d questions already done, skipping them.", len(done_ids))

    pending = [q for q in questions if q["id"] not in done_ids]
    logger.info("Evaluating %d questions (%d remaining)…", len(questions), len(pending))

    # ── Build shared LLM client ───────────────────────────────────────────────
    llm_client = _build_llm_client(config)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    if stage == 2:
        llm_client_for_binding = llm_client if config.llm_binding_extraction else None
        pipeline = Stage2Pipeline(
            index, graph, verifier, nlp, config, runs_dir,
            llm_client=llm_client_for_binding,
        )
        logger.info(
            "Using Stage2Pipeline (gap fill=%s, eviction=%s, verify=%s, llm_binding=%s)",
            config.gap_fill_enabled, config.register_eviction_enabled,
            config.answer_verify_enabled, config.llm_binding_extraction,
        )
    else:
        pipeline = Stage1Pipeline(index, graph, verifier, nlp, config, runs_dir)

    # Open results file in append mode so each result is flushed immediately
    with open(results_path, "a", encoding="utf-8") as results_f:
        total = len(questions)
        done_so_far = len(done_ids)
        for q in pending:
            done_so_far += 1
            logger.info("[%d/%d] id=%s", done_so_far, total, q["id"])
            try:
                result = pipeline.run_question(
                    question_id=q["id"],
                    question=q["question"],
                    gold_answer=q["answer"],
                    supporting_facts=q.get("supporting_facts", []),
                )
                r = result.__dict__
            except Exception as e:
                logger.error("Question %s failed: %s", q["id"], e, exc_info=True)
                r = {
                    "question_id": q["id"],
                    "question": q["question"],
                    "gold_answer": q["answer"],
                    "predicted_answer": "",
                    "em": 0.0, "f1": 0.0,
                    "doc_recall": 0.0,
                    "latency_s": 0.0,
                    "total_llm_calls": 0,
                    "failure_type": "error",
                    "error": str(e),
                }
            all_results.append(r)
            results_f.write(json.dumps(r) + "\n")
            results_f.flush()

    summary = aggregate_results(all_results)
    summary["failure_analysis"] = failure_analysis(all_results)

    summary_path = os.path.join(runs_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _print_summary(summary)
    logger.info("All results saved to: %s", runs_dir)


def cmd_single(args, config):
    """Run one question by ID and print the trace path."""
    import spacy
    from registerrag.data_loaders import (
        load_hotpotqa, load_2wiki, load_generic_factual
    )
    from registerrag.corpus.indexer import HybridIndex
    from registerrag.corpus.graph import EvidenceGraph
    from registerrag.pipeline.verifier import CrossEncoderVerifier
    from registerrag.pipeline.stage1 import Stage1Pipeline

    logger.info(config.summary())
    config.validate()

    index    = HybridIndex.load(config.index_dir, config)
    graph    = EvidenceGraph.load(os.path.join(config.index_dir, "graph.pkl"))
    nlp      = _load_spacy(config.spacy_model)
    verifier = CrossEncoderVerifier(
        config.verifier_model, config.embedding_device, config.verifier_batch_size
    )

    runs_dir = os.path.join(config.runs_dir, "single")
    pipeline = Stage1Pipeline(index, graph, verifier, nlp, config, runs_dir)

    questions = _load_questions(args.data, args.format, getattr(args, "corpus_dir", None))
    target = next((q for q in questions if q["id"] == args.question_id), None)
    if target is None:
        print(f"ERROR: question id {args.question_id!r} not found in dataset.")
        sys.exit(1)

    result = pipeline.run_question(
        question_id=target["id"],
        question=target["question"],
        gold_answer=target["answer"],
        supporting_facts=target.get("supporting_facts", []),
    )

    print(f"\n{'─'*60}")
    print(f"  Question : {target['question']}")
    print(f"  Generated: {result.predicted_answer!r}")
    print(f"  Gold     : {result.gold_answer!r}")
    print(f"  EM={result.em:.0f}  F1={result.f1:.3f}  Recall={result.doc_recall:.3f}")
    print(f"  Failure  : {result.failure_type}")
    print(f"  Trace    : {result.trace_path}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm_client(config):
    """Build an OpenAI-compatible client (shared), routed through the LLM cache."""
    from registerrag.llm_cache import CachedChatClient
    return CachedChatClient(_raw_llm_client(config), config.llm_provider)


def _raw_llm_client(config):
    from openai import OpenAI
    if config.llm_provider == "ollama":
        return OpenAI(
            base_url=config.ollama_base_url.rstrip("/") + "/v1",
            api_key="ollama",
        )
    elif config.llm_provider == "anthropic":
        return OpenAI(
            base_url="https://api.anthropic.com/v1",
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
    else:  # openai (default)
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


def _load_spacy(model_name: str):
    import spacy
    nlp = spacy.load(model_name, disable=["parser", "lemmatizer"])
    if "sentencizer" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def _load_questions(data_path: str, fmt: str, corpus_dir=None) -> list[dict]:
    from registerrag.data_loaders import (
        load_hotpotqa, load_2wiki, load_generic_factual, load_repolicyqa
    )
    if fmt == "repolicyqa":
        questions, _ = load_repolicyqa(data_path, corpus_dir)
    elif fmt == "hotpotqa":
        questions, _ = load_hotpotqa(data_path)
    elif fmt == "2wiki":
        questions, _ = load_2wiki(data_path)
    elif fmt == "musique":
        from registerrag.data_loaders import load_musique
        questions, _ = load_musique(data_path)
    else:
        questions, _ = load_generic_factual(data_path)
    return questions


def _filter_questions(questions: list[dict], args) -> list[dict]:
    """
    Apply --max-questions and optional --question-ids filtering.
    --max-questions 5  → first 5 questions
    --question-ids a,b,c → only those IDs
    """
    if getattr(args, "question_ids", None):
        ids = set(args.question_ids.split(","))
        questions = [q for q in questions if q["id"] in ids]
        logger.info("Filtered to %d question(s) by --question-ids", len(questions))

    if getattr(args, "max_questions", None) and args.max_questions > 0:
        questions = questions[: args.max_questions]

    return questions


def _save_config_snapshot(config, runs_dir: str) -> None:
    """Save config to JSON for reproducibility."""
    snap = {k: v for k, v in config.__dict__.items()
            if not k.endswith("_key")}  # don't write API keys to disk
    with open(os.path.join(runs_dir, "config.json"), "w") as f:
        json.dump(snap, f, indent=2)


def _print_summary(summary: dict) -> None:
    sep = "─" * 52
    print(f"\n{sep}")
    print("  RegisterRAG Stage 1 — Evaluation Summary")
    print(sep)
    print(f"  Questions      : {summary.get('total_questions', 0)}")
    print(f"  EM             : {summary.get('em', 0):.4f}")
    print(f"  F1             : {summary.get('f1', 0):.4f}")
    print(f"  Doc Recall     : {summary.get('doc_recall', 0):.4f}")
    print(sep)
    fa = summary.get("failure_analysis", {})
    print(f"  Correct        : {fa.get('none',              {}).get('count', 0)}")
    print(f"  Retrieval fail : {fa.get('retrieval',         {}).get('count', 0)}")
    print(f"  Evidence select: {fa.get('evidence_selection', {}).get('count', 0)}")
    print(f"  Partial recall : {fa.get('partial_retrieval', {}).get('count', 0)}")
    print(f"  Generation fail: {fa.get('generation',        {}).get('count', 0)}")
    print(sep)
    print(f"  LLM calls/Q    : {summary.get('mean_llm_calls', 0):.1f}")
    print(f"  Latency mean   : {summary.get('mean_latency_s', 0):.2f}s")
    print(f"  Latency p95    : {summary.get('p95_latency_s', 0):.2f}s")
    print(f"  Mean req/Q     : {summary.get('mean_requirements', 0):.1f}")
    print(f"  Mean proof/Q   : {summary.get('mean_proof_units', 0):.1f}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI definition
# ─────────────────────────────────────────────────────────────────────────────

def _add_llm_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("LLM / generator")
    g.add_argument("--provider",
                   choices=["openai", "anthropic", "ollama"],
                   help="LLM provider  (default: openai)")
    g.add_argument("--model",
                   metavar="MODEL",
                   help="Model name e.g. gpt-4o-mini | qwen2.5:7b | claude-haiku-3-5")
    g.add_argument("--temperature", type=float,
                   help="Sampling temperature  (default: 0.0)")
    g.add_argument("--ollama-url", metavar="URL",
                   help="Ollama server base URL  (default: http://localhost:11434)")


def _add_embedding_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Embedding")
    g.add_argument("--embedding-model", metavar="MODEL",
                   help="sentence-transformers model  (default: all-MiniLM-L6-v2)")
    g.add_argument("--embedding-device", choices=["cpu", "cuda"],
                   help="Device for embeddings  (default: cpu)")


def _add_retrieval_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Retrieval")
    g.add_argument("--top-k-initial", type=int, metavar="N",
                   help="Candidates before cross-encoder  (default: 20)")
    g.add_argument("--top-k", type=int, metavar="N",
                   help="Kept after rerank — labelled SUPPORT  (default: 10)")
    g.add_argument("--hybrid-lambda", type=float, metavar="λ",
                   help="λ·dense + (1-λ)·BM25  [0=BM25 only, 1=dense only]  (default: 0.5)")


def _add_graph_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Graph expansion")
    g.add_argument("--graph-hops", type=int, metavar="N",
                   help="Graph expansion hops from seed units  (default: 1)")
    g.add_argument("--graph-top-n", type=int, metavar="N",
                   help="Expand neighbours of top-N retrieved  (default: 5)")
    g.add_argument("--graph-sem-threshold", type=float, metavar="T",
                   help="Cosine threshold for semantic edges  (default: 0.70)")


def _add_verifier_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Cross-encoder verifier")
    g.add_argument("--verifier-model", metavar="MODEL",
                   help="HuggingFace cross-encoder model  "
                        "(default: cross-encoder/ms-marco-MiniLM-L-6-v2)")
    g.add_argument("--verifier-batch-size", type=int, metavar="N",
                   help="Inference batch size  (default: 32)")


def _add_register_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Evidence register")
    g.add_argument("--register-budget", type=int, metavar="TOKENS",
                   help="Max token budget for register  (default: 600)")


def _add_path_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Paths")
    g.add_argument("--index-dir", metavar="DIR",
                   help="Directory for BM25/FAISS/graph artefacts  (default: indexes)")
    g.add_argument("--runs-dir",  metavar="DIR",
                   help="Parent directory for run outputs  (default: runs)")


def _add_corpus_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Corpus / profile")
    g.add_argument("--corpus-dir", metavar="DIR",
                   help="repolicyqa: folder of Markdown files (default: <data dir>/final)")
    g.add_argument("--profile", metavar="JSON",
                   help="JSON file of Config overrides, e.g. profiles/policy_baseline.json")
    g.add_argument("--llm-cache", metavar="SQLITE", dest="llm_cache",
                   help="Persistent LLM response cache; reruns replay stored completions "
                        "so runs are reproducible (e.g. cache/llm_qwen7b.sqlite)")


def apply_profile(path: str, config) -> None:
    """Apply a JSON {config_field: value} profile; unknown keys are an error."""
    with open(path, encoding="utf-8") as f:
        prof = json.load(f)
    for k, v in prof.items():
        if k.startswith("_"):
            continue                      # comments / descriptions
        if not hasattr(config, k):
            raise ValueError(f"Profile {path}: unknown config field '{k}'")
        setattr(config, k, v)
    logger.info("Applied profile %s: %s", path, {k: v for k, v in prof.items() if not k.startswith("_")})


def apply_args_to_config(args, config) -> None:
    """
    Copy every CLI override into config.
    None-valued args are skipped (config default wins).
    A --profile is applied first, so explicit CLI flags still win.
    """
    if getattr(args, "profile", None):
        apply_profile(args.profile, config)
    mapping = {
        # LLM
        "provider":           "llm_provider",
        "model":              "llm_model",
        "temperature":        "llm_temperature",
        "ollama_url":         "ollama_base_url",
        # Embedding
        "embedding_model":    "embedding_model",
        "embedding_device":   "embedding_device",
        # Retrieval
        "top_k_initial":      "top_k_initial",
        "top_k":              "top_k_final",
        "hybrid_lambda":      "hybrid_lambda",
        # Graph
        "graph_hops":         "graph_expansion_hops",
        "graph_top_n":        "graph_expansion_top_n",
        "graph_sem_threshold":"graph_semantic_threshold",
        # Verifier
        "verifier_model":     "verifier_model",
        "verifier_batch_size":"verifier_batch_size",
        # Register
        "register_budget":    "register_token_budget",
        # Paths
        "index_dir":          "index_dir",
        "runs_dir":           "runs_dir",
        "llm_cache":          "llm_cache_path",
    }
    for arg_attr, cfg_attr in mapping.items():
        val = getattr(args, arg_attr, None)
        if val is not None:
            setattr(config, cfg_attr, val)

    # Stage 2 boolean flags (store_true → only set when explicitly passed)
    if getattr(args, "no_gap_fill", False):
        config.gap_fill_enabled = False
    if getattr(args, "no_eviction", False):
        config.register_eviction_enabled = False
    if getattr(args, "no_verify", False):
        config.answer_verify_enabled = False
    if getattr(args, "llm_binding", False):
        config.llm_binding_extraction = True
    if getattr(args, "self_consistency", False):
        config.self_consistency_enabled = True
    if getattr(args, "gap_fill_iters", None) is not None:
        config.gap_fill_max_iter = args.gap_fill_iters
    if getattr(args, "gap_fill_threshold", None) is not None:
        config.gap_fill_confidence_threshold = args.gap_fill_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    from registerrag.config import Config
    config = Config.from_env()

    parser = argparse.ArgumentParser(
        prog="registerrag",
        description="RegisterRAG Stage 1 — full CLI control, no code edits needed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── build ──────────────────────────────────────────────────────────────────
    p_build = sub.add_parser(
        "build",
        help="Build corpus indexes and evidence graph (one-time)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_build.add_argument("--data",   required=True,
                         help="Dataset JSON file (or policy doc directory)")
    p_build.add_argument("--format",
                         choices=["hotpotqa", "2wiki", "musique", "generic", "policy", "repolicyqa"],
                         default="hotpotqa")
    _add_corpus_args(p_build)
    _add_embedding_args(p_build)
    _add_graph_args(p_build)
    _add_path_args(p_build)

    # ── eval ───────────────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "eval",
        help="Evaluate Stage 1 on a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_eval.add_argument("--data",   required=True,
                        help="Dataset JSON file")
    p_eval.add_argument("--format",
                        choices=["hotpotqa", "2wiki", "musique", "generic", "repolicyqa"],
                        default="hotpotqa")
    _add_corpus_args(p_eval)
    p_eval.add_argument("--max-questions", type=int, default=None,
                        metavar="N",
                        help="Limit to first N questions (e.g. 5 for a smoke test)")
    p_eval.add_argument("--question-ids", metavar="ID1,ID2,...",
                        help="Run only these comma-separated question IDs")
    p_eval.add_argument("--run-name", metavar="NAME",
                        help="Label for this run; auto-generated if omitted")
    p_eval.add_argument("--stage", type=int, choices=[1, 2], default=1,
                        help="Pipeline stage: 1=single-pass, 2=gap-fill+eviction+verify (default: 1)")
    p_eval.add_argument("--resume", action="store_true",
                        help="Skip questions already in results.jsonl and append new results")
    # Stage 2 knobs
    s2 = p_eval.add_argument_group("Stage 2 options")
    s2.add_argument("--gap-fill-iters", type=int, metavar="N",
                    help="Max gap-fill iterations per question (default: 2)")
    s2.add_argument("--gap-fill-threshold", type=float, metavar="T",
                    help="Confidence below which a binding is treated as a gap (default: 0.5)")
    s2.add_argument("--no-gap-fill", action="store_true",
                    help="Disable gap filling even in Stage 2")
    s2.add_argument("--no-eviction", action="store_true",
                    help="Disable register eviction (revert to Stage 1 warn-only)")
    s2.add_argument("--no-verify", action="store_true",
                    help="Disable answer verification + retry")
    s2.add_argument("--llm-binding", action="store_true",
                    help="Use an LLM call to extract each binding instead of spaCy NER "
                         "(more accurate for multi-hop, +1 LLM call per requirement)")
    s2.add_argument("--self-consistency", action="store_true",
                    help="On comparison questions, sample the answer k=3 times at "
                         "temperature 0.35 and take the majority vote")
    _add_llm_args(p_eval)
    _add_embedding_args(p_eval)
    _add_retrieval_args(p_eval)
    _add_graph_args(p_eval)
    _add_verifier_args(p_eval)
    _add_register_args(p_eval)
    _add_path_args(p_eval)

    # ── single ─────────────────────────────────────────────────────────────────
    p_single = sub.add_parser(
        "single",
        help="Run and trace a single question by ID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_single.add_argument("--data",        required=True)
    p_single.add_argument("--format",
                          choices=["hotpotqa", "2wiki", "musique", "generic", "repolicyqa"],
                          default="hotpotqa")
    _add_corpus_args(p_single)
    p_single.add_argument("--question-id", required=True,
                          help="Question ID to run")
    _add_llm_args(p_single)
    _add_embedding_args(p_single)
    _add_retrieval_args(p_single)
    _add_graph_args(p_single)
    _add_verifier_args(p_single)
    _add_register_args(p_single)
    _add_path_args(p_single)

    # ── split ──────────────────────────────────────────────────────────────────
    p_split = sub.add_parser(
        "split",
        help="Write stratified dev/test splits of a QA file (same raw format)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_split.add_argument("--data", required=True)
    p_split.add_argument("--stratify-key", default="question_type",
                         help="Field to stratify on (2wiki: type, repolicyqa: question_type)")
    p_split.add_argument("--dev-per-group", type=int, default=20)
    p_split.add_argument("--seed", type=int, default=13)
    p_split.add_argument("--out-dir", default="splits")

    # ── score ──────────────────────────────────────────────────────────────────
    p_score = sub.add_parser(
        "score",
        help="Score a finished run: per-type EM/F1/ROUGE-L/yes-no accuracy + evidence hit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_score.add_argument("--data", required=True)
    p_score.add_argument("--format", choices=["hotpotqa", "2wiki", "musique", "generic", "repolicyqa"],
                         default="hotpotqa")
    p_score.add_argument("--run-name", required=True)
    _add_corpus_args(p_score)
    _add_path_args(p_score)

    args = parser.parse_args()
    apply_args_to_config(args, config)

    if args.command == "split":
        from registerrag.evaluation.splits import write_splits
        write_splits(args.data, args.stratify_key, args.dev_per_group, args.seed, args.out_dir)
        return
    if args.command == "score":
        from registerrag.evaluation.scoring import score_run
        questions = _load_questions(args.data, args.format, getattr(args, "corpus_dir", None))
        score_run(os.path.join(config.runs_dir, args.run_name), questions, config.index_dir)
        return

    from registerrag import llm_cache
    if config.llm_cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(config.llm_cache_path)), exist_ok=True)
    cache = llm_cache.configure(config.llm_cache_path or None)

    if args.command == "build":
        cmd_build(args, config)
    elif args.command == "eval":
        cmd_eval(args, config)
    elif args.command == "single":
        cmd_single(args, config)
    if cache:
        logger.info(cache.stats())


if __name__ == "__main__":
    main()
