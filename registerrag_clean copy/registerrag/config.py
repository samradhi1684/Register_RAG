"""
RegisterRAG configuration dataclass.
Every tunable hyperparameter lives here.
All fields can be overridden from the CLI — see run.py apply_args_to_config().
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os


@dataclass
class Config:
    # ── LLM / Generator ───────────────────────────────────────────────────────
    llm_provider: str = "openai"
    # "openai"    → needs OPENAI_API_KEY
    # "anthropic" → needs ANTHROPIC_API_KEY
    # "ollama"    → needs Ollama running locally (no key)
    llm_model: str = "gpt-4o-mini"
    # Examples:
    #   openai:    gpt-4o-mini  |  gpt-4o
    #   anthropic: claude-haiku-3-5  |  claude-sonnet-4-5
    #   ollama:    qwen2.5:7b  |  llama3.1:8b  |  mistral:7b
    llm_temperature: float = 0.0

    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    # ── Embedding model ────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    # Other fast options: "all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"   # "cuda" for GPU
    embedding_batch_size: int = 256

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k_initial: int = 20     # hybrid candidates before cross-encoder
    top_k_final: int = 10       # kept after rerank; labelled SUPPORT
    hybrid_lambda: float = 0.5  # λ·dense + (1-λ)·BM25  [0=pure BM25, 1=pure dense]

    # ── Graph ─────────────────────────────────────────────────────────────────
    graph_max_entity_units: int = 50   # cap per entity in inverted index
    graph_entity_link_k: int = 5       # max pairs linked per entity
    graph_semantic_top_k: int = 5      # semantic neighbours per unit (FAISS NN)
    graph_semantic_threshold: float = 0.70
    graph_expansion_hops: int = 1
    graph_expansion_top_n: int = 5     # expand graph neighbours of top-N retrieved

    # ── Cross-encoder verifier ────────────────────────────────────────────────
    verifier_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Other options: "cross-encoder/ms-marco-MiniLM-L-12-v2" (slower, better)
    verifier_batch_size: int = 32

    # ── Evidence register ─────────────────────────────────────────────────────
    register_token_budget: int = 600
    max_requirements: int = 6

    # ── Binding extraction — NER type mapping ─────────────────────────────────
    answer_type_to_ner: dict = field(default_factory=lambda: {
        "person":     ["PERSON"],
        "place":      ["GPE", "LOC", "FAC", "NORP"],
        "movie":      ["WORK_OF_ART", "ORG"],
        "film":       ["WORK_OF_ART", "ORG"],
        "university": ["ORG", "GPE"],
        "org":        ["ORG"],
        "date":       ["DATE"],
        "year":       ["DATE"],
        "number":     ["CARDINAL", "ORDINAL", "QUANTITY"],
        "amount":     ["MONEY", "QUANTITY", "CARDINAL"],
    })

    # ── spaCy ─────────────────────────────────────────────────────────────────
    spacy_model: str = "en_core_web_sm"
    # Retrieval unit for every corpus: "sentence" = one unit per sentence (paragraphs,
    # list items; a table row / metadata field stays one unit, a run-on sentence of
    # >160 words is windowed). "passage" = sentences packed into <=80-word passages
    # (the earlier MuSiQue/DSIRE setting, kept for comparison). Applied at build time.
    retrieval_unit: str = "sentence"

    # ── Paths ─────────────────────────────────────────────────────────────────
    index_dir: str = "indexes"
    runs_dir: str = "runs"

    # ── Retrieval quality (multi-query RRF + title-boost + LLM binding) ──────
    multi_query_enabled: bool = True          # build up to 3 query variants, fuse with RRF
    multi_query_max_variants: int = 3         # dep+need, concrete-need, entity-title
    top_k_multi_query: int = 30               # per-variant depth + fused pool size when multi_query on
    title_boost_enabled: bool = True          # +title_boost_weight for doc-title entity match
    title_boost_weight: float = 0.20
    entity_anchor_enabled: bool = True        # inject dep entity's own article into dependent-step candidates
    root_title_anchor_enabled: bool = True    # inject articles whose title the root step names (loose match)
    title_step_collapse_enabled: bool = True  # drop "What is the title of X?" planner steps
    hub_filter_enabled: bool = True           # dependent steps: skip pronoun-subject sentences from other articles
    answer_kind_router_enabled: bool = True   # route by answer kind (boolean/choice/fact/description): binding style, answer format, voting
    answer_normalize_enabled: bool = True     # trim honorifics / counts / place qualifiers from final answer
    yesno_compare_enabled: bool = True        # decide "same country/nationality?" by comparing leaf bindings
    kinship_templates_enabled: bool = True    # deterministic decomposition of grandparent / in-law questions
    anchor_v1_fallback_enabled: bool = True   # dependent step empty -> v1 prompt over the dep entity's own article
    date_compare_enabled: bool = True         # decide "which came first / who is younger" from date bindings
    refusal_fallback_enabled: bool = True     # "Not provided"-style answers -> final binding
    v1_article_title: bool = True             # v1 binding prompt names the source article (resolves "She was a wife of …")
    llm_binding_extraction: bool = False      # use LLM to extract binding (--llm-binding)
    self_consistency_enabled: bool = False    # majority vote over k samples on comparison Qs (--self-consistency)
    self_consistency_k: int = 3
    self_consistency_temperature: float = 0.35

    # ── Stage 2 features ──────────────────────────────────────────────────────
    # Gap detection + iterative retrieval
    gap_fill_enabled: bool = True
    gap_fill_max_iter: int = 2            # max gap-fill iterations per question
    gap_fill_confidence_threshold: float = 0.5   # entry with conf < this is a gap
    gap_fill_top_k_boost: int = 10        # extra candidates on gap re-query

    # Register eviction
    register_eviction_enabled: bool = True  # False = Stage 1 warn-only behaviour

    # Answer verification via cross-encoder
    answer_verify_enabled: bool = True
    answer_verify_threshold: float = -2.0  # ms-marco logit; below = likely hallucination
    answer_verify_max_retries: int = 1

    # ── Run limits ────────────────────────────────────────────────────────────
    max_questions: int = -1   # -1 = all; set e.g. 5 for a quick smoke test
    seed: int = 42
    # ── Scope anchoring (pipeline/scope.py) ──────────────────────────────────
    scope_anchor_enabled: bool = False    # fuzzy question → scope fallback anchor (on in *_scope profiles)
    scope_min_coverage: float = 0.5       # share of the scope name's idf mass matched
    anchor_inject_all_max: int = 40       # docs up to this size are injected whole
    anchor_scope_top_n: int = 30          # else: top-n of a hybrid search inside them

    # Steps left without a binding: show their top SUPPORT passages to the
    # answer generator anyway, so it can still infer the answer
    missing_step_evidence_enabled: bool = False
    missing_step_evidence_k: int = 3

    # Planner: extra rules + examples for nested descriptions (inside-out, one relation per step)
    planner_nested_rules_enabled: bool = False

    # Register: when per-passage binding extraction finds nothing, one joint
    # call over the top-k SUPPORT passages (answer must occur in them)
    joint_binding_fallback_enabled: bool = False
    joint_binding_k: int = 5
    joint_binding_first: bool = False       # try the joint call before per-passage extraction

    # Answer prompt: ask for the minimal answer span (no titles / added places / units)
    answer_minimal_span_enabled: bool = False

    # Unified step 2 (pipeline/operators.py): general comparison handling
    compare_operator_enabled: bool = False   # >=2 leaf bindings -> LLM states the comparison, code computes it
    choice_snap_enabled: bool = False        # "…, A or B?" -> answer snapped to the named option
    answer_type_check_enabled: bool = False  # choice answer naming no option -> one constrained retry (replaces year guard)

    # Description answers: ask for every stated condition/amount/deadline/exception
    description_complete_enabled: bool = False

    # Span binding prompt returns {"answer": str} ("" = none) instead of
    # {"has_candidate": bool, ...}, which made the 7B model refuse before reading
    binding_answer_first: bool = False
    binding_answer_check: bool = False   # with answer_first: {"answer", "answers_need"} — judge after reading
    binding_type_check: bool = False     # reject bare numbers for person/place/org, digit-less dates/numbers

    # Binding beam by backtracking (stage2._backtrack): a NOT FOUND dependent step
    # retries with alternative values of the step it depends on
    backtrack_enabled: bool = False
    backtrack_width: int = 3              # alternatives tried per dependency
    backtrack_extract_calls: int = 3      # LLM extraction calls to find alternatives

    # v3 candidate (pipeline/nli.py): NLI check of each extracted short binding
    # against its source passage. Contradiction -> try the next passage;
    # with nli_lookahead > 0, a neutral value is kept only if none of the next
    # nli_lookahead passages gives an entailed one.
    nli_verify_enabled: bool = False
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_lookahead: int = 0
    nli_contradiction_min: float = 0.5   # p(contradiction) needed to reject

    # Unified step 3 (pipeline/replan.py): re-plan steps still NOT FOUND after gap fill
    replan_enabled: bool = False
    replan_max_steps: int = 2            # sub-steps per rewrite
    replan_max_per_question: int = 2     # rewrites per question

    # Persistent LLM response cache (SQLite) for reproducible runs; '' = off
    llm_cache_path: str = ""

    # ─────────────────────────────────────────────────────────────────────────
    def validate(self) -> None:
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Export it or use --provider ollama.")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        # ollama: no key needed — just a running server

    @classmethod
    def from_env(cls) -> "Config":
        """Load defaults then apply RRAG_* environment overrides."""
        cfg = cls()
        env_map = {
            "RRAG_LLM_PROVIDER":     "llm_provider",
            "RRAG_LLM_MODEL":        "llm_model",
            "RRAG_EMBEDDING_MODEL":  "embedding_model",
            "RRAG_EMBEDDING_DEVICE": "embedding_device",
            "RRAG_INDEX_DIR":        "index_dir",
            "RRAG_RUNS_DIR":         "runs_dir",
            "OLLAMA_BASE_URL":       "ollama_base_url",
        }
        for env_key, attr in env_map.items():
            val = os.getenv(env_key)
            if val is not None:
                setattr(cfg, attr, val)
        return cfg

    def summary(self) -> str:
        """Human-readable one-block config dump printed at run start."""
        lines = [
            "┌─ Config ──────────────────────────────────────────",
            f"│  provider/model     : {self.llm_provider} / {self.llm_model}",
            f"│  embedding          : {self.embedding_model} on {self.embedding_device}",
            f"│  verifier           : {self.verifier_model}",
            f"│  top_k_initial/final: {self.top_k_initial} / {self.top_k_final}",
            f"│  hybrid λ           : {self.hybrid_lambda}",
            f"│  multi_query        : {'on' if self.multi_query_enabled else 'off'}  title_boost={'on' if self.title_boost_enabled else 'off'}({self.title_boost_weight})  llm_binding={'on' if self.llm_binding_extraction else 'off'}",
            f"│  graph hops/top_n   : {self.graph_expansion_hops} / {self.graph_expansion_top_n}",
            f"│  register budget    : {self.register_token_budget} tokens  eviction={'on' if self.register_eviction_enabled else 'off'}",
            f"│  gap fill           : {'on' if self.gap_fill_enabled else 'off'}  iters={self.gap_fill_max_iter}  conf_thresh={self.gap_fill_confidence_threshold}",
            f"│  answer verify      : {'on' if self.answer_verify_enabled else 'off'}  threshold={self.answer_verify_threshold}",
            f"│  max_questions      : {'all' if self.max_questions < 0 else self.max_questions}",
            f"│  index_dir          : {self.index_dir}",
            f"│  runs_dir           : {self.runs_dir}",
            "└───────────────────────────────────────────────────",
        ]
        return "\n".join(lines)
