"""
IRCoT Baseline — aligned with StonyBrookNLP/ircot (ACL 2023)
=============================================================
Pipeline (ircot_qa system, per base_configs/ircot_qa_*.jsonnet):

  Step 0  – Initial BM25 retrieval with the question as the query
  Loop    – Interleave CoT generation (one sentence) + BM25 retrieval
            using query_source="question_or_last_generated_sentence"
  Exit    – StepByStepExitControllerParticipant fires when (in order):
              1. last generated sentence is empty, OR
              2. len(generated_sentences) >= max_sentences (default 10), OR
              3. answer_extractor_regex matches the last generated sentence
            global_max_num_paras (15) caps paragraph accumulation only; it
            does NOT add a fourth exit condition from the CoT loop.
            Retrieval is only performed when NONE of the three exit
            conditions are true — exit is checked fully before retrieving.
  Step N  – Separate final QA LLM call with all collected paragraphs as
             context, using the same cot-style prompt (rc_qa_type="cot").
             The regex is applied to the QA output; if it does not match,
             match_all_on_failure=True returns the full raw output as the
             answer (after stripping a trailing period if present).
             Both CoT and final QA steps use the SAME prompt file:
               prompts/{dataset}/gold_with_2_distractors_context_cot_qa_codex.txt

Key settings sourced from base_configs/ircot_qa_codex_*.jsonnet:
  • answer_extractor_regex  : r".* answer is:? (.*)\\.?"   (note ":?" and "\\."?)
  • answer_extractor_remove_last_fullstop : True
  • match_all_on_failure     : True  (fallback = full string, also strips period)
  • global_max_num_paras    : 15
  • bm25_retrieval_count    : 6  (hits per retrieval step)
  • rc_context_type         : "gold_with_2_distractors"
  • rc_qa_type              : "cot"  (both CoT gen and final QA use the cot prompt)
  • document_type           : "title_paragraph_text"  (BM25 retrieval)
  • cumulate_titles         : True  (paragraphs accumulate across steps)
  • question_prefix         : ""   (codex configs omit this key entirely)
  • Prompt files            : prompts/{dataset}/gold_with_2_distractors_context_cot_qa_codex.txt
                              (the SAME file for both CoT generation and final QA)

Settings that cannot be matched (model substitution):
  • Original engine is code-davinci-002 (Codex); gpt-4o-mini / qwen2.5:14b
    are substituted. Prompt format and stop sequences (["\n"]) are preserved.

NOTE: This implementation targets the ircot_qa pipeline, not ircot-only
(retrieval). It does not reproduce the original Codex/Flan-T5 results;
those models are unavailable. gpt-4o-mini and qwen2.5:14b are substituted.

Generators supported:
  - gpt-4o-mini via OpenAI Chat API  (OPENAI_API_KEY env var)
  - qwen2.5:14b  via Ollama           (ollama must be running locally)

Datasets supported:
  - hotpotqa
  - 2wikimultihopqa
  - musique
  - policyqa   (domain-specific; uses its own context field, no retriever)

Remaining differences from the original Codex experiment:
  - LLM: code-davinci-002 replaced by gpt-4o-mini or qwen2.5:14b.
    The Completion API (raw text continuation) is replaced by the Chat API;
    prompt format and stop=["\n"] are preserved but token distributions differ.
  - policyqa is not part of the original IRCoT benchmark; it uses local context
    and its own prompt, so no official prompt file exists.
  - The original uses Elasticsearch 7.10 + the IRCoT retriever_server;
    this script calls the same server via HTTP so retrieval is identical if
    the same index is running.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import List, Optional

import requests
from rapidfuzz import fuzz
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Retrieval / text helpers — verbatim from commaqa/inference/ircot.py
# ---------------------------------------------------------------------------

random.seed(100)  # Match original ircot.py: random.seed(100) — Don't change.

_WH_WORDS = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}

# Original is_reasoning_sentence() starters
_REASONING_STARTERS = ["thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence"]

# Original math-equation regex
_REASONING_MATH_RE = re.compile(
    r"(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)"
)

# Answer-extractor regex — taken verbatim from base_configs/ircot_qa_codex_*.jsonnet:
#   "answer_extractor_regex": ".* answer is:? (.*)\\.?"
# The ":?" makes the colon optional; "\\.?" makes the trailing period optional.
# This matches the jsonnet configs exactly; the default in ircot.py itself omits
# ":?" but the codex configs override it to include the optional colon.
_ANSWER_RE = re.compile(r".* answer is:? (.*)\.?")


def remove_wh_words(text: str) -> str:
    """Verbatim from commaqa/inference/ircot.py."""
    words = [w for w in text.split(" ") if w.strip().lower() not in _WH_WORDS]
    return " ".join(words)


def is_reasoning_sentence(sentence: str) -> bool:
    """Verbatim from commaqa/inference/ircot.py."""
    for starter in _REASONING_STARTERS:
        if sentence.lower().startswith(starter):
            return True
    return bool(_REASONING_MATH_RE.match(sentence))


def remove_reasoning_sentences(sentences: List[str]) -> List[str]:
    """Verbatim from commaqa/inference/ircot.py."""
    return [s for s in sentences if not is_reasoning_sentence(s)]


def is_para_closely_matching(
    existing_titles: List[str],
    existing_paras: List[str],
    new_title: str,
    new_para: str,
    match_threshold: float = 90,
) -> bool:
    """Verbatim from commaqa/inference/ircot.py."""
    if new_title in existing_titles and new_para in existing_paras:
        return True
    for et, ep in zip(existing_titles, existing_paras):
        if (fuzz.ratio(et, new_title) >= match_threshold
                and fuzz.ratio(ep, new_para) >= match_threshold):
            return True
    return False


def para_to_text(title: str, para: str, max_num_words: int = 350) -> str:
    """Verbatim from commaqa/inference/ircot.py.

    Note: split/join on spaces (not split()) to preserve newlines,
    then attach title prefix if not already present.
    """
    para = " ".join(para.split(" ")[:max_num_words])
    if not para.strip().startswith("Wikipedia Title: "):
        para = "Wikipedia Title: " + title + "\n" + para.strip()
    return para


def extract_answer(sentence: str, remove_last_fullstop: bool = True) -> Optional[str]:
    """Apply the answer-extractor regex from base_configs/ircot_qa_codex_*.jsonnet.

    Regex: ".* answer is:? (.*)\\.?"
    answer_extractor_remove_last_fullstop: True

    Returns the captured group, stripping a trailing period if present.
    match_all_on_failure=True is handled at the call site (not here).
    """
    m = _ANSWER_RE.match(sentence.strip())
    if m:
        ans = m.group(1).strip()
        if remove_last_fullstop and ans.endswith("."):
            ans = ans[:-1]
        return ans
    return None


# ---------------------------------------------------------------------------
# spaCy sentence splitter — same as StepByStepCOTGenParticipant
# ---------------------------------------------------------------------------

_SPACY_NLP = None


def get_first_sentence(text: str) -> str:
    """Return the first spaCy sentence — identical to the original."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy  # type: ignore
        _SPACY_NLP = spacy.load("en_core_web_sm")
    doc = _SPACY_NLP(text)
    sents = list(doc.sents)
    return sents[0].text if sents else text


# ---------------------------------------------------------------------------
# BM25 retrieval via the IRCoT retriever_server
# ---------------------------------------------------------------------------

def safe_post(url: str, params: dict, retries: int = 10) -> dict:
    for attempt in range(retries):
        try:
            r = requests.post(url, json=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[retriever] attempt {attempt+1}/{retries} failed: {e}. Retrying in 20s…")
            time.sleep(20)
    raise RuntimeError("Retriever unreachable after several attempts.")


def bm25_retrieve(
    query: str,
    corpus_name: str,
    retriever_url: str,
    retrieval_count: int,
    document_type: str = "title_paragraph_text",  # matches base_configs
) -> List[dict]:
    """BM25 retrieval via the IRCoT retriever_server (Elasticsearch backend).

    document_type is "title_paragraph_text" per base_configs/*.jsonnet.
    wh-word removal is applied here (same as in the original query() method).
    """
    query = remove_wh_words(query)
    if not query.strip():
        return []
    params = {
        "retrieval_method": "retrieve_from_elasticsearch",
        "query_text": query,
        "max_hits_count": retrieval_count,
        "corpus_name": corpus_name,
        "document_type": document_type,
    }
    result = safe_post(retriever_url + "/retrieve", params)
    return result.get("retrieval", [])


def accumulate_paragraphs(
    retrieved_titles: List[str],
    retrieved_paras: List[str],
    hits: List[dict],
    global_max_num_paras: int = 15,
) -> None:
    """Add new hits to the running lists, respecting the global cap.

    Mirrors RetrieveAndResetParagraphsParticipant with cumulate_titles=True
    and global_max_num_paras=15 (from base_configs).
    Paragraphs longer than 600 words are skipped (same as original).
    """
    for hit in hits:
        if len(retrieved_paras) >= global_max_num_paras:
            break
        title = hit["title"]
        para  = hit.get("paragraph_text", "")
        if len(para.split()) > 600:
            print("[retriever] WARNING: skipping excessively long paragraph.")
            continue
        if not is_para_closely_matching(retrieved_titles, retrieved_paras, title, para):
            retrieved_titles.append(title)
            retrieved_paras.append(para)


# ---------------------------------------------------------------------------
# LLM generators
# ---------------------------------------------------------------------------

class OpenAIGenerator:
    """Chat wrapper for gpt-4o-mini.

    Generation parameters held constant to match original spirit:
      temperature=0, max_tokens=300, stop=["\n"]
    (Original used Completion API; we use Chat API with the same settings.)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 300,
        stop: Optional[List[str]] = None,
    ):
        import openai  # type: ignore
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop if stop is not None else ["\n"]

    def generate(self, prompt: str) -> str:
        prompt = prompt.rstrip()
        for attempt in range(10):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stop=self.stop,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                print(f"[openai] attempt {attempt+1}: {e}. Retrying in 20s…")
                time.sleep(20)
        raise RuntimeError("OpenAI call failed after retries.")


class OllamaGenerator:
    """Wrapper for local Ollama (qwen2.5:14b).
    Same parameters as OpenAIGenerator: temperature=0, max_tokens=300, stop=["\n"].
    """

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        temperature: float = 0.0,
        max_tokens: int = 300,
        stop: Optional[List[str]] = None,
        ollama_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop if stop is not None else ["\n"]
        self.base_url = ollama_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        prompt = prompt.rstrip()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "stop": self.stop,
            },
        }
        for attempt in range(10):
            try:
                r = requests.post(
                    f"{self.base_url}/api/generate", json=payload, timeout=120
                )
                r.raise_for_status()
                return r.json().get("response", "").strip()
            except Exception as e:
                print(f"[ollama] attempt {attempt+1}: {e}. Retrying in 20s…")
                time.sleep(20)
        raise RuntimeError("Ollama call failed after retries.")


def build_generator(generator_name: str, **kwargs):
    if generator_name == "gpt-4o-mini":
        return OpenAIGenerator(model="gpt-4o-mini", **kwargs)
    elif generator_name.startswith("qwen") or generator_name == "ollama":
        model = kwargs.pop("model", "qwen2.5:14b")
        return OllamaGenerator(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown generator: {generator_name}")


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def load_prompt(prompt_path: str) -> str:
    """Load a prompt file, stripping '# METADATA:' lines.
    Matches the original read_prompt() filter.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("# METADATA:")]
    return "".join(lines).strip()


def build_cot_prompt(
    few_shot_prompt: str,
    question: str,
    titles: List[str],
    paras: List[str],
    cot_so_far: str,
    question_prefix: str = "",
    max_para_num_words: int = 350,
) -> str:
    """Build the prompt for the CoT-generation step.

    Mirrors StepByStepCOTGenParticipant.query() with add_context=True:
      <context paragraphs>

      Q: <question_prefix><question>
      A: <cot so far>

    Separated from the few-shot block by \\n\\n\\n (same as original).
    The question_prefix defaults to "" because codex-based base_configs omit
    the question_prefix key (StepByStepCOTGenParticipant default is "").
    Pass "Answer the following question by reasoning step-by-step.\\n" to
    match flan-t5 configs.
    """
    context_parts = [para_to_text(t, p, max_para_num_words) for t, p in zip(titles, paras)]
    context = "\n\n".join(context_parts).strip()

    # question with prefix prepended — same as original question_prefix logic
    full_question = question_prefix + question

    test_block = context + "\n\n" + f"Q: {full_question}" + "\n" + f"A: {cot_so_far}"
    test_block = re.sub(r"\n\n+", "\n\n", test_block)

    if few_shot_prompt:
        return "\n\n\n".join([few_shot_prompt, test_block]).strip()
    return test_block.strip()


def build_qa_prompt(
    few_shot_prompt: str,
    question: str,
    titles: List[str],
    paras: List[str],
    question_prefix: str = "",
    max_para_num_words: int = 350,
) -> str:
    """Build the prompt for the final QA step (answer_main_question).

    rc_qa_type="cot": uses the same cot prompt file as the CoT generation step.
    The prompt format is identical to build_cot_prompt except the "A:" line
    has no cot_so_far (the model must generate from scratch).

    Mirrors the llmqa participant with rc_qa_type="cot":
      <context paragraphs>

      Q: <question_prefix><question>
      A:

    The question_prefix defaults to "" because codex-based base_configs omit
    the qa_question_prefix key.  Flan-t5 configs use
    "Answer the following question by reasoning step-by-step.\\n" (cot) or
    "Answer the following question.\\n" (direct).
    """
    context_parts = [para_to_text(t, p, max_para_num_words) for t, p in zip(titles, paras)]
    context = "\n\n".join(context_parts).strip()

    full_question = question_prefix + question
    test_block = context + "\n\n" + f"Q: {full_question}" + "\n" + "A:"
    test_block = re.sub(r"\n\n+", "\n\n", test_block)

    if few_shot_prompt:
        return "\n\n\n".join([few_shot_prompt, test_block]).strip()
    return test_block.strip()


# ---------------------------------------------------------------------------
# Dataset readers
# ---------------------------------------------------------------------------

def _load_raw(path: str) -> list:
    """Load a file that is either a JSON array or JSONL (one object per line).

    The official IRCoT processed data files (dev_subsampled.jsonl, test_subsampled.jsonl)
    are JSONL format.  policyqa data is typically a JSON array.  This helper accepts both.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)          # JSON array
    # JSONL: one JSON object per non-empty line
    return [json.loads(ln) for ln in content.splitlines() if ln.strip()]


def load_dataset(path: str, dataset_name: str) -> List[dict]:
    """Normalise raw JSON/JSONL to {id, question, answer, context_titles, context_paras}.

    Accepts both JSON arrays (legacy / policyqa) and JSONL files (official IRCoT
    processed_data/{dataset}/dev_subsampled.jsonl and test_subsampled.jsonl).
    """
    raw = _load_raw(path)
    items = []

    if dataset_name == "hotpotqa":
        for ex in raw:
            titles = ex["context"]["title"]
            paras  = ["".join(sents) for sents in ex["context"]["sentences"]]
            items.append({
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "context_titles": titles,
                "context_paras": paras,
            })

    elif dataset_name == "2wikimultihopqa":
        for ex in raw:
            titles = [c[0] for c in ex["context"]]
            paras  = ["".join(c[1]) for c in ex["context"]]
            items.append({
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex["answer"],
                "context_titles": titles,
                "context_paras": paras,
            })

    elif dataset_name == "musique":
        for ex in raw:
            titles = [p["title"] for p in ex["paragraphs"]]
            paras  = [p["paragraph_text"] for p in ex["paragraphs"]]
            items.append({
                "id": ex["id"],
                "question": ex["question"],
                "answer": ex.get("answer", ""),
                "context_titles": titles,
                "context_paras": paras,
            })

    elif dataset_name == "iirc":
        # Official processed IIRC format (output of processing_scripts/process_iirc.py):
        # Each example has a "question", "answer" (dict with "answer_type" and "answer_value"
        # or a flat string), "context" list of {title, paragraph_text} dicts, and "pid".
        for ex in raw:
            titles = [c["title"] for c in ex.get("context", [])]
            paras  = [c.get("paragraph_text", "") for c in ex.get("context", [])]
            # answer may be a string or {"answer_type": ..., "answer_value": ...}
            raw_answer = ex.get("answer", "")
            if isinstance(raw_answer, dict):
                answer = str(raw_answer.get("answer_value", ""))
            else:
                answer = str(raw_answer)
            items.append({
                "id": ex.get("pid", ex.get("id", "")),
                "question": ex["question"],
                "answer": answer,
                "context_titles": titles,
                "context_paras": paras,
            })

    elif dataset_name == "policyqa":
        for ex in raw:
            chunk_title = f"Document {ex.get('document_id', '')}"
            chunk_text  = ex.get("chunk", "")
            extra_ctx   = ex.get("context", "")
            titles = [chunk_title]
            paras  = [chunk_text + (" " + extra_ctx if extra_ctx else "")]
            items.append({
                "id": str(ex.get("document_id", "")),
                "question": ex["question"],
                "answer": ex["answer"],
                "context_titles": titles,
                "context_paras": paras,
            })
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return items


# ---------------------------------------------------------------------------
# IRCoT-QA pipeline
# ---------------------------------------------------------------------------

def ircot_qa_predict(
    item: dict,
    generator,
    cot_prompt: str,
    qa_prompt: str,
    retriever_url: Optional[str],
    corpus_name: Optional[str],
    retrieval_count: int = 6,
    max_sentences: int = 10,
    max_para_num_words: int = 350,
    global_max_num_paras: int = 15,
    use_local_context: bool = False,
    cot_question_prefix: str = "",
    qa_question_prefix: str = "",
) -> dict:
    """Run the ircot_qa pipeline for one example.

    Pipeline (mirrors base_configs/ircot_qa_codex_*.jsonnet):

    1. Initial retrieval with the question (query_source="question_or_last_generated_sentence",
       first call: generated_sentences is empty → use question).
    2. Loop — exit conditions checked in the exact order of
       StepByStepExitControllerParticipant, BEFORE any retrieval:
       a. Generate one CoT sentence (stop at \\n).
       b. Exit if sentence is empty.                      [condition 1]
       c. Exit if len(generated_sentences) >= max_sentences. [condition 2]
       d. Exit if answer_extractor_regex matches.         [condition 3]
       e. Only if NOT exiting: retrieve again using the last non-reasoning
          sentence (or question if all sentences are reasoning sentences).
          global_max_num_paras is enforced inside accumulate_paragraphs only;
          hitting the cap does NOT exit the CoT loop.
    3. Final QA call (rc_qa_type="cot"): uses the same prompt file as the
       CoT step with all collected paragraphs (answer_main_question).
    4. Extract answer from final QA output; fall back to raw output if regex
       does not match (match_all_on_failure=True in extract_answer node).

    Settings from base_configs/ircot_qa_codex_*.jsonnet:
      • bm25_retrieval_count = 6  (hits per step)
      • global_max_num_paras = 15
      • cumulate_titles = True  (paragraphs accumulate — never reset)
      • document_type = "title_paragraph_text"
      • question_prefix = "" for both CoT and QA (codex config default)
      • rc_qa_type = "cot"  (both steps share the same prompt file)
    """
    question = item["question"]
    retrieved_titles: List[str] = []
    retrieved_paras:  List[str] = []
    generated_sentences: List[str] = []

    # For policyqa / --use_local_context: seed from the item's context paragraphs
    if use_local_context or retriever_url is None:
        retrieved_titles = list(item["context_titles"])
        retrieved_paras  = list(item["context_paras"])

    # -----------------------------------------------------------------------
    # Step 0: Initial retrieval using the question
    # -----------------------------------------------------------------------
    if retriever_url and corpus_name and not use_local_context:
        hits = bm25_retrieve(
            query=question,
            corpus_name=corpus_name,
            retriever_url=retriever_url,
            retrieval_count=retrieval_count,
        )
        accumulate_paragraphs(retrieved_titles, retrieved_paras, hits, global_max_num_paras)

    # -----------------------------------------------------------------------
    # CoT loop — exit conditions are checked in the order defined by
    # StepByStepExitControllerParticipant before any retrieval is done.
    # -----------------------------------------------------------------------
    for _step in range(max_sentences):
        # --- Generate one CoT sentence ---
        cot_so_far = " ".join(generated_sentences)
        prompt = build_cot_prompt(
            cot_prompt, question,
            retrieved_titles, retrieved_paras,
            cot_so_far, cot_question_prefix, max_para_num_words,
        )
        raw_gen = generator.generate(prompt)
        new_sentence = get_first_sentence(raw_gen.strip())

        # --- Exit condition 1: empty sentence ---
        if not new_sentence:
            break

        generated_sentences.append(new_sentence)

        # --- Exit condition 2: max sentence limit reached ---
        # (StepByStepExitControllerParticipant checks len >= max_num_sentences)
        if len(generated_sentences) >= max_sentences:
            break

        # --- Exit condition 3: answer regex matches last sentence ---
        # Retrieval is skipped if any exit condition fires.
        if extract_answer(new_sentence) is not None:
            break

        # --- No exit: retrieve for next step ---
        # global_max_num_paras is enforced inside accumulate_paragraphs only;
        # hitting the cap does NOT exit the CoT loop (no such check in the
        # official StepByStepExitControllerParticipant).
        # query_source = "question_or_last_generated_sentence":
        # use the last non-reasoning sentence, or fall back to the question.
        if retriever_url and corpus_name and not use_local_context:
            non_reasoning = remove_reasoning_sentences(generated_sentences)
            query = non_reasoning[-1].strip() if non_reasoning else question
            hits = bm25_retrieve(
                query=query,
                corpus_name=corpus_name,
                retriever_url=retriever_url,
                retrieval_count=retrieval_count,
            )
            accumulate_paragraphs(retrieved_titles, retrieved_paras, hits, global_max_num_paras)

    # -----------------------------------------------------------------------
    # Final QA step — separate LLM call with all collected paragraphs.
    # rc_qa_type="cot": uses the SAME prompt file as the CoT generation step
    # (answer_main_question / llmqa participant with rc_qa_type="cot").
    # -----------------------------------------------------------------------
    qa_prompt_full = build_qa_prompt(
        qa_prompt, question,
        retrieved_titles, retrieved_paras,
        qa_question_prefix, max_para_num_words,
    )
    qa_raw = generator.generate(qa_prompt_full)
    qa_raw = qa_raw.strip()

    # Extract answer from QA output.
    # match_all_on_failure=True: if regex fails, use the full output as the answer.
    # answer_extractor_remove_last_fullstop=True: strip trailing period either way.
    predicted_answer = extract_answer(qa_raw)
    if predicted_answer is None:
        # match_all_on_failure fallback
        predicted_answer = qa_raw
        if predicted_answer.endswith("."):
            predicted_answer = predicted_answer[:-1]

    return {
        "id": item["id"],
        "question": question,
        "answer": item["answer"],
        "predicted_answer": predicted_answer,
        "generated_cot": " ".join(generated_sentences),
        "qa_raw_output": qa_raw,
        "retrieved_titles": retrieved_titles,
        "num_retrieved_paras": len(retrieved_paras),
        "num_cot_sentences": len(generated_sentences),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str) -> dict:
    done: dict = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[str(rec["id"])] = rec
    return done


def save_checkpoint_line(checkpoint_path: str, result: dict) -> None:
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(result) + "\n")


# ---------------------------------------------------------------------------
# Dataset / corpus config
# ---------------------------------------------------------------------------

DATASET_CORPUS_NAMES = {
    "hotpotqa":        "hotpotqa",          # ES index name used by IRCoT
    "2wikimultihopqa": "2wikimultihopqa",
    "musique":         "musique",
    "iirc":            "iirc",
    "policyqa":        None,                 # local context only
}

DATASET_ALIASES = {
    "hotpot":          "hotpotqa",
    "hotpotqa":        "hotpotqa",
    "2wiki":           "2wikimultihopqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique":         "musique",
    "iirc":            "iirc",
    "policy":          "policyqa",
    "policyqa":        "policyqa",
}

# Official prompt file paths (relative to the script's directory or --prompt_dir).
#
# Per base_configs/ircot_qa_codex_*.jsonnet, rc_qa_type="cot": both the CoT
# generation step and the final QA step use the SAME prompt file:
#   prompts/{dataset}/gold_with_2_distractors_context_cot_qa_codex.txt
#
# policyqa is not part of the original IRCoT benchmark; its prompt paths are
# placeholders and must be supplied by the user via --cot_prompt / --qa_prompt.
_OFFICIAL_PROMPT = {
    "hotpotqa":        "prompts/hotpotqa/gold_with_2_distractors_context_cot_qa_codex.txt",
    "2wikimultihopqa": "prompts/2wikimultihopqa/gold_with_2_distractors_context_cot_qa_codex.txt",
    "musique":         "prompts/musique/gold_with_2_distractors_context_cot_qa_codex.txt",
    "iirc":            "prompts/iirc/gold_with_2_distractors_context_cot_qa_codex.txt",
    "policyqa":        None,  # no official prompt; must supply --cot_prompt / --qa_prompt
}

# Both CoT generation and final QA use the same official prompt file.
DEFAULT_COT_PROMPTS = _OFFICIAL_PROMPT
DEFAULT_QA_PROMPTS  = _OFFICIAL_PROMPT


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="IRCoT-QA baseline — aligned with StonyBrookNLP/ircot (ACL 2023)"
    )
    p.add_argument("--dataset",      required=True,
                   help="Dataset: hotpotqa | 2wiki | musique | policyqa")
    p.add_argument("--input",        required=True,
                   help="Path to input JSON file")
    p.add_argument("--output_dir",   default="outputs",
                   help="Directory for outputs & checkpoints")

    # Two separate prompt arguments — both default to the same official file.
    # For the standard ircot_qa_codex config (rc_qa_type="cot"), both point
    # to gold_with_2_distractors_context_cot_qa_codex.txt and can be left unset.
    p.add_argument("--cot_prompt",   default=None,
                   help="Path to CoT-generation few-shot prompt file "
                        "(default: prompts/{dataset}/gold_with_2_distractors_context_cot_qa_codex.txt)")
    p.add_argument("--qa_prompt",    default=None,
                   help="Path to final QA few-shot prompt file "
                        "(default: same as --cot_prompt; rc_qa_type=\"cot\" uses one shared file)")

    p.add_argument("--generator",    default="gpt-4o-mini",
                   choices=["gpt-4o-mini", "qwen2.5:14b"],
                   help="LLM generator to use")
    p.add_argument("--retriever_url", default="http://localhost:8000",
                   help="IRCoT retriever_server URL")

    # Hyperparameters from base_configs
    p.add_argument("--retrieval_count",      type=int, default=6,
                   help="BM25 hits per step (base_configs default: 6)")
    p.add_argument("--max_sentences",        type=int, default=10,
                   help="Max CoT sentences per example (original: 10)")
    p.add_argument("--max_para_num_words",   type=int, default=350,
                   help="Word cap per paragraph in context (original: 350)")
    p.add_argument("--global_max_num_paras", type=int, default=15,
                   help="Max paragraphs accumulated total (base_configs: 15)")

    p.add_argument("--use_local_context", action="store_true",
                   help="Skip Elasticsearch; use only context from the JSON file")
    p.add_argument("--ollama_url",   default="http://localhost:11434",
                   help="Ollama server URL (for qwen2.5:14b)")
    p.add_argument("--limit",        type=int, default=None,
                   help="Only process first N examples (testing)")
    return p.parse_args()


def _resolve_prompt(path_arg: Optional[str], dataset_name: str, label: str,
                    script_dir: str) -> str:
    """Resolve and load a prompt file; abort with a clear message if missing.

    For datasets with an official prompt (hotpotqa, 2wikimultihopqa, musique)
    the default is the released gold_with_2_distractors_context_cot_qa_codex.txt
    file.  Both CoT and QA default to the same file (rc_qa_type="cot").

    For policyqa (no official prompt) the user MUST supply --cot_prompt and
    --qa_prompt explicitly; omitting either is a fatal error.
    """
    default_map = DEFAULT_COT_PROMPTS if label == "cot" else DEFAULT_QA_PROMPTS
    default_rel  = default_map.get(dataset_name)

    if path_arg:
        candidate = path_arg
    elif default_rel is None:
        # policyqa has no official prompt
        print(
            f"ERROR: no default {label} prompt exists for '{dataset_name}'. "
            f"Supply --{'cot' if label == 'cot' else 'qa'}_prompt explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        candidate = os.path.join(script_dir, default_rel)

    if not os.path.exists(candidate):
        print(
            f"ERROR: {label} prompt file not found: {candidate}\n"
            f"  Clone the official prompts from StonyBrookNLP/ircot and place them at:\n"
            f"    {os.path.join(script_dir, default_rel) if default_rel else '<path>'}\n"
            f"  or supply an explicit path with --{'cot' if label == 'cot' else 'qa'}_prompt.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompt = load_prompt(candidate)
    print(f"[IRCoT] {label} prompt: {candidate} ({len(prompt)} chars)")
    return prompt


def main():
    args = parse_args()

    # Resolve dataset name
    dataset_name = DATASET_ALIASES.get(args.dataset.lower())
    if dataset_name is None:
        print(f"ERROR: unknown dataset '{args.dataset}'. "
              f"Choose from: {list(DATASET_ALIASES.keys())}", file=sys.stderr)
        sys.exit(1)

    corpus_name   = DATASET_CORPUS_NAMES[dataset_name]
    use_local     = args.use_local_context or (corpus_name is None)
    retriever_url = None if use_local else args.retriever_url

    # Output paths
    os.makedirs(args.output_dir, exist_ok=True)
    gen_slug        = args.generator.replace(":", "_").replace(".", "_")
    run_name        = f"ircot_{gen_slug}_{dataset_name}"
    checkpoint_path = os.path.join(args.output_dir, f"{run_name}_checkpoint.jsonl")
    output_path     = os.path.join(args.output_dir, f"{run_name}_predictions.json")

    print(f"[IRCoT] dataset={dataset_name}  generator={args.generator}")
    print(f"[IRCoT] retrieval_count={args.retrieval_count}  "
          f"max_sentences={args.max_sentences}  "
          f"global_max_num_paras={args.global_max_num_paras}")
    print(f"[IRCoT] checkpoint → {checkpoint_path}")
    print(f"[IRCoT] output     → {output_path}")

    # Load data
    print(f"[IRCoT] loading {args.input} …")
    items = load_dataset(args.input, dataset_name)
    if args.limit:
        items = items[: args.limit]
    print(f"[IRCoT] {len(items)} examples loaded.")

    # Resolve prompt files — aborts on missing file instead of silently going zero-shot.
    # Both CoT and QA default to the same official file (rc_qa_type="cot").
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cot_prompt = _resolve_prompt(args.cot_prompt, dataset_name, "cot", script_dir)
    qa_prompt  = _resolve_prompt(args.qa_prompt,  dataset_name, "qa",  script_dir)

    # Generator
    gen_kwargs: dict = {}
    if args.generator == "qwen2.5:14b":
        gen_kwargs["ollama_url"] = args.ollama_url
        gen_kwargs["model"]      = "qwen2.5:14b"
    generator = build_generator(args.generator, **gen_kwargs)

    # Resume from checkpoint
    done = load_checkpoint(checkpoint_path)
    print(f"[IRCoT] {len(done)} examples already done (checkpoint).")

    results = list(done.values())
    pending = [item for item in items if str(item["id"]) not in done]

    for item in tqdm(pending, desc=f"IRCoT {dataset_name}", unit="ex"):
        try:
            result = ircot_qa_predict(
                item=item,
                generator=generator,
                cot_prompt=cot_prompt,
                qa_prompt=qa_prompt,
                retriever_url=retriever_url,
                corpus_name=corpus_name,
                retrieval_count=args.retrieval_count,
                max_sentences=args.max_sentences,
                max_para_num_words=args.max_para_num_words,
                global_max_num_paras=args.global_max_num_paras,
                use_local_context=use_local,
            )
        except Exception as e:
            print(f"\n[IRCoT] ERROR on id={item['id']}: {e}")
            result = {
                "id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "predicted_answer": "ERROR",
                "generated_cot": "",
                "qa_raw_output": "",
                "retrieved_titles": [],
                "num_retrieved_paras": 0,
                "num_cot_sentences": 0,
            }

        results.append(result)
        save_checkpoint_line(checkpoint_path, result)

    # Write final output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[IRCoT] Done. {len(results)} predictions written to {output_path}")

    # -----------------------------------------------------------------------
    # Inline evaluation — SQuAD-style EM and F1 (same normalization as the
    # official evaluate.py used by the IRCoT repo).
    # Skips examples whose predicted_answer is "ERROR".
    # -----------------------------------------------------------------------
    import string

    def _normalize(text: str) -> str:
        """Lowercase, strip punctuation and articles, collapse whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = [t for t in text.split() if t not in {"a", "an", "the"}]
        return " ".join(tokens)

    def _token_f1(pred: str, gold: str) -> float:
        pred_tokens = _normalize(pred).split()
        gold_tokens = _normalize(gold).split()
        if not pred_tokens or not gold_tokens:
            return float(pred_tokens == gold_tokens)
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            return 0.0
        p = len(common) / len(pred_tokens)
        r = len(common) / len(gold_tokens)
        return 2 * p * r / (p + r)

    def _exact_match(pred: str, gold: str) -> float:
        return float(_normalize(pred) == _normalize(gold))

    evaluated = [r for r in results if r.get("predicted_answer", "ERROR") != "ERROR"]
    if evaluated:
        em = sum(_exact_match(r["predicted_answer"], r["answer"]) for r in evaluated)
        f1 = sum(_token_f1(r["predicted_answer"],    r["answer"]) for r in evaluated)
        n  = len(evaluated)
        print(f"[IRCoT] Evaluation on {n} examples  (skipped {len(results)-n} ERRORs)")
        print(f"[IRCoT] Exact Match : {100*em/n:.2f}")
        print(f"[IRCoT] Token F1    : {100*f1/n:.2f}")
    else:
        print("[IRCoT] No valid predictions to evaluate.")


if __name__ == "__main__":
    main()