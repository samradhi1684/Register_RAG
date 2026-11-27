"""
Dataset loaders.

Supported formats
-----------------
1. HotpotQA  — JSON array of records with
   {"id", "question", "answer", "type", "supporting_facts", "context"}

1b. MuSiQue — {id, question, answer, answer_aliases, paragraphs[...]}

2. Generic factual  — JSON array of records with
   {"id", "question", "answer",
    "contexts": [{"id", "title", "sentences": ["s0","s1",…]}]}

3. Policy docs  — directory of Markdown files; one file per document
"""
from __future__ import annotations
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API — returns dicts the corpus processor can consume
# ─────────────────────────────────────────────────────────────────────────────

def _hotpot_pairs(ctx) -> list:
    """HotpotQA context in either layout -> [(title, [sentences])].
    List layout: [[title, [s0, s1, …]], …]; HuggingFace layout:
    {"title": [...], "sentences": [[...], …]}."""
    if isinstance(ctx, dict):
        return list(zip(ctx.get("title", []), ctx.get("sentences", [])))
    return [(t, s) for t, s in ctx]


def _hotpot_facts(sf) -> list:
    """supporting_facts in either layout -> [[title, sent_idx], …]."""
    if isinstance(sf, dict):
        return [[t, int(i)] for t, i in zip(sf.get("title", []), sf.get("sent_id", []))]
    return sf or []


def load_hotpotqa(path: str) -> tuple[list[dict], list[dict]]:
    """
    Load a HotpotQA JSON file (original list layout or HuggingFace dict layout).

    Returns
    -------
    questions : list of {"id", "question", "answer", "type", "supporting_facts"}
                supporting_facts: [["title", sent_idx], …] (sentence-level gold)
    documents : list of {"id", "title", "sentences": [str, …]}
                (deduplicated across all questions)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    doc_map: dict[str, dict] = {}  # title → doc

    for item in data:
        qid = item.get("_id") or item.get("id", "")
        questions.append({
            "id": qid,
            "question": item["question"],
            "answer": item["answer"],
            "type": item.get("type", ""),            # bridge / comparison
            "supporting_facts": _hotpot_facts(item.get("supporting_facts", [])),
        })
        for title, sents in _hotpot_pairs(item.get("context", [])):
            if title not in doc_map:
                doc_map[title] = {
                    "id": _slugify(title),
                    "title": title,
                    "sentences": list(sents),
                }

    documents = list(doc_map.values())
    logger.info("Loaded %d questions, %d unique documents from %s",
                len(questions), len(documents), path)
    return questions, documents


def load_musique(path: str, sentence_units: bool = True) -> tuple[list[dict], list[dict]]:
    """
    Load MuSiQue (answerable split records):
      {id, question, answer, answer_aliases, answerable, question_decomposition,
       paragraphs: [{idx, title, paragraph_text, is_supporting}]}

    Documents: one per distinct paragraph.  MuSiQue paragraphs are passages
    of Wikipedia articles, so one title can carry several different texts;
    the id is "<title slug>_<text hash>" and identical paragraphs shared by
    several questions are indexed once.  Units: one per sentence (regex split,
    corpus/structured.py), like 2Wiki/HotpotQA; with sentence_units=False the
    sentences are packed into passages of at most 80 words instead.

    Questions: type = hop count ("2hop", "3hop", "4hop", from the id);
    gold is document-level, one entry per supporting paragraph:
      supporting_facts = [{"doc_ids": [id]}, …]
    answer_aliases are kept for scoring (max over answer + aliases).
    """
    import hashlib
    from registerrag.corpus.structured import split_sentences, _pack_passages

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions, doc_map = [], {}
    for item in data:
        gold = []
        for p in item.get("paragraphs", []):
            text = " ".join(p.get("paragraph_text", "").split())
            title = p.get("title", "")
            h = hashlib.sha1(f"{title}|{text}".encode("utf-8")).hexdigest()[:8]
            did = f"{_slugify(title)}_{h}"
            if did not in doc_map:
                sents = split_sentences(text)
                doc_map[did] = {"id": did, "title": title,
                                "sentences": (sents if sentence_units else _pack_passages(sents, 80)) or [text]}
            if p.get("is_supporting"):
                gold.append({"doc_ids": [did]})
        hop = re.match(r"(\d)hop", item.get("id", ""))
        questions.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "answer_aliases": list(item.get("answer_aliases", []) or []),
            "type": f"{hop.group(1)}hop" if hop else "",
            "decomposition": item.get("question_decomposition", []),
            "supporting_facts": gold,
        })

    documents = list(doc_map.values())
    logger.info("Loaded %d MuSiQue questions, %d unique paragraphs from %s",
                len(questions), len(documents), path)
    return questions, documents


def load_2wiki(path: str) -> tuple[list[dict], list[dict]]:
    """
    Load a 2WikiMultiHopQA JSON file.

    Format is HotpotQA-compatible: same `context` and `supporting_facts`
    structure.  Extra fields (`type`, `evidences`) are preserved in the
    question dict for downstream inspection but not used by the pipeline.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    doc_map: dict[str, dict] = {}

    for item in data:
        qid = item.get("id") or item.get("_id", "")
        questions.append({
            "id": qid,
            "question": item["question"],
            "answer": item["answer"],
            "supporting_facts": item.get("supporting_facts", []),
            "type": item.get("type", ""),          # compositional / bridge_comparison / inference
            "evidences": item.get("evidences", []), # Wikidata triples — not used for retrieval
        })
        for title, sents in item.get("context", []):
            if title not in doc_map:
                doc_map[title] = {
                    "id": _slugify(title),
                    "title": title,
                    "sentences": sents,
                }

    documents = list(doc_map.values())
    logger.info("Loaded %d questions, %d unique documents from 2Wiki: %s",
                len(questions), len(documents), path)
    return questions, documents


def load_generic_factual(path: str) -> tuple[list[dict], list[dict]]:
    """
    Load a generic factual JSON.
    Flexible: works with the HotpotQA schema too.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    doc_map: dict[str, dict] = {}

    for item in data:
        qid = item.get("id") or item.get("_id", "")
        questions.append({
            "id": qid,
            "question": item["question"],
            "answer": item["answer"],
            "supporting_facts": item.get("supporting_facts", []),
        })
        # support both schemas
        contexts = item.get("contexts") or []
        if not contexts and "context" in item:
            # HotpotQA style
            for title, sents in item["context"]:
                if title not in doc_map:
                    doc_map[title] = {
                        "id": _slugify(title),
                        "title": title,
                        "sentences": sents,
                    }
        else:
            for ctx in contexts:
                title = ctx.get("title", ctx.get("id", ""))
                if title not in doc_map:
                    doc_map[title] = {
                        "id": ctx.get("id", _slugify(title)),
                        "title": title,
                        "sentences": ctx.get("sentences", [ctx.get("text", "")]),
                    }

    documents = list(doc_map.values())
    logger.info("Loaded %d questions, %d documents from %s",
                len(questions), len(documents), path)
    return questions, documents


def default_corpus_dir(qa_path: str) -> str:
    """RePolicyQA convention: the corpus folder `final/` sits next to the QA file."""
    return str(Path(qa_path).resolve().parent / "final")


def load_repolicyqa(path: str, corpus_dir: Optional[str] = None) -> tuple[list[dict], list[dict]]:
    """
    Load RePolicyQA / DSIRE: a QA JSON array plus a folder of Markdown files.

    QA records: {document_id, question, answer, context, conditions,
                 question_type, program_category, chunk}
    Corpus:     <program_id>_meta.md            program metadata + summary
                <program_id>.md, <program_id>_<k>.md   program content

    Documents
      - one per meta file:            id "p<pid>_meta"
      - one per content file:         id "p<pid>_c<k>"
      - byte-identical content shared by several programs (e.g. the same
        federal statute) is indexed once:  id "shared_<hash>"
      Titles are "<program name> (<state>)" so programs that share a name
      across states stay distinguishable; a shared file is titled by its
      first heading.  Each document is {"id","title","markdown","metadata"}.
      metadata["scopes"] names the program(s) a document belongs to (a shared
      file belongs to every program that attaches it); see write_scopes().

    Questions get id "<document_id>_<n>" and document-level gold:
      supporting_facts = [{"doc_ids": [all documents of that program]}]
    """
    import hashlib
    from registerrag.corpus.structured import parse_meta_fields, normalize_answer_text

    corpus_dir = corpus_dir or default_corpus_dir(path)
    files = sorted(Path(corpus_dir).glob("*.md"))
    by_prog: dict[str, dict[str, list[Path]]] = {}
    for f in files:
        stem = f.stem.strip()
        pid = stem.split("_")[0].strip()
        kind = "meta" if stem.endswith("_meta") else "content"
        by_prog.setdefault(pid, {"meta": [], "content": []})[kind].append(f)

    # program titles from meta files
    prog_title: dict[str, str] = {}
    prog_meta: dict[str, dict] = {}
    for pid, g in by_prog.items():
        title, fields = "", {}
        for m in g["meta"]:
            title, fields = parse_meta_fields(m.read_text(encoding="utf-8"))
        if not title and g["content"]:   # no meta file: the first heading of the first content file
            first = next((l.strip("# *").strip() for l in g["content"][0].read_text(encoding="utf-8").splitlines()
                          if l.strip()), "")
            title = first[:120]
        state = fields.get("state", "").strip()
        title = title or f"Program {pid}"
        prog_title[pid] = f"{title} ({state})" if state and state.lower() not in title.lower() else title
        prog_meta[pid] = fields

    # content de-duplication by normalised hash
    def _h(text: str) -> str:
        return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:12]

    content_hash: dict[Path, str] = {}
    hash_progs: dict[str, set[str]] = {}
    for pid, g in by_prog.items():
        for f in g["content"]:
            h = _h(f.read_text(encoding="utf-8"))
            content_hash[f] = h
            hash_progs.setdefault(h, set()).add(pid)

    documents: list[dict] = []
    prog_docs: dict[str, list[str]] = {pid: [] for pid in by_prog}
    emitted_shared: set[str] = set()
    for pid, g in sorted(by_prog.items()):
        for m in g["meta"]:
            did = f"p{pid}_meta"
            documents.append({"id": did, "title": prog_title[pid],
                              "markdown": m.read_text(encoding="utf-8"),
                              "metadata": {"program_id": pid, "file": m.name, **prog_meta[pid],
                                           "scopes": [prog_title[pid]]}})
            prog_docs[pid].append(did)
        for k, f in enumerate(g["content"]):
            h = content_hash[f]
            text = f.read_text(encoding="utf-8")
            if len(hash_progs[h]) > 1:
                did = f"shared_{h}"
                if h not in emitted_shared:
                    emitted_shared.add(h)
                    first = next((l.strip("# ").strip() for l in text.splitlines() if l.strip()), did)
                    documents.append({"id": did, "title": first[:120], "markdown": text,
                                      "metadata": {"programs": sorted(hash_progs[h]), "file": f.name,
                                                   "scopes": sorted({prog_title[p] for p in hash_progs[h]})}})
            else:
                did = f"p{pid}_c{k}"
                documents.append({"id": did, "title": prog_title[pid], "markdown": text,
                                  "metadata": {"program_id": pid, "file": f.name,
                                               "scopes": [prog_title[pid]]}})
            prog_docs[pid].append(did)

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    questions, seen = [], {}
    for item in data:
        pid = str(item.get("document_id", "")).strip()
        n = seen[pid] = seen.get(pid, -1) + 1
        questions.append({
            "id": item.get("id") or f"{pid}_{n}",
            "question": normalize_answer_text(item["question"]),
            "answer": normalize_answer_text(item["answer"]),
            "type": item.get("question_type", ""),
            "program_category": item.get("program_category", ""),
            "conditions": item.get("conditions", ""),
            "gold_context": normalize_answer_text(item.get("context", "")),
            "document_id": pid,
            "supporting_facts": [{"doc_ids": prog_docs.get(pid, [])}],
        })

    missing = sum(1 for q in questions if not q["supporting_facts"][0]["doc_ids"])
    logger.info("Loaded %d RePolicyQA questions, %d programs, %d documents (%d shared) from %s%s",
                len(questions), len(by_prog), len(documents), len(emitted_shared), corpus_dir,
                f" — {missing} questions have no corpus documents" if missing else "")
    return questions, documents


def load_policy_docs(directory: str) -> list[dict]:
    """
    Load all Markdown files in `directory`.
    Returns documents: [{"id", "title", "content", "sections": [...]}]
    Sections: [{"heading": str, "level": int, "text": str}]
    """
    docs = []
    for md_path in sorted(Path(directory).glob("**/*.md")):
        try:
            doc = _parse_markdown_doc(md_path)
            docs.append(doc)
        except Exception as exc:
            logger.warning("Skipping %s: %s", md_path, exc)

    logger.info("Loaded %d policy documents from %s", len(docs), directory)
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", text.lower())[:60]


def _parse_markdown_doc(path: Path) -> dict:
    """Split an MD file into sections by heading hierarchy."""
    content = path.read_text(encoding="utf-8")
    title = path.stem.replace("_", " ").replace("-", " ").title()

    sections: list[dict] = []
    current_heading = ""
    current_level = 0
    current_lines: list[str] = []

    for line in content.splitlines():
        heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if heading_match:
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "level": current_level,
                    "text": "\n".join(current_lines).strip(),
                })
            current_level = len(heading_match.group(1))
            current_heading = heading_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "heading": current_heading,
            "level": current_level,
            "text": "\n".join(current_lines).strip(),
        })

    return {
        "id": _slugify(path.stem),
        "title": title,
        "content": content,
        "sections": sections,
        "path": str(path),
    }


def write_scopes(documents: list[dict], index_dir: str) -> str:
    """
    Write <index_dir>/scopes.json: {scope name: [document ids]}.

    A scope is the unit a question is "about" — a Wikipedia article, a policy
    program.  Ingestion decides scope membership (doc metadata "scopes",
    default: the document's own title); the core only matches question text
    against scope names (pipeline/scope.py), the same way for every dataset.
    """
    scopes: dict[str, list[str]] = {}
    for d in documents:
        names = (d.get("metadata") or {}).get("scopes") or [d.get("title", "")]
        for n in names:
            if n:
                scopes.setdefault(n, []).append(d["id"])
    out = os.path.join(index_dir, "scopes.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(scopes, f, ensure_ascii=False, indent=0)
    return out
