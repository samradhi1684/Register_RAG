"""
Build one corpus file per Wikipedia dataset from the official dev split, in the
layout the existing loaders read (registerrag/data_loaders.py).

  2Wiki   : data/2wikimultihopqa/data/dev.json                  -> data/2wikimultihopqa/corpus_dev.json
  HotpotQA: data/hotpotqa/distractor/validation-*.parquet       -> data/hotpotqa/corpus_dev.json
  MuSiQue : data/musique/data/musique_ans_v1.0_dev.jsonl        -> data/musique/corpus_dev.json
            (answerable version, the one the 500 come from; the "full" version samples
             different distractors and adds unanswerable twins, so it is not used)

The loaders keep one text per title (first one seen). Some titles occur with
slightly different texts in dev, so the 500 benchmark questions (qa-pairs/) are
written first: the corpus then holds exactly the passages those questions came with.
Only the contexts are used for indexing; questions in these files are not evaluated.

Usage (from registerrag_clean/):  python tools/prepare_corpus.py
"""
import ast
import json
import os

import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "registerrag")
DATA, QA = os.path.join(ROOT, "data"), os.path.join(ROOT, "qa-pairs")


def _p(x):
    """qa-pairs files store nested fields as Python-literal strings."""
    return ast.literal_eval(x) if isinstance(x, str) else x


def _load(path):
    return json.load(open(path, encoding="utf-8"))


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"{path}: {len(rows)} records")


def wiki2():
    mine = [dict(r, context=_p(r["context"]), supporting_facts=_p(r["supporting_facts"]))
            for r in _load(os.path.join(QA, "2wiki_500.json"))]
    ids = {r["id"] for r in mine}
    dev = [r for r in _load(os.path.join(DATA, "2wikimultihopqa", "data", "dev.json")) if r["_id"] not in ids]
    _write(os.path.join(DATA, "2wikimultihopqa", "corpus_dev.json"), mine + dev)


def hotpot():
    mine = [dict(r, context=_p(r["context"]), supporting_facts=_p(r["supporting_facts"]))
            for r in _load(os.path.join(QA, "hotpot_500.json"))]
    ids = {r["id"] for r in mine}
    dev = [r for r in pq.read_table(os.path.join(DATA, "hotpotqa", "distractor",
                                                 "validation-00000-of-00001.parquet")).to_pylist()
           if r["id"] not in ids]
    _write(os.path.join(DATA, "hotpotqa", "corpus_dev.json"), mine + dev)


def musique():
    mine = [dict(r, paragraphs=_p(r["paragraphs"]), question_decomposition=_p(r["question_decomposition"]),
                 answer_aliases=_p(r["answer_aliases"]))
            for r in _load(os.path.join(QA, "musique_500.json"))]
    ids = {r["id"] for r in mine}
    dev = [json.loads(l) for l in open(os.path.join(DATA, "musique", "data", "musique_ans_v1.0_dev.jsonl"),
                                       encoding="utf-8")]
    _write(os.path.join(DATA, "musique", "corpus_dev.json"), mine + [r for r in dev if r["id"] not in ids])


if __name__ == "__main__":
    wiki2()
    hotpot()
    musique()
