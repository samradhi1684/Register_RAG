"""
DSIRE corpus of only the source passages of the 500 questions: one Markdown
document per program (document_id), holding that program's distinct question
chunks (the text each question was written from), nothing else.
  python tools/prepare_dsire_92.py  ->  registerrag/data/final_92docs/<pid>.md
The program's metadata chunk (if one of its chunks) is placed first so the
document opens with the program name.
"""
import json, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
qa = json.load(open(os.path.join(ROOT, "registerrag/qa-pairs/repolicyqa_500.json"), encoding="utf-8"))
out = os.path.join(ROOT, "registerrag/data/final_92docs")
os.makedirs(out, exist_ok=True)
chunks = {}
for q in qa:
    pid = str(q["document_id"]).strip()
    lst = chunks.setdefault(pid, [])
    norm = " ".join(q["chunk"].split())
    if all(" ".join(c.split()) != norm for c in lst):
        lst.append(q["chunk"])
is_meta = lambda c: bool(re.search(r"\*\*\*?\s*program category", c, re.I))
for pid, lst in chunks.items():
    lst.sort(key=lambda c: not is_meta(c))
    open(os.path.join(out, f"{pid}.md"), "w", encoding="utf-8").write("\n\n".join(c.strip() for c in lst) + "\n")
print(len(chunks), "documents,", sum(map(len, chunks.values())), "chunks ->", out)
