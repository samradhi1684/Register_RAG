"""Pivot runs_progressive/results_by_qtype.csv into an xlsx: one sheet per dataset,
one row per run, and for each metric one column per question type (ALL first).
Needs openpyxl. Run scripts/qtype_csv.py first to refresh the CSV."""
import argparse, csv, os
from collections import OrderedDict
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="runs_progressive/results_by_qtype.csv")
ap.add_argument("--out", default="runs_progressive/results_by_qtype.xlsx")
a = ap.parse_args()

ID = ["run_dir", "status", "generator", "system", "label", "variant"]
rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
METRICS = [c for c in rows[0] if c not in ID + ["dataset", "qtype"]]
COUNTS = {"generation_failures", "reranking_failures", "retrieval_failures", "no_retrieval",
          "system_errors", "correct", "n", "peak_prompt_tokens_max"}
FONT, BOLD = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
HFILL, SFILL = PatternFill("solid", fgColor="DDE4EE"), PatternFill("solid", fgColor="F2F2F2")
EDGE = Border(left=Side(style="thin", color="999999"))

wb = Workbook()
wb.remove(wb.active)
for ds in sorted({r["dataset"] for r in rows}):
    sub = [r for r in rows if r["dataset"] == ds]
    qts = ["ALL"] + sorted({r["qtype"] for r in sub} - {"ALL"})
    runs = OrderedDict()
    for r in sub:  # CSV is already sorted by generator, then system order
        runs.setdefault(r["run_dir"], {"id": r, "q": {}})["q"][r["qtype"]] = r
    runs = OrderedDict(sorted(runs.items(), key=lambda kv: kv[1]["id"]["generator"]))
    ws = wb.create_sheet(ds)
    for j, c in enumerate(ID, 1):
        ws.cell(1, j, c).font = BOLD
        ws.merge_cells(start_row=1, start_column=j, end_row=2, end_column=j)
    col = len(ID) + 1
    for m in METRICS:
        ws.cell(1, col, m).font = BOLD
        ws.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col + len(qts) - 1)
        for k, q in enumerate(qts):
            c = ws.cell(2, col + k, q)
            c.font = BOLD
            if k == 0:
                c.border = ws.cell(1, col).border = EDGE
        col += len(qts)
    for rr in (1, 2):
        for cc in range(1, col):
            cell = ws.cell(rr, cc)
            cell.fill, cell.alignment = HFILL, Alignment(horizontal="center", vertical="center", wrap_text=True)
    n_qt = {q: next((v["q"][q]["n"] for v in runs.values() if q in v["q"]), "") for q in qts}
    for i, (rd, v) in enumerate(runs.items(), 3):
        for j, c in enumerate(ID, 1):
            ws.cell(i, j, v["id"][c]).font = FONT
        col = len(ID) + 1
        for m in METRICS:
            for k, q in enumerate(qts):
                raw = v["q"].get(q, {}).get(m, "")
                cell = ws.cell(i, col + k)
                if raw != "":
                    cell.value = float(raw) if m not in COUNTS else int(float(raw))
                    cell.number_format = "0" if m in COUNTS else "0.000"
                cell.font = FONT
                if k == 0:
                    cell.border, cell.fill = EDGE, SFILL
            col += len(qts)
    note = len(runs) + 4
    ws.cell(note, 1, f"Question types and sizes (from a complete run): "
            + "; ".join(f"{q} n={n_qt[q]}" for q in qts)).font = Font(name="Arial", size=9, italic=True)
    ws.cell(note + 1, 1, "Metrics computed per question-type subset with registerrag.runner.summarize. "
            "status=incomplete rows cover only the questions answered so far."
            + (" MuSiQue qtype = qid prefix (official hop-pattern type)." if ds == "musique" else "")
            ).font = Font(name="Arial", size=9, italic=True)
    ws.freeze_panes = ws.cell(3, len(ID) + 1)
    for j, w in enumerate([24, 10, 11, 10, 34, 30], 1):
        ws.column_dimensions[get_column_letter(j)].width = w
    for j in range(len(ID) + 1, col):
        ws.column_dimensions[get_column_letter(j)].width = 11
    ws.row_dimensions[2].height = 30
wb.save(a.out)
print("wrote", a.out, [ws.title for ws in wb])
