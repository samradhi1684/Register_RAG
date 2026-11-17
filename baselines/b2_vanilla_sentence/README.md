# B2 - vanilla RAG, sentence units

Identical to B1 except the retrieval unit is a single sentence (native sentence boundaries for 2Wiki/HotpotQA, pysbd for MuSiQue).
This is the baseline every RegisterRAG component is measured against: it separates the effect of a finer unit from everything the method adds.

- **Code:** shared `registerrag.pipelines.OneShotRAG`.

## Run it
```powershell
..\.venv\Scripts\Activate.ps1
python scripts/run_baselines.py --methods sentence
python scripts/run_baselines.py --methods sentence --generator qwen2.5:7b
python scripts/run_baselines.py --methods sentence --rep 1               # stability repetition
```
Results: `results/`.
