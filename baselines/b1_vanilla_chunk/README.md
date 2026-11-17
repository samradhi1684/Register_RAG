# B1 - vanilla RAG, chunk units

Question -> hybrid retrieval (BM25 + BGE-small, lambda 0.5) over 256-word chunks (50-word overlap) -> MiniLM cross-encoder rerank -> top-5 chunks
in the prompt -> generator. The standard deployed setup; every other row must beat it. No paper to reproduce: it is the control.

- **Code:** shared `registerrag.pipelines.OneShotRAG` (identical to B2; only the retrieval unit differs).
- **Frozen settings:** title prepended, forced-answer prompt v2, top-k 5, no token budget. Budget sweeps use `--top-k 20 --budget 512|1024|2048|4096`.

## Run it
```powershell
..\.venv\Scripts\Activate.ps1
python scripts/run_baselines.py --methods chunk                          # 3B, 2wiki + musique, 100 questions each
python scripts/run_baselines.py --methods chunk --generator qwen2.5:7b   # 7B
python scripts/run_baselines.py --methods chunk --openai                 # + GPT-4o-mini reference row (needs $env:OPENAI_API_KEY)
python scripts/run_baselines.py --methods chunk --rep 1                  # stability repetition -> results/stability/rep1 (then 2, 3)
python scripts/stability.py --baseline b1_vanilla_chunk
```
Results: `results/<system>_<dataset>_<config hash>/{results.jsonl, summary.json, config.json}` (config, seed, data SHA-256, model, prompt version, latency, tokens, per-question evidence).
