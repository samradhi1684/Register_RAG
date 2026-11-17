# B7b - LycheeMemory (ACL 2026): the authors' released 7B checkpoint, inference only

- **Paper:** "Dynamic Long Context Reasoning over Compressed Memory via End-to-End Reinforcement Learning", Chen et al., ACL 2026 (long, pp. 8064-8083), [arXiv 2602.08382](https://arxiv.org/abs/2602.08382). This is a **different paper** from B7 (LycheeMemory V2, arXiv 2608.12990).
- **Official code:** github.com/owoakuma/LycheeMemory (pinned commit `e4b93d8f`); **weights:** `lerverson/LycheeMemory-7B` (pinned revision `17bb642d`, Apache-2.0, ~16 GB bf16).
- **Status: reference row, NOT a reproduction of the paper's results.** The repo README says only the *inference code without the Gate* and the weights are released; Compressor pre-training, RL optimisation and the Gate are "released soon". There is **no evaluation script, no prompt file and no RULER-HQA script** in the repo. Read this before citing a B7b number.
- **Code here:** `lychee_acl.py` (pipeline, in the main venv), `worker.py` (model, in an isolated venv), `tools/setup_env.ps1`. Tests: `python -m pytest baselines/b7b_lychee_acl -q` (fake worker, no GPU).

## What it is, and how it differs from the other rows
Every other row retrieves top-5 units and prompts the frozen qwen2.5. B7b does **neither**: the released model compresses the question's whole context into KV-cache memory (ratio 4, window/stride 2048, stream chunk 4096) and generates the answer itself. So it is a *long-context-model* reference, not a RAG method:
- it receives **all of the question's own paragraphs** (`ex.doc_ids`, the same pool that per-question scope would retrieve from) as `Document i:\n<title>\n<text>` blocks - the layout of the authors' `test.json`;
- it ignores `--generator`; the 7B checkpoint (Qwen2-based, own weights) is the generator, so **do not compare it with the 3B rows as "the same model"**;
- retrieval metrics (recall/precision, failure categories that depend on retrieval) are **not applicable** (empty by construction);
- `peak_prompt_tokens` is the *full context length*, not what the transformer attends to after compression.

## Fidelity table
| Item | Authors | Here |
|---|---|---|
| Compressor + Reasoner | released checkpoint | the same checkpoint, authors' `src/` code |
| Gate (LoRA recall filter) | in the paper; **not released** | not used (the released inference path has none) |
| Recurrent working memory with RL-trained reasoning | paper | whatever the released weights do in `memory_mode="recurrent"`; unverified against the paper |
| Attention backend | `flash_attention_2` (recommended) | `sdpa` by default: flash-attn does not build on Windows. Numerics may differ slightly; `--method-arg attn=flash_attention_2` if you have it |
| Decoding | `example.py` defaults to top-k 50 sampling only with `--do_sample` | greedy (`do_sample=False`), `max_new_tokens=1024` |
| Prompt | none: raw context + raw question, no chat template (as in `example.py`) | identical. Whether the RL-trained model expects a template is **unknown** |
| Answer format | not documented | `extract_answer`: last `\boxed{..}`, else text after a final `Answer:`, else the output's first line; the **raw output is always stored** (`extra.raw_output`) so extraction can be revised without re-running |
| Benchmarks | RULER-HQA etc. (data/scripts not released) | our 2Wiki / MuSiQue 100-question sets (short contexts, ~10-20 paragraphs, far below the paper's regime) |

## Vendor patch and measured behaviour (read this)
- **Vendor patch (`vendor_patches/sdpa_position_ids.patch`, applied by `setup_env.ps1`).** The authors' `generate` loop overwrites `position_ids` with `cumsum(attention_mask) - 1`, valid only for the 2D mask of the flash-attn path. With `sdpa` the memory module returns a 4D mask *and its own correct position ids*, and the override crashed (`size of tensor a (28) must match b (2560)`). The patch applies the override only when the mask is 2D. Without flash-attn (not buildable on Windows) the released code cannot run unpatched. Equivalence with the flash path is **unverified**; smoke outputs were correct ("Paris", "the Seine").
- **Speed (measured, RTX 5080 16 GB):** the bf16 7B is ~15 GB, so the card is full and Windows spills to shared memory: 33-token context 139 s, 5,041-token context 327-707 s per question. A 100-question run is therefore many hours.
- Extra packages the authors' `src/utils.py` imports at module level: `tensordict` (installed `--no-deps`, so torch is never replaced), `openai`, `pyvers`.

## Set up (once)
Needs ~16 GB free disk for the weights and the GPU free of other jobs (the 7B in bf16 is ~15 GB; the RTX 5080 has 16 GB, so **stop Ollama models first**: `ollama stop qwen2.5:3b`).
```powershell
powershell -ExecutionPolicy Bypass -File baselines\b7b_lychee_acl\tools\setup_env.ps1
```
Creates `..\.venv_lychee` (isolated: `transformers==4.53.2`, same torch build as the main venv), clones the pinned code to `baselines/b7b_lychee_acl/vendor/`, downloads the pinned weights and checks that the authors' modelling code imports. The main venv is not modified.

## Run it
```powershell
..\.venv\Scripts\Activate.ps1
python -m pytest baselines/b7b_lychee_acl -q
python scripts/run_baselines.py --methods lycheeacl --limit 3 --out runs_smoke    # smoke test first; READ extra.raw_output
python scripts/run_baselines.py --methods lycheeacl                                # full: 2wiki + musique, resumable per question
```
Knobs (`--method-arg`): `attn` (sdpa), `ratio` 4, `window` 2048, `stride` 2048, `stream_chunk_size` 4096, `max_new_tokens` 1024, `timeout_s` 1500, `revision`, `model`, `python`, `vendor`.
A crashed or hung worker is restarted once per question; an out-of-memory error is logged as that question's error and the run continues (failed questions are retried on resume). Each answer is fsynced before the next question.

## Reading the result
Per question: `prediction` (extracted), `extra.raw_output` (full generation), `text_tokens`, `worker_seconds`, `worker_peak_mem_mb`. First check that the raw outputs are short answers; if the model writes long reasoning, adjust `extract_answer` and re-score from `raw_output` rather than re-running.
