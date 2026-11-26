#!/usr/bin/env bash
# Full rerun, self-contained: no gitignored file is needed (no logs/ folder, no cached index, no old results).
# ALL baselines (B1-B9, references, B7b) x 2wiki, musique, dsire x qwen2.5:3b and :7b. n=100.
# Output: runs_rerun/ (results + logs). Existing results are not touched. Resumable: rerun the same script after a crash.
# Indexes (B4 graph, B6 tuples, B7 memory) are always rebuilt: any cached ones are moved to index.bak_<time>.
cd "$(dirname "$0")"
# Python: $PY if set, else the active venv, else ../.venv (Windows or Linux layout), else python3
if [ -z "$PY" ]; then
  if [ -n "$VIRTUAL_ENV" ]; then PY="$VIRTUAL_ENV/bin/python"; [ -x "$PY" ] || PY="$VIRTUAL_ENV/Scripts/python.exe"
  elif [ -x ../.venv/bin/python ]; then PY=../.venv/bin/python
  elif [ -x ../.venv/Scripts/python.exe ]; then PY=../.venv/Scripts/python.exe
  else PY=$(command -v python3 || command -v python); fi
fi
echo "using python: $PY"; "$PY" -c "import registerrag, sys" || { echo "python cannot import registerrag / deps (pip install -r requirements.txt)"; exit 1; }
OUT=runs_rerun
DATASETS=${DATASETS:-2wiki,musique,dsire}   # e.g. DATASETS=hotpot to run only HotpotQA (n=100)
B7B_ENV=""; [ "${B7B_CPU:-0}" = "1" ] && B7B_ENV='env CUDA_VISIBLE_DEVICES='   # B7b on GPU by default; B7B_CPU=1 forces CPU
export PYTHONPATH="$PWD" PYTHONUTF8=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT/logs"

# fresh indexes (only on the first start; a marker keeps resumes from discarding a half-built index)
if [ ! -f "$OUT/.indexes_reset" ]; then
  for d in b4_hlg_rag b6_evimem b7_lycheememory; do
    [ -d "baselines/$d/index" ] && mv "baselines/$d/index" "baselines/$d/index.bak_$(date +%Y%m%d_%H%M%S)"
  done
  date > "$OUT/.indexes_reset"
fi

# Part 1: all Ollama-based methods
for g in qwen2.5:3b qwen2.5:7b; do
  for attempt in $(seq 1 30); do
    echo "##### part1 $g attempt $attempt $(date)"
    $PY scripts/run_baselines.py --datasets $DATASETS --generator $g --attempts 5 \
      --methods none,chunk,sentence,iterative,pyrag,evimem,georag,surerag,liveledger,hlg,lychee \
      --out $OUT && break
    echo "part1 $g attempt $attempt failed; finished work is kept, retrying"; sleep 10
  done
done >> "$OUT/logs/rerun_all.log" 2>&1

# Part 2: B7b (own 7B checkpoint; GPU must be free of Ollama models; harness side on CPU)
ollama stop qwen2.5:3b >/dev/null 2>&1; ollama stop qwen2.5:7b >/dev/null 2>&1
ollama stop qwen2.5-coder:3b >/dev/null 2>&1; ollama stop qwen2.5-coder:7b >/dev/null 2>&1
for attempt in $(seq 1 30); do
  echo "##### b7b attempt $attempt $(date)"
  $B7B_ENV $PY scripts/run_baselines.py --datasets $DATASETS --methods lycheeacl \
    --out $OUT && break
  echo "b7b attempt $attempt failed; finished questions are kept, retrying"; sleep 10
done >> "$OUT/logs/rerun_b7b.log" 2>&1
echo "##### ALL DONE $(date)" >> "$OUT/logs/rerun_all.log"
