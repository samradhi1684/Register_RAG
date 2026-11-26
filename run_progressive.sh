#!/usr/bin/env bash
# Progressive study: all baselines x 2wiki, musique, dsire on the 500-question sets; report for N = 100, 200, 300, 400, 500.
# One full run per configuration; the report takes the first N questions of each run (nested sets). Output: runs_progressive/
# Resumable (rerun after any crash). The report can be generated at any time, also while the run is going:
#   python scripts/progressive_report.py runs_progressive
# Settings (env):  GENS="qwen2.5:3b qwen2.5:7b"   METHODS=...   WITH_B7B=1 (0 skips B7b)   DATASETS=2wiki,musique,dsire   B7B_CPU=1 (force B7b onto CPU; default GPU)
cd "$(dirname "$0")"
if [ -z "$PY" ]; then
  if [ -n "$VIRTUAL_ENV" ]; then PY="$VIRTUAL_ENV/bin/python"; [ -x "$PY" ] || PY="$VIRTUAL_ENV/Scripts/python.exe"
  elif [ -x ../.venv/bin/python ]; then PY=../.venv/bin/python
  elif [ -x ../.venv/Scripts/python.exe ]; then PY=../.venv/Scripts/python.exe
  else PY=$(command -v python3 || command -v python); fi
fi
echo "using python: $PY"; "$PY" -c "import registerrag" || { echo "cannot import registerrag (pip install -r requirements.txt)"; exit 1; }
OUT=runs_progressive
B7B_ENV=""; [ "${B7B_CPU:-0}" = "1" ] && B7B_ENV='env CUDA_VISIBLE_DEVICES='
GENS=${GENS:-"qwen2.5:3b qwen2.5:7b"}
DATASETS=${DATASETS:-2wiki,musique,dsire}   # e.g. DATASETS=hotpot for the HotpotQA study only
METHODS=${METHODS:-none,chunk,sentence,iterative,pyrag,evimem,georag,surerag,liveledger,hlg,lychee}
export PYTHONPATH="$PWD" PYTHONUTF8=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT/logs"

$PY scripts/make_progressive_sets.py --total 500 || exit 1

for g in $GENS; do
  for attempt in $(seq 1 30); do
    echo "##### $g attempt $attempt $(date)"
    $PY scripts/run_baselines.py --suffix prog500 --datasets $DATASETS --generator $g --attempts 5 \
      --methods $METHODS --out $OUT && break
    echo "$g attempt $attempt failed; finished questions are kept, retrying"; sleep 10
  done
  $PY scripts/progressive_report.py $OUT >/dev/null 2>&1
done >> "$OUT/logs/progressive.log" 2>&1

if [ "${WITH_B7B:-1}" = "1" ]; then
  for m in qwen2.5:3b qwen2.5:7b qwen2.5-coder:3b qwen2.5-coder:7b; do ollama stop $m >/dev/null 2>&1; done
  for attempt in $(seq 1 30); do
    echo "##### b7b attempt $attempt $(date)"
    $B7B_ENV $PY scripts/run_baselines.py --suffix prog500 --datasets $DATASETS --methods lycheeacl --out $OUT && break
    echo "b7b attempt $attempt failed; retrying"; sleep 10
  done >> "$OUT/logs/progressive_b7b.log" 2>&1
fi
$PY scripts/progressive_report.py $OUT >> "$OUT/logs/progressive.log" 2>&1
echo "##### ALL DONE $(date)" >> "$OUT/logs/progressive.log"
