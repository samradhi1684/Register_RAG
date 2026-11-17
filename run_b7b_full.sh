#!/bin/bash
# B7b full run (2wiki + musique, 100 q each) with the released 7B. Resumable: re-run after any crash/reboot.
# Harness side runs on CPU (CUDA_VISIBLE_DEVICES=""): the 7B alone fills the 16 GB card. Never run while other GPU jobs are active.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
export PYTHONPATH="$PWD" PYTHONUTF8=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
ollama stop qwen2.5:3b >/dev/null 2>&1; ollama stop qwen2.5-coder:3b >/dev/null 2>&1
for ds in 2wiki musique; do
  data=datasets/$( [ $ds = 2wiki ] && echo wiki2_data/2wiki_100.json || echo musique_data/musique_100.json )
  for attempt in $(seq 1 30); do
    echo "##### b7b $ds attempt $attempt $(date)"
    CUDA_VISIBLE_DEVICES="" $PY -m registerrag.cli run --system lycheeacl --dataset $ds --data "$data" \
      --out baselines/b7b_lychee_acl/results --prepend-title --prompt v2 --top-k 5 --generator qwen2.5:3b \
      --method-arg timeout_s=3600 && break
    echo "attempt $attempt failed; finished questions are kept, retrying..."; sleep 5
  done
done
echo "##### B7B DONE $(date)"
