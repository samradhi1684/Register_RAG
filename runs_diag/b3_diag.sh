#!/bin/bash
cd "$(dirname "$0")/.."
PY=../.venv/Scripts/python.exe
export PYTHONPATH="$PWD" PYTHONUTF8=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for ds in 2wiki musique; do
  data=datasets/$( [ $ds = 2wiki ] && echo wiki2_data/2wiki_100.json || echo musique_data/musique_100.json )
  for attempt in $(seq 1 25); do
    echo "##### b3diag $ds attempt $attempt $(date)"
    $PY -m registerrag.cli run --system georag --dataset $ds --data "$data" --out runs_diag/b3_prior005 \
      --prepend-title --prompt v2 --top-k 5 --generator qwen2.5:3b --limit 40 --method-arg prior_temp=0.05 && break
    sleep 3
  done
done
echo "##### B3DIAG DONE $(date)"
