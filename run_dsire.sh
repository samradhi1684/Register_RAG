#!/bin/bash
# All baselines on the DSIRE (RePolicyQA) dataset. Re-run after any crash: finished runs skip, partial runs resume.
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
for g in qwen2.5:3b qwen2.5:7b; do
  echo "##### $g $(date)"
  $PY scripts/run_baselines.py --datasets dsire --generator $g --attempts 5 \
      --methods none,chunk,sentence,iterative,pyrag,evimem,georag,surerag,liveledger,hlg,lychee
done
echo "##### ALL DONE $(date)"
