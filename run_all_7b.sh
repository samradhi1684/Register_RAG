#!/bin/bash
# Sequential 7B baseline runs. Re-run after any crash/reboot: finished runs are skipped, partial runs resume per question.
# Each method is re-invoked up to 40 times: the intermittent native crashes can exhaust one invocation's attempts.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
for m in none chunk sentence iterative pyrag evimem georag surerag liveledger hlg lychee; do
  for round in $(seq 1 40); do
    echo "##### $m round $round $(date)"
    $PY scripts/run_baselines.py --methods $m --generator qwen2.5:7b --attempts 5 && break
  done
done
echo "##### ALL DONE $(date)"
