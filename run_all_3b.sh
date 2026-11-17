#!/bin/bash
# Sequential 3B baseline runs. Re-run this script after any crash/reboot: finished runs are skipped, partial runs resume per question.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
for m in pyrag evimem georag surerag liveledger hlg lychee; do
  echo "##### $m $(date)" 
  $PY scripts/run_baselines.py --methods $m --generator qwen2.5:3b --attempts 5
done
echo "##### ALL DONE $(date)"
