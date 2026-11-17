#!/bin/bash
# Waits for run_all_3b.sh to finish, frees the GPU, then runs the 3-question B7b smoke test. Safe to re-run.
cd "$(dirname "$0")/../../.."
until grep -q "ALL DONE" logs/run_all_3b.log; do sleep 60; done
ollama stop qwen2.5:3b >/dev/null 2>&1; ollama stop qwen2.5:7b >/dev/null 2>&1
PYTHONUNBUFFERED=1 ../.venv/Scripts/python.exe scripts/run_baselines.py --methods lycheeacl --limit 3 --out runs_smoke_b7b --no-prebuild
echo "##### B7b SMOKE DONE $(date)"
