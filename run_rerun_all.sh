#!/usr/bin/env bash
# Full rerun: ALL baselines (B1-B9, B7b, references) x 2wiki, musique, dsire x qwen2.5:3b and :7b.
# Results go to runs_rerun/ (existing results are not touched). Resumable: rerun the same script after a crash.
# Usage:  bash run_rerun_all.sh            (keeps cached indexes)
#         FRESH_INDEX=1 bash run_rerun_all.sh   (deletes cached HLG/EviMem/Lychee indexes first = rebuilt from scratch)
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
OUT=runs_rerun
mkdir -p logs

if [ "$FRESH_INDEX" = "1" ]; then
  for d in b4_hlg_rag b6_evimem b7_lycheememory; do
    [ -d "baselines/$d/index" ] && mv "baselines/$d/index" "baselines/$d/index.bak_$(date +%Y%m%d_%H%M%S)"
  done
fi

# Part 1: everything that runs through Ollama (needs the main venv + Ollama)
for g in qwen2.5:3b qwen2.5:7b; do
  $PY scripts/run_baselines.py --datasets 2wiki,musique,dsire --generator $g --attempts 5 \
    --methods none,chunk,sentence,iterative,pyrag,evimem,georag,surerag,liveledger,hlg,lychee \
    --out $OUT
done >> logs/rerun_all.log 2>&1

# Part 2: B7b (own 7B checkpoint, ignores --generator, needs the GPU free of Ollama models)
ollama stop qwen2.5:3b 2>/dev/null; ollama stop qwen2.5:7b 2>/dev/null
$PY scripts/run_baselines.py --datasets 2wiki,musique,dsire --methods lycheeacl --out $OUT \
  >> logs/rerun_b7b.log 2>&1
