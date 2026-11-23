#!/usr/bin/env bash
# Linux/macOS version of setup_env.ps1: isolated venv ../.venv_lychee, pinned authors' code + weights, vendor patch.
# Run from anywhere with the MAIN venv active (its torch version/CUDA build is copied).  Idempotent.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"; B="$(dirname "$HERE")"; REPO="$(dirname "$(dirname "$B")")"; PARENT="$(dirname "$REPO")"
VENV="$PARENT/.venv_lychee"; PY="$VENV/bin/python"; VENDOR="$B/vendor"
GIT_SHA=e4b93d8fa09c1258a14c708d75551d2bca58c446; HF_SHA=17bb642d067328d6c54b6f4e12bf70528c8c5c22
[ -x "$PY" ] || python3 -m venv "$VENV"
TV=$(python -c "import torch;print(torch.__version__.split('+')[0])"); CU=$(python -c "import torch;print((torch.version.cuda or '').replace('.',''))")
echo "torch $TV (cu$CU) as in the main venv"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install "torch==$TV" --index-url "https://download.pytorch.org/whl/cu$CU"
"$PY" -m pip install "transformers==4.53.2" numpy huggingface_hub
"$PY" -m pip install --no-deps tensordict
"$PY" -m pip install openai cloudpickle packaging orjson importlib_metadata "pyvers>=0.2,<0.3"
[ -d "$VENDOR/src" ] || git clone https://github.com/owoakuma/LycheeMemory.git "$VENDOR"
git -C "$VENDOR" checkout "$GIT_SHA"
if git -C "$VENDOR" apply --check "$B/vendor_patches/sdpa_position_ids.patch" 2>/dev/null; then git -C "$VENDOR" apply "$B/vendor_patches/sdpa_position_ids.patch"; fi
"$PY" -c "from huggingface_hub import snapshot_download as s; print(s('lerverson/LycheeMemory-7B', revision='$HF_SHA'))"
cd "$VENDOR" && "$PY" -c "import torch,transformers;from src.modeling_qwen2 import Qwen2ForCausalLM;print('ok: torch',torch.__version__,'cuda',torch.cuda.is_available())"
echo "B7b environment ready."
