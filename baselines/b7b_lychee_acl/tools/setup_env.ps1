# One-time setup for B7b. Creates an ISOLATED venv (.venv_lychee next to .venv) so transformers==4.53.2 never touches the main stack.
# Run from the repo folder:   powershell -ExecutionPolicy Bypass -File baselines\b7b_lychee_acl\tools\setup_env.ps1
# Idempotent: safe to re-run after an interruption.
$ErrorActionPreference = "Stop"
$here   = Split-Path -Parent $MyInvocation.MyCommand.Path
$b      = Split-Path -Parent $here
$repo   = Split-Path -Parent (Split-Path -Parent $b)
$parent = Split-Path -Parent $repo
$main   = Join-Path $parent ".venv\Scripts\python.exe"
$venv   = Join-Path $parent ".venv_lychee"
$py     = Join-Path $venv "Scripts\python.exe"
$vendor = Join-Path $b "vendor"
$GIT_SHA = "e4b93d8fa09c1258a14c708d75551d2bca58c446"      # github.com/owoakuma/LycheeMemory (pinned)
$HF_SHA  = "17bb642d067328d6c54b6f4e12bf70528c8c5c22"      # huggingface.co/lerverson/LycheeMemory-7B (pinned)

if (-not (Test-Path $py)) { & $main -m venv $venv }
$torchVer = (& $main -c "import torch;print(torch.__version__.split('+')[0])").Trim()
$cuda     = (& $main -c "import torch;print((torch.version.cuda or '').replace('.',''))").Trim()
Write-Host "torch $torchVer (cu$cuda) as in the main venv"
& $py -m pip install --upgrade pip
& $py -m pip install "torch==$torchVer" --index-url "https://download.pytorch.org/whl/cu$cuda"
& $py -m pip install "transformers==4.53.2" numpy huggingface_hub
# the authors' src/utils.py imports tensordict + openai at module level; --no-deps so pip never replaces torch
& $py -m pip install --no-deps tensordict
& $py -m pip install openai cloudpickle packaging orjson importlib_metadata "pyvers>=0.2,<0.3"

if (-not (Test-Path (Join-Path $vendor "src"))) { git clone https://github.com/owoakuma/LycheeMemory.git $vendor }
git -C $vendor checkout $GIT_SHA
# our one-condition patch (see README 'Vendor patch'); skipped if already applied
git -C $vendor apply --check (Join-Path $b 'vendor_patches\sdpa_position_ids.patch') 2>$null
if ($LASTEXITCODE -eq 0) { git -C $vendor apply (Join-Path $b 'vendor_patches\sdpa_position_ids.patch') }
& $py -c "from huggingface_hub import snapshot_download as s; print(s('lerverson/LycheeMemory-7B', revision='$HF_SHA'))"

# import check on CPU (no GPU needed): the authors' modelling code must load under the pinned transformers
Push-Location $vendor
& $py -c "import torch,transformers;from src.modeling_qwen2 import Qwen2ForCausalLM;print('ok: torch',torch.__version__,'transformers',transformers.__version__,'cuda',torch.cuda.is_available())"
Pop-Location
Write-Host "B7b environment ready."
