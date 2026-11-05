# Runs every baseline on every dataset with the FROZEN settings (title on, forced-answer prompt v2, top-k 5).
# Finished runs are skipped; an interrupted or crashed run is retried (up to 3 attempts) and RESUMES where it stopped.
# A configuration that still fails does not stop the others; failures are listed at the end.
#   powershell -ExecutionPolicy Bypass -File scripts\run_baselines.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_baselines.ps1 -Datasets musique -Limit 10 -Out runs_smoke
#   powershell -ExecutionPolicy Bypass -File scripts\run_baselines.ps1 -Out runs_stab\rep1
#   powershell -ExecutionPolicy Bypass -File scripts\run_baselines.ps1 -Suffix 500
param(
  [string]$Generator = "qwen2.5:3b",
  [string[]]$Datasets = @("2wiki", "musique"),
  [string]$Out = "runs",
  [string]$Suffix = "100",
  [int]$Limit = 0,
  [int]$Attempts = 3
)
$env:PYTHONPATH = "."
$env:PYTHONUTF8 = "1"
$files = @{
  "2wiki"   = "datasets/wiki2_data/2wiki_$Suffix.json"
  "musique" = "datasets/musique_data/musique_$Suffix.json"
}
$script:failed = @()

function Run-One([string]$label, [string[]]$cliArgs) {
  Write-Host ""
  Write-Host "=== $label ===" -ForegroundColor Cyan
  for ($try = 1; $try -le $Attempts; $try++) {
    python -m registerrag.cli run @cliArgs
    if ($LASTEXITCODE -eq 0) { return }
    Write-Host "attempt $try of $Attempts failed (exit code $LASTEXITCODE); finished questions are kept, retrying..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
  }
  $script:failed += $label
}

foreach ($ds in $Datasets) {
  $data = $files[$ds]
  if (-not (Test-Path $data)) { Write-Host "missing $data - skipping $ds" -ForegroundColor Yellow; continue }
  $common = @("--dataset", $ds, "--data", $data, "--out", $Out, "--prepend-title", "--prompt", "v2")
  if ($Limit -gt 0) { $common += @("--limit", "$Limit") }
  $slm = @("--generator", $Generator)

  Run-One "$ds | floor: no retrieval"                  (@("--system", "none") + $common + $slm)
  Run-One "$ds | Baseline 1: chunk, top-5"             (@("--system", "chunk", "--top-k", "5") + $common + $slm)
  Run-One "$ds | Baseline 2: sentence, top-5"          (@("--system", "sentence", "--top-k", "5") + $common + $slm)
  Run-One "$ds | Baseline 3: iterative (top-5 per sub-question)" (@("--system", "iterative", "--top-k", "5") + $common + $slm)
  if ($env:OPENAI_API_KEY) {
    Run-One "$ds | Baseline 4: GPT-4o-mini + chunk, top-5" (@("--system", "chunk", "--top-k", "5") + $common + @("--generator", "openai:gpt-4o-mini"))
  } else {
    Write-Host "Baseline 4 skipped (set `$env:OPENAI_API_KEY to enable)" -ForegroundColor Yellow
  }
}
python scripts\summarize_runs.py $Out --hops
if ($script:failed.Count -gt 0) {
  Write-Host ""
  Write-Host "STILL FAILING after $Attempts attempts:" -ForegroundColor Red
  $script:failed | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
  exit 1
}
