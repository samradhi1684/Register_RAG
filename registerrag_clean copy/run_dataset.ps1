# Run the RegisterRAG pipeline on one dataset's 100-question test set, score, judge, report.
#   .\run_dataset.ps1 -Dataset 2wiki                       (run1: official-dev corpus)
#   .\run_dataset.ps1 -Dataset 2wiki -Run run2_500ctx      (run2: corpus = contexts of the 500 questions)
#   .\run_dataset.ps1 -Dataset musique -Model qwen2.5:14b -Name test100_14b
#   .\run_dataset.ps1 -Dataset 2wiki -Run run2_500ctx -Data registerrag\qa-pairs\2wiki_500.json -Name all500
# Index:  runs\<Run>\<Dataset>\index      Output: runs\<Run>\<Dataset>\<Name>\
#   (results.jsonl, traces\, score.json, judge_rows.jsonl, report.md, failures.csv)
param(
    [Parameter(Mandatory)] [ValidateSet("2wiki", "hotpotqa", "musique", "repolicyqa")] [string] $Dataset,
    [string] $Run = "run1",
    [string] $Model = "qwen2.5:7b",
    [string] $Profile = "profiles\registerrag_v2.json",
    [string] $Name = "test100",
    [string] $Cache = "",              # optional cache file (default: one per run folder)
    [string] $Data = ""                # optional question file (default: the 100-question test set)
)
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONHASHSEED = "0"          # repeatable candidate order
Set-Location $PSScriptRoot
$py = "..\.venv\Scripts\python.exe"
$data = if ($Data) { $Data } else { "registerrag\qa-pairs\$Dataset\${Dataset}_100.json" }
$base = "runs\$Run\$Dataset"
$idx = "$base\index"
# DSIRE corpus folder per run (run1: all ~1k documents; run2: the 92 source documents of the 500 questions,
# only the passages the questions were written from — see tools\prepare_dsire_92.py)
$corpus = @{ "run1" = "registerrag\data\final"; "run2_500ctx" = "registerrag\data\final_92docs" }[$Run]
$extra = @(); if ($Dataset -eq "repolicyqa") { $extra = @("--corpus-dir", $corpus) }
# one LLM cache per run folder, so a run's latency is never shortened by another run's cached calls
$m = $Model -replace "[:/]", "-"
$cache = if ($Cache) { $Cache } elseif ($Run -eq "run1") { "cache\$m.sqlite" } else { "cache\$Run-$m.sqlite" }
$log = "$base\$Name.log"
Write-Host "=== $Run / $Dataset, $Model -> $base\$Name"
& $py -m registerrag.run eval --data $data --format $Dataset --provider ollama --model $Model --stage 2 `
    --index-dir $idx --runs-dir $base --profile $Profile --llm-cache $cache --run-name $Name @extra *> $log
& $py -m registerrag.run score --data $data --format $Dataset --index-dir $idx --runs-dir $base --run-name $Name @extra *>> $log
& $py tools\judge.py "$base\$Name" *>> $log
& $py tools\report.py "$base\$Name" $Dataset $data $idx $corpus
