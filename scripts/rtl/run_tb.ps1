# Re-run the whole RTL verification plan in one command (PowerShell).
#
#   .\scripts\rtl\run_tb.ps1                  # V-4, V-3, V-5, then V-1 (the long one)
#   .\scripts\rtl\run_tb.ps1 v4 v3 v5         # skip V-1
#   $env:VIVADO_BIN = 'D:\Xilinx\2024.1\bin'; .\scripts\rtl\run_tb.ps1
#
# V-1 is 16,211 windows at about 1.74 million cycles each and takes hours; it is
# run in resumable chunks across $env:SHARDS (default 2, measured) XSim processes over disjoint window ranges
# (docs/rtl_declarations.md D-6 allows splitting and forbids subsampling).
#
# Everything lands in artifacts/rtl/verify.json; docs/results_rtl.md is generated
# from it by scripts/rtl/render_results.py.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Stages)

$ErrorActionPreference = 'Stop'
# $readmemh paths in ds_gen.vh are relative to the repository root, so both XSim
# and Vivado have to be launched from there.
Set-Location (Join-Path $PSScriptRoot '..\..')

$py = '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
if (-not $Stages) { $Stages = @('all') }
$shards = if ($env:SHARDS) { $env:SHARDS } else { '2' }

& $py scripts\rtl\run_verify.py --stages compile @Stages --shards $shards
exit $LASTEXITCODE
