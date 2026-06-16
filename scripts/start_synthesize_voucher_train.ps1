param(
    [int]$Total = 500,
    [string]$Output = "data\synthesize_voucher_train.jsonl",
    [int]$Seed = 42,
    [int]$MaxDocs = 100000,
    [int]$MaxAttempts = 5000,
    [string]$Model = "mimo-v2.5",
    [string]$BaseUrl = "https://token-plan-cn.xiaomimimo.com/v1",
    [int]$RequestTimeout = 60,
    [int]$LlmRetries = 2,
    [int]$MaxCompletionTokens = 512
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo

if (-not $env:MIMO_API_KEY) {
    throw "MIMO_API_KEY is not set. Set it in this PowerShell session before starting generation."
}

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$logs = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logs | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logs "synthesize_voucher_train_$stamp.out.log"
$stderr = Join-Path $logs "synthesize_voucher_train_$stamp.err.log"

$argsList = @(
    "scripts\sample_coupon_budget.py",
    "--total", $Total,
    "--with-llm",
    "--max-docs", $MaxDocs,
    "--output", $Output,
    "--model", $Model,
    "--base-url", $BaseUrl,
    "--max-attempts", $MaxAttempts,
    "--request-timeout", $RequestTimeout,
    "--llm-retries", $LlmRetries,
    "--max-completion-tokens", $MaxCompletionTokens,
    "--seed", $Seed
)

$process = Start-Process `
    -FilePath $python `
    -ArgumentList $argsList `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

Write-Host "Started synthesize voucher generation."
Write-Host "PID: $($process.Id)"
Write-Host "Output: $Output"
Write-Host "Stdout log: $stdout"
Write-Host "Stderr log: $stderr"
Write-Host "Check progress:"
Write-Host "  (Get-Content $Output | Measure-Object -Line).Lines"
