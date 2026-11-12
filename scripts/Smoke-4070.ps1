# Reports PASS or NOT_RUN for each piece. FAIL only if the memory self-test fails.
#   powershell -ExecutionPolicy Bypass -File .\scripts\Smoke-4070.ps1
# One command for each local speech server (faster-whisper and Piper, no LFM3):
#   powershell -ExecutionPolicy Bypass -File .\scripts\Smoke-4070.ps1 -Mode ears
#   powershell -ExecutionPolicy Bypass -File .\scripts\Smoke-4070.ps1 -Mode mouth
# Fetch Whisper base and the Piper voices once. This does not start a server:
#   powershell -ExecutionPolicy Bypass -File .\scripts\Smoke-4070.ps1 -Mode setup

param(
    [ValidateSet("smoke", "ears", "mouth", "setup")]
    [string]$Mode = "smoke"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"

$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $python = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = "python"
} else {
    Write-Output "python NOT_RUN"
    exit 3
}

switch ($Mode) {
    "ears" {
        & $python -3 -m hyper_ambient_maxi --serve-ears
        exit $LASTEXITCODE
    }
    "mouth" {
        & $python -3 -m hyper_ambient_maxi --serve-mouth
        exit $LASTEXITCODE
    }
    "setup" {
        & $python -3 -m hyper_ambient_maxi --download-models
        exit $LASTEXITCODE
    }
    default {
        & $python -3 -m hyper_ambient_maxi --smoke
        exit $LASTEXITCODE
    }
}
