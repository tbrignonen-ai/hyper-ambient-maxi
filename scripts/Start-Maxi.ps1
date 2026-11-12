# One command for the RTX 4070 checkout.
#   powershell -ExecutionPolicy Bypass -File .\scripts\Start-Maxi.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\Start-Maxi.ps1 -Mode smoke
#   powershell -ExecutionPolicy Bypass -File .\scripts\Start-Maxi.ps1 -Mode presence
# Text mode uses the thomas-4070 profile. llama-server is not started here.

param(
    [ValidateSet("text", "smoke", "presence", "selftest")]
    [string]$Mode = "text",
    [string]$Profile = "thomas-4070",
    [string]$Home = ""
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "src"
if (-not $Home) {
    $Home = Join-Path $Root "home"
}

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
    "smoke" {
        & $python -3 -m hyper_ambient_maxi --smoke
        exit $LASTEXITCODE
    }
    "selftest" {
        & $python -3 -m hyper_ambient_maxi --selftest
        exit $LASTEXITCODE
    }
    "presence" {
        & $python -3 -m hyper_ambient_maxi --presence --hotkey script --profile $Profile --home $Home
        exit $LASTEXITCODE
    }
    default {
        & $python -3 -m hyper_ambient_maxi --profile $Profile --home $Home --text
        exit $LASTEXITCODE
    }
}
