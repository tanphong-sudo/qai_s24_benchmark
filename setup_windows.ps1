$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Qualcomm S24 ASR Benchmark - Windows ==="
Write-Host "Working folder: $PSScriptRoot"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:QAI_BENCHMARK_ROOT) {
    $env:QAI_BENCHMARK_ROOT = Join-Path $PSScriptRoot "qai_asr_s24_benchmark"
}

function Invoke-PythonLauncher {
    param([string[]]$Args)
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 @Args
        if ($LASTEXITCODE -eq 0) { return }
        & py -3 @Args
        if ($LASTEXITCODE -eq 0) { return }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python @Args
        if ($LASTEXITCODE -eq 0) { return }
    }
    throw "Python 3.11/3.x not found. Install 64-bit Python 3.11 from python.org, enable 'Add python.exe to PATH', then run again."
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating Python virtual environment..."
    Invoke-PythonLauncher -Args @("-m", "venv", ".venv")
}

$venvScripts = Join-Path $PSScriptRoot ".venv\Scripts"
$env:PATH = "$venvScripts;$env:PATH"

Write-Host "Installing/updating dependencies..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "dependency installation failed" }

Write-Host ""
Write-Host "Setup complete."
Write-Host "If QAI_HUB_API_TOKEN is not already set, Python will ask for it securely."
Write-Host "The token is NOT written into the evidence bundle."
Write-Host ""
Write-Host "Starting benchmark: 100 samples per benchmark, 3 models, exact hosted Samsung Galaxy S24, NPU requested..."
Write-Host ""

& $venvPython run_benchmark.py
$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Host ""
    Write-Host "Benchmark stopped with exit code $code. Re-run RUN_WINDOWS.bat to resume from checkpoints." -ForegroundColor Red
    exit $code
}

$bundle = Join-Path $env:QAI_BENCHMARK_ROOT "QAI_S24_BENCHMARK_SUBMISSION.zip"
Write-Host ""
Write-Host "=== DONE ===" -ForegroundColor Green
Write-Host "Upload this ONE file to Drive:" -ForegroundColor Green
Write-Host $bundle -ForegroundColor Cyan
Write-Host ""
Write-Host "The ZIP includes the final table plus Qualcomm job IDs/URLs, exact S24 device evidence, sample selection, profile data, console log and SHA-256 hashes."
