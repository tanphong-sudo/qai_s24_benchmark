$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "=== Qualcomm S24 ASR Benchmark - Windows ==="
Write-Host "Working folder: $PSScriptRoot"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not $env:QAI_RUN_MODE) {
    $env:QAI_RUN_MODE = "benchmark"
}
if (-not $env:QAI_ARTIFACT_POLICY) {
    $env:QAI_ARTIFACT_POLICY = "separate_qnn_dlc"
}
if (-not $env:QAI_HUB_JOB_RETRIES) {
    $env:QAI_HUB_JOB_RETRIES = "3"
}
if (-not $env:QAI_ENABLE_PROFILING) {
    $env:QAI_ENABLE_PROFILING = "1"
}

if (-not $env:QAI_BENCHMARK_ROOT) {
    $env:QAI_BENCHMARK_ROOT = Join-Path $PSScriptRoot "qai_asr_s24_benchmark"
}

# ------------------------------------------------------------
# Find installed 64-bit Python.
# Prefer 3.11 / 3.12, but also allow the user's Python 3.13.
# ------------------------------------------------------------

$candidates = @(
    @{Exe="py";     Prefix=@("-3.11")},
    @{Exe="py";     Prefix=@("-3.12")},
    @{Exe="py";     Prefix=@("-3.10")},
    @{Exe="py";     Prefix=@("-3.13")},
    @{Exe="python"; Prefix=@()}
)

$selectedExe = $null
$selectedPrefix = @()

foreach ($candidate in $candidates) {

    $exe = $candidate.Exe
    $prefixArgs = @($candidate.Prefix)

    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
        continue
    }

    try {
        $probe = & $exe @prefixArgs -c "import sys, platform; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor)+'.'+str(sys.version_info.micro)+'|'+platform.machine()+'|'+sys.executable)" 2>$null

        if ($LASTEXITCODE -ne 0) {
            continue
        }

        if (-not $probe) {
            continue
        }

        $line = ($probe | Select-Object -Last 1).Trim()
        $parts = $line.Split("|")

        if ($parts.Count -lt 3) {
            continue
        }

        $arch = $parts[1]

        if ($arch -notmatch "AMD64|x86_64") {
            continue
        }

        $selectedExe = $exe
        $selectedPrefix = $prefixArgs

        Write-Host ""
        Write-Host "Selected Python: $($parts[0]) [$arch]" -ForegroundColor Green
        Write-Host "Interpreter: $($parts[2])"
        Write-Host ""

        break
    }
    catch {
        continue
    }
}

if (-not $selectedExe) {
    Write-Host ""
    Write-Host "ERROR: No usable 64-bit Python found." -ForegroundColor Red
    Write-Host "Install Python 3.11 64-bit, then run RUN_WINDOWS.bat again."
    exit 1
}

# ------------------------------------------------------------
# Virtual environment
# ------------------------------------------------------------

$venvDir = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {

    if (Test-Path $venvDir) {
        Write-Host "Removing incomplete .venv from previous attempt..."
        Remove-Item -Recurse -Force $venvDir
    }

    Write-Host "Creating Python virtual environment..."

    & $selectedExe @selectedPrefix -m venv $venvDir

    if ($LASTEXITCODE -ne 0) {
        throw "Python failed to create virtual environment."
    }

    if (-not (Test-Path $venvPython)) {
        throw ".venv was not created correctly."
    }
}

Write-Host ""
Write-Host "Virtual environment Python:"

& $venvPython -c "import sys,platform; print(sys.version); print('arch=' + platform.machine()); print('exe=' + sys.executable)"

if ($LASTEXITCODE -ne 0) {
    throw "Virtual environment Python failed."
}

# ------------------------------------------------------------
# Install dependencies
# ------------------------------------------------------------

Write-Host ""
Write-Host "Upgrading pip..."

& $venvPython -m pip install --upgrade pip

if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed."
}

Write-Host ""
Write-Host "Installing benchmark dependencies..."
Write-Host "This can take a while on the first run."
Write-Host ""

& $venvPython -m pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

# ------------------------------------------------------------
# Preflight checks before any paid/remote Workbench jobs start
# ------------------------------------------------------------

Write-Host ""
Write-Host "Validating installed dependencies and benchmark code..."

& $venvPython -m pip check

if ($LASTEXITCODE -ne 0) {
    throw "Installed Python dependencies are inconsistent."
}

& $venvPython -m py_compile run_benchmark.py

if ($LASTEXITCODE -ne 0) {
    throw "run_benchmark.py failed the Python syntax check."
}

& $venvPython -m unittest discover -s tests -v

if ($LASTEXITCODE -ne 0) {
    throw "Benchmark regression preflight failed."
}

# ------------------------------------------------------------
# Start benchmark
# ------------------------------------------------------------

Write-Host ""
Write-Host "============================================================"
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "============================================================"
Write-Host ""
Write-Host "Benchmark configuration:"
Write-Host "  Models : Whisper Tiny / Whisper Small / PhoWhisper Base"
Write-Host "  Device : exact hosted Samsung Galaxy S24"
Write-Host "  Compute: Qualcomm NPU requested"
Write-Host "  Samples: 100 per benchmark"
Write-Host "  Artifact: separate QNN DLC encoder/decoder (no context link required)"
Write-Host "  Retries: $env:QAI_HUB_JOB_RETRIES per inference/profile job"
Write-Host "  Profile: required S24 latency + Peak RAM"
Write-Host ""
Write-Host "If Qualcomm API token is not configured yet,"
Write-Host "the benchmark will ask for it."
Write-Host ""
Write-Host "Starting benchmark..."
Write-Host ""

& $venvPython run_benchmark.py

$code = $LASTEXITCODE

if ($code -ne 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "Benchmark stopped with exit code $code." -ForegroundColor Red
    Write-Host "You can run RUN_WINDOWS.bat again to resume checkpoints." -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Red
    exit $code
}

$bundle = Join-Path $env:QAI_BENCHMARK_ROOT "QAI_S24_BENCHMARK_SUBMISSION.zip"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "BENCHMARK COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Upload this file to Drive:"
Write-Host ""
Write-Host $bundle -ForegroundColor Cyan
Write-Host ""
Write-Host "The result ZIP contains:"
Write-Host "  - FINAL_BENCHMARK_TABLE.csv"
Write-Host "  - Qualcomm inference job IDs"
Write-Host "  - Qualcomm job URLs"
Write-Host "  - exact Samsung Galaxy S24 device evidence"
Write-Host "  - sample selection"
Write-Host "  - required S24 latency / Peak RAM profile fields"
Write-Host "  - console log"
Write-Host "  - SHA-256 hashes"
Write-Host ""
