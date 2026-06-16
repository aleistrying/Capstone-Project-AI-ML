# CineAssist — Windows PowerShell runner
#
# Usage:
#   .\run.ps1            setup deps, run tests, launch Streamlit (default)
#   .\run.ps1 setup      install/update dependencies only
#   .\run.ps1 test       run tests only
#   .\run.ps1 app        launch Streamlit UI
#   .\run.ps1 api        launch FastAPI backend (uvicorn)
#   .\run.ps1 all        Streamlit + API in two separate windows
#
# First-time: you may need to allow script execution:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

param(
    [string]$Command = "default"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Paths ─────────────────────────────────────────────────────────────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = Split-Path -Parent $ScriptDir
$VenvDir    = Join-Path $RepoRoot ".venv"
$PythonExe  = $null

# ── Colour helpers ────────────────────────────────────────────────────────────
function Info($msg)  { Write-Host "[cineassist] $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "[warn] $msg"       -ForegroundColor Yellow }
function Err($msg)   { Write-Host "[error] $msg"      -ForegroundColor Red; exit 1 }

# ── Resolve Python ────────────────────────────────────────────────────────────
function Find-Python {
    $candidates = @(
        (Join-Path $VenvDir "Scripts\python.exe"),
        (Join-Path $VenvDir "Scripts\python3.exe"),
        (Get-Command python  -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source),
        (Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    return $null
}

# ── Step 1: venv ──────────────────────────────────────────────────────────────
function Setup-Venv {
    if (-not (Test-Path $VenvDir)) {
        Info "Creating virtual environment at $VenvDir ..."
        $base = (Get-Command python -ErrorAction SilentlyContinue)?.Source `
             ?? (Get-Command python3 -ErrorAction SilentlyContinue)?.Source
        if (-not $base) { Err "Python 3 not found. Install from https://python.org" }
        & $base -m venv $VenvDir
    }

    $script:PythonExe = Find-Python
    if (-not $PythonExe) { Err "Could not find Python inside venv." }

    $version = & $PythonExe --version 2>&1
    Info "Python: $PythonExe ($version)"

    # Bootstrap pip if missing
    $pipOk = & $PythonExe -m pip --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Info "Bootstrapping pip..."
        & $PythonExe -m ensurepip --upgrade
    }
}

# ── Step 2: install deps ──────────────────────────────────────────────────────
function Install-Deps {
    Info "Installing dependencies from requirements.txt ..."
    & $PythonExe -m pip install -r "$ScriptDir\requirements.txt" -q
    Info "Dependencies installed."
}

# ── Step 3: preflight checks ──────────────────────────────────────────────────
function Preflight-Checks {
    $ok = $true

    $csvFiles = Get-ChildItem "$ScriptDir\data\processed\*.csv" -ErrorAction SilentlyContinue
    if (-not $csvFiles) {
        Warn "No CSV in data\processed\ — run notebooks 01 and 02 first."
        $ok = $false
    } else {
        Info "Dataset: $($csvFiles[0].FullName)"
    }

    if (-not (Test-Path "$ScriptDir\models\tfidf_vectorizer.pkl")) {
        Warn "models\tfidf_vectorizer.pkl missing — run notebook 03_Vectorization."
        $ok = $false
    }
    if (-not (Test-Path "$ScriptDir\models\tfidf_matrix.pkl") -and
        -not (Test-Path "$ScriptDir\models\tfidf_matrix.npz")) {
        Warn "models\tfidf_matrix (.pkl/.npz) missing — run notebook 03_Vectorization."
        $ok = $false
    }

    if (-not $ok) {
        Warn "Some assets are missing. The app will show an error until they are generated."
        Warn "Run: python src\data\preprocess.py   (downloads and builds everything)"
    }
}

# ── Step 4: tests ─────────────────────────────────────────────────────────────
function Run-Tests {
    Info "Running tests ..."
    Push-Location $ScriptDir
    & $PythonExe -m pytest tests\ -v
    if ($LASTEXITCODE -ne 0) { Warn "Some tests failed. Check output above." }
    Pop-Location
}

# ── Launchers ─────────────────────────────────────────────────────────────────
function Launch-App {
    Info "Starting Streamlit app → http://localhost:8501"
    Push-Location $ScriptDir
    & $PythonExe -m streamlit run app\streamlit_app.py `
        --server.headless true `
        --browser.gatherUsageStats false
    Pop-Location
}

function Launch-Api {
    Info "Starting FastAPI → http://localhost:8000  (docs: /docs)"
    Push-Location $ScriptDir
    & $PythonExe -m uvicorn backend.api.routes:app --reload --host 0.0.0.0 --port 8000
    Pop-Location
}

function Launch-All {
    Info "Opening Streamlit and FastAPI in separate PowerShell windows ..."
    $appCmd = "Set-Location '$ScriptDir'; & '$PythonExe' -m streamlit run app\streamlit_app.py --server.headless true --browser.gatherUsageStats false"
    $apiCmd = "Set-Location '$ScriptDir'; & '$PythonExe' -m uvicorn backend.api.routes:app --reload --host 0.0.0.0 --port 8000"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $appCmd
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $apiCmd
    Info "Both services launched. Close the windows to stop."
}

# ── Main ──────────────────────────────────────────────────────────────────────
switch ($Command.ToLower()) {
    "setup" {
        Setup-Venv
        Install-Deps
    }
    "test" {
        Setup-Venv
        Install-Deps
        Run-Tests
    }
    "app" {
        Setup-Venv
        Install-Deps
        Preflight-Checks
        Launch-App
    }
    "api" {
        Setup-Venv
        Install-Deps
        Launch-Api
    }
    "all" {
        Setup-Venv
        Install-Deps
        Preflight-Checks
        Launch-All
    }
    { $_ -in "default", "" } {
        Setup-Venv
        Install-Deps
        Run-Tests
        Preflight-Checks
        Launch-App
    }
    default {
        Write-Host "Usage: .\run.ps1 [setup|test|app|api|all]"
        exit 1
    }
}
