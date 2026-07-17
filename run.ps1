# CineAssist — Windows PowerShell runner
#
# Usage:
#   .\run.ps1            setup deps, run tests, launch Streamlit (default)
#   .\run.ps1 setup      install/update dependencies only
#   .\run.ps1 test       run tests only
#   .\run.ps1 app        launch Streamlit UI
#   .\run.ps1 api        launch FastAPI backend (uvicorn)
#   .\run.ps1 all        Streamlit + API in two separate windows
#   .\run.ps1 rebuild    rebuild the TF-IDF models/dataset from source (SLOW)
#   .\run.ps1 -Rebuild   (same as the 'rebuild' command)
#   .\run.ps1 app -NonInteractive   never prompt (CI-safe)
#
# Model/dataset assets (models\*.pkl, data\processed\movies_final.csv) live in
# Git LFS. This runner NEVER silently rebuilds them: it detects whether they are
# real files, un-smudged LFS pointers (needs `git lfs pull`), or missing.
#
# Compatible with Windows PowerShell 5.1 AND PowerShell 7 (no ?? / ?. / ternary).
#
# First-time: you may need to allow script execution:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

param(
    [string]$Command = "default",
    [switch]$Rebuild,
    [switch]$NonInteractive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Paths ─────────────────────────────────────────────────────────────────────
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot   = $ScriptDir   # run.ps1 lives at the repo root
$VenvDir    = Join-Path $RepoRoot ".venv"
$PythonExe  = $null

# ── LFS-backed assets the app needs at runtime ─────────────────────────────────
$Vectorizer  = Join-Path $RepoRoot "models\tfidf_vectorizer.pkl"
$MatrixPkl   = Join-Path $RepoRoot "models\tfidf_matrix.pkl"
$MatrixNpz   = Join-Path $RepoRoot "models\tfidf_matrix.npz"
$Dataset     = Join-Path $RepoRoot "data\processed\movies_final.csv"
$LfsSpecLine = "version https://git-lfs.github.com/spec/v1"

if ($Rebuild) { $Command = "rebuild" }

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
        # Windows PowerShell 5.1 has no ?. / ?? operators, so resolve explicitly.
        $base = Get-Command python  -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
        if (-not $base) {
            $base = Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
        }
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

# ── Asset detection ────────────────────────────────────────────────────────────
# Classify one path as: real | pointer | missing
#   pointer = an un-smudged Git LFS pointer (tiny text file whose first line is
#             the LFS spec URL) — the real bytes were never downloaded.
function Get-AssetState {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return "missing" }
    $len = (Get-Item $Path).Length
    # Real LFS objects are many MB; a pointer is < ~200 bytes. Only sniff the
    # first line when the file is small enough to possibly be a pointer.
    if ($len -lt 1024) {
        $first = Get-Content -Path $Path -TotalCount 1 -ErrorAction SilentlyContinue
        if ($first -eq $LfsSpecLine) { return "pointer" }
    }
    return "real"
}

# Is the raw TMDB dataset present (so preprocess.py can run)?
function Test-RawData {
    return ((Get-AssetState (Join-Path $RepoRoot "data\raw\tmdb_5000_movies.csv")) -eq "real")
}

# Prompt helper. The default governs an empty <Enter> only. Non-interactive
# (no console, or -NonInteractive) always declines so CI never hangs and never
# auto-triggers downloads or rebuilds.
function Confirm-YesNo {
    param([string]$Message, [string]$Default = "N")
    if ((-not [Environment]::UserInteractive) -or $NonInteractive) { return $false }
    $reply = Read-Host $Message
    if ([string]::IsNullOrWhiteSpace($reply)) { $reply = $Default }
    return ($reply -match '^[Yy]')
}

# Run `git lfs pull`, degrading gracefully if git-lfs is not installed.
function Invoke-LfsPull {
    $hasLfs = $false
    try {
        & git lfs version 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $hasLfs = $true }
    } catch {
        $hasLfs = $false
    }
    if (-not $hasLfs) {
        Warn "git-lfs is not installed. Install from https://git-lfs.com, then run: git lfs pull"
        return $false
    }
    Info "Running: git lfs pull ..."
    Push-Location $RepoRoot
    # Pipe to Out-Host so progress shows but does not pollute the boolean return.
    & git lfs pull | Out-Host
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -eq 0) { return $true }
    Warn "git lfs pull failed — check your network / LFS access."
    return $false
}

# Rebuild the models/dataset from source, picking the right builder.
function Invoke-Rebuild {
    Info "Rebuilding assets from source (this can take several minutes) ..."
    Push-Location $ScriptDir
    try {
        if (Test-RawData) {
            Info "Raw data found -> running src\data\preprocess.py"
            # Out-Host so build output shows but does not pollute the boolean return.
            & $PythonExe src\data\preprocess.py | Out-Host
        } elseif ((Get-AssetState $Dataset) -eq "real") {
            Info "Processed dataset found -> running src\data\retrain_from_final.py"
            & $PythonExe src\data\retrain_from_final.py | Out-Host
        } else {
            Warn "Cannot rebuild: no raw data in data\raw\ and no real processed"
            Warn "dataset at data\processed\movies_final.csv."
            Warn "Fetch the LFS assets instead:  git lfs pull"
            return $false
        }
        if ($LASTEXITCODE -ne 0) {
            Warn "Build script exited with code $LASTEXITCODE."
            return $false
        }
    } finally {
        Pop-Location
    }
    Info "Rebuild complete."
    return $true
}

# ── Step 3: preflight checks ──────────────────────────────────────────────────
# Detects the three asset states (real / LFS pointer / missing) and guides the
# user WITHOUT ever silently rebuilding. Returns $true when the app is safe to
# launch, $false (with guidance already printed) when it is not.
function Preflight-Checks {
    $v = Get-AssetState $Vectorizer
    $d = Get-AssetState $Dataset
    $m = Get-AssetState $MatrixPkl
    if ($m -eq "missing") { $m = Get-AssetState $MatrixNpz }

    # (a) everything real → proceed, no rebuild.
    if (($v -eq "real") -and ($m -eq "real") -and ($d -eq "real")) {
        Info "Assets OK — using existing models/dataset (no rebuild needed)."
        return $true
    }

    # (b) any LFS pointer → the real bytes were never downloaded.
    if (($v -eq "pointer") -or ($m -eq "pointer") -or ($d -eq "pointer")) {
        Warn "One or more assets are un-smudged Git LFS pointers (not real files):"
        if ($v -eq "pointer") { Warn "  - $Vectorizer" }
        if ($m -eq "pointer") { Warn "  - models\tfidf_matrix.pkl (or .npz)" }
        if ($d -eq "pointer") { Warn "  - $Dataset" }
        Warn "These need to be downloaded with Git LFS. Do NOT rebuild for this."
        if (Confirm-YesNo "Run 'git lfs pull' now to download them? [Y/n] " "Y") {
            if (Invoke-LfsPull) {
                Info "Re-checking assets after git lfs pull ..."
                return (Preflight-Checks)
            }
        } else {
            Warn "Skipped. Download them yourself with:  git lfs pull"
        }
        Warn "App not launched — assets are still LFS pointers."
        return $false
    }

    # (c) genuinely missing (no file at all) → never auto-rebuild; offer it.
    Warn "Some assets are missing:"
    if ($v -eq "missing") { Warn "  - $Vectorizer" }
    if ($m -eq "missing") { Warn "  - models\tfidf_matrix.pkl (or .npz)" }
    if ($d -eq "missing") { Warn "  - $Dataset" }
    Warn "Options:"
    Warn "  1) git lfs pull          — fetch the pre-built assets we uploaded (recommended)"
    Warn "  2) .\run.ps1 rebuild     — rebuild from source (SLOW)"
    if (Confirm-YesNo "Rebuild models now from source? [y/N] " "N") {
        if (Invoke-Rebuild) {
            Info "Re-checking assets after rebuild ..."
            return (Preflight-Checks)
        }
    } else {
        Warn "Not rebuilding. Run 'git lfs pull' or '.\run.ps1 rebuild' when ready."
    }
    Warn "App not launched — required assets are unavailable."
    return $false
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
    "rebuild" {
        Setup-Venv
        Install-Deps
        if (-not (Invoke-Rebuild)) { Err "Rebuild did not complete." }
    }
    "app" {
        Setup-Venv
        Install-Deps
        if (-not (Preflight-Checks)) { exit 1 }
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
        if (-not (Preflight-Checks)) { exit 1 }
        Launch-All
    }
    { $_ -in "default", "" } {
        Setup-Venv
        Install-Deps
        Run-Tests
        if (-not (Preflight-Checks)) { exit 1 }
        Launch-App
    }
    default {
        Write-Host "Usage: .\run.ps1 [setup|test|app|api|all|rebuild]"
        exit 1
    }
}
