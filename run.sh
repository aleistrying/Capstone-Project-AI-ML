#!/usr/bin/env bash
# CineAssist — project runner
#
# Usage:
#   ./run.sh            setup deps, run tests, launch Streamlit (default)
#   ./run.sh setup      install/update dependencies only
#   ./run.sh test       run tests only
#   ./run.sh app        launch Streamlit UI
#   ./run.sh api        launch FastAPI backend (uvicorn)
#   ./run.sh all        Streamlit + API together (two panes, requires tmux)
#   ./run.sh rebuild    rebuild the TF-IDF models/dataset from source (SLOW)
#
# Model/dataset assets (models/*.pkl, data/processed/movies_final.csv) are
# stored in Git LFS. This runner NEVER silently rebuilds them: it detects
# whether they are real files, un-smudged LFS pointers (needs `git lfs pull`),
# or genuinely missing, and guides you accordingly.

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"   # run.sh lives at the repo root
VENV_DIR="$REPO_ROOT/.venv"
PYTHON=""
PIP=""

# ── LFS-backed assets the app needs at runtime ───────────────────────────────
VECTORIZER="$SCRIPT_DIR/models/tfidf_vectorizer.pkl"
MATRIX_PKL="$SCRIPT_DIR/models/tfidf_matrix.pkl"
MATRIX_NPZ="$SCRIPT_DIR/models/tfidf_matrix.npz"
DATASET="$SCRIPT_DIR/data/processed/movies_final.csv"

# First line of every Git LFS pointer file.
LFS_SPEC_LINE="version https://git-lfs.github.com/spec/v1"

# ── Colour helpers ──────────────────────────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"; RESET="\033[0m"
info()    { echo -e "${GREEN}[cineassist]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[warn]${RESET} $*"; }
error()   { echo -e "${RED}[error]${RESET} $*"; exit 1; }

# ── Resolve interpreter ─────────────────────────────────────────────────────
find_python() {
    for candidate in \
        "$VENV_DIR/bin/python3" \
        "$VENV_DIR/bin/python" \
        "$(which python3 2>/dev/null)" \
        "$(which python 2>/dev/null)"; do
        if [ -x "$candidate" ] 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# ── Step 1: venv ────────────────────────────────────────────────────────────
setup_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment at $VENV_DIR ..."
        py=$(which python3 2>/dev/null || which python 2>/dev/null) || error "Python 3 not found."
        "$py" -m venv "$VENV_DIR"
    fi

    PYTHON=$(find_python) || error "Could not find Python in venv."
    info "Python: $PYTHON ($($PYTHON --version))"

    # Bootstrap pip if missing
    if ! "$PYTHON" -m pip --version &>/dev/null; then
        info "Bootstrapping pip ..."
        "$PYTHON" -m ensurepip --upgrade
    fi
    PIP="$PYTHON -m pip"
}

# ── Step 2: install deps ────────────────────────────────────────────────────
install_deps() {
    info "Installing dependencies from requirements.txt ..."
    $PIP install -r "$SCRIPT_DIR/requirements.txt" -q
    info "Dependencies installed."
}

# ── Asset detection ──────────────────────────────────────────────────────────
# Classify one path as: real | pointer | missing
#   pointer = an un-smudged Git LFS pointer (tiny text file whose first line is
#             the LFS spec URL) — the real bytes were never downloaded.
classify_asset() {
    local f="$1"
    if [ ! -f "$f" ]; then
        echo "missing"; return 0
    fi
    local size first
    size=$(wc -c < "$f" 2>/dev/null | tr -d '[:space:]')
    # Real LFS objects are many MB; a pointer is < ~200 bytes. Only bother
    # sniffing the first line when the file is small enough to be a pointer.
    if [ "${size:-0}" -lt 1024 ]; then
        first=$(head -n 1 "$f" 2>/dev/null || true)
        if [ "$first" = "$LFS_SPEC_LINE" ]; then
            echo "pointer"; return 0
        fi
    fi
    echo "real"; return 0
}

# Is the raw TMDB dataset present (so preprocess.py can run)?
have_raw_data() {
    [ "$(classify_asset "$SCRIPT_DIR/data/raw/tmdb_5000_movies.csv")" = "real" ]
}

# Prompt helper. confirm "message [y/N] " N  → 0 for yes, 1 for no.
# The default governs an empty <Enter> only. Non-interactive (no TTY) always
# declines, so CI / piped runs never hang and never auto-trigger downloads or
# rebuilds — the caller prints instructions instead.
confirm() {
    local msg="$1" default="${2:-N}" reply
    if [ ! -t 0 ]; then
        return 1
    fi
    read -r -p "$msg" reply || reply=""
    reply="${reply:-$default}"
    case "$reply" in [Yy]*) return 0;; *) return 1;; esac
}

# Run `git lfs pull`, degrading gracefully if git-lfs is not installed.
run_lfs_pull() {
    if git lfs version >/dev/null 2>&1; then
        info "Running: git lfs pull ..."
        if ( cd "$REPO_ROOT" && git lfs pull ); then
            return 0
        fi
        warn "git lfs pull failed — check your network / LFS access."
        return 1
    fi
    warn "git-lfs is not installed. Install it from https://git-lfs.com, then run: git lfs pull"
    return 1
}

# Rebuild the models/dataset from source. Chooses the right builder:
#   raw data present            → src/data/preprocess.py (full rebuild)
#   only processed csv present  → src/data/retrain_from_final.py (models only)
do_rebuild() {
    info "Rebuilding assets from source (this can take several minutes) ..."
    cd "$SCRIPT_DIR"
    if have_raw_data; then
        info "Raw data found → running src/data/preprocess.py"
        "$PYTHON" src/data/preprocess.py
    elif [ "$(classify_asset "$DATASET")" = "real" ]; then
        info "Processed dataset found → running src/data/retrain_from_final.py"
        "$PYTHON" src/data/retrain_from_final.py
    else
        warn "Cannot rebuild: no raw data in data/raw/ and no real processed"
        warn "dataset at data/processed/movies_final.csv."
        warn "Fetch the LFS assets instead:  git lfs pull"
        return 1
    fi
    info "Rebuild complete."
    return 0
}

# ── Step 3: preflight checks ────────────────────────────────────────────────
# Detects the three asset states (real / LFS pointer / missing) and guides the
# user WITHOUT ever silently rebuilding. Returns 0 when the app is safe to
# launch, non-zero (with guidance already printed) when it is not.
preflight_checks() {
    local v m d
    v=$(classify_asset "$VECTORIZER")
    d=$(classify_asset "$DATASET")
    # Matrix may be .pkl or .npz — pick whichever is the "best" state present.
    m=$(classify_asset "$MATRIX_PKL")
    if [ "$m" = "missing" ]; then
        m=$(classify_asset "$MATRIX_NPZ")
    fi

    # (a) everything real → proceed, no rebuild.
    if [ "$v" = "real" ] && [ "$m" = "real" ] && [ "$d" = "real" ]; then
        info "Assets OK — using existing models/dataset (no rebuild needed)."
        return 0
    fi

    # (b) any LFS pointer → the real bytes were never downloaded.
    if [ "$v" = "pointer" ] || [ "$m" = "pointer" ] || [ "$d" = "pointer" ]; then
        warn "One or more assets are un-smudged Git LFS pointers (not real files):"
        [ "$v" = "pointer" ] && warn "  • $VECTORIZER"
        [ "$m" = "pointer" ] && warn "  • models/tfidf_matrix.pkl (or .npz)"
        [ "$d" = "pointer" ] && warn "  • $DATASET"
        warn "These need to be downloaded with Git LFS. Do NOT rebuild for this."
        if confirm "Run 'git lfs pull' now to download them? [Y/n] " Y; then
            if run_lfs_pull; then
                info "Re-checking assets after git lfs pull ..."
                if preflight_checks; then return 0; fi
                return 1
            fi
        else
            warn "Skipped. Download them yourself with:  git lfs pull"
        fi
        warn "App not launched — assets are still LFS pointers."
        return 1
    fi

    # (c) genuinely missing (no file at all) → never auto-rebuild; offer it.
    warn "Some assets are missing:"
    [ "$v" = "missing" ] && warn "  • $VECTORIZER"
    [ "$m" = "missing" ] && warn "  • models/tfidf_matrix.pkl (or .npz)"
    [ "$d" = "missing" ] && warn "  • $DATASET"
    warn "Options:"
    warn "  1) git lfs pull            — fetch the pre-built assets we uploaded (recommended)"
    warn "  2) ./run.sh rebuild        — rebuild from source (SLOW)"
    if confirm "Rebuild models now from source? [y/N] " N; then
        if do_rebuild; then
            info "Re-checking assets after rebuild ..."
            if preflight_checks; then return 0; fi
            return 1
        fi
    else
        warn "Not rebuilding. Run 'git lfs pull' or './run.sh rebuild' when ready."
    fi
    warn "App not launched — required assets are unavailable."
    return 1
}

# ── Step 4: tests ───────────────────────────────────────────────────────────
run_tests() {
    info "Running tests ..."
    cd "$SCRIPT_DIR"
    $PYTHON -m pytest tests/ -v || {
        warn "Some tests failed. Check output above."
    }
}

# ── Launchers ───────────────────────────────────────────────────────────────
launch_app() {
    info "Starting Streamlit app → http://localhost:8501"
    cd "$SCRIPT_DIR"
    exec "$PYTHON" -m streamlit run app/streamlit_app.py \
        --server.headless true \
        --browser.gatherUsageStats false
}

launch_api() {
    info "Starting FastAPI → http://localhost:8000  (docs: /docs)"
    cd "$SCRIPT_DIR"
    exec "$PYTHON" -m uvicorn backend.api.routes:app --reload --host 0.0.0.0 --port 8000
}

launch_all() {
    if ! command -v tmux &>/dev/null; then
        warn "tmux not found — launching Streamlit only (run './run.sh api' in a second terminal for the API)."
        launch_app
        return
    fi
    info "Launching Streamlit + FastAPI in tmux (session: cineassist)"
    tmux new-session -d -s cineassist -x 220 -y 50 \
        "$BASH" -c "cd $SCRIPT_DIR && $PYTHON -m streamlit run app/streamlit_app.py --server.headless true" \; \
        split-window -h \
        "$BASH" -c "cd $SCRIPT_DIR && $PYTHON -m uvicorn backend.api.routes:app --reload --host 0.0.0.0 --port 8000" \; \
        attach
}

# ── Main ─────────────────────────────────────────────────────────────────────
CMD="${1:-default}"

case "$CMD" in
    setup)
        setup_venv
        install_deps
        ;;
    test)
        setup_venv
        install_deps
        run_tests
        ;;
    rebuild)
        setup_venv
        install_deps
        do_rebuild || error "Rebuild did not complete."
        ;;
    app)
        setup_venv
        install_deps
        preflight_checks || exit 1
        launch_app
        ;;
    api)
        setup_venv
        install_deps
        launch_api
        ;;
    all)
        setup_venv
        install_deps
        preflight_checks || exit 1
        launch_all
        ;;
    default|"")
        setup_venv
        install_deps
        run_tests
        preflight_checks || exit 1
        launch_app
        ;;
    *)
        echo "Usage: ./run.sh [setup|test|app|api|all|rebuild]"
        exit 1
        ;;
esac
