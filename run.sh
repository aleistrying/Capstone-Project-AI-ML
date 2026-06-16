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

set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON=""
PIP=""

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

# ── Step 3: preflight checks ────────────────────────────────────────────────
preflight_checks() {
    local ok=true

    # Processed data
    if [ -z "$(ls -A "$SCRIPT_DIR/data/processed"/*.csv 2>/dev/null)" ]; then
        warn "No CSV found in data/processed/ — run notebooks 01 & 02 first."
        ok=false
    else
        info "Dataset: $(ls "$SCRIPT_DIR/data/processed"/*.csv | head -1)"
    fi

    # TF-IDF models
    if [ ! -f "$SCRIPT_DIR/models/tfidf_vectorizer.pkl" ]; then
        warn "models/tfidf_vectorizer.pkl missing — run notebook 03_Vectorization."
        ok=false
    fi
    if [ ! -f "$SCRIPT_DIR/models/tfidf_matrix.pkl" ] && \
       [ ! -f "$SCRIPT_DIR/models/tfidf_matrix.npz" ]; then
        warn "models/tfidf_matrix (.pkl/.npz) missing — run notebook 03_Vectorization."
        ok=false
    fi

    if [ "$ok" = false ]; then
        warn "Some assets are missing. The app will show an error until they are generated."
        warn "Run the notebooks in order: 01 → 02 → 03 → 04"
    fi
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
    app)
        setup_venv
        install_deps
        preflight_checks
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
        preflight_checks
        launch_all
        ;;
    default|"")
        setup_venv
        install_deps
        run_tests
        preflight_checks
        launch_app
        ;;
    *)
        echo "Usage: ./run.sh [setup|test|app|api|all]"
        exit 1
        ;;
esac
