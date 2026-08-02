"""
Build the slim assets used by the hosted (Streamlit Community Cloud) app.

Streamlit Cloud does not resolve Git LFS pointers and caps the container at
about 1 GB of RAM, so the full-size artifacts committed on `main` cannot be
served as-is. This script rewrites them into two smaller files that carry the
same information the serving path actually reads:

  data/processed/movies_final.csv  (109 MB, LFS)
      -> data/processed/movies_final.parquet  (~26 MB, plain blob)
      `combined_features` is dropped: it exists only so preprocess.py and
      retrain_from_final.py can *fit* the TF-IDF. Nothing that answers a user
      query reads it.

  models/tfidf_matrix.pkl  (95 MB, LFS)
      -> models/tfidf_matrix.npz  (~45 MB, plain blob)
      Downcast to float32 and saved compressed. Cosine similarity over
      TF-IDF weights does not need float64 precision, and the app already
      prefers the .npz when present.

Both outputs land under GitHub's 100 MB per-file limit, which is what lets the
deploy branch drop LFS entirely.

Run from the project root:  python src/data/build_deploy_assets.py
"""

from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import load_npz, save_npz

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"

# Only needed to build the TF-IDF, never to serve a recommendation.
FIT_ONLY_COLUMNS = ["combined_features"]


def _mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB"


def build_dataframe() -> None:
    src = DATA_PATH / "movies_final.csv"
    dst = DATA_PATH / "movies_final.parquet"

    if not src.exists():
        raise SystemExit(
            f"ERROR: {src} not found. If you are on the deploy branch the CSV is "
            "intentionally absent — run this script on main, where the LFS asset lives."
        )

    print(f"Reading {src.name} ({_mb(src)}) ...")
    df = pd.read_csv(src)

    dropped = [c for c in FIT_ONLY_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)
    print(f"Dropped fit-only columns: {dropped or '(none present)'}")

    df.to_parquet(dst, compression="zstd", index=False)
    print(f"Wrote {dst.name} ({_mb(dst)}) — {len(df):,} rows x {len(df.columns)} cols")
    print(f"In-memory footprint: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")


def build_matrix() -> None:
    src = MODELS_PATH / "tfidf_matrix.pkl"
    dst = MODELS_PATH / "tfidf_matrix.npz"

    if dst.exists() and not src.exists():
        print(f"{dst.name} already present and no .pkl to rebuild from — skipping.")
        return
    if not src.exists():
        raise SystemExit(
            f"ERROR: {src} not found. Run this script on main, where the LFS asset lives."
        )

    print(f"Loading {src.name} ({_mb(src)}) ...")
    matrix = joblib.load(src)
    print(f"  shape={matrix.shape} dtype={matrix.dtype} nnz={matrix.nnz:,}")

    save_npz(str(dst), matrix.astype("float32"), compressed=True)
    print(f"Wrote {dst.name} ({_mb(dst)})")

    # Round-trip check: the deploy branch has no .pkl to fall back on, so a
    # corrupt .npz would be an outage rather than a degraded mode.
    reloaded = load_npz(str(dst))
    if reloaded.shape != matrix.shape:
        raise RuntimeError("matrix shape changed on round-trip")
    if reloaded.nnz != matrix.nnz:
        raise RuntimeError("matrix nonzero count changed on round-trip")
    print("Round-trip verified ✓")


if __name__ == "__main__":
    build_dataframe()
    build_matrix()
    print("\nDone. Both outputs are plain blobs under 100 MB — safe to commit without LFS.")
