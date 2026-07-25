"""
Single cached entry point for the dataset and TF-IDF models.

The app and each page under app/pages/ used to define their own
`@st.cache_resource load_assets()`. Streamlit keys that cache per function, so
visiting three pages loaded three independent copies of a ~460 MB working set —
fine on a laptop, fatal inside Streamlit Cloud's ~1 GB container. Everything now
imports the one loader below, so the assets are read once per process.

Format preference, highest first:
  data/processed/movies_final.parquet  then any *.csv in data/processed/
  models/tfidf_matrix.npz             then models/tfidf_matrix.pkl

The deploy branch ships only the parquet and the .npz (see
src/data/build_deploy_assets.py); main still carries the CSV and .pkl, so both
checkouts work from the same code.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"

_MISSING_ASSETS_MESSAGE = (
    "Dataset or models not found.\n\n"
    "For a local checkout, build them with:\n\n"
    "```bash\npython src/data/preprocess.py\n```"
)


def _load_dataframe():
    parquet_path = DATA_PATH / "movies_final.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    csv_files = sorted(DATA_PATH.glob("*.csv"))
    if csv_files:
        return pd.read_csv(csv_files[0])

    return None


def _load_matrix():
    npz_path = MODELS_PATH / "tfidf_matrix.npz"
    if npz_path.exists():
        from scipy.sparse import load_npz

        return load_npz(str(npz_path))

    pkl_path = MODELS_PATH / "tfidf_matrix.pkl"
    if pkl_path.exists():
        return joblib.load(pkl_path)

    return None


@st.cache_resource(show_spinner="Loading dataset & models…")
def load_assets():
    """Return (movies_df, vectorizer, tfidf_matrix); any element may be None."""
    movies_df = _load_dataframe()
    if movies_df is None:
        return None, None, None

    vec_path = MODELS_PATH / "tfidf_vectorizer.pkl"
    if not vec_path.exists():
        return movies_df, None, None
    vectorizer = joblib.load(vec_path)

    return movies_df, vectorizer, _load_matrix()


def load_assets_or_stop():
    """Same as load_assets(), but render an error and halt if anything is missing."""
    movies_df, vectorizer, tfidf_matrix = load_assets()

    if movies_df is None or vectorizer is None or tfidf_matrix is None:
        st.error(_MISSING_ASSETS_MESSAGE)
        st.stop()

    if tfidf_matrix.shape[0] != len(movies_df):
        st.error(
            f"Model/data mismatch: matrix has {tfidf_matrix.shape[0]:,} rows but the "
            f"dataset has {len(movies_df):,}. Rebuild with "
            "`python src/data/preprocess.py`."
        )
        st.stop()

    return movies_df, vectorizer, tfidf_matrix


def dataset_label() -> str:
    """Name of the file the dataframe was actually read from, for diagnostics."""
    parquet_path = DATA_PATH / "movies_final.parquet"
    if parquet_path.exists():
        return parquet_path.name
    csv_files = sorted(DATA_PATH.glob("*.csv"))
    return csv_files[0].name if csv_files else "none"
