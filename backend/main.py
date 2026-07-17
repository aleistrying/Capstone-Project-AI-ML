"""
Central controller for CineAssist (API path).

Thin adapter over the ONE centralized pipeline in
`src/chatbot/chatbot_flow.py::run_pipeline`. The Streamlit chat app, the NLP
Inspector page, and this /recommend endpoint all run the same code — there is
no separate merge/normalization logic here anymore.

This module's only extra responsibilities are:
  1. Load the dataset + TF-IDF assets from disk once (cached per process).
  2. Map the PipelineTrace into the JSON shape the API contract expects.
"""

import sys
import os
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chatbot.chatbot_flow import run_pipeline, initialize_conversation_state

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"

# Process-wide asset cache: (movies_df, vectorizer, tfidf_matrix).
_assets: tuple | None = None


def _load_assets() -> tuple:
    """
    Load and cache the dataset and TF-IDF models from disk.

    Returns (movies_df, vectorizer, tfidf_matrix); any element is None if its
    file is missing. Cached after the first call so repeated /recommend requests
    don't reload the (large) matrix.
    """
    global _assets
    if _assets is not None:
        return _assets

    csv_files = list(DATA_PATH.glob("*.csv"))
    movies_df = pd.read_csv(csv_files[0]) if csv_files else None

    vec_path = MODELS_PATH / "tfidf_vectorizer.pkl"
    vectorizer = joblib.load(vec_path) if vec_path.exists() else None

    tfidf_matrix = None
    npz_path = MODELS_PATH / "tfidf_matrix.npz"
    pkl_path = MODELS_PATH / "tfidf_matrix.pkl"
    if npz_path.exists():
        from scipy.sparse import load_npz
        tfidf_matrix = load_npz(str(npz_path))
    elif pkl_path.exists():
        tfidf_matrix = joblib.load(pkl_path)

    _assets = (movies_df, vectorizer, tfidf_matrix)
    return _assets


def _safe_rating(value) -> float:
    """Coerce a vote_average cell to float, tolerating None/NaN."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def handle_user_message(
    raw_text: str,
    form_data: dict | None = None,
    movies_df=None,
    top_n: int = 5,
) -> dict:
    """
    Full request pipeline: text in → recommendations out.

    Delegates the entire NLP + recommendation flow to run_pipeline (the same
    pipeline the chat UI runs), then reshapes the trace into the API response.

    Args:
        raw_text: Free-text user input (any supported language).
        form_data: Optional starter-question answers with keys:
                   genre, mood, year_range, language, min_rating, similar_to.
        movies_df: Optional DataFrame override; defaults to the on-disk dataset.
        top_n: Number of movie recommendations to return.

    Returns:
        {
            "detected_language": "es",
            "normalized_query": "I want a funny family movie from the 2000s",
            "preferences": { ... },
            "recommendations": [ { title, year, genres, rating, score,
                                    poster_url, explanation } ],
            "metadata": { "model", "reranker", "query_text",
                          "max_similarity", "broadened", "status" }
        }
    """
    df, vectorizer, tfidf_matrix = _load_assets()
    if movies_df is not None:
        df = movies_df

    trace = run_pipeline(
        raw_text,
        initialize_conversation_state(),
        df,
        vectorizer,
        tfidf_matrix,
        top_n=top_n,
        form_data=form_data,
    )

    recommendations = []
    for r in trace.recommendations:
        recommendations.append({
            "title": r["title"],
            "year": r["year"],
            "genres": r["genres"],
            "rating": _safe_rating(r["rating"]),
            "score": round(float(r["similarity"]), 4),
            "poster_url": r.get("poster_url"),
            "explanation": r["explanation"],
        })

    return {
        "detected_language": trace.ui_language,
        "normalized_query": trace.english_input,
        "preferences": trace.prefs,
        "recommendations": recommendations,
        "metadata": {
            "model": "tfidf_cosine",
            "reranker": "none",
            "query_text": trace.query_text,
            "max_similarity": round(trace.max_similarity, 4),
            "broadened": trace.broadened,
            "status": trace.status,
        },
    }
