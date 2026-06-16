"""
Recommender service: wraps TF-IDF + cosine similarity engine.

Loads the vectorizer and TF-IDF matrix once at import time so repeated
calls in the same process do not reload from disk.
"""

import os
import sys
import joblib
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.recommender.recommender_engine import recommend_on_the_fly

_MODELS_DIR = Path(__file__).parent.parent.parent / "models"

_vectorizer = None
_tfidf_matrix = None
_movies_df = None


def _load_assets(movies_df=None):
    """Lazy-load models. Pass movies_df explicitly when running standalone."""
    global _vectorizer, _tfidf_matrix, _movies_df

    if movies_df is not None:
        _movies_df = movies_df

    if _vectorizer is None:
        vec_path = _MODELS_DIR / "tfidf_vectorizer.pkl"
        if vec_path.exists():
            _vectorizer = joblib.load(vec_path)

    if _tfidf_matrix is None:
        mat_path = _MODELS_DIR / "tfidf_matrix.pkl"
        npz_path = _MODELS_DIR / "tfidf_matrix.npz"
        if npz_path.exists():
            from scipy.sparse import load_npz
            _tfidf_matrix = load_npz(npz_path)
        elif mat_path.exists():
            _tfidf_matrix = joblib.load(mat_path)


def recommend(preferences: dict, movies_df=None, top_n: int = 5) -> list[dict]:
    """
    Run TF-IDF cosine similarity recommendation.

    Args:
        preferences: Structured preference dict from nlp_service.extract().
        movies_df: DataFrame with movie data (required on first call).
        top_n: Number of recommendations to return.

    Returns:
        List of recommendation dicts with title, year, genres, rating, score,
        overview fields.
    """
    _load_assets(movies_df)

    if _vectorizer is None or _movies_df is None:
        return []

    query_parts = []
    query_parts.extend(preferences.get("genres") or [])
    query_parts.extend(preferences.get("mood") or [])
    if preferences.get("similar_to"):
        query_parts.append(preferences["similar_to"])
    query_parts.append(preferences.get("free_text", ""))
    query_text = " ".join(p for p in query_parts if p)

    state_dict = {
        "language": preferences.get("language"),
        "rating": preferences.get("min_rating"),
        "year": preferences["year_range"][0] if preferences.get("year_range") else None,
    }

    df_result = recommend_on_the_fly(
        query_text, _movies_df, _vectorizer, _tfidf_matrix,
        state_dict=state_dict, top_n=top_n
    )

    if df_result is None or df_result.empty:
        return []

    results = []
    for _, row in df_result.iterrows():
        results.append({
            "title": row.get("title", "Unknown"),
            "year": int(row["release_year"]) if "release_year" in row and row["release_year"] else None,
            "genres": row.get("genres_list", []),
            "rating": float(row.get("vote_average", 0)),
            "score": round(float(row.get("similarity_score", 0)), 4),
            "overview": row.get("overview", ""),
            "poster_url": row.get("poster_url", None),
        })

    return results
