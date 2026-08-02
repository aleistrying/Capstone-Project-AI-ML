"""TF-IDF movie ranking with optional language, year, and rating preferences."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.utils.text_cleaning import clean_text

DISPLAY_COLUMNS = (
    "movieId",
    "title",
    "genres_list",
    "vote_average",
    "overview",
)
VALID_YEAR_MODES = {"filter", "soft"}


def _validate_inputs(movies_df, tfidf_matrix, top_n, year_mode, year_decay):
    """Reject invalid model/data combinations with actionable messages."""
    missing = [column for column in DISPLAY_COLUMNS if column not in movies_df.columns]
    if missing:
        raise ValueError(f"movies_df is missing required columns: {', '.join(missing)}")
    if tfidf_matrix is None:
        raise ValueError("tfidf_matrix is required")
    if tfidf_matrix.shape[0] != len(movies_df):
        raise ValueError(
            "tfidf_matrix row count must match movies_df: "
            f"{tfidf_matrix.shape[0]} != {len(movies_df)}"
        )
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if year_mode not in VALID_YEAR_MODES:
        raise ValueError(f"year_mode must be one of {sorted(VALID_YEAR_MODES)}")
    if not 0 <= year_decay <= 1:
        raise ValueError("year_decay must be between 0 and 1")


def _build_scored_frame(movies_df, similarity_scores):
    """Build a narrow frame used only for filtering and ranking."""
    scored = pd.DataFrame(
        {"similarity_score": similarity_scores}, index=movies_df.index
    )
    for column in ("original_language", "vote_average"):
        if column in movies_df.columns:
            scored[column] = movies_df[column]

    if "release_year" in movies_df.columns:
        scored["release_year"] = movies_df["release_year"]
    elif "release_date" in movies_df.columns:
        scored["release_year"] = pd.to_datetime(
            movies_df["release_date"], errors="coerce"
        ).dt.year
    return scored


def _apply_soft_year_score(scored, target_year, year_decay):
    """Apply decade-distance decay without removing out-of-decade movies."""
    years = pd.to_numeric(scored["release_year"], errors="coerce").to_numpy()
    decade_center = target_year + 4.5
    in_decade = (years >= target_year) & (years <= target_year + 9)
    decades_away = np.abs(years - decade_center) / 10.0
    decay = np.where(in_decade, 1.0, year_decay**decades_away)
    decay = np.where(np.isnan(years), 0.5, decay)
    result = scored.copy()
    result["similarity_score"] = result["similarity_score"].to_numpy() * decay
    return result


def _apply_filters(scored, filters: Mapping, year_mode, year_decay):
    """Apply optional filters and return the narrowed scoring frame."""
    filtered = scored
    language = filters.get("language")
    if language:
        if "original_language" not in filtered.columns:
            raise ValueError("language filtering requires original_language")
        filtered = filtered[filtered["original_language"] == language]

    year = filters.get("year")
    if year and "release_year" in filtered.columns:
        target_year = int(year)
        if year_mode == "soft":
            filtered = _apply_soft_year_score(filtered, target_year, year_decay)
        else:
            filtered = filtered[
                filtered["release_year"].between(target_year - 2, target_year + 5)
            ]

    rating = filters.get("rating")
    if rating:
        if "vote_average" not in filtered.columns:
            raise ValueError("rating filtering requires vote_average")
        filtered = filtered[filtered["vote_average"] >= float(rating)]
    return filtered


def recommend_on_the_fly(
    query_text,
    movies_df,
    vectorizer,
    tfidf_matrix,
    state_dict=None,
    top_n=5,
    year_mode="filter",
    year_decay=0.6,
):
    """Recommend movies using TF-IDF cosine similarity and optional filters.

    ``year_mode="filter"`` hard-filters a requested period. ``"soft"`` keeps
    every candidate and applies a decade-distance score decay. If filters remove
    every result, the function falls back to raw similarity ranking.
    """
    _validate_inputs(movies_df, tfidf_matrix, top_n, year_mode, year_decay)
    query_vector = vectorizer.transform([clean_text(query_text)])
    similarity_scores = (tfidf_matrix @ query_vector.T).toarray().ravel()
    scored = _build_scored_frame(movies_df, similarity_scores)
    filtered = _apply_filters(scored, state_dict or {}, year_mode, year_decay)

    ranked = filtered.sort_values("similarity_score", ascending=False).head(top_n)
    if ranked.empty:
        ranked = scored.sort_values("similarity_score", ascending=False).head(top_n)

    recommendations = movies_df.loc[ranked.index].copy()
    for column in ranked.columns:
        recommendations[column] = ranked[column]

    columns = list(DISPLAY_COLUMNS)
    columns.insert(4, "similarity_score")
    if "release_year" in recommendations.columns:
        columns.insert(2, "release_year")
    if "original_language" in recommendations.columns:
        columns.append("original_language")
    return recommendations[columns]
