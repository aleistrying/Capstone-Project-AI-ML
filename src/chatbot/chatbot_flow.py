"""
Chatbot flow module — conversational wrapper around the CineAssist pipeline.

Used by the Streamlit app directly. For API use, prefer backend.main.handle_user_message.
"""

import sys
import os
import ast

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.nlp.nlp_preferences import extract_preferences
from src.nlp.keyword_extractor import build_genre_vocabulary, build_query
from src.utils.explanation_generator import generate_explanation
from src.recommender.recommender_engine import recommend_on_the_fly

# Genre vocabulary is derived once from the dataset and reused across turns.
_GENRE_VOCAB: set[str] | None = None


def _get_genre_vocab(movies_df) -> set[str]:
    """Lazily build and cache the dataset genre vocabulary (used by build_query)."""
    global _GENRE_VOCAB
    if _GENRE_VOCAB is None:
        _GENRE_VOCAB = build_genre_vocabulary(movies_df)
    return _GENRE_VOCAB

# Below this top similarity score we treat the result set as a broadened/low-confidence
# fallback rather than a strong match (e.g. queries with no in-vocabulary terms).
_LOW_CONFIDENCE_THRESHOLD = 0.02

_GREETING = (
    "Hi! Tell me what you're in the mood for — e.g. a funny space "
    "adventure, a dark thriller from the 90s, something like Inception…"
)


def _parse_genres(raw) -> list[str]:
    """Normalize the genres_list field (a stringified list or a real list) to list[str]."""
    if isinstance(raw, list):
        return [str(g) for g in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return [str(g) for g in parsed]
        except (ValueError, SyntaxError):
            # Not a Python literal — fall back to a comma split.
            return [g.strip() for g in raw.split(",") if g.strip()]
    return []


def _safe_year(value) -> int | None:
    """Convert a release_year cell to int, tolerating NaN/None/bad values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_chat_recommendations(
    user_input: str,
    state_dict: dict,
    movies_df,
    vectorizer,
    tfidf_matrix=None,
) -> tuple[str, list[dict], dict, dict]:
    """
    Process one user turn and return structured, card-ready recommendations.

    Returns (intro_text, recommendations, updated_state, meta) where:
      - intro_text: short message to show above the cards (or a greeting/no-match note)
      - recommendations: list of dicts with keys
          title, year, rating, genres, overview, similarity, explanation
      - updated_state: the mutated conversation state
      - meta: {"max_similarity": float, "broadened": bool}
    """
    prefs = extract_preferences(user_input)
    state_dict.update(prefs)

    # Build a FOCUSED query instead of dumping the whole sentence into TF-IDF.
    # Lili's build_query() (src/nlp/keyword_extractor.py) strips filler/stopwords
    # and keeps meaningful content words — e.g. "psychological" survives and is
    # passed straight to the cosine model, which carries the real intent. The
    # tuned version also drops residual noise and expands meaning words with
    # thematic synonyms (filtered to the model vocabulary), which roughly doubles
    # match similarity for relevant films. We combine that with our own detected
    # genres/moods for extra signal.
    model_vocab = set(getattr(vectorizer, "vocabulary_", {})) or None
    extracted = build_query(user_input, _get_genre_vocab(movies_df), vocab=model_vocab)
    query_parts = (
        (state_dict.get("genres") or [])
        + (state_dict.get("mood") or [])
        + extracted["entities"]["genres"]
        + extracted["query"].split()
    )
    # De-duplicate while preserving order; drop empties.
    query_text = " ".join(dict.fromkeys(p for p in query_parts if p)).strip()

    if not query_text:
        return _GREETING, [], state_dict, {"max_similarity": 0.0, "broadened": False}

    legacy_state = {
        "language": state_dict.get("language"),
        "rating":   state_dict.get("min_rating") or state_dict.get("rating"),
        "year": state_dict["year_range"][0] if state_dict.get("year_range") else None,
    }

    # year_mode="soft": prefer the requested decade without hard-excluding strong
    # matches just outside it (raises match quality and avoids the edge-of-decade
    # cliff). similarity_score returned is still the raw cosine value.
    recommendations = recommend_on_the_fly(
        query_text, movies_df, vectorizer, tfidf_matrix,
        state_dict=legacy_state, year_mode="soft",
    )

    if recommendations is None or recommendations.empty:
        return (
            "No matches found. Try different words or a broader search.",
            [],
            state_dict,
            {"max_similarity": 0.0, "broadened": False},
        )

    recs: list[dict] = []
    for _, movie in recommendations.iterrows():
        movie_dict = movie.to_dict()
        recs.append(
            {
                "title": movie_dict.get("title", "Untitled"),
                "year": _safe_year(movie_dict.get("release_year")),
                "rating": movie_dict.get("vote_average"),
                "genres": _parse_genres(movie_dict.get("genres_list")),
                "overview": str(movie_dict.get("overview") or "").strip(),
                "similarity": float(movie_dict.get("similarity_score") or 0.0),
                "explanation": generate_explanation(movie_dict, state_dict),
            }
        )

    max_similarity = max((r["similarity"] for r in recs), default=0.0)
    broadened = max_similarity < _LOW_CONFIDENCE_THRESHOLD

    intro = (
        "I couldn't find a strong match, so here are the closest movies I have:"
        if broadened
        else "Here are your recommendations:"
    )
    meta = {"max_similarity": max_similarity, "broadened": broadened}
    return intro, recs, state_dict, meta


def chatbot_response(
    user_input: str,
    state_dict: dict,
    movies_df,
    vectorizer,
    tfidf_matrix=None,
) -> tuple[str, dict]:
    """
    Legacy text wrapper around get_chat_recommendations.

    Returns (response_text, updated_state_dict). Kept for backward compatibility;
    the Streamlit app uses get_chat_recommendations for rich card rendering.
    """
    intro, recs, state_dict, _meta = get_chat_recommendations(
        user_input, state_dict, movies_df, vectorizer, tfidf_matrix
    )

    if not recs:
        return intro, state_dict

    response = intro + "\n\n"
    for movie in recs:
        year_str = f" ({movie['year']})" if movie["year"] else ""
        rating = movie["rating"] if movie["rating"] is not None else "N/A"
        response += f"**{movie['title']}**{year_str} — {rating}/10\n"
        response += f"> {movie['explanation']}\n\n"

    return response, state_dict


def initialize_conversation_state() -> dict:
    return {
        "genres":    [],
        "language":  None,
        "year_range": None,
        "mood":      [],
        "min_rating": None,
        "similar_to": None,
        "free_text":  "",
    }
