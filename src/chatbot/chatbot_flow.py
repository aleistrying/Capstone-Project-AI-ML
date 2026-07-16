"""
Chatbot flow module — conversational wrapper around the CineAssist pipeline.

FULL PIPELINE (one user turn):

  [User types in any language]
        ↓
  1. detect_language()        ← Brayan — src/translation/lang_detector.py
        Statistical (langdetect) + stopword fallback for short text.
        ↓
  2. translate_to_english()   ← Brayan — src/translation/translator.py (MarianMT)
        ↓
  3. extract_preferences()    ← Team — src/nlp/nlp_preferences.py
  3b. build_query()           ← Lili (ported) — src/nlp/keyword_extractor.py
        Focused keyword query + thematic synonym expansion.
        ↓
  4. recommend_on_the_fly()   ← Alejandro/David — src/recommender/recommender_engine.py
        TF-IDF cosine + soft-decade ranking.
        ↓
  5. generate_explanation()   ← Team — src/utils/explanation_generator.py
        ↓
  6. translate_from_english() ← Brayan — translate intro + explanations back
        ↓
  [Chatbot replies in the same language the user wrote in]

USED BY: app/streamlit_app.py (Streamlit chat UI).

Translation degrades gracefully: if the ML translation deps/models are not
available, the pipeline runs in English-only mode instead of crashing.
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

# --- Translation layer (Brayan). Imported defensively: torch/transformers/
# langdetect may be missing in some environments, and MarianMT models download
# on first use. Any failure falls back to English-only so the MVP never breaks.
try:
    from src.translation import (
        detect_language as _detect_language,
        translate_to_english as _translate_to_english,
        translate_from_english as _translate_from_english,
    )
    _TRANSLATION_AVAILABLE = True
except Exception:  # missing deps, etc.
    _TRANSLATION_AVAILABLE = False

try:
    from backend.services.language_service import normalize as _domain_normalize
    _DOMAIN_AVAILABLE = True
except Exception:
    _DOMAIN_AVAILABLE = False

# Genre vocabulary is derived once from the dataset and reused across turns.
_GENRE_VOCAB: set[str] | None = None


def _get_genre_vocab(movies_df) -> set[str]:
    """Lazily build and cache the dataset genre vocabulary (used by build_query)."""
    global _GENRE_VOCAB
    if _GENRE_VOCAB is None:
        _GENRE_VOCAB = build_genre_vocabulary(movies_df)
    return _GENRE_VOCAB


# Below this top similarity score we treat the result set as broadened/low-confidence.
_LOW_CONFIDENCE_THRESHOLD = 0.02

_GREETING = (
    "Hi! Tell me what you're in the mood for — e.g. a funny space "
    "adventure, a dark thriller from the 90s, something like Inception…"
)


# ---------------------------------------------------------------------------
# Translation helpers — all safe no-ops when translation is unavailable.
# ---------------------------------------------------------------------------

def _safe_domain_normalize(text: str) -> dict:
    """Domain vocabulary mapping + stopword language detection (no heavy deps)."""
    if _DOMAIN_AVAILABLE:
        try:
            return _domain_normalize(text)
        except Exception:
            pass
    return {
        "detected_language": "en", "mapped_genres": [], "mapped_moods": [],
        "mapped_year_range": None, "mapped_language": None,
    }


def _detect_ui_language(text: str, domain: dict) -> str:
    """langdetect primary, stopword detector fallback (Brayan's combined approach)."""
    if not _TRANSLATION_AVAILABLE:
        return domain.get("detected_language", "en")
    try:
        lang_ngram = _detect_language(text)
    except Exception:
        lang_ngram = "en"
    return lang_ngram if lang_ngram != "en" else domain.get("detected_language", "en")


def _to_english(text: str, ui_lang: str) -> str:
    """Translate user input to English for processing; safe fallback to original."""
    if ui_lang == "en" or not _TRANSLATION_AVAILABLE:
        return text
    try:
        return _translate_to_english(text, ui_lang)
    except Exception:
        return text


def _from_english(text: str, ui_lang: str) -> str:
    """Translate an English string back to the user's language; safe fallback."""
    if ui_lang == "en" or not _TRANSLATION_AVAILABLE or not text:
        return text
    try:
        return _translate_from_english(text, ui_lang)
    except Exception:
        return text


def _fill_pref_gaps(prefs: dict, domain: dict) -> None:
    """Fill preference gaps with domain-vocabulary mappings from the original text."""
    if not prefs.get("genres") and domain.get("mapped_genres"):
        prefs["genres"] = domain["mapped_genres"]
    if not prefs.get("mood") and domain.get("mapped_moods"):
        prefs["mood"] = domain["mapped_moods"]
    if not prefs.get("year_range") and domain.get("mapped_year_range"):
        prefs["year_range"] = domain["mapped_year_range"]
    if not prefs.get("language") and domain.get("mapped_language"):
        prefs["language"] = domain["mapped_language"]


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

    Multilingual: the user may write in any supported language. Input is
    translated to English for processing; the intro and per-movie explanations
    are translated back to the user's language. Movie titles are left as-is.

    Returns (intro_text, recommendations, updated_state, meta) where each rec
    has: title, year, rating, genres, overview, similarity, explanation.
    meta = {"max_similarity": float, "broadened": bool, "ui_language": str}.
    """
    # ── Translation front: detect language, translate input to English ────────
    domain = _safe_domain_normalize(user_input)
    ui_lang = _detect_ui_language(user_input, domain)
    state_dict["ui_language"] = ui_lang
    english_input = _to_english(user_input, ui_lang)

    prefs = extract_preferences(english_input)
    _fill_pref_gaps(prefs, domain)   # backfill from original-language domain terms
    state_dict.update(prefs)

    def _meta(max_sim, broadened):
        return {"max_similarity": max_sim, "broadened": broadened, "ui_language": ui_lang}

    # Build a FOCUSED query (keyword extraction + synonym expansion) on the
    # English text instead of dumping the whole sentence into TF-IDF.
    model_vocab = set(getattr(vectorizer, "vocabulary_", {})) or None
    extracted = build_query(english_input, _get_genre_vocab(movies_df), vocab=model_vocab)
    query_parts = (
        (state_dict.get("genres") or [])
        + (state_dict.get("mood") or [])
        + extracted["entities"]["genres"]
        + extracted["query"].split()
    )
    query_text = " ".join(dict.fromkeys(p for p in query_parts if p)).strip()

    if not query_text:
        return _from_english(_GREETING, ui_lang), [], state_dict, _meta(0.0, False)

    legacy_state = {
        "language": state_dict.get("language"),
        "rating":   state_dict.get("min_rating") or state_dict.get("rating"),
        "year": state_dict["year_range"][0] if state_dict.get("year_range") else None,
    }

    # year_mode="soft": prefer the requested decade without hard-excluding strong
    # matches just outside it. similarity_score is the decade-aware match score.
    recommendations = recommend_on_the_fly(
        query_text, movies_df, vectorizer, tfidf_matrix,
        state_dict=legacy_state, year_mode="soft",
    )

    if recommendations is None or recommendations.empty:
        msg = "No matches found. Try different words or a broader search."
        return _from_english(msg, ui_lang), [], state_dict, _meta(0.0, False)

    recs: list[dict] = []
    for _, movie in recommendations.iterrows():
        movie_dict = movie.to_dict()
        explanation_en = generate_explanation(movie_dict, state_dict)
        recs.append(
            {
                "title": movie_dict.get("title", "Untitled"),
                "year": _safe_year(movie_dict.get("release_year")),
                "rating": movie_dict.get("vote_average"),
                "genres": _parse_genres(movie_dict.get("genres_list")),
                "overview": str(movie_dict.get("overview") or "").strip(),
                "similarity": float(movie_dict.get("similarity_score") or 0.0),
                # Translate the explanation back to the user's language (no-op for English).
                "explanation": _from_english(explanation_en, ui_lang),
            }
        )

    max_similarity = max((r["similarity"] for r in recs), default=0.0)
    broadened = max_similarity < _LOW_CONFIDENCE_THRESHOLD

    intro_en = (
        "I couldn't find a strong match, so here are the closest movies I have:"
        if broadened
        else "Here are your recommendations:"
    )
    return _from_english(intro_en, ui_lang), recs, state_dict, _meta(max_similarity, broadened)


def chatbot_response(
    user_input: str,
    state_dict: dict,
    movies_df,
    vectorizer,
    tfidf_matrix=None,
) -> tuple[str, dict]:
    """
    Legacy text wrapper around get_chat_recommendations.

    Returns (response_text, updated_state_dict). The Streamlit app uses
    get_chat_recommendations directly for rich card rendering; this is kept for
    backward compatibility (and any text-only API caller). The returned text is
    already in the user's language (translation handled upstream).
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
    """
    Create a fresh state dict for a new conversation.

    Keys:
      genres, language (movie language), year_range, mood, min_rating,
      similar_to, free_text, ui_language (the language the USER writes in).
    """
    return {
        "genres":      [],
        "language":    None,
        "year_range":  None,
        "mood":        [],
        "min_rating":  None,
        "similar_to":  None,
        "free_text":   "",
        "ui_language": None,
    }
