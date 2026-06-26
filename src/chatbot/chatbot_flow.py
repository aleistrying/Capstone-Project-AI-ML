"""
Chatbot flow module — conversational wrapper around the CineAssist pipeline.

WHAT THIS FILE DOES
--------------------
This module handles one full "turn" of the conversation:
  user message in → chatbot response out.

It is the glue layer that connects every other module together.

FULL PIPELINE (one user turn):

  [User types in any language]
          ↓
  1. detect_language()          ← Brayan — src/translation/lang_detector.py
     Identifies whether the user is writing in Spanish, French, etc.
     Uses the 'langdetect' library (Google's statistical language detector).
     Result is stored in state so it only runs on the first message.
          ↓
  2. translate_to_english()     ← Brayan — src/translation/translator.py
     If the user is not writing in English, the full sentence is translated
     using a Helsinki-NLP MarianMT neural model from HuggingFace.
     This gives the NLP extractor clean English text to work with.
          ↓
  3. extract_preferences()      ← Carlos/Team — src/nlp/nlp_preferences.py
     Extracts structured preferences from the English text:
     genres, mood, year, rating, free_text.
          ↓
  4. recommend_on_the_fly()     ← Alejandro — src/recommender/recommender_engine.py
     TF-IDF cosine similarity search over the movie dataset.
          ↓
  5. generate_explanation()     ← Team — src/utils/explanation_generator.py
     Builds a natural-language sentence explaining why each movie was chosen.
          ↓
  6. translate_from_english()   ← Brayan — src/translation/translator.py
     Translates the complete response back into the user's original language.
          ↓
  [Chatbot replies in the same language the user wrote in]

USED BY
--------
  app/streamlit_app.py  (Streamlit chat UI)

NOTE: For API use, prefer backend.main.handle_user_message, which uses
the faster stopword-based language_service instead of full ML translation.
"""

import sys
import os

# Make the project root importable from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.nlp.nlp_preferences import extract_preferences
from src.utils.explanation_generator import generate_explanation
from src.recommender.recommender_engine import recommend_on_the_fly

# Brayan's translation module:
#   detect_language        → statistical language identification (langdetect)
#   translate_to_english   → translates any supported language → English
#   translate_from_english → translates English → any supported language
from src.translation import detect_language, translate_to_english, translate_from_english

# language_service provides two things we also use here:
#   1. A stopword-based language detector — more reliable for short/accent-free text
#   2. Domain vocabulary mapping — maps "accion"→"action", "años 90"→[1990,1999], etc.
#      This catches movie terms that langdetect or NLP might miss after translation.
from backend.services.language_service import (
    normalize as _domain_normalize,
    detect_language as _sw_detect,
)


def chatbot_response(
    user_input: str,
    state_dict: dict,
    movies_df,
    vectorizer,
    tfidf_matrix=None,
) -> tuple[str, dict]:
    """
    Process one user turn and return the chatbot's response.

    This is called once per message the user sends. State is passed in and
    out so the conversation remembers what was said in previous turns
    (e.g. a genre chosen in turn 1 is remembered in turn 2).

    Args:
        user_input:   The raw text the user typed (any language).
        state_dict:   Current conversation state. See initialize_conversation_state().
        movies_df:    Pandas DataFrame with movie data.
        vectorizer:   Fitted TF-IDF vectorizer (loaded from models/).
        tfidf_matrix: Pre-computed TF-IDF matrix (optional; computed on the fly if None).

    Returns:
        (response_text, updated_state_dict)
        response_text is in the same language the user wrote in.
    """

    # ── Step 1: Domain normalization (always runs, any language) ─────────────
    # language_service.normalize() does two things regardless of language:
    #   a) Stopword-based language detection (reliable even for short text)
    #   b) Domain vocabulary mapping: "accion"→"action", "años 90"→[1990,1999]
    # This runs BEFORE translation so we always capture movie-domain terms.
    domain = _domain_normalize(user_input)

    # ── Step 2: Detect user language (combined approach) ──────────────────────
    # langdetect (statistical n-gram) is more accurate on long text.
    # Stopword detector is more reliable on short or accent-free text (e.g. "colombiana accion").
    # We use langdetect as primary and fall back to stopwords when it returns 'en'
    # but stopwords found a non-English match.
    lang_ngram = detect_language(user_input)
    lang_sw    = domain["detected_language"]
    ui_lang    = lang_ngram if lang_ngram != "en" else lang_sw
    state_dict["ui_language"] = ui_lang

    # ── Step 3: Translate input to English ────────────────────────────────────
    # Uses MarianMT (cached after first call). No-op when already English.
    if ui_lang != "en":
        english_input = translate_to_english(user_input, ui_lang)
    else:
        english_input = user_input

    # ── Step 4: Extract preferences from English text ─────────────────────────
    prefs = extract_preferences(english_input)

    # ── Step 5: Fill gaps with domain mapping results ─────────────────────────
    # If NLP found no genres after translation, use what language_service mapped
    # from the original text (e.g. "accion" → "action", "comedia" → "comedy").
    # Same logic for mood and year range.
    if not prefs["genres"] and domain["mapped_genres"]:
        prefs["genres"] = domain["mapped_genres"]
    if not prefs["mood"] and domain["mapped_moods"]:
        prefs["mood"] = domain["mapped_moods"]
    if not prefs.get("year_range") and domain["mapped_year_range"]:
        prefs["year_range"] = domain["mapped_year_range"]
    if not prefs.get("language") and domain["mapped_language"]:
        prefs["language"] = domain["mapped_language"]

    state_dict.update(prefs)

    # ── Step 4: Conversational prompting ──────────────────────────────────────
    # If we don't have a genre yet, ask for one.
    # The response is translated back to the user's language.
    if not state_dict.get("genres"):
        response = (
            "Hi! To help you find something, what genre would you like? "
            "(e.g. action, comedy, drama, thriller…)"
        )
        return _reply(response, ui_lang), state_dict

    # ── Step 5: Build query and get recommendations ───────────────────────────
    query_parts = state_dict.get("genres", []) + (state_dict.get("mood") or [])
    query_parts.append(state_dict.get("free_text", ""))
    query_text = " ".join(p for p in query_parts if p)

    # Pass a flat dict with the fields recommender_engine expects
    legacy_state = {
        "language":  state_dict.get("language"),
        "rating":    state_dict.get("min_rating") or state_dict.get("rating"),
        "year":      state_dict["year_range"][0] if state_dict.get("year_range") else None,
    }

    recommendations = recommend_on_the_fly(
        query_text, movies_df, vectorizer, tfidf_matrix, state_dict=legacy_state
    )

    if recommendations is None or recommendations.empty:
        response = "No matches found. Try different words or a broader search."
        return _reply(response, ui_lang), state_dict

    # ── Step 6: Build response with explanations ──────────────────────────────
    response = "Here are your recommendations:\n\n"
    for _, movie in recommendations.iterrows():
        explanation = generate_explanation(movie.to_dict(), state_dict)
        year_str = (
            f" ({int(movie['release_year'])})"
            if "release_year" in movie and movie["release_year"]
            else ""
        )
        response += f"**{movie['title']}**{year_str} — {movie.get('vote_average', 'N/A')}/10\n"
        response += f"> {explanation}\n\n"

    # ── Step 7: Translate response back ───────────────────────────────────────
    return _reply(response, ui_lang), state_dict


def _reply(english_text: str, ui_lang: str) -> str:
    """
    Translate an English response to the user's language.

    If the user is already English, returns the text unchanged.
    Otherwise uses translate_from_english() (MarianMT model).

    This helper exists so every early-return in chatbot_response()
    goes through the same translation path without repeating code.
    """
    if ui_lang == "en":
        return english_text
    return translate_from_english(english_text, ui_lang)


def initialize_conversation_state() -> dict:
    """
    Create a fresh state dict for a new conversation.

    Each key stores one piece of information about the user's preferences
    gathered across multiple turns. The Streamlit UI keeps this in
    st.session_state.chat_state and passes it to chatbot_response() each turn.

    Keys:
      genres      — list of genre strings e.g. ['action', 'comedy']
      language    — preferred movie language code e.g. 'es' (in what language the MOVIE is)
      year_range  — [start_year, end_year] e.g. [1990, 1999]
      mood        — list of mood strings e.g. ['dark', 'intense']
      min_rating  — minimum vote_average float e.g. 7.5
      similar_to  — reference movie title string
      free_text   — raw user text, kept for TF-IDF query building
      ui_language — ISO 639-1 code of the INTERFACE language (what the USER writes in)
                    e.g. 'es' means the user writes Spanish, so responses are Spanish too
    """
    return {
        "genres":      [],
        "language":    None,
        "year_range":  None,
        "mood":        [],
        "min_rating":  None,
        "similar_to":  None,
        "free_text":   "",
        "ui_language": None,   # set on first message by detect_language()
    }
