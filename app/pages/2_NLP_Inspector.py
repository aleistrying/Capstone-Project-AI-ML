"""
CineAssist — NLP Pipeline Inspector

Shows each stage of the NLP extraction pipeline for any input text:
  1. Language detection + domain normalization  (language_service)
  2. Preference extraction                       (nlp_service → nlp_preferences)
  3. Translation status                          (translation_service — stub)

This page documents WHERE entity extraction happens and exposes it
for debugging / demos.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services import language_service, nlp_service, translation_service

st.set_page_config(
    page_title="CineAssist — NLP Inspector",
    page_icon="🔍",
    layout="wide",
)
st.title("🔍 NLP Pipeline Inspector")
st.caption(
    "Trace any input text through the full NLP pipeline: "
    "language detection → domain normalization → preference extraction."
)

# ---------------------------------------------------------------------------
# Sample queries
# ---------------------------------------------------------------------------

SAMPLES = {
    "(type your own)": "",
    "English — action with explosions": "I want an action movie with lots of fights and explosions from the 90s",
    "Spanish — comedy family 2000s": "quiero una pelicula chistosa para familia de los 2000",
    "Spanish — terror 80s": "busco algo de terror o suspenso de los 80",
    "French — romance": "je cherche un film romantique avec de l'humour",
    "Portuguese — animation": "um filme de animação para crianças muito legal",
    "Mixed — sci-fi with year": "I want sci-fi movies from 2010s, películas de ciencia ficción",
    "Min rating constraint": "show me movies with rating above 7.5 action thriller",
}

with st.sidebar:
    st.subheader("Sample Queries")
    selected = st.selectbox("Load a sample", list(SAMPLES.keys()))
    st.divider()
    st.caption("NLP Pipeline Order")
    st.markdown("""
1. **translation_service** *(stub)*
   Full sentence translation — not yet implemented.
   Planned: Google Translate / DeepL / HuggingFace.

2. **language_service.normalize()**
   Detects language via stopword heuristics.
   Maps domain vocabulary to canonical English
   (genre, mood, decade, language synonyms).

3. **nlp_service.extract()**
   Wraps `src/nlp/nlp_preferences.py`.
   Regex + keyword matching on the normalized text.
   Returns: `genres`, `mood`, `year_range`, `language`,
   `min_rating`, `similar_to`, `free_text`.
""")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

default_text = SAMPLES[selected]
user_input = st.text_area(
    "Input text",
    value=default_text,
    height=80,
    placeholder="Type a movie request in any language…",
)

run = st.button("Analyze", type="primary", disabled=not user_input.strip())

if run and user_input.strip():
    # ── Stage 0: Translation (stub) ──────────────────────────────────────────
    with st.expander("Stage 0 — Translation (stub)", expanded=True):
        trans = translation_service.translate(user_input)
        col1, col2, col3 = st.columns([2, 1, 1])
        col1.write("**Input text**")
        col1.code(trans["original_text"])
        col2.metric("Detected language (stub)", trans["detected_language"])
        col3.metric("Was translated", str(trans["was_translated"]))
        st.warning(
            "Translation is a stub — `translate_to_english()` currently returns input unchanged. "
            "See `src/translation/translator.py` for the TODO integration points."
        )

    # ── Stage 1: Language service ────────────────────────────────────────────
    with st.expander("Stage 1 — Language detection + domain normalization", expanded=True):
        lang_result = language_service.normalize(user_input)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected language", lang_result.get("detected_language", "—"))
            st.write("**Normalized text**")
            st.code(lang_result.get("normalized_text", ""))
        with col2:
            st.write("**Mapped entities**")
            mapped = {
                "genres":     lang_result.get("mapped_genres", []),
                "moods":      lang_result.get("mapped_moods", []),
                "year_range": lang_result.get("mapped_year_range"),
                "language":   lang_result.get("mapped_language"),
            }
            st.json(mapped)

    # ── Stage 2: NLP service ─────────────────────────────────────────────────
    with st.expander("Stage 2 — Preference extraction (NLP)", expanded=True):
        prefs = nlp_service.extract(lang_result.get("normalized_text", user_input))
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Extracted preferences**")
            st.json(prefs)
        with col2:
            st.write("**Extraction notes**")
            notes = []
            if prefs.get("genres"):
                notes.append(f"Genres found: {prefs['genres']}")
            if prefs.get("mood"):
                notes.append(f"Mood: {prefs['mood']}")
            if prefs.get("year_range"):
                yr = prefs["year_range"]
                notes.append(f"Year range: {yr[0]}–{yr[1]}" if isinstance(yr, (list, tuple)) and len(yr) == 2 else f"Year: {yr}")
            if prefs.get("language"):
                notes.append(f"Language filter: {prefs['language']}")
            if prefs.get("min_rating"):
                notes.append(f"Min rating: {prefs['min_rating']}")
            if prefs.get("similar_to"):
                notes.append(f"Similar to: {prefs['similar_to']}")
            if not notes:
                notes.append("No structured preferences extracted — will use free-text TF-IDF query.")
            for note in notes:
                st.markdown(f"- {note}")

    # ── Stage 3: Merged pipeline view ────────────────────────────────────────
    with st.expander("Stage 3 — Merged preference object (backend/main.py)", expanded=False):
        merged_genres = list(dict.fromkeys(
            (prefs.get("genres") or []) + (lang_result.get("mapped_genres") or [])
        ))
        merged_moods = list(dict.fromkeys(
            (prefs.get("mood") or []) + (lang_result.get("mapped_moods") or [])
        ))
        lang_yr = lang_result.get("mapped_year_range")
        nlp_yr  = prefs.get("year_range")
        if lang_yr and (not nlp_yr or (
            isinstance(lang_yr, (list, tuple)) and
            isinstance(nlp_yr, (list, tuple)) and
            lang_yr[0] != lang_yr[1] and
            nlp_yr[0] == nlp_yr[1]
        )):
            final_yr = lang_yr
            yr_source = "language_service (decade range preferred over exact-year match)"
        else:
            final_yr = nlp_yr
            yr_source = "nlp_service"

        merged = {
            "genres":      merged_genres,
            "mood":        merged_moods,
            "year_range":  final_yr,
            "year_source": yr_source,
            "language":    prefs.get("language") or lang_result.get("mapped_language"),
            "min_rating":  prefs.get("min_rating"),
            "similar_to":  prefs.get("similar_to"),
            "free_text":   prefs.get("free_text", ""),
        }
        st.json(merged)
        st.caption(
            "This is the object passed to `recommender_service.recommend()`. "
            "Year range source shows which stage won when both extracted a year."
        )

    # ── Architecture callout ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Where NLP entity extraction happens")
    st.markdown("""
| Stage | File | What it extracts |
|-------|------|-----------------|
| Language detection | `backend/services/language_service.py` | Language (EN/ES/FR/PT), genre synonyms, mood synonyms, decade expressions |
| NLP extraction | `src/nlp/nlp_preferences.py` | Genres, moods, year range, min rating, similar_to, free_text via regex + keyword lists |
| Translation *(stub)* | `src/translation/translator.py` | Full sentence translation — not yet implemented |
| Merge + priority | `backend/main.py` | Combines both, resolves conflicts (e.g. decade range vs exact year) |
    """)

elif not run:
    st.info("Enter a query above and click **Analyze** to trace it through the NLP pipeline.")
