"""
CineAssist — NLP Pipeline Inspector

Traces any input text through the EXACT pipeline the live chat app runs, stage
by stage, using the centralized `run_pipeline()` in
`src/chatbot/chatbot_flow.py`. There is no duplicated logic here: whatever the
recommender does, this page shows — real translation, the query fed to TF-IDF,
the filters applied, and the resulting cosine-similarity recommendations.

  0. Language detection + translation  (langdetect + MarianMT, real)
  1. Domain normalization              (language_service)
  2. Preference extraction             (nlp_preferences)
  3. Query building                    (keyword_extractor → text fed to TF-IDF)
  4. Filters                           (language / year / min-rating)
  5. Cosine similarity recommendations (recommender_engine)
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.assets import load_assets_or_stop
from src.chatbot.chatbot_flow import run_pipeline, initialize_conversation_state

st.set_page_config(
    page_title="CineAssist — NLP Inspector",
    page_icon="🔍",
    layout="wide",
)
st.title("🔍 NLP Pipeline Inspector")
st.caption(
    "Trace any input through the exact pipeline the chat app runs: "
    "language detection → translation → normalization → preference extraction → "
    "query building → TF-IDF cosine similarity → recommendations."
)

# ---------------------------------------------------------------------------
# Assets — same models/data the live app and Metrics page use, one shared cache
# ---------------------------------------------------------------------------

movies_df, vectorizer, tfidf_matrix = load_assets_or_stop()

# ---------------------------------------------------------------------------
# Sample queries + sidebar reference
# ---------------------------------------------------------------------------

SAMPLES = {
    "(type your own)": "",
    "English — action 90s": "I want an action movie with lots of fights and explosions from the 90s",
    "Spanish — comedy family 2000s": "quiero una pelicula chistosa para familia de los 2000",
    "Spanish — terror 80s": "busco algo de terror o suspenso de los 80",
    "French — romance": "je cherche un film romantique avec de l'humour",
    "Portuguese — animation": "um filme de animação para crianças muito legal",
    "Similar-to query": "something like Inception",
    "Min rating constraint": "show me action thriller movies with rating above 7.5",
}

with st.sidebar:
    st.subheader("Sample Queries")
    selected = st.selectbox("Load a sample", list(SAMPLES.keys()))
    st.divider()
    st.caption("Pipeline (single source of truth)")
    st.markdown("""
`src/chatbot/chatbot_flow.py → run_pipeline()`

0. **Translation** — `langdetect` + Helsinki-NLP MarianMT.
   English input is passed through unchanged.
1. **Domain normalization** — `language_service`.
   Maps genre/mood/decade synonyms to canonical English.
2. **Preference extraction** — `nlp_preferences`.
   Regex + keyword matching → structured prefs.
3. **Query building** — `keyword_extractor`.
   The focused text that TF-IDF cosine runs against.
4. **Filters** — language / year / min-rating.
5. **Recommendation** — `recommender_engine`.
   TF-IDF cosine + soft-decade ranking.
""")
    st.caption("For aggregate P@5 / Recall / MRR see the **Metrics** page.")

default_text = SAMPLES[selected]
user_input = st.text_area(
    "Input text",
    value=default_text,
    height=90,
    placeholder="Type a movie request in any language…",
)

run = st.button("Analyze", type="primary", disabled=not user_input.strip())

if not run:
    st.info(
        "Enter a query above and click **Analyze** to trace it through the full pipeline."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Run the ONE centralized pipeline
# ---------------------------------------------------------------------------

with st.spinner(
    "Running pipeline (first non-English query downloads a translation model)…"
):
    trace = run_pipeline(
        user_input,
        initialize_conversation_state(),
        movies_df,
        vectorizer,
        tfidf_matrix,
    )

# ── Top-line summary metrics ────────────────────────────────────────────────
st.subheader("Summary")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Detected language", trace.ui_language)
m2.metric("Translated?", "Yes" if trace.was_translated else "No")
m3.metric("Query terms in vocab", f"{len(trace.vocab_hits)}/{trace.vocab_total}")
m4.metric("Top match score", f"{trace.max_similarity:.4f}")
m5.metric("Results", len(trace.recommendations))

if trace.status == "empty_query":
    st.warning(
        "No usable query terms were extracted — the app would reply with a "
        "greeting/prompt instead of running the recommender."
    )
elif trace.status == "no_matches":
    st.warning("The filters removed every candidate and no fallback matched.")

# ── Stage 0: Translation (real) ─────────────────────────────────────────────
with st.expander("Stage 0 — Language detection & translation", expanded=True):
    if not trace.translation_available:
        st.info(
            "Neural translation deps (torch / transformers / langdetect) are not "
            "installed in this environment, so the pipeline ran in English-only "
            "mode. Language was detected with the stopword heuristic in "
            "`language_service`. Install `requirements.txt` to enable MarianMT."
        )
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original input**")
        st.code(trace.raw_input)
        st.caption(
            f"Stopword-detected language (language_service): `{trace.domain.get('detected_language', '—')}`"
        )
    with col2:
        st.write("**English input (fed to NLP)**")
        st.code(trace.english_input)
        if trace.was_translated:
            st.success(f"Translated `{trace.ui_language}` → `en` via MarianMT.")
        elif trace.ui_language == "en":
            st.caption("Input already English — translation skipped (no-op).")
        else:
            st.caption(f"Detected `{trace.ui_language}` but text returned unchanged.")

# ── Stage 1: Domain normalization ───────────────────────────────────────────
with st.expander("Stage 1 — Domain normalization (language_service)", expanded=True):
    st.write("**Mapped entities** (multilingual synonym → canonical English)")
    st.json(
        {
            "genres": trace.domain.get("mapped_genres", []),
            "moods": trace.domain.get("mapped_moods", []),
            "year_range": trace.domain.get("mapped_year_range"),
            "language": trace.domain.get("mapped_language"),
        }
    )
    st.caption(
        "These backfill any preference the extractor missed on the English text "
        "(e.g. genre words that only appeared in the original language)."
    )

# ── Stage 2: Preference extraction ──────────────────────────────────────────
with st.expander("Stage 2 — Preference extraction (nlp_preferences)", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Structured preferences**")
        st.json(trace.prefs)
    with col2:
        st.write("**Notes**")
        notes = []
        if trace.prefs.get("genres"):
            notes.append(f"Genres: {trace.prefs['genres']}")
        if trace.prefs.get("mood"):
            notes.append(f"Mood: {trace.prefs['mood']}")
        yr = trace.prefs.get("year_range")
        if yr:
            notes.append(
                f"Year range: {yr[0]}–{yr[1]}"
                if isinstance(yr, (list, tuple)) and len(yr) == 2
                else f"Year: {yr}"
            )
        if trace.prefs.get("language"):
            notes.append(f"Movie-language filter: {trace.prefs['language']}")
        if trace.prefs.get("min_rating"):
            notes.append(f"Min rating: {trace.prefs['min_rating']}")
        if trace.prefs.get("similar_to"):
            notes.append(f"Similar to: {trace.prefs['similar_to']}")
        if not notes:
            notes.append("No structured filters — relies on the free-text query.")
        for n in notes:
            st.markdown(f"- {n}")

# ── Stage 3: Query building (what cosine runs against) ──────────────────────
with st.expander(
    "Stage 3 — Query building (what TF-IDF cosine runs against)", expanded=True
):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Focused query** (genres + mood + extracted keywords)")
        st.code(trace.query_text or "(empty)")
        st.write("**Cleaned query** (same cleaning as the movie corpus)")
        st.code(trace.cleaned_query or "(empty)")
    with col2:
        st.write("**Vocabulary coverage**")
        st.metric(
            "Terms matching the TF-IDF vocabulary",
            f"{len(trace.vocab_hits)}/{trace.vocab_total}",
        )
        st.write(trace.vocab_hits or "—")
        if trace.vocab_total and not trace.vocab_hits:
            st.error(
                "No query term is in the vocabulary → every cosine score will be ~0."
            )
    st.caption(
        "Cosine similarity is computed between this cleaned query vector and each "
        "movie's TF-IDF vector. Only terms present in the vocabulary contribute — "
        "this is why coverage matters."
    )
    with st.popover("Keyword-extractor detail"):
        st.json(trace.keyword_extraction)

# ── Stage 4: Filters ─────────────────────────────────────────────────────────
with st.expander("Stage 4 — Filters applied inside the recommender", expanded=True):
    st.write("**Filter state passed to `recommend_on_the_fly()`**")
    st.json(trace.filters)
    st.markdown("""
- **language** → hard filter on `original_language` (rows in other languages are dropped).
- **year** → `year_mode="soft"`: not a hard cut. Movies inside the decade keep full
  cosine; films outside are multiplied by a decade-distance decay, so a strong 1999
  match still surfaces for a "90s" query.
- **rating** → hard filter: `vote_average ≥ min_rating`.

If every candidate is filtered out, the engine falls back to the closest matches by
raw cosine so the UI is never empty.
    """)

# ── Stage 5: Cosine similarity recommendations ──────────────────────────────
with st.expander("Stage 5 — Cosine similarity recommendations", expanded=True):
    if trace.recommendations:
        rec_rows = [
            {
                "#": i + 1,
                "Title": r["title"],
                "Year": r["year"],
                "Rating": r["rating"],
                "Match score": round(r["similarity"], 4),
                "Genres": ", ".join(r["genres"][:3]),
            }
            for i, r in enumerate(trace.recommendations)
        ]
        st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

        st.bar_chart(
            pd.DataFrame(rec_rows).set_index("Title")[["Match score"]],
            horizontal=True,
        )

        if trace.broadened:
            st.warning(
                f"Top score {trace.max_similarity:.4f} is below the confidence "
                "threshold (0.02) — the app labels these as broadened/low-confidence."
            )

        st.write("**Why these? (English explanations)**")
        for r in trace.recommendations:
            st.markdown(f"- **{r['title']}** — {r['explanation']}")
    else:
        st.info("No recommendations produced for this query.")

# ---------------------------------------------------------------------------
# Architecture callout
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Where each stage lives (single centralized pipeline)")
st.markdown("""
| Stage | File | Responsibility |
|-------|------|----------------|
| Orchestration | `src/chatbot/chatbot_flow.py` → `run_pipeline()` | Runs every stage; **this page and the chat app both call it** |
| Translation | `src/translation/` (`lang_detector`, `translator`) | langdetect + Helsinki-NLP MarianMT |
| Domain normalization | `backend/services/language_service.py` | Multilingual synonym → canonical English |
| Preference extraction | `src/nlp/nlp_preferences.py` | Genres, mood, year range, min rating, similar_to |
| Query building | `src/nlp/keyword_extractor.py` | Focused query + synonym expansion for TF-IDF |
| Recommendation | `src/recommender/recommender_engine.py` | TF-IDF cosine + soft-decade ranking + filters |
""")
