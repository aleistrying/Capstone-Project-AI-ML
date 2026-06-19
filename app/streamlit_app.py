"""
CineAssist Streamlit UI.

Runs locally:  streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import joblib
import sys
import os
from pathlib import Path

# Resolve project root regardless of where the script is launched from
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.chatbot_flow import chatbot_response, initialize_conversation_state

# ---------------------------------------------------------------------------
# Asset loading (cached across reruns)
# ---------------------------------------------------------------------------

DATA_PATH    = PROJECT_ROOT / "data" / "processed"
MODELS_PATH  = PROJECT_ROOT / "models"


@st.cache_resource
def load_assets():
    csv_files = list(DATA_PATH.glob("*.csv"))
    if not csv_files:
        st.error(
            "No processed dataset found in data/processed/. "
            "Run the preprocessing notebook first."
        )
        st.stop()
    movies_df = pd.read_csv(csv_files[0])

    vec_path = MODELS_PATH / "tfidf_vectorizer.pkl"
    if not vec_path.exists():
        st.error("tfidf_vectorizer.pkl not found in models/. Run notebook 03_Vectorization first.")
        st.stop()
    vectorizer = joblib.load(vec_path)

    mat_path = MODELS_PATH / "tfidf_matrix.pkl"
    npz_path = MODELS_PATH / "tfidf_matrix.npz"
    if npz_path.exists():
        from scipy.sparse import load_npz
        tfidf_matrix = load_npz(str(npz_path))
    elif mat_path.exists():
        tfidf_matrix = joblib.load(mat_path)
    else:
        tfidf_matrix = None

    return movies_df, vectorizer, tfidf_matrix


movies_df, vectorizer, tfidf_matrix = load_assets()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CineAssist", page_icon="🎬", layout="centered")
st.title("🎬 CineAssist")
st.caption("Describe what you want to watch and get personalized movie recommendations.")

# ---------------------------------------------------------------------------
# Optional starter questions (sidebar)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Quick filters (optional)")
    st.caption("Fill in any or all — or just describe your mood in the chat.")
    starter_genre = st.selectbox(
        "Genre",
        ["", "Action", "Comedy", "Drama", "Horror", "Romance",
         "Sci-Fi", "Thriller", "Animation", "Fantasy", "Family"],
    )
    starter_mood = st.selectbox(
        "Mood",
        ["", "Feel-good", "Dark", "Intense", "Relaxing", "Nostalgic", "Romantic"],
    )
    starter_decade = st.selectbox(
        "Decade",
        ["", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"],
    )
    starter_lang = st.selectbox(
        "Original language",
        ["", "English", "Spanish", "French", "Italian", "Japanese", "Korean"],
    )
    min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5)

    # -----------------------------------------------------------------------
    # Model diagnostics — quick check that data and model line up
    # -----------------------------------------------------------------------
    with st.expander("🔧 Model diagnostics"):
        n_movies = len(movies_df)
        n_docs, n_terms = (tfidf_matrix.shape if tfidf_matrix is not None else (0, 0))
        vocab_size = len(getattr(vectorizer, "vocabulary_", {}))

        c1, c2 = st.columns(2)
        c1.metric("Movies (rows)", f"{n_movies:,}")
        c2.metric("Matrix rows", f"{n_docs:,}")
        c1.metric("Matrix terms", f"{n_terms:,}")
        c2.metric("Vocabulary", f"{vocab_size:,}")

        if tfidf_matrix is not None and n_docs == n_movies:
            st.success(f"Matrix rows match movies ({n_movies:,}) ✓")
        else:
            st.error(
                f"Mismatch: matrix has {n_docs:,} rows but data has {n_movies:,}. "
                "Rebuild with `python src/data/preprocess.py`."
            )

        _loaded = list(DATA_PATH.glob("*.csv"))
        st.caption(f"Dataset: `{_loaded[0].name if _loaded else 'none'}`")

        # Query inspector — see how text is cleaned and how many terms hit the vocab
        probe = st.text_input("Inspect a query", "batman superhero gotham")
        if probe:
            from src.utils.text_cleaning import clean_text
            cleaned = clean_text(probe)
            vec_tokens = set(getattr(vectorizer, "vocabulary_", {}))
            toks = cleaned.split()
            hits = [t for t in toks if t in vec_tokens]
            st.write(f"Cleaned → `{cleaned or '(empty)'}`")
            st.write(f"Vocab hits: **{len(hits)}/{len(toks)}** → {hits or '—'}")
            if toks and not hits:
                st.warning("No query terms in vocabulary → similarity will be ~0.")

# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_state" not in st.session_state:
    st.session_state.chat_state = initialize_conversation_state()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("e.g. I want a funny family movie from the 90s"):
    # Inject starter answers into the state before the turn
    state = st.session_state.chat_state
    if starter_genre:
        state["genres"] = state.get("genres") or [starter_genre.lower()]
    if starter_mood:
        state["mood"] = state.get("mood") or [starter_mood.lower()]
    if starter_decade:
        decade_start = int(starter_decade[:4])
        state["year_range"] = state.get("year_range") or [decade_start, decade_start + 9]
    if starter_lang:
        state["language"] = state.get("language") or starter_lang[:2].lower()
    if min_rating > 0:
        state["min_rating"] = state.get("min_rating") or min_rating

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text, updated_state = chatbot_response(
            prompt, state, movies_df, vectorizer, tfidf_matrix
        )
        st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.chat_state = updated_state

# ---------------------------------------------------------------------------
# Reset button
# ---------------------------------------------------------------------------

if st.button("Start over"):
    st.session_state.messages = []
    st.session_state.chat_state = initialize_conversation_state()
    st.rerun()
