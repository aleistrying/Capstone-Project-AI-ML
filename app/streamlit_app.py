"""
CineAssist Streamlit UI.

Runs locally:  streamlit run app/streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Resolve project root regardless of where the script is launched from
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.chatbot_flow import get_chat_recommendations, initialize_conversation_state
from app.movie_cards import inject_css, render_recommendations
from app.assets import dataset_label, load_assets_or_stop

# ---------------------------------------------------------------------------
# Asset loading — shared cache with every page under app/pages/ (see app/assets.py)
# ---------------------------------------------------------------------------

movies_df, vectorizer, tfidf_matrix = load_assets_or_stop()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CineAssist", page_icon="🎬", layout="centered")
inject_css()
st.title("🎬 CineAssist")
st.caption("Describe what you want to watch and get personalized movie recommendations.")

# ---------------------------------------------------------------------------
# Sidebar — CineAssist is chat-first, so the sidebar only holds developer tools.
# (The old "Quick filters" were removed: they had no effect because the chat
# pipeline re-derives all preferences from the typed message each turn.)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Developer tools")

    # -----------------------------------------------------------------------
    # Model diagnostics — quick check that data and model line up
    # -----------------------------------------------------------------------
    with st.expander("🔧 Model diagnostics"):
        n_movies = len(movies_df)
        n_docs, n_terms = tfidf_matrix.shape
        vocabulary = getattr(vectorizer, "vocabulary_", {})

        c1, c2 = st.columns(2)
        c1.metric("Movies (rows)", f"{n_movies:,}")
        c2.metric("Matrix rows", f"{n_docs:,}")
        c1.metric("Matrix terms", f"{n_terms:,}")
        c2.metric("Vocabulary", f"{len(vocabulary):,}")

        # load_assets_or_stop() already halts on a row-count mismatch, so reaching
        # this point means the two line up.
        st.success(f"Matrix rows match movies ({n_movies:,}) ✓")
        st.caption(f"Dataset: `{dataset_label()}`")

        # Query inspector — see how text is cleaned and how many terms hit the vocab
        probe = st.text_input("Inspect a query", "batman superhero gotham")
        if probe:
            from src.utils.text_cleaning import clean_text
            cleaned = clean_text(probe)
            toks = cleaned.split()
            # Probe the vocabulary dict directly — copying its ~857K keys into a
            # set on every rerun costs more memory than the container can spare.
            hits = [t for t in toks if t in vocabulary]
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

def _render_message(message: dict) -> None:
    """Render one stored turn: rich cards for assistant recs, markdown otherwise."""
    if message.get("recommendations") is not None:
        render_recommendations(
            message.get("intro", ""),
            message["recommendations"],
            message.get("meta"),
        )
    else:
        st.markdown(message.get("content", ""))


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        _render_message(message)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("e.g. I want a funny family movie from the 90s"):
    state = st.session_state.chat_state

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        intro, recs, updated_state, meta = get_chat_recommendations(
            prompt, state, movies_df, vectorizer, tfidf_matrix
        )
        render_recommendations(intro, recs, meta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "intro": intro,
            "recommendations": recs,
            "meta": meta,
            "content": intro,  # fallback text if recommendations is empty
        }
    )
    st.session_state.chat_state = updated_state

# ---------------------------------------------------------------------------
# Reset button
# ---------------------------------------------------------------------------

if st.button("Start over"):
    st.session_state.messages = []
    st.session_state.chat_state = initialize_conversation_state()
    st.rerun()
