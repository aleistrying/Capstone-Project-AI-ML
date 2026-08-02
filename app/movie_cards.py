"""
Movie recommendation card rendering for the CineAssist Streamlit UI.

Fully self-contained: placeholder "posters" are CSS boxes, no external image
fetches or network calls, so the UI renders reliably offline / in a live demo.
"""

import html

import streamlit as st

# A small palette to give each placeholder poster a distinct, stable colour.
_POSTER_COLORS = [
    "#6C5CE7",
    "#00B894",
    "#0984E3",
    "#E17055",
    "#E84393",
    "#FDCB6E",
    "#00CEC9",
    "#D63031",
]


def inject_css() -> None:
    """Inject the card styling once per session render."""
    st.markdown(
        """
        <style>
        .cine-poster {
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            width: 100%; aspect-ratio: 2 / 3; border-radius: 10px;
            color: #fff; text-align: center; padding: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .cine-poster .emoji { font-size: 2rem; line-height: 1; }
        .cine-poster .initial { font-size: 2.4rem; font-weight: 700; margin-top: 4px; }
        .cine-rating {
            display: inline-block; background: #FFC107; color: #1a1a1a;
            font-weight: 700; padding: 2px 10px; border-radius: 12px;
            font-size: 0.85rem; margin-left: 6px;
        }
        .cine-chip {
            display: inline-block; background: rgba(108,92,231,0.15);
            color: #6C5CE7; border: 1px solid rgba(108,92,231,0.35);
            padding: 2px 10px; border-radius: 12px;
            font-size: 0.78rem; margin: 2px 4px 2px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _poster_html(title: str, index: int) -> str:
    color = _POSTER_COLORS[index % len(_POSTER_COLORS)]
    initial = (title.strip()[:1] or "?").upper()
    return (
        f'<div class="cine-poster" style="background:{color};">'
        f'<span class="emoji">🎬</span>'
        f'<span class="initial">{html.escape(initial)}</span>'
        f"</div>"
    )


def _render_card(rec: dict, index: int) -> None:
    poster_col, info_col = st.columns([1, 3])

    with poster_col:
        st.markdown(_poster_html(rec.get("title", "?"), index), unsafe_allow_html=True)

    with info_col:
        title = rec.get("title", "Untitled")
        year = rec.get("year")
        title_line = f"**{title}**" + (f" ({year})" if year else "")

        rating = rec.get("rating")
        if rating is not None:
            try:
                title_line += (
                    f' <span class="cine-rating">⭐ {float(rating):.1f}</span>'
                )
            except (TypeError, ValueError):
                pass
        st.markdown(title_line, unsafe_allow_html=True)

        genres = rec.get("genres") or []
        if genres:
            chips = "".join(
                f'<span class="cine-chip">{html.escape(str(g))}</span>' for g in genres
            )
            st.markdown(chips, unsafe_allow_html=True)

        # Similarity as a 0–100% match indicator. Clamp to [0, 1] for the bar.
        similarity = float(rec.get("similarity") or 0.0)
        pct = max(0.0, min(1.0, similarity))
        st.progress(pct, text=f"Match: {pct * 100:.0f}%")

        with st.expander("Why this & overview"):
            explanation = rec.get("explanation")
            if explanation:
                st.markdown(explanation)
            overview = rec.get("overview")
            if overview:
                st.caption(overview)
            else:
                st.caption("No overview available.")


def render_recommendations(
    intro: str, recs: list[dict], meta: dict | None = None
) -> None:
    """
    Render an assistant turn: intro line, optional low-confidence notice, and cards.

    Args:
        intro: short message shown above the cards.
        recs: list of recommendation dicts from get_chat_recommendations().
        meta: {"broadened": bool, "max_similarity": float} or None.
    """
    if intro:
        st.markdown(intro)

    if not recs:
        return

    if meta and meta.get("broadened"):
        st.info(
            "No strong matches — showing the closest movies I have. "
            "Try different or more specific words (genre, mood, decade)."
        )

    for i, rec in enumerate(recs):
        _render_card(rec, i)
        if i < len(recs) - 1:
            st.divider()
