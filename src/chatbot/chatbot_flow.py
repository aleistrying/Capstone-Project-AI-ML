"""
Chatbot flow module — conversational wrapper around the CineAssist pipeline.

Used by the Streamlit app directly. For API use, prefer backend.main.handle_user_message.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.nlp.nlp_preferences import extract_preferences
from src.utils.explanation_generator import generate_explanation
from src.recommender.recommender_engine import recommend_on_the_fly


def chatbot_response(
    user_input: str,
    state_dict: dict,
    movies_df,
    vectorizer,
    tfidf_matrix=None,
) -> tuple[str, dict]:
    """
    Process one user turn: extract preferences, recommend, explain.

    Returns (response_text, updated_state_dict).
    """
    prefs = extract_preferences(user_input)
    state_dict.update(prefs)

    # Build the query from any signal we have. free_text is the full user input,
    # so detected genres/mood just reinforce those tokens — we no longer dead-end
    # when no genre is recognized; we still run similarity on what was typed.
    query_parts = (state_dict.get("genres") or []) + (state_dict.get("mood") or [])
    query_parts.append(state_dict.get("free_text", ""))
    query_text = " ".join(p for p in query_parts if p).strip()

    if not query_text:
        return (
            "Hi! Tell me what you're in the mood for — e.g. a funny space "
            "adventure, a dark thriller from the 90s, something like Inception…",
            state_dict,
        )

    legacy_state = {
        "language": state_dict.get("language"),
        "rating":   state_dict.get("min_rating") or state_dict.get("rating"),
        "year": state_dict["year_range"][0] if state_dict.get("year_range") else None,
    }

    recommendations = recommend_on_the_fly(
        query_text, movies_df, vectorizer, tfidf_matrix, state_dict=legacy_state
    )

    if recommendations is None or recommendations.empty:
        return "No matches found. Try different words or a broader search.", state_dict

    response = "Here are your recommendations:\n\n"
    for _, movie in recommendations.iterrows():
        explanation = generate_explanation(movie.to_dict(), state_dict)
        year_str = f" ({int(movie['release_year'])})" if "release_year" in movie and movie["release_year"] else ""
        response += f"**{movie['title']}**{year_str} — {movie.get('vote_average', 'N/A')}/10\n"
        response += f"> {explanation}\n\n"

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
