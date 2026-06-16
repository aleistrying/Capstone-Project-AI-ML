"""
Central controller for CineAssist.

Orchestrates language normalization → preference extraction → recommendation
→ explanation and returns a clean response dict.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.services import language_service, nlp_service, recommender_service, explanation_service


def handle_user_message(
    raw_text: str,
    form_data: dict | None = None,
    movies_df=None,
    top_n: int = 5,
) -> dict:
    """
    Full request pipeline: text in → recommendations out.

    Args:
        raw_text: Free-text user input (any supported language).
        form_data: Optional starter-question answers with keys:
                   genre, mood, year_range, language, min_rating, similar_to.
        movies_df: Pandas DataFrame with movie data. Must be provided at least
                   on the first call so the recommender can cache it.
        top_n: Number of movie recommendations to return.

    Returns:
        {
            "detected_language": "es",
            "normalized_query": "funny family comedy movie from the 2000s",
            "preferences": { ... },
            "recommendations": [ { title, year, genres, rating, score,
                                    poster_url, explanation } ],
            "metadata": { "model": "tfidf_cosine", "reranker": "none" }
        }
    """
    language_result = language_service.normalize(raw_text)

    merged_text = language_result["normalized_text"]
    prefs = nlp_service.extract(merged_text, form_data)

    # Prefer mapped values from language_service when nlp_service found nothing
    if not prefs.get("genres") and language_result.get("mapped_genres"):
        prefs["genres"] = language_result["mapped_genres"]
    if not prefs.get("mood") and language_result.get("mapped_moods"):
        prefs["mood"] = language_result["mapped_moods"]
    # Prefer the language service's decade range when it's wider than an exact-year match
    lang_yr = language_result.get("mapped_year_range")
    if lang_yr and (not prefs.get("year_range") or (lang_yr[0] != lang_yr[1] and prefs["year_range"][0] == prefs["year_range"][1])):
        prefs["year_range"] = lang_yr
    if not prefs.get("language") and language_result.get("mapped_language"):
        prefs["language"] = language_result["mapped_language"]

    raw_recommendations = recommender_service.recommend(prefs, movies_df=movies_df, top_n=top_n)

    recommendations = []
    for movie in raw_recommendations:
        explanation = explanation_service.explain(movie, prefs)
        recommendations.append({**movie, "explanation": explanation})

    return {
        "detected_language": language_result["detected_language"],
        "normalized_query": merged_text,
        "preferences": prefs,
        "recommendations": recommendations,
        "metadata": {
            "model": "tfidf_cosine",
            "reranker": "none",
        },
    }
