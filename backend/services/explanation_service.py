"""
Explanation service: generates a natural-language explanation for each
recommendation based on how it matches the user's preferences.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.explanation_generator import generate_explanation


def explain(movie: dict, preferences: dict) -> str:
    """
    Generate a short English explanation for one recommended movie.

    Args:
        movie: Dict from recommender_service.recommend() — keys: title, year,
               genres, rating, score, overview.
        preferences: Structured preference dict from nlp_service.extract().

    Returns:
        Human-readable explanation string.
    """
    return generate_explanation(movie, preferences)
