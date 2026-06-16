"""
NLP service: wraps the core preference extractor and merges form data.

Accepts language-normalized text from language_service plus optional
starter-question answers and returns a unified preference dict.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.nlp.nlp_preferences import extract_preferences


def extract(normalized_text: str, form_data: dict | None = None) -> dict:
    """
    Extract structured preferences from normalized text, then overlay form_data.

    Args:
        normalized_text: English-normalized text from language_service.
        form_data: Optional dict with keys genre, mood, year_range, language,
                   min_rating, similar_to.

    Returns:
        {
            "genres": [...],
            "mood": [...],
            "year_range": [start, end] | None,
            "language": "en" | None,
            "min_rating": float | None,
            "similar_to": str | None,
            "free_text": str,
        }
    """
    prefs = extract_preferences(normalized_text)

    # Normalize the legacy "year" field to "year_range" if needed
    if "year" in prefs and prefs["year"] and "year_range" not in prefs:
        y = int(prefs["year"])
        prefs["year_range"] = [y, y]
    prefs.setdefault("year_range", None)
    prefs.setdefault("min_rating", prefs.pop("rating", None))
    prefs.setdefault("similar_to", None)
    prefs.pop("year", None)

    if form_data:
        if form_data.get("genre") and not prefs["genres"]:
            prefs["genres"] = [form_data["genre"]]
        if form_data.get("mood") and not prefs["mood"]:
            prefs["mood"] = [form_data["mood"]]
        if form_data.get("year_range") and not prefs["year_range"]:
            prefs["year_range"] = form_data["year_range"]
        if form_data.get("language") and not prefs["language"]:
            prefs["language"] = form_data["language"]
        if form_data.get("min_rating") and not prefs["min_rating"]:
            prefs["min_rating"] = float(form_data["min_rating"])
        if form_data.get("similar_to") and not prefs["similar_to"]:
            prefs["similar_to"] = form_data["similar_to"]

    return prefs
