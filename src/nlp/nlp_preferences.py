"""Extract structured movie preferences from natural-language requests."""

import re

from src.contracts import Preferences

GENRE_KEYWORDS = {
    "action": ["action", "thrilling", "explosion", "fight", "adventure"],
    "comedy": ["comedy", "funny", "hilarious", "laugh", "humor"],
    "drama": ["drama", "emotional", "sad", "touching", "life"],
    "horror": ["horror", "scary", "spooky", "creep", "terror"],
    "sci-fi": ["sci-fi", "science fiction", "space", "alien", "future", "robot"],
    "romance": ["romance", "romantic", "love", "couple", "heart"],
    "thriller": ["thriller", "suspense", "mystery", "crime"],
    "animation": ["animation", "animated", "cartoon", "anime"],
    "fantasy": ["fantasy", "magic", "wizard", "dragon"],
    "family": ["family", "kids", "children", "child-friendly"],
}

MOOD_KEYWORDS = {
    "dark": ["dark", "gloomy", "intense", "gritty"],
    "feel-good": [
        "feel-good",
        "feel good",
        "happy",
        "inspiring",
        "uplifting",
        "cheerful",
    ],
    "intense": ["intense", "fast-paced", "edge of my seat", "breathless"],
    "relaxing": ["relaxing", "chill", "calm", "light-hearted"],
    "nostalgic": ["nostalgic", "old days", "classic feel"],
    "funny": ["funny", "hilarious", "laugh", "humor", "comic"],
    "romantic": ["romantic", "love story"],
}

LANGUAGES = {
    "spanish": "es",
    "english": "en",
    "french": "fr",
    "italian": "it",
    "portuguese": "pt",
    "german": "de",
    "japanese": "ja",
    "korean": "ko",
}


def _matching_labels(text: str, vocabulary: dict[str, list[str]]) -> list[str]:
    """Return labels whose label or synonym occurs in the normalized text."""
    return [
        label
        for label, keywords in vocabulary.items()
        if label in text or any(keyword in text for keyword in keywords)
    ]


def _extract_year_range(original: str, normalized: str) -> list[int] | None:
    """Extract an exact year or a decade range."""
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", original)
    if year_match:
        year = int(year_match.group(1))
        return [year, year]

    decade_match = re.search(r"\b(20\d{2}|[5-9]\d)s\b", normalized)
    if not decade_match:
        return None
    raw = decade_match.group(1)
    start = int(f"19{raw}") if len(raw) == 2 else int(raw)
    return [start, start + 9]


def _extract_rating(text: str) -> float | None:
    """Extract a requested minimum rating."""
    match = re.search(
        r"(?:rating|score|above|min(?:imum)?)\s*:?\s*(\d+(?:\.\d+)?)",
        text,
    )
    return float(match.group(1)) if match else None


def _extract_similar_title(text: str) -> str | None:
    """Extract the title following a 'similar to' expression."""
    match = re.search(
        r"(?:similar to|like|as good as)\s+([A-Z][^,.!?]+?)(?:[,.]|$)",
        text,
    )
    return match.group(1).strip() if match else None


def extract_preferences(user_input: str) -> Preferences:
    """Extract genres, language, year, mood, rating, and title preferences."""
    normalized = user_input.lower()
    language = next(
        (code for name, code in LANGUAGES.items() if name in normalized), None
    )
    return {
        "genres": _matching_labels(normalized, GENRE_KEYWORDS),
        "language": language,
        "year_range": _extract_year_range(user_input, normalized),
        "mood": _matching_labels(normalized, MOOD_KEYWORDS),
        "min_rating": _extract_rating(normalized),
        "similar_to": _extract_similar_title(user_input),
        "free_text": user_input,
    }
