import re


def extract_preferences(user_input: str) -> dict:
    """
    Extract structured movie preferences from English (or lightly mixed) text.

    Returns a dict with keys: genres, language, year_range, mood, min_rating,
    similar_to, free_text.
    """
    user_input_lower = user_input.lower()

    genres_keywords = {
        "action":    ["action", "thrilling", "explosion", "fight", "adventure"],
        "comedy":    ["comedy", "funny", "hilarious", "laugh", "humor"],
        "drama":     ["drama", "emotional", "sad", "touching", "life"],
        "horror":    ["horror", "scary", "spooky", "creep", "terror"],
        "sci-fi":    ["sci-fi", "science fiction", "space", "alien", "future", "robot"],
        "romance":   ["romance", "romantic", "love", "couple", "heart"],
        "thriller":  ["thriller", "suspense", "mystery", "crime"],
        "animation": ["animation", "animated", "cartoon", "anime"],
        "fantasy":   ["fantasy", "magic", "wizard", "dragon"],
        "family":    ["family", "kids", "children", "child-friendly"],
    }

    moods_keywords = {
        "dark":      ["dark", "gloomy", "intense", "gritty"],
        "feel-good": ["feel-good", "feel good", "happy", "inspiring", "uplifting", "cheerful"],
        "intense":   ["intense", "fast-paced", "edge of my seat", "breathless"],
        "relaxing":  ["relaxing", "chill", "calm", "light-hearted"],
        "nostalgic": ["nostalgic", "old days", "classic feel"],
        "funny":     ["funny", "hilarious", "laugh", "humor", "comic"],
        "romantic":  ["romantic", "love story"],
    }

    languages = {
        "spanish": "es", "english": "en", "french": "fr",
        "italian": "it", "portuguese": "pt", "german": "de",
        "japanese": "ja", "korean": "ko",
    }

    preferences: dict = {
        "genres":    [],
        "language":  None,
        "year_range": None,
        "mood":      [],
        "min_rating": None,
        "similar_to": None,
        "free_text":  user_input,
    }

    for genre, keywords in genres_keywords.items():
        if genre in user_input_lower or any(w in user_input_lower for w in keywords):
            if genre not in preferences["genres"]:
                preferences["genres"].append(genre)

    for lang_name, lang_code in languages.items():
        if lang_name in user_input_lower:
            preferences["language"] = lang_code
            break

    for mood, keywords in moods_keywords.items():
        if mood in user_input_lower or any(w in user_input_lower for w in keywords):
            if mood not in preferences["mood"]:
                preferences["mood"].append(mood)

    # Explicit 4-digit year → exact range
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", user_input)
    if year_match:
        y = int(year_match.group(1))
        preferences["year_range"] = [y, y]

    # Decade pattern: "90s", "80s", "2000s"
    decade_match = re.search(r"\b(20\d{2}|[5-9]\d)s\b", user_input_lower)
    if decade_match and not preferences["year_range"]:
        raw = decade_match.group(1)
        if len(raw) == 2:
            start = int(f"19{raw}")
        else:
            start = int(raw)
        preferences["year_range"] = [start, start + 9]

    # Minimum rating
    rating_match = re.search(
        r"(?:rating|score|above|min(?:imum)?)\s*:?\s*(\d+(?:\.\d+)?)",
        user_input_lower,
    )
    if rating_match:
        preferences["min_rating"] = float(rating_match.group(1))

    # Similar-to reference: "similar to Inception", "like The Matrix"
    similar_match = re.search(
        r"(?:similar to|like|as good as)\s+([A-Z][^,.!?]+?)(?:[,.]|$)",
        user_input,
    )
    if similar_match:
        preferences["similar_to"] = similar_match.group(1).strip()

    return preferences
