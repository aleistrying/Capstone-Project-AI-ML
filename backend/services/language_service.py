"""
Domain-specific multilingual normalization for the movie domain.

Detects language, tokenizes, and maps domain vocabulary to canonical
English labels used by the recommendation dataset. Does not train a
general-purpose translator — scope is movie-domain terms only.
"""

import re
import unicodedata

# Language detection: stopword-based heuristic

_STOPWORDS = {
    "es": {
        "quiero",
        "una",
        "un",
        "de",
        "que",
        "me",
        "para",
        "con",
        "por",
        "los",
        "las",
        "como",
        "algo",
        "es",
        "en",
        "del",
        "al",
        "más",
        "muy",
        "película",
        "pelicula",
        "recomienda",
        "recomiéndame",
        "recomiendame",
        "ver",
        "busco",
        "dame",
    },
    "fr": {
        "je",
        "veux",
        "un",
        "une",
        "de",
        "du",
        "des",
        "les",
        "la",
        "le",
        "avec",
        "pour",
        "que",
        "est",
        "dans",
        "film",
        "quelque",
        "chose",
        "comme",
        "voire",
    },
    "pt": {
        "quero",
        "um",
        "uma",
        "de",
        "que",
        "para",
        "com",
        "por",
        "os",
        "as",
        "como",
        "algo",
        "em",
        "do",
        "da",
        "ao",
        "mais",
        "muito",
        "filme",
    },
}


def detect_language(text: str) -> str:
    """Return ISO 639-1 code ('en', 'es', 'fr', 'pt'). Defaults to 'en'."""
    tokens = set(re.findall(r"[a-záéíóúüñàèìòùâêîôûçã]+", text.lower()))
    scores = {lang: len(tokens & words) for lang, words in _STOPWORDS.items()}
    best_lang, best_score = max(scores.items(), key=lambda x: x[1])
    return best_lang if best_score >= 2 else "en"


# Domain synonym dictionaries: maps multilingual terms → canonical English
_GENRE_SYNONYMS: dict[str, list[str]] = {
    "comedy": [
        "comedy",
        "comedia",
        "chistosa",
        "graciosa",
        "graciosa",
        "comédie",
        "funny",
        "humor",
        "humorística",
        "humour",
    ],
    "action": [
        "action",
        "acción",
        "accion",
        "action",
        "ação",
        "thrilling",
        "explosion",
        "fight",
    ],
    "drama": ["drama", "emotional", "dramática", "dramatique", "sad", "touching"],
    "horror": [
        "horror",
        "terror",
        "scary",
        "spooky",
        "horreur",
        "miedo",
        "creepy",
        "aterradora",
    ],
    "sci-fi": [
        "sci-fi",
        "science fiction",
        "ciencia ficción",
        "ciencia ficcion",
        "science-fiction",
        "space",
        "alien",
        "future",
        "futurista",
        "robot",
    ],
    "romance": [
        "romance",
        "romántica",
        "romantica",
        "romantique",
        "love",
        "amor",
        "couple",
        "amorosa",
        "heart",
    ],
    "thriller": [
        "thriller",
        "suspenso",
        "suspense",
        "mystery",
        "misterio",
        "crime",
        "crimen",
        "intriga",
    ],
    "animation": [
        "animation",
        "animación",
        "animacion",
        "animé",
        "anime",
        "animated",
        "cartoon",
        "dibujos",
    ],
    "fantasy": [
        "fantasy",
        "fantasía",
        "fantasia",
        "fantaisie",
        "magic",
        "magia",
        "wizard",
        "dragon",
        "mágica",
    ],
    "family": [
        "family",
        "familia",
        "familial",
        "niños",
        "children",
        "kids",
        "infantil",
        "familiar",
    ],
    "adventure": ["adventure", "aventura", "aventure", "explorer"],
    "documentary": ["documentary", "documental", "documentaire", "real story"],
    "musical": ["musical", "música", "musique", "singing", "dance", "baile"],
    "western": ["western", "vaqueros", "cowboys"],
    "war": ["war", "guerra", "guerre", "battle", "batalla"],
    "crime": ["crime", "crimen", "policier", "detective", "policial"],
}

_MOOD_SYNONYMS: dict[str, list[str]] = {
    "dark": [
        "dark",
        "oscura",
        "sombre",
        "gloomy",
        "intense",
        "gritty",
        "oscuro",
        "pesada",
    ],
    "feel-good": [
        "feel-good",
        "happy",
        "feliz",
        "inspiring",
        "uplifting",
        "cheerful",
        "alegre",
        "positiva",
        "divertida",
        "feel good",
    ],
    "intense": [
        "intense",
        "fast-paced",
        "intensa",
        "breathless",
        "edge of my seat",
        "emocionante",
        "acelerada",
    ],
    "relaxing": [
        "relaxing",
        "chill",
        "calm",
        "light-hearted",
        "relajante",
        "tranquila",
        "ligera",
        "suave",
    ],
    "nostalgic": [
        "nostalgic",
        "nostálgica",
        "nostalgique",
        "classic feel",
        "clásica",
        "clásico",
        "old days",
    ],
    "romantic": [
        "romantic",
        "romántica",
        "romantique",
        "love story",
        "historia de amor",
        "amorosa",
    ],
    "scary": [
        "scary",
        "aterradora",
        "de miedo",
        "que da miedo",
        "terror",
        "espeluznante",
    ],
    "funny": [
        "funny",
        "chistosa",
        "cómica",
        "comica",
        "graciosa",
        "de risa",
        "jocosa",
        "comique",
    ],
}

_DECADE_MAP: dict[str, tuple[int, int]] = {
    "50s": (1950, 1959),
    "60s": (1960, 1969),
    "70s": (1970, 1979),
    "80s": (1980, 1989),
    "90s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020s": (2020, 2029),
    # Spanish/Portuguese decade phrases: "de los 2000", "los 90", "años 90" …
    "los 2000": (2000, 2009),
    "los 2010": (2010, 2019),
    "los 2020": (2020, 2029),
    "los 90": (1990, 1999),
    "los 80": (1980, 1989),
    "los 70": (1970, 1979),
    "los 60": (1960, 1969),
    "los 50": (1950, 1959),
    "años 50": (1950, 1959),
    "años 60": (1960, 1969),
    "años 70": (1970, 1979),
    "años 80": (1980, 1989),
    "años 90": (1990, 1999),
    "años 2000": (2000, 2009),
}

_LANGUAGE_SYNONYMS: dict[str, str] = {
    "spanish": "es",
    "español": "es",
    "espanol": "es",
    "en español": "es",
    "english": "en",
    "inglés": "en",
    "ingles": "en",
    "en inglés": "en",
    "french": "fr",
    "francés": "fr",
    "frances": "fr",
    "en francés": "fr",
    "italian": "it",
    "italiano": "it",
    "en italiano": "it",
    "portuguese": "pt",
    "portugués": "pt",
    "portugues": "pt",
    "german": "de",
    "alemán": "de",
    "aleman": "de",
    "japanese": "ja",
    "japonés": "ja",
    "japones": "ja",
    "korean": "ko",
    "coreano": "ko",
}


def _normalize_text(text: str) -> str:
    """Lowercase and strip accents for matching."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _mapped_labels(normalized: str, vocabulary: dict) -> list[str]:
    """Map any matching synonym to its canonical domain label."""
    return [
        label
        for label, synonyms in vocabulary.items()
        if any(_normalize_text(synonym) in normalized for synonym in synonyms)
    ]


def _mapped_year_range(raw_text: str, normalized: str) -> list[int] | None:
    """Map a localized decade expression or explicit year to a range."""
    for decade_key, year_range in _DECADE_MAP.items():
        if _normalize_text(decade_key) in normalized:
            return list(year_range)
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", raw_text)
    if year_match:
        year = int(year_match.group(1))
        return [year, year]
    return None


def normalize(raw_text: str) -> dict:
    """
    Detect language and map domain vocabulary to canonical English labels.

    Returns:
        {
            "detected_language": "es",
            "normalized_text": "funny family comedy movie from the 2000s",
            "mapped_genres": ["comedy", "family"],
            "mapped_moods": ["funny"],
            "mapped_year_range": [2000, 2009],
            "mapped_language": None,
        }
    """
    detected_lang = detect_language(raw_text)
    normalized = _normalize_text(raw_text)

    mapped_genres = _mapped_labels(normalized, _GENRE_SYNONYMS)
    mapped_moods = _mapped_labels(normalized, _MOOD_SYNONYMS)
    mapped_year_range = _mapped_year_range(raw_text, normalized)
    mapped_language = next(
        (
            code
            for phrase, code in _LANGUAGE_SYNONYMS.items()
            if _normalize_text(phrase) in normalized
        ),
        None,
    )

    genre_part = " ".join(mapped_genres) if mapped_genres else ""
    mood_part = " ".join(mapped_moods) if mapped_moods else ""
    year_part = (
        f"from the {mapped_year_range[0]}s"
        if mapped_year_range and mapped_year_range[0] != mapped_year_range[1]
        else (f"from {mapped_year_range[0]}" if mapped_year_range else "")
    )
    normalized_text = " ".join(
        p for p in [genre_part, mood_part, year_part, raw_text] if p
    )

    return {
        "detected_language": detected_lang,
        "normalized_text": normalized_text,
        "mapped_genres": mapped_genres,
        "mapped_moods": mapped_moods,
        "mapped_year_range": mapped_year_range,
        "mapped_language": mapped_language,
    }
