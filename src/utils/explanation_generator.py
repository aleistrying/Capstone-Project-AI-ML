"""Create concise, evidence-based explanations for movie recommendations."""


def _genre_text(movie_data: dict) -> str:
    """Normalize a movie's genre representation for case-insensitive matching."""
    raw_genres = movie_data.get("genres_list") or movie_data.get("genres") or ""
    if isinstance(raw_genres, list):
        return " ".join(str(genre) for genre in raw_genres).lower()
    return str(raw_genres).lower()


def _genre_reason(movie_genres: str, user_prefs: dict) -> str | None:
    """Describe requested genres that occur in the movie metadata."""
    matches = [
        genre
        for genre in user_prefs.get("genres") or []
        if genre.lower() in movie_genres
    ]
    return f"matches your interest in **{', '.join(matches)}**" if matches else None


def _year_reason(movie_data: dict, user_prefs: dict) -> str | None:
    """Describe an exact-year or decade match."""
    movie_year = movie_data.get("release_year") or movie_data.get("year")
    year_range = user_prefs.get("year_range")
    if not movie_year or not year_range:
        return None
    start, end = year_range[0], year_range[-1]
    if not start <= int(movie_year) <= end:
        return None
    era = f"{start}s" if start != end else str(start)
    return f"fits the **{era}** era you asked for"


def _mood_reason(movie_data: dict, movie_genres: str, user_prefs: dict) -> str | None:
    """Describe the first requested mood supported by overview or genre text."""
    overview = str(movie_data.get("overview", "")).lower()
    mood = next(
        (
            item
            for item in user_prefs.get("mood") or []
            if item.lower() in overview or item.lower() in movie_genres
        ),
        None,
    )
    return f"captures the **{mood}** tone you mentioned" if mood else None


def _rating_value(movie_data: dict) -> float:
    """Return a safe numeric rating for explanation text."""
    raw_rating = movie_data.get("vote_average") or movie_data.get("rating") or 0
    try:
        return float(raw_rating)
    except (TypeError, ValueError):
        return 0.0


def _rating_reason(rating: float) -> str | None:
    """Describe ratings that provide a meaningful quality signal."""
    if rating >= 8.0:
        return f"is critically acclaimed (rating **{rating:.1f}**)"
    if rating >= 7.0:
        return f"has a solid rating of **{rating:.1f}**"
    return None


def generate_explanation(movie_data: dict, user_prefs: dict) -> str:
    """Generate a short English explanation for a movie recommendation."""
    movie_genres = _genre_text(movie_data)
    similar_to = user_prefs.get("similar_to")
    rating = _rating_value(movie_data)
    reasons = [
        _genre_reason(movie_genres, user_prefs),
        _year_reason(movie_data, user_prefs),
        _mood_reason(movie_data, movie_genres, user_prefs),
        f"has a narrative style similar to **{similar_to}**" if similar_to else None,
        _rating_reason(rating),
    ]
    present_reasons = [reason for reason in reasons if reason is not None]
    if not present_reasons:
        return f"Recommended based on its unique story and a rating of {rating:.1f}."
    if len(present_reasons) == 1:
        return f"Recommended because it {present_reasons[0]}."
    return (
        f"Recommended because it {', '.join(present_reasons[:-1])} "
        f"and {present_reasons[-1]}."
    )
