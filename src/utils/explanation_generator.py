def generate_explanation(movie_data: dict, user_prefs: dict) -> str:
    """
    Generate a short English explanation for a movie recommendation.

    Args:
        movie_data: Dict or Series with keys title, genres / genres_list,
                    release_year / year, vote_average / rating, overview.
        user_prefs: Preferences dict from nlp_service.extract().
    """
    reasons = []

    # Genre overlap
    raw_genres = movie_data.get("genres_list") or movie_data.get("genres") or ""
    if isinstance(raw_genres, list):
        movie_genres_str = " ".join(str(g) for g in raw_genres).lower()
    else:
        movie_genres_str = str(raw_genres).lower()

    requested_genres = user_prefs.get("genres") or []
    matching_genres = [g for g in requested_genres if g.lower() in movie_genres_str]
    if matching_genres:
        reasons.append(f"matches your interest in **{', '.join(matching_genres)}**")

    # Year / decade match
    movie_year = movie_data.get("release_year") or movie_data.get("year")
    year_range = user_prefs.get("year_range")
    if movie_year and year_range:
        start, end = year_range[0], year_range[-1]
        if start <= int(movie_year) <= end:
            decade_label = f"{start}s" if start != end else str(start)
            reasons.append(f"fits the **{decade_label}** era you asked for")

    # Mood match
    moods = user_prefs.get("mood") or []
    overview = str(movie_data.get("overview", "")).lower()
    for mood in moods:
        if mood.lower() in overview or mood.lower() in movie_genres_str:
            reasons.append(f"captures the **{mood}** tone you mentioned")
            break

    # Similar-to reference
    similar_to = user_prefs.get("similar_to")
    if similar_to:
        reasons.append(f"has a narrative style similar to **{similar_to}**")

    # Quality signal
    rating = movie_data.get("vote_average") or movie_data.get("rating") or 0
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        rating = 0.0
    if rating >= 8.0:
        reasons.append(f"is critically acclaimed (rating **{rating:.1f}**)")
    elif rating >= 7.0:
        reasons.append(f"has a solid rating of **{rating:.1f}**")

    if not reasons:
        return f"Recommended based on its unique story and a rating of {rating:.1f}."

    if len(reasons) == 1:
        return f"Recommended because it {reasons[0]}."
    return f"Recommended because it {', '.join(reasons[:-1])} and {reasons[-1]}."
