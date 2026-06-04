def generate_explanation(movie_data, user_prefs):
    """
    Genera una justificación profesional en lenguaje natural para una recomendación.
    
    Args:
        movie_data (pd.Series/dict): Datos de la película recomendada.
        user_prefs (dict): Preferencias extraídas (Phase 4).
    """
    reasons = []
    
    # 1. Coincidencia de Géneros (Phase 13: "Matches genre") [2]
    movie_genres = str(movie_data.get('genres_list', '')).lower()
    common_genres = [g for g in user_prefs.get('genres', []) if g.lower() in movie_genres]
    if common_genres:
        # Usamos negrita para resaltar en la UI de Streamlit [5]
        reasons.append(f"coincide con tu interés en el cine de **{', '.join(common_genres)}**")
    
    # 2. Coincidencia de Época o Año [4]
    movie_year = str(movie_data.get('release_year', ''))
    req_year = str(user_prefs.get('year', ''))
    if req_year and req_year[:3] in movie_year:
        reasons.append(f"es un título destacado de la época que buscas (**{movie_year}**)")

    # 3. Coincidencia de Mood y Temas (Phase 13: "Similar tone") [2, 6]
    mood = user_prefs.get('mood')
    overview = str(movie_data.get('overview', '')).lower()
    keywords = str(movie_data.get('keywords', '')).lower()
    
    if mood:
        if mood.lower() in overview or mood.lower() in keywords:
            reasons.append(f"presenta esa atmósfera **{mood}** que mencionaste")

    # 4. Referencia a Película Similar (Si el usuario dio una referencia) [4, 6]
    similar_to = user_prefs.get('similar_to')
    if similar_to:
        reasons.append(f"tiene una narrativa similar a **{similar_to}**")

    # 5. Factor de Calidad (Basado en Ratings de MovieLens/TMDB) [3, 7]
    vote_avg = movie_data.get('vote_average', 0)
    if vote_avg >= 8.0:
        reasons.append("es una de las obras mejor valoradas por la crítica")

    # --- Construcción Gramatical de la Frase ---
    if not reasons:
        return f"Te la recomiendo por su trama única y su sólida calificación de {vote_avg}."

    # Unir las razones usando comas y una "y" al final para que suene natural
    if len(reasons) == 1:
        explanation = f"Te la recomiendo porque {reasons}."
    else:
        explanation = f"Te la recomiendo porque {', '.join(reasons[:-1])} y {reasons[-1]}."
    
    return explanation