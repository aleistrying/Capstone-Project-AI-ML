from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def recommend_on_the_fly(query_text, movies_df, vectorizer, tfidf_matrix, state_dict=None, top_n=5):
    """
    Genera recomendaciones de películas en tiempo real con filtrado de preferencias.
    """
    # 1. Vectorización de la consulta
    query_vector = vectorizer.transform([query_text])
    
    # 2. Cálculo de Similitud de Coseno
    # Se genera un puntaje para CADA película del dataset [5], [4]
    cosine_sim_scores = linear_kernel(query_vector, tfidf_matrix).flatten()
    
    # 3. Creación de un pool temporal de candidatos
    # En lugar de tomar solo 5, tomamos un pool más grande (ej. 50) para poder aplicar filtros
    temp_df = movies_df.copy()
    temp_df['similarity_score'] = cosine_sim_scores
    
    # 4. Aplicación de Filtros "Duros" (Refinamiento según Phase 4/Step 3) [4]
    if state_dict:
        # Filtro por Idioma (si se extrajo en la Fase 4)
        if state_dict.get('language'):
            temp_df = temp_df[temp_df['original_language'] == state_dict['language']]
            
        # Filtro por Año/Década
        if state_dict.get('year'):
            # Permite un margen de +/- 2 años para no ser demasiado restrictivo
            target_year = int(state_dict['year'])
            temp_df = temp_df[temp_df['release_year'].between(target_year - 2, target_year + 5)]
            
        # Filtro por Calificación mínima (Garantía de calidad del MVP) [6]
        if state_dict.get('rating'):
            temp_df = temp_df[temp_df['vote_average'] >= float(state_dict['rating'])]

    # 5. Ranking Final y Selección del Top N
    # Ordenamos por similitud y devolvemos los mejores resultados tras el filtrado [3]
    recommendations = temp_df.sort_values(by='similarity_score', ascending=False).head(top_n)
    
    # Verificación de resultados vacíos (Manejo de Incertidumbre)
    if recommendations.empty:
        # Si los filtros son muy estrictos, devolvemos los top 5 globales por similitud pura
        related_indices = cosine_sim_scores.argsort()[-top_n:][::-1]
        return movies_df.iloc[related_indices].copy()

    # Retornar columnas esenciales para la Fase 8 (Explicabilidad) [7], [8]
    return recommendations[['title', 'genres_list', 'release_year', 'vote_average', 'similarity_score', 'overview']]