"""
Módulo del flujo principal del chatbot de recomendación de películas CineAssist.
"""

import sys
# Asegura que Databricks encuentre tus módulos en la carpeta src
sys.path.append('/Workspace/Users/d.esteban.am@gmail.com/AI-and-ML-Laboratory-Databricks/src')

import pandas as pd
from nlp.nlp_preferences import extract_preferences
from utils.explanation_generator import generate_explanation
from recommender.recommender_engine import recommend_on_the_fly

def chatbot_response(user_input, state_dict, movies_df, vectorizer, tfidf_matrix):
    """
    Procesa la entrada del usuario, gestiona el estado y devuelve recomendaciones con explicaciones.
    """
    # 1. Extract preferences from nlp.py file --> extract_preferences function
    prefs = extract_preferences(user_input)
    
    # 2. Actualization of the global state of the conversation
    state_dict.update(prefs)
    
    # 3. Aclaration logic, in the case of missing at least basic information (Control questions)
    if not state_dict.get('genres'):
        return "¡Hello! to help you to find something, ¿what gender would you like to see?", state_dict
    
    if state_dict.get('year') is None:
        return "¿Do you have any idea of year o time that you would like to see, or do you prefer something new?", state_dict

    # 4. Motor de Recomendación "On-the-fly"
    # Combinamos el texto libre, géneros y mood para crear una consulta robusta
    query_text = f"{state_dict.get('free_text', '')} {' '.join(state_dict.get('genres', []))} {state_dict.get('mood') or ''}"
    
    # Llamada al motor que calcula la similitud de coseno en tiempo real
    recommendations = recommend_on_the_fly(query_text, movies_df, vectorizer, tfidf_matrix)
    
    if recommendations.empty:
        return "Lo siento, no encontré películas que coincidan exactamente. ¿Podrías intentar con otros términos?", state_dict

    # 5. Generación de Respuesta Final con Explicabilidad (Fase 8)
    # El MVP requiere: Top 5, puntaje de similitud y una breve explicación
    response = "### 🎬 ¡He encontrado estas opciones para ti!\n\n"
    
    for _, movie in recommendations.iterrows():
        # Generar la justificación personalizada para cada recomendación
        explanation = generate_explanation(movie, state_dict)
        
        response += f"**{movie['title']}** (Rating: {movie.get('vote_average', 'N/A')})\n"
        response += f"> 💡 {explanation}\n\n"
        
    return response, state_dict

def initialize_conversation_state():
    """
    Inicializa el esquema de preferencias para el chatbot.
    """
    return {
        "genres": [],
        "language": None,
        "year_range": None,
        "mood": None,
        "rating": None,
        "year": None,
        "free_text": ""
    }
