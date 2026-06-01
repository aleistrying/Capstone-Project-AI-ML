"""
Módulo del flujo principal del chatbot de recomendación de películas.
"""

# Importar funciones de otros módulos del proyecto
import sys
sys.path.append('/Workspace/Users/d.esteban.am@gmail.com/AI-and-ML-Laboratory-Databricks/src')

from nlp.nlp_preferences import extract_preferences

# TODO: Estas funciones deben ser creadas/importadas según tu implementación
# from recommender.recommender_engine import recommend_on_the_fly
# from utils.explanation_generator import generate_explanation


def chatbot_response(user_input, state_dict, movies_df, vectorizer, tfidf_matrix):
    """
    Función principal del chatbot que procesa la entrada del usuario y genera recomendaciones.
    
    Args:
        user_input (str): Texto ingresado por el usuario
        state_dict (dict): Diccionario con el estado de la conversación
        movies_df (pd.DataFrame): DataFrame con las películas
        vectorizer: Vectorizador TF-IDF entrenado
        tfidf_matrix: Matriz TF-IDF de las películas
        
    Returns:
        tuple: (respuesta del chatbot, estado actualizado)
    """
    # 1. Extraer preferencias usando el módulo de la Fase 4
    prefs = extract_preferences(user_input)
    
    # 2. Actualizar el estado global de la conversación
    state_dict.update(prefs)
    
    # 3. Lógica de aclaración (Preguntas de control)
    if not state_dict.get('genres'):
        return "¡Hola! Para ayudarte mejor, ¿qué género te apetece ver hoy?", state_dict
    
    if state_dict.get('year') is None:
        return "¿Tienes alguna preferencia de época? Por ejemplo, algo de los '90s' o algo 'reciente'.", state_dict

    # 4. Si tenemos suficiente info, llamar al recomendador (Fase 3)
    # query_text combina las preferencias para la búsqueda "on-the-fly"
    query_text = f"{state_dict.get('free_text', '')} {' '.join(state_dict.get('genres', []))} {state_dict.get('mood') or ''}"
    
    # TODO: Implementar recommend_on_the_fly
    # recommendations = recommend_on_the_fly(query_text, movies_df, vectorizer, tfidf_matrix)
    
    # Por ahora, retornamos un mensaje placeholder
    response = f"Buscaría películas de: {', '.join(state_dict.get('genres', []))}"
    if state_dict.get('year'):
        response += f" del año {state_dict.get('year')}"
    
    # 5. Generar respuesta final con explicaciones (cuando tengas recommend_on_the_fly)
    # response = "¡He encontrado estas opciones para ti!\n"
    # for _, movie in recommendations.iterrows():
    #     explanation = generate_explanation(movie, state_dict)
    #     response += f"- {movie['title']}: {explanation}\n"
        
    return response, state_dict


# Función auxiliar para inicializar el estado
def initialize_conversation_state():
    """
    Inicializa el diccionario de estado de la conversación.
    
    Returns:
        dict: Estado inicial vacío
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
