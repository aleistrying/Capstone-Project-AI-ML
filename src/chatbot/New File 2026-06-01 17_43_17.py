"""
Módulo del flujo principal del chatbot de recomendación de películas.
"""

# Importar funciones de otros módulos del proyecto
import sys
sys.path.append('/Workspace/Users/d.esteban.am@gmail.com/AI-and-ML-Laboratory-Databricks/src')

from nlp.nlp_preferences import extract_preferences
from utils.explanation_generator import generate_explanation

# TODO: Estas funciones deben ser creadas/importadas según tu implementación
# from recommender.recommender_engine import recommend_on_the_fly // It will be created on the fly becasuse to process the whole dataset is too long and requires too many resources
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
    # 1. Extract preferences from nlp.py file --> extract_preferences function
    prefs = extract_preferences(user_input)
    
    # 2. Actualization of the global state of the conversation
    state_dict.update(prefs)
    
    # 3. Aclaration logic, in the case of missing at least basic information (Control questions)
    if not state_dict.get('genres'):
        return "¡Hello! to help you to find something, ¿what gender would you like to see?", state_dict
    
    if state_dict.get('year') is None:
        return "¿Do you have any idea of year o time that you would like to see, or do you prefer something new?", 
        state_dict

    # # 4. Now, if the information is enought, call the RECOMENENDER>> Recomender on-the-fly
    # # query_text combine preferences for the search "on-the-fly"
    # query_text = f"{state_dict.get('free_text', '')} {' '.join(state_dict.get('genres', []))} {state_dict.get('mood') or ''}"
    
    # 4. Obtener las recomendaciones (asegúrate de tener implementado recommend_on_the_fly)
    recommendations = recommend_on_the_fly(query_text, movies_df, vectorizer, tfidf_matrix)

    
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
