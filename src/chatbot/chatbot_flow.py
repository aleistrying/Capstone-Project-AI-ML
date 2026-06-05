"""
Módulo del flujo principal del chatbot de recomendación de películas CineAssist.
"""

import sys
# Functions and modules
sys.path.append('/Workspace/Users/d.esteban.am@gmail.com/AI-and-ML-Laboratory-Databricks/src')

import pandas as pd
from nlp.nlp_preferences import extract_preferences
from utils.explanation_generator import generate_explanation
from recommender.recommender_engine import recommend_on_the_fly

def chatbot_response(user_input, state_dict, movies_df, vectorizer, tfidf_matrix):
    prefs = extract_preferences(user_input)
    
    state_dict.update(prefs)
        
    if not state_dict.get('genres'):
        return "¡Hello! to help you to find something, ¿what gender would you like to see?", state_dict
    
    if state_dict.get('year') is None:
        return "¿Do you have any idea of year o time that you would like to see, or do you prefer something new?", state_dict
    
    query_text = f"{state_dict.get('free_text', '')} {' '.join(state_dict.get('genres', []))} {state_dict.get('mood') or ''}"
        
    recommendations = recommend_on_the_fly(query_text, movies_df, vectorizer, tfidf_matrix)
    
    if recommendations.empty:
        return "Sorry, could you try with other words", state_dict

    
    response = "Movie Founded!\n\n"
    
    for _, movie in recommendations.iterrows():  
        explanation = generate_explanation(movie, state_dict)
        
        response += f"**{movie['title']}** (Rating: {movie.get('vote_average', 'N/A')})\n"
        response += f"> 💡 {explanation}\n\n"
        
    return response, state_dict

def initialize_conversation_state():

    return {
        "genres": [],
        "language": None,
        "year_range": None,
        "mood": None,
        "rating": None,
        "year": None,
        "free_text": ""
    }
