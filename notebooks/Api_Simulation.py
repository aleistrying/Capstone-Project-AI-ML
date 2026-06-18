import sys
import os
import pandas as pd
import joblib

# Configuration of modules and SRC folders path 
sys.path.append(os.path.abspath("../src"))

# Import Core Functions 
import importlib
from nlp import nlp_preferences
from recommender import recommender_engine
from utils import explanation_generator
from chatbot import chatbot_flow

# Reload modules to catch changes
importlib.reload(nlp_preferences)
importlib.reload(recommender_engine)
importlib.reload(explanation_generator)
importlib.reload(chatbot_flow)

# Import Specific Functions
from nlp.nlp_preferences import extract_preferences
from recommender.recommender_engine import recommend_on_the_fly
from utils.explanation_generator import generate_explanation
from chatbot.chatbot_flow import initialize_conversation_state

from chatbot.chatbot_flow import chatbot_response, initialize_conversation_state

# Load of dataset cleaned and trained modules (Vectorizar and Matrix)
# movies_df = pd.read_csv("../data/movies_clean.csv")
movies_df = spark.read.table("workspace.datasets.movies_final").toPandas()
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("../models/tfidf_matrix.pkl")

print(f"✅ Dataset cargado: {len(movies_df)} películas")
print(f"✅ Vectorizer cargado: {vectorizer}")
print(f"✅ TF-IDF Matrix shape: {tfidf_matrix.shape}")

print("\n" + "="*60)
print("🎬 CHATBOT DE RECOMENDACIÓN DE PELÍCULAS")
print("="*60)
print("Escribe 'exit' o 'quit' para salir de la conversación\n")

# Inicializar el estado de la conversación
state = initialize_conversation_state()

# Primer mensaje del usuario
user_input = input("👤 Tú: ")

# Bucle interactivo de conversación
while user_input.lower() not in ['exit', 'quit', 'salir']:
    # Obtener respuesta del chatbot
    response, state = chatbot_response(user_input, state, movies_df, vectorizer, tfidf_matrix)
    
    # Mostrar respuesta del bot
    print(f"\n🤖 Bot: {response}\n")
    
    # Verificar si hay recomendaciones en el estado
    # Asumiendo que el estado tiene una clave 'recommendations' cuando hay recomendaciones finales
    if 'recommendations' in state and state['recommendations']:
        print("\n" + "="*60)
        print("✅ Recomendaciones generadas exitosamente")
        print("="*60)
        
        # Preguntar si quiere continuar con otra búsqueda
        continue_chat = input("\n¿Quieres buscar más películas? (s/n): ").lower()
        if continue_chat in ['s', 'si', 'sí', 'yes', 'y']:
            # Reiniciar estado para nueva conversación
            state = initialize_conversation_state()
            print("\n" + "-"*60)
            print("Nueva búsqueda iniciada")
            print("-"*60 + "\n")
        else:
            print("\n👋 ¡Hasta luego! Disfruta tus películas.")
            break
    
    # Solicitar siguiente input del usuario
    user_input = input("👤 Tú: ")

if user_input.lower() in ['exit', 'quit', 'salir']:
    print("\n👋 ¡Hasta luego! Disfruta tus películas.")
