import streamlit as st
import pandas as pd
import joblib
import sys
import os

# 1. Configuración de Rutas para Databricks
WORKSPACE_PATH = "/Workspace/Users/d.esteban.am@gmail.com/AI-and-ML-Laboratory-Databricks"
sys.path.append(os.path.join(WORKSPACE_PATH, "src"))

from chatbot.chatbot_flow import chatbot_response, initialize_conversation_state

# 2. Carga Optimizada de Recursos (Se cargan una sola vez al inicio)
@st.cache_resource
def load_assets():
    # Rutas absolutas para Databricks
    movies_df = spark.read.table("workspace.datasets.movies_final").toPandas()
    vectorizer = joblib.load(f"{WORKSPACE_PATH}/models/tfidf_vectorizer.pkl")
    # Si guardaste la matriz TF-IDF, descomenta:
    # tfidf_matrix = joblib.load(f"{WORKSPACE_PATH}/models/tfidf_matrix.pkl")
    tfidf_matrix = None  # Por ahora
    return movies_df, vectorizer, tfidf_matrix

movies_df, vectorizer, tfidf_matrix = load_assets()

# 3. Configuración Visual de la Página
st.set_page_config(page_title="CineAssist AI", page_icon="🎬", layout="centered")
st.title("🎬 CineAssist: Tu Asistente de Cine")
st.markdown("Descríbeme qué tipo de película buscas y te daré las mejores opciones.")

# 4. Gestión del Estado de la Conversación (Memoria del Chat)
if "messages" not in st.session_state:
    st.session_state.messages = []  # Historial visual
if "chat_state" not in st.session_state:
    st.session_state.chat_state = initialize_conversation_state()  # Estado interno de preferencias

# Mostrar mensajes previos del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Interacción del Usuario
if prompt := st.chat_input("Ej: Quiero una película de acción similar a John Wick"):
    # Agregar mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta del asistente usando el flujo de la Fase 7
    with st.chat_message("assistant"):
        # La función chatbot_response orquesta NLP -> Recomendador -> Explicación
        response_text, updated_state = chatbot_response(
            prompt, 
            st.session_state.chat_state, 
            movies_df, 
            vectorizer, 
            tfidf_matrix
        )
        
        st.markdown(response_text)
        
        # Actualizar historial y estado de preferencias
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.session_state.chat_state = updated_state
