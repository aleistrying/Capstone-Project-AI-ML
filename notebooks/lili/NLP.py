# %%
import pandas as pd

movies_df = pd.read_csv("movies_final.csv")
movies_df.head()

# %%
import nltk
import string

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')




# %%
import json
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

with open("intents.json", "r") as file:
    intents_data = json.load(file)

movies_df = pd.read_csv("movies_final.csv")

movies_df["combined_features"] = (
    movies_df["combined_features"]
    .fillna("")
)

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies_df["combined_features"])

# %%
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(query):

    query_vector = tfidf.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten() # Flatten to 1D array

    top_indices = similarities.argsort()[-5:][::-1] # Get indices of top 5 similar movies

    recommendations = (
        movies_df.iloc[top_indices]["title"]
        .tolist()
    )

    response = f"Based on your interest in '{query.lower()}', I recommend these movies:\n\n"

    for movie in recommendations:
        response += f"- {movie}\n"

    return response

# %%
waiting_for_recommendation = False

# %%
# Detectar intención
def get_intent(user_input):

    user_words = set(user_input.lower().split())

    best_match = None
    max_matches = 0

    for intent in intents_data["intents"]:

        for pattern in intent["patterns"]:

            pattern_words = set(pattern.lower().split())

            matches = len(
                user_words.intersection(pattern_words)
            )

            if matches > max_matches:
                max_matches = matches
                best_match = intent

    return best_match

# %%
# Procesar mensaje
def process_input(text):

    global waiting_for_recommendation

    if waiting_for_recommendation:

        waiting_for_recommendation = False

        return recommend_movies(text)

    intent = get_intent(text)

    if intent is None:

        noanswer = next(
            item
            for item in intents_data["intents"]
            if item["tag"] == "noanswer"
        )

        return random.choice(
            noanswer["responses"]
        )

    if intent["tag"] == "movies":

        waiting_for_recommendation = True

        return random.choice(
            intent["responses"]
        )

    return random.choice(
        intent["responses"]
    )

# %%
# Chatbot
print("🎬 MovieBot is running!")
print("Type 'quit' to exit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() in ["quit", "exit"]:

        print("Bot: Goodbye!")
        break

    response = process_input(user_input)

    print(f"\nBot: {response}\n")


