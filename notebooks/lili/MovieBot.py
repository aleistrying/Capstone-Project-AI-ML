import pandas as pd
import json
import random
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

#print("Script folder:", BASE_DIR)

#print("Loading CSV...")
#start = time.time()
pd.options.mode.string_storage = "python"

print("Loading CSV...")

movies_df = pd.read_csv(BASE_DIR / "movies_final.csv",
    engine="c",
    low_memory=False
)

print("CSV loaded")

movies_df = pd.read_csv(
    BASE_DIR / "movies_final.csv"
)

print(movies_df.head())

#print("CSV loaded")
#print("Rows:", len(movies_df))
#print("Time:", time.time() - start, "seconds")

movies_df["release_date"] = pd.to_datetime(
    movies_df["release_date"],
    errors="coerce"
).dt.year

waiting_for_details = False
last_request = ""

with open(
    BASE_DIR / "intents.json",
    "r",
    encoding="utf-8"
) as file:
    intents_data = json.load(file)

ignored_words = set()
for intent in intents_data["intents"]:

    if intent["tag"] == "movies":

        ignored_words.update(
            intent.get(
                "ignored_words",
                []
            )
        )

        break
    
movies_df["release_year"] = (
    movies_df["release_date"]
    .astype("Int64")
)

movies_df["combined_features"] = (
    movies_df["combined_features"]
    .fillna("")
)

tfidf = TfidfVectorizer(stop_words="english")

tfidf_matrix = tfidf.fit_transform(movies_df["combined_features"])

genres = set()

for genre_string in movies_df["genres_list"].dropna():

    for genre in str(genre_string).split():

        genres.add(genre.lower())


# %%
from nltk.corpus import stopwords

stop_words = set(
    stopwords.words("english")
)

def extract_keywords(
    text,
    entities
):

    text = text.lower()

    for genre in entities["genres"]:
        text = text.replace(genre,"")

    for year in entities["years"]:
        text = text.replace(year,"" )

    words = re.findall(
        r"\w+",
        text
    )
    
    keywords = [

        word

    for word in words

    if word not in stop_words
    and word not in ignored_words
    and len(word) > 2

    ]

    return keywords

# %%
def get_response_by_tag(tag):

    for intent in intents_data["intents"]:

        if intent["tag"] == tag:

            return random.choice(
                intent["responses"]
            )

    return f"ERROR: Tag '{tag}' not found"

# %%
# Detectar intención
def get_intent(user_input):

# Eliminar signos de puntuación
    user_input = re.sub(
        r"[^\w\s]",
        "",
        user_input.lower()
    )

    user_words = set(user_input.lower().split())

    best_match = None
    max_matches = 0

    for intent in intents_data["intents"]:
        #print(intent["tag"])

        for pattern in intent["patterns"]:

            pattern_words = set(pattern.lower().split())

            matches = len(
                user_words.intersection(pattern_words)
            )

            if matches > max_matches:

                max_matches = matches
                best_match = intent

    #print(
     #   "Detected intent:",
     #   best_match["tag"] if best_match else None
   # )

    return best_match

# %%
def extract_entities(text):

    text = text.lower()

    entities = {
        "genres": [],
        "years": [],
    }    

    for genre in genres:
        if genre in ignored_words:
            continue

        if re.search(
            rf"\b{re.escape(genre)}\b",
            text
        ):
            entities["genres"].append(genre)

# Años
    entities["years"] = re.findall(
        r"\b(19\d{2}|20\d{2})\b",
        text
    )


    return entities

# %%
def build_query(user_input):

    entities = extract_entities(
        user_input
    )

    keywords = extract_keywords(
        user_input,
        entities
    )
    
    
    query_parts = []

    query_parts.extend(
        entities["genres"]
    )

    query_parts.extend(
        keywords
    )
    
    return {
    "entities": entities,
    "keywords": keywords,
    "query": " ".join(query_parts)
}

# %%
from sklearn.metrics.pairwise import linear_kernel

def recommend_on_the_fly(
    query_text,
    movies_df,
    vectorizer,
    tfidf_matrix,
    state_dict=None,
    top_n=5
):

    query_vector = vectorizer.transform([query_text])

    similarity_scores = linear_kernel(
        query_vector,
        tfidf_matrix
    ).flatten()

    temp_df = movies_df.copy()

    temp_df["similarity_score"] = similarity_scores

    if state_dict:

        if state_dict.get("year"):

            target_year = int(
                state_dict["year"]
            )

            temp_df =  temp_df[
        temp_df["release_year"]
        == target_year
            ]

            #print("Movies in year:",
                    #len(
                       # temp_df[
                           # temp_df["release_year"] == target_year
                       # ]
                    #)
              #  )

        if state_dict.get("genre"):

            temp_df = temp_df[
                temp_df["genres_list"]
                .str.contains(
                    state_dict["genre"],
                    case=False,
                    na=False
                )
            ]

    recommendations = (
        temp_df
        .sort_values(
            "similarity_score",
            ascending=False
        )
        .head(20)
        .sort_values(
        by='vote_average',
        ascending=False
    )
    .head(top_n)
    )

    return recommendations

# %%
def actor_match(cast_list, person_query):

    if pd.isna(cast_list):
        return False

    actors = [
        actor.strip().lower()
        for actor in str(cast_list).split(",")
    ]

    return person_query.lower() in actors

# %%
# función para filtros
def apply_filters(df, entities):

    filtered_df = df.copy()

    if entities["genres"]:

        filtered_df = filtered_df[
            filtered_df["genres_list"]
            .str.contains(
                entities["genres"][0],
                case=False,
                na=False
            )
        ]

    if entities["years"]:

        year = int(
            entities["years"][0]
        )

        filtered_df = filtered_df[
            filtered_df["release_date"]
            == year
        ]

    return filtered_df

# %%
# función para buscar personas
def search_person(filtered_df, keywords):

    if len(keywords) < 2:
        return None

    person_query = " ".join(keywords)

    #print("Searching for:", person_query)

    matches = filtered_df[
        filtered_df["Director"]
        .fillna("")
        .str.lower()
        .str.contains(
            person_query.lower(),
            regex=False,
            na=False
        )
    ]

   # print("Matches found:", len(matches))

    if len(matches) == 0:
        return None

    return (
        matches
        .sort_values(
            by="vote_average",
            ascending=False
        )
        .head(5)
    )

# %%
# función para formatear respuesta
def build_response(results):

    response = (
        get_response_by_tag(
            "recommendation_response"
        )
        + "\n\n"
    )

    for rank, (_, row) in enumerate(
        results.iterrows(),
        start=1
    ):

        year = row["release_year"]

        response += (
            f"{rank}. {row['title']} "
            f"({year}) "
            f"⭐ {row['vote_average']:.1f}\n"
        )

    return response

# %%
from sklearn.metrics.pairwise import cosine_similarity

def recommend_movies(user_input):

    global waiting_for_details
    global last_request

    search_data = build_query(
        user_input
    )

    entities = search_data["entities"]
    keywords = search_data["keywords"]
    smart_query = search_data["query"]

    has_information = (
        len(entities["genres"]) > 0
        or len(entities["years"]) > 0
        or len(keywords) > 0
    )

    #print("Entities:", entities)
    #print("Keywords:", keywords)

    if not has_information:

        waiting_for_details = True
        last_request = ""

        return get_response_by_tag(
            "movies"
        )

    filtered_df = apply_filters(
        movies_df,
        entities
    )

    if len(filtered_df) == 0:
        filtered_df = movies_df

    person_results = search_person(
        filtered_df,
        keywords
    )

    if person_results is not None:

        return build_response(
            person_results
        )

    state_dict = {}

    if entities["years"]:
        state_dict["year"] = entities["years"][0]

    if entities["genres"]:
        state_dict["genre"] = entities["genres"][0]

    filtered_matrix = tfidf.transform(
        filtered_df["combined_features"]
    )

    results = recommend_on_the_fly(
        smart_query,
        filtered_df,
        tfidf,
        filtered_matrix,
        state_dict,
        top_n=5
    )

    #print("Results shape:", results.shape)
    #print(results.head())

    #print("Smart query:", smart_query)
    #print("Recommendations found:", len(results))
    #print("Results shape:", results.shape)
    #print(results[["title", "similarity_score"]].head())
    
    return build_response(
        results
    )

# %%
def process_input(text):
  
    global waiting_for_details
    global last_request

    if waiting_for_details:

        waiting_for_details = False

        full_request = last_request + " " + text

        #print("Full request:", full_request)

        return recommend_movies(full_request)

    intent = get_intent(text)
    
    #print("User text:", text)
    
    if intent is not None:

        if intent["tag"] == "movies":

            search_data = build_query(text)

            has_information = (
                len(search_data["entities"]["genres"]) > 0
                or len(search_data["entities"]["years"]) > 0
                or len(search_data["keywords"]) > 0
            )
            #print("Keywords found:", search_data["keywords"])
            #print("Has information:", has_information)
        
            if has_information:
                return recommend_movies(text)

            waiting_for_details = True
            last_request = ""

            return random.choice(
                intent["responses"]
            )

        if intent["tag"] in [
            "greeting",
            "goodbye",
            "thanks",
            "options"
        ]:

            return random.choice(
                intent["responses"]
            )

    return recommend_movies(text)

# %%
# Chatbot
print("🎬 MovieBot is running!")
print("Type 'quit' to exit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() in ["quit", "exit"]:

        print(f"MovieBot: {get_response_by_tag('departure')}")
        break

    response = process_input(user_input)

    print(f"\nMovieBot: {response}\n")


