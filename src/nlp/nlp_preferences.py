import re

def extract_preferences(user_input):
    user_input = user_input.lower()
    
    genres_keywords = {
    'action': ['action', 'thrilling', 'explosion', 'fight', 'adventure'],
    'comedy': ['comedy', 'funny', 'hilarious', 'laugh', 'humor'],
    'drama': ['drama', 'emotional', 'sad', 'touching', 'life'],
    'horror': ['horror', 'scary', 'spooky', 'creep', 'terror'],
    'sci-fi': ['sci-fi', 'science fiction', 'space', 'alien', 'future', 'robot'],
    'romance': ['romance', 'romantic', 'love', 'couple', 'heart'],
    'thriller': ['thriller', 'suspense', 'mystery', 'crime'],
    'animation': ['animation', 'animated', 'cartoon', 'anime'],
    'fantasy': ['fantasy', 'magic', 'wizard', 'dragon']
}

    languages = {'spanish': 'es', 'english': 'en', 'french': 'fr', 'italian': 'it'}
    
    moods_keywords = {
    'dark': ['dark', 'gloomy', 'intense', 'gritty'],
    'feel-good': ['feel-good', 'happy', 'inspiring', 'uplifting', 'cheerful'],
    'intense': ['intense', 'fast-paced', 'edge of my seat', 'breathless'],
    'relaxing': ['relaxing', 'chill', 'calm', 'light-hearted'],
    'nostalgic': ['nostalgic', 'old days', 'classic feel']
}

    preferences = {
        "genres": [],
        "language": None,
        "year_range": None,
        "mood": None,
        "rating": None,
        "free_text": user_input
        
    }


    #Genre Extraction
    for genre, keywords in genres_keywords.items():
        if genre in user_input or any(word in user_input for word in keywords):
            if genre not in preferences["genres"]:
                preferences["genres"].append(genre)

    # Language Extraction
    for lang_name, lang_code in languages.items():
        if lang_name in user_input:
            preferences["language"] = lang_code

    # Mood Extraction
    for mood, keywords in moods_keywords.items():
        if mood in user_input or any(word in user_input for word in keywords):
            preferences["mood"] = mood
            break  

    # Year
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', user_input)
    if year_match:
        preferences["year"] = int(year_match.group(1))
    
    # Decade ("90s", "80s")
    decade_match = re.search(r'\b(\d{2})s\b', user_input)
    if decade_match:
        
        preferences["year"] = int(f"19{decade_match.group(1)}")

    # Ranking
    rating_match = re.search(r'(?:rating|score|puntuación|más de|above)\s*(\d+(?:\.\d+)?)', user_input)
    if rating_match:
        preferences["rating"] = float(rating_match.group(1))

    return preferences
