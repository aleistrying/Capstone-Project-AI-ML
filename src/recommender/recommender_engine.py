from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
import re

def preprocess_query(text):
    """Apply stemming to match training data preprocessing."""
    text = text.lower()
    stem_map = {
        'comedy': 'comedi', 'comedies': 'comedi',
        'romance': 'romanc', 'romantic': 'romant',
        'adventure': 'adventur', 'adventures': 'adventur',
        'family': 'famili', 'families': 'famili',
        'mystery': 'mysteri', 'mysteries': 'mysteri',
        'history': 'histori', 'historical': 'histor',
        'documentary': 'documentari', 'documentaries': 'documentari',
        'fantasy': 'fantasi', 'fantasies': 'fantasi',
    }
    for word, stem in stem_map.items():
        text = re.sub(r'\b' + word + r'\b', stem, text)
    return text

def recommend_on_the_fly(query_text, movies_df, vectorizer, tfidf_matrix, state_dict=None, top_n=5):
    # Check if user wants movies similar to a specific title
    similar_to = state_dict.get('similar_to') if state_dict else None
    
    if similar_to:
        # Search for the reference movie in the dataset
        mask = movies_df['title'].str.contains(similar_to, case=False, na=False)
        matches = movies_df[mask]
        
        if not matches.empty:
            # Use the first match as reference
            ref_idx = matches.index[0]
            # Use the TF-IDF vector of the reference movie as query
            query_vector = tfidf_matrix[ref_idx]
        else:
            # If movie not found, fall back to text-based search
            query_text = preprocess_query(query_text)
            query_vector = vectorizer.transform([query_text])
    else:
        # Standard text-based search
        query_text = preprocess_query(query_text)
        query_vector = vectorizer.transform([query_text])

    cosine_sim_scores = linear_kernel(query_vector, tfidf_matrix).flatten()

    temp_df = movies_df.copy()
    temp_df['similarity_score'] = cosine_sim_scores
    
    # If searching by similar_to, exclude the reference movie from results
    if similar_to:
        temp_df = temp_df[~temp_df['title'].str.contains(similar_to, case=False, na=False)]

    # Extract year from release_date if release_year doesn't exist
    if 'release_year' not in temp_df.columns and 'release_date' in temp_df.columns:
        temp_df['release_year'] = pd.to_datetime(temp_df['release_date'], errors='coerce').dt.year

    if state_dict:
        # Language filter
        if state_dict.get('language'):
            temp_df = temp_df[temp_df['original_language'] == state_dict['language']]

        # Year filter - using year_range from NLP extraction
        if state_dict.get('year_range') and 'release_year' in temp_df.columns:
            year_min, year_max = state_dict['year_range']
            temp_df = temp_df[temp_df['release_year'].between(year_min, year_max)]

        # Min Ranking
        if state_dict.get('rating'):
            temp_df = temp_df[temp_df['vote_average'] >= float(state_dict['rating'])]


    recommendations = temp_df.sort_values(by='similarity_score', ascending=False).head(top_n)


    if recommendations.empty:
        related_indices = cosine_sim_scores.argsort()[-top_n:][::-1]
        return movies_df.iloc[related_indices].copy()

    # Select columns, handling missing release_year
    cols = ['title', 'genres_list', 'vote_average', 'similarity_score', 'overview']
    if 'release_year' in recommendations.columns:
        cols.insert(2, 'release_year')

    return recommendations[cols]