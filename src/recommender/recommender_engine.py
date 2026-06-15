from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def recommend_on_the_fly(query_text, movies_df, vectorizer, tfidf_matrix, state_dict=None, top_n=5):

    query_vector = vectorizer.transform([query_text])

    cosine_sim_scores = linear_kernel(query_vector, tfidf_matrix).flatten()

    temp_df = movies_df.copy()
    temp_df['similarity_score'] = cosine_sim_scores

    # Extract year from release_date if release_year doesn't exist
    if 'release_year' not in temp_df.columns and 'release_date' in temp_df.columns:
        temp_df['release_year'] = pd.to_datetime(temp_df['release_date'], errors='coerce').dt.year

    if state_dict:
        # Language filter
        if state_dict.get('language'):
            temp_df = temp_df[temp_df['original_language'] == state_dict['language']]

        # Year filter
        if state_dict.get('year') and 'release_year' in temp_df.columns:
            target_year = int(state_dict['year'])
            temp_df = temp_df[temp_df['release_year'].between(target_year - 2, target_year + 5)]

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