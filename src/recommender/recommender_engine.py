import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

from src.utils.text_cleaning import clean_text


def recommend_on_the_fly(
    query_text, movies_df, vectorizer, tfidf_matrix, state_dict=None, top_n=5,
    year_mode="filter", year_decay=0.6,
):
    """
    Recommend movies for a free-text query via TF-IDF cosine similarity.

    year_mode controls how a requested decade is applied:
      - "filter" (default): hard-exclude movies outside the decade window.
        Kept as default so the metrics module / backend behave as before.
      - "soft": never exclude on year. Instead score each movie as cosine × a
        decade-distance decay (full credit inside the decade, `year_decay` per
        decade away) and rank by that. This avoids a hard cliff that was wrongly
        dropping relevant edge-of-decade films (e.g. a 1999 match for a "90s"
        query) and surfaces stronger matches. `similarity_score` becomes this
        decade-aware score, so the displayed match % and the ordering stay
        consistent; for in-decade films decay is 1.0 so it equals the raw cosine.
    """

    # Clean the query exactly like the corpus was cleaned, otherwise the query
    # tokens won't match the TF-IDF vocabulary and similarity collapses.
    query_vector = vectorizer.transform([clean_text(query_text)])

    cosine_sim_scores = linear_kernel(query_vector, tfidf_matrix).flatten()

    temp_df = movies_df.copy()
    temp_df['similarity_score'] = cosine_sim_scores

    # Extract year from release_date if release_year doesn't exist
    if 'release_year' not in temp_df.columns and 'release_date' in temp_df.columns:
        temp_df['release_year'] = pd.to_datetime(temp_df['release_date'], errors='coerce').dt.year

    rank_col = 'similarity_score'

    if state_dict:
        # Language filter
        if state_dict.get('language'):
            temp_df = temp_df[temp_df['original_language'] == state_dict['language']]

        # Year handling — hard filter or soft decade-distance ranking
        if state_dict.get('year') and 'release_year' in temp_df.columns:
            target_year = int(state_dict['year'])
            if year_mode == "soft":
                yr = pd.to_numeric(temp_df['release_year'], errors='coerce').to_numpy()
                decade_center = target_year + 4.5  # e.g. 1990 -> 1994.5
                in_decade = (yr >= target_year) & (yr <= target_year + 9)
                decades_away = np.abs(yr - decade_center) / 10.0
                decay = np.where(in_decade, 1.0, year_decay ** decades_away)
                decay = np.where(np.isnan(yr), 0.5, decay)  # unknown year: mild penalty
                # Fold the decade decay into similarity_score so the displayed
                # match % and the ranking stay consistent.
                temp_df['similarity_score'] = temp_df['similarity_score'].to_numpy() * decay
            else:
                temp_df = temp_df[temp_df['release_year'].between(target_year - 2, target_year + 5)]

        # Min Ranking
        if state_dict.get('rating'):
            temp_df = temp_df[temp_df['vote_average'] >= float(state_dict['rating'])]

    recommendations = temp_df.sort_values(by=rank_col, ascending=False).head(top_n)

    if recommendations.empty:
        # Filters removed every candidate. Fall back to the closest matches by raw
        # similarity (ignoring filters) so the UI still has something to show. Build
        # this from the same scored frame so the returned columns — including
        # similarity_score — stay identical to the normal path.
        scored_df = movies_df.copy()
        scored_df['similarity_score'] = cosine_sim_scores
        if 'release_year' not in scored_df.columns and 'release_date' in scored_df.columns:
            scored_df['release_year'] = pd.to_datetime(
                scored_df['release_date'], errors='coerce'
            ).dt.year
        recommendations = scored_df.sort_values(
            by='similarity_score', ascending=False
        ).head(top_n)

    # Select columns, handling missing release_year
    cols = ['movieId','title', 'genres_list', 'vote_average', 'similarity_score', 'overview']
    if 'release_year' in recommendations.columns:
        cols.insert(2, 'release_year')
    if 'original_language' in recommendations.columns:
        cols.append('original_language')

    return recommendations[cols]