import numpy as np
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

    # Equivalent to linear_kernel(query_vector, tfidf_matrix) — TF-IDF rows are
    # L2-normalized, so the dot product IS the cosine — but multiplying in this
    # direction avoids transposing the 87K x 857K matrix, which allocated ~190 MB
    # per call. Results are bit-for-bit identical; this just picks the cheap side.
    cosine_sim_scores = (tfidf_matrix @ query_vector.T).toarray().ravel()

    # Score, filter and rank on a narrow numeric frame rather than on a copy of
    # movies_df. Deep-copying all ~87K x 10 columns per query and then sorting
    # every one of them cost ~250 MB of peak RSS — more than the hosted
    # container can spare. The display columns are pulled from movies_df once
    # the top-N rows are known, so the returned frame is unchanged.
    scored_df = pd.DataFrame(
        {'similarity_score': cosine_sim_scores}, index=movies_df.index
    )
    for col in ('original_language', 'vote_average'):
        if col in movies_df.columns:
            scored_df[col] = movies_df[col]

    # Extract year from release_date if release_year doesn't exist
    if 'release_year' in movies_df.columns:
        scored_df['release_year'] = movies_df['release_year']
    elif 'release_date' in movies_df.columns:
        scored_df['release_year'] = pd.to_datetime(
            movies_df['release_date'], errors='coerce'
        ).dt.year

    # scored_df stays pristine and unfiltered for the no-results fallback below;
    # temp_df is the working frame the filters narrow down.
    temp_df = scored_df
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
                # match % and the ranking stay consistent. Copy first when no
                # filter has replaced temp_df yet, so the decay never leaks into
                # scored_df — the fallback below must rank on raw similarity.
                if temp_df is scored_df:
                    temp_df = temp_df.copy()
                temp_df['similarity_score'] = temp_df['similarity_score'].to_numpy() * decay
            else:
                temp_df = temp_df[temp_df['release_year'].between(target_year - 2, target_year + 5)]

        # Min Ranking
        if state_dict.get('rating'):
            temp_df = temp_df[temp_df['vote_average'] >= float(state_dict['rating'])]

    ranked = temp_df.sort_values(by=rank_col, ascending=False).head(top_n)

    if ranked.empty:
        # Filters removed every candidate. Fall back to the closest matches by raw
        # similarity (ignoring filters) so the UI still has something to show. Rank
        # the unfiltered scored frame, so the returned columns — including
        # similarity_score — stay identical to the normal path.
        ranked = scored_df.sort_values(by='similarity_score', ascending=False).head(top_n)

    # Re-attach the display columns for just the handful of surviving rows.
    recommendations = movies_df.loc[ranked.index].copy()
    for col in ranked.columns:
        recommendations[col] = ranked[col]

    # Select columns, handling missing release_year
    cols = ['movieId','title', 'genres_list', 'vote_average', 'similarity_score', 'overview']
    if 'release_year' in recommendations.columns:
        cols.insert(2, 'release_year')
    if 'original_language' in recommendations.columns:
        cols.append('original_language')

    return recommendations[cols]