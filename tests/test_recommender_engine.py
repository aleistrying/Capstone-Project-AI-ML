"""Unit tests for TF-IDF ranking, validation, filters, and fallback behavior."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.recommender.recommender_engine import recommend_on_the_fly


class FixedVectorizer:
    """Return a deterministic one-term query vector."""

    def transform(self, _texts):
        return csr_matrix([[1.0]])


@pytest.fixture(name="movies")
def movies_fixture():
    """Provide the minimum valid movie schema plus filter columns."""
    return pd.DataFrame(
        [
            {
                "movieId": 1,
                "title": "Best Match",
                "genres_list": "['Comedy']",
                "vote_average": 8.5,
                "overview": "First",
                "release_year": 1995,
                "original_language": "en",
            },
            {
                "movieId": 2,
                "title": "Second Match",
                "genres_list": "['Drama']",
                "vote_average": 7.0,
                "overview": "Second",
                "release_year": 2015,
                "original_language": "fr",
            },
        ]
    )


@pytest.fixture(name="matrix")
def matrix_fixture():
    """Return similarity weights matching the two fixture rows."""
    return csr_matrix(np.array([[0.9], [0.4]]))


def test_ranks_by_similarity(movies, matrix):
    """The largest cosine score appears first."""
    result = recommend_on_the_fly("comedy", movies, FixedVectorizer(), matrix, top_n=2)

    assert result["movieId"].tolist() == [1, 2]
    assert result["similarity_score"].tolist() == pytest.approx([0.9, 0.4])


def test_empty_filter_result_falls_back_to_raw_ranking(movies, matrix):
    """Overly restrictive filters still produce the closest raw match."""
    result = recommend_on_the_fly(
        "comedy",
        movies,
        FixedVectorizer(),
        matrix,
        state_dict={"language": "es"},
        top_n=1,
    )

    assert result.iloc[0]["movieId"] == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_n": 0}, "top_n must be a positive integer"),
        ({"year_mode": "unknown"}, "year_mode must be one of"),
        ({"year_decay": 1.5}, "year_decay must be between 0 and 1"),
    ],
)
def test_rejects_invalid_options(movies, matrix, kwargs, message):
    """Invalid ranking options fail before expensive model work."""
    with pytest.raises(ValueError, match=message):
        recommend_on_the_fly("query", movies, FixedVectorizer(), matrix, **kwargs)


def test_rejects_model_dataset_row_mismatch(movies):
    """A matrix trained on different rows cannot silently rank the dataset."""
    with pytest.raises(ValueError, match="row count must match"):
        recommend_on_the_fly("query", movies, FixedVectorizer(), csr_matrix([[0.9]]))


def test_reports_missing_required_columns(movies, matrix):
    """Invalid dataset schemas produce an actionable validation error."""
    with pytest.raises(ValueError, match="overview"):
        recommend_on_the_fly(
            "query", movies.drop(columns="overview"), FixedVectorizer(), matrix
        )
