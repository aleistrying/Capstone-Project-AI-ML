"""Property-based tests for NLP parsing and recommendation invariants."""

import math

from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.nlp.nlp_preferences import extract_preferences
from src.recommender.recommender_engine import recommend_on_the_fly


class OneTermVectorizer:
    """Minimal deterministic vectorizer used by property tests."""

    vocabulary_ = {"query": 0}

    def transform(self, _texts):
        return csr_matrix([[1.0]])


@given(st.text())
def test_preference_extraction_is_total_for_unicode(text):
    """Every Unicode input produces a complete, stable preference contract."""
    result = extract_preferences(text)

    assert result["free_text"] == text
    assert isinstance(result["genres"], list)
    assert isinstance(result["mood"], list)
    assert result["year_range"] is None or len(result["year_range"]) == 2


@given(
    scores=st.lists(
        st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=12,
    ),
    requested_top_n=st.integers(min_value=1, max_value=15),
)
def test_recommender_output_is_bounded_and_finite(scores, requested_top_n):
    """Ranking never exceeds top_n and always returns finite similarity values."""
    movies = pd.DataFrame(
        {
            "movieId": range(len(scores)),
            "title": [f"Movie {index}" for index in range(len(scores))],
            "genres_list": ["['Drama']"] * len(scores),
            "vote_average": [7.0] * len(scores),
            "overview": ["query"] * len(scores),
        }
    )
    matrix = csr_matrix(np.asarray(scores).reshape(-1, 1))

    result = recommend_on_the_fly(
        "query",
        movies,
        OneTermVectorizer(),
        matrix,
        top_n=requested_top_n,
    )

    assert len(result) <= min(requested_top_n, len(scores))
    assert all(math.isfinite(score) for score in result["similarity_score"])
    assert result["similarity_score"].is_monotonic_decreasing
