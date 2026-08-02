"""Small deterministic microbenchmark for recommendation regression tracking."""

import pandas as pd
from scipy.sparse import csr_matrix

from src.recommender.recommender_engine import recommend_on_the_fly


class BenchmarkVectorizer:
    """Return the same sparse query vector for each benchmark iteration."""

    vocabulary_ = {"query": 0}

    def transform(self, _texts):
        return csr_matrix([[1.0]])


def test_recommendation_microbenchmark(benchmark):
    """Record latency for ranking a small in-memory candidate set."""
    row_count = 500
    movies = pd.DataFrame(
        {
            "movieId": range(row_count),
            "title": [f"Movie {index}" for index in range(row_count)],
            "genres_list": ["['Drama']"] * row_count,
            "vote_average": [7.0] * row_count,
            "overview": ["query"] * row_count,
        }
    )
    matrix = csr_matrix([[1.0]] * row_count)

    result = benchmark(
        recommend_on_the_fly,
        "query",
        movies,
        BenchmarkVectorizer(),
        matrix,
    )

    assert len(result) == 5
