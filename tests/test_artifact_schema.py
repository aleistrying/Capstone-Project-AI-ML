"""Data-quality and ML artifact consistency tests."""

import pandas as pd
import pytest
from pandera.errors import SchemaError
from scipy.sparse import csr_matrix

from src.data.schema import validate_model_alignment, validate_movie_dataframe


class TwoFeatureVectorizer:
    """Vectorizer metadata double with two fitted features."""

    vocabulary_ = {"space": 0, "comedy": 1}


@pytest.fixture(name="valid_movies")
def valid_movies_fixture():
    """Return a small DataFrame satisfying the production schema."""
    return pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["A", "B"],
            "genres_list": ["['Comedy']", "['Drama']"],
            "vote_average": [7.5, 8.0],
            "overview": ["A story", "Another story"],
            "release_year": [2000.0, 2001.0],
            "original_language": ["en", "fr"],
        }
    )


def test_valid_movie_schema_and_artifacts(valid_movies):
    """Compatible rows, columns, vocabulary, and matrix pass validation."""
    matrix = csr_matrix([[1.0, 0.0], [0.0, 1.0]])

    assert len(validate_movie_dataframe(valid_movies)) == 2
    validate_model_alignment(valid_movies, TwoFeatureVectorizer(), matrix)


def test_schema_rejects_rating_outside_ten_point_scale(valid_movies):
    """Impossible ratings are caught before recommendation or training."""
    invalid = valid_movies.copy()
    invalid.loc[0, "vote_average"] = 12.0

    with pytest.raises(SchemaError):
        validate_movie_dataframe(invalid)


def test_alignment_rejects_vectorizer_matrix_mismatch(valid_movies):
    """Stale vectorizer and matrix artifacts cannot be used together."""
    with pytest.raises(ValueError, match="vectorizer vocabulary"):
        validate_model_alignment(
            valid_movies,
            TwoFeatureVectorizer(),
            csr_matrix([[1.0], [0.0]]),
        )
