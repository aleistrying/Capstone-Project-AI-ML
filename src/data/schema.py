"""Runtime data and model-alignment validation for CineAssist artifacts."""

import pandera.pandas as pa

MOVIE_SCHEMA = pa.DataFrameSchema(
    {
        "movieId": pa.Column(int, nullable=True, coerce=True),
        "title": pa.Column(str, nullable=False),
        "genres_list": pa.Column(None, nullable=True),
        "vote_average": pa.Column(
            float,
            checks=pa.Check.in_range(0, 10),
            nullable=True,
            coerce=True,
        ),
        "overview": pa.Column(str, nullable=True),
        "release_year": pa.Column(
            float,
            checks=pa.Check.in_range(1870, 2200),
            nullable=True,
            coerce=True,
            required=False,
        ),
        "original_language": pa.Column(str, nullable=True, required=False),
    },
    strict=False,
    coerce=False,
)


def validate_movie_dataframe(movies_df):
    """Validate and return a movie DataFrame using the serving-data contract."""
    return MOVIE_SCHEMA.validate(movies_df)


def validate_model_alignment(movies_df, vectorizer, tfidf_matrix) -> None:
    """Ensure dataset rows and vectorizer features match the TF-IDF matrix."""
    validate_movie_dataframe(movies_df)
    if tfidf_matrix.shape[0] != len(movies_df):
        raise ValueError(
            "TF-IDF matrix row count does not match the movie dataset: "
            f"{tfidf_matrix.shape[0]} != {len(movies_df)}"
        )
    feature_count = len(getattr(vectorizer, "vocabulary_", {}))
    if tfidf_matrix.shape[1] != feature_count:
        raise ValueError(
            "TF-IDF matrix column count does not match the vectorizer vocabulary: "
            f"{tfidf_matrix.shape[1]} != {feature_count}"
        )
