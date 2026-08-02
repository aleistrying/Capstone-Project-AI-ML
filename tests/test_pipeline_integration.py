"""Miniature end-to-end test for the real NLP and TF-IDF recommendation path."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.chatbot import chatbot_flow
from src.utils.text_cleaning import clean_text


def test_pipeline_ranks_a_matching_movie_without_production_assets(monkeypatch):
    """Text flows through extraction, query building, ranking, and serialization."""
    movies = pd.DataFrame(
        [
            {
                "movieId": 1,
                "title": "Galaxy Laughs",
                "genres_list": "['Comedy', 'Science Fiction']",
                "vote_average": 8.1,
                "overview": "A funny comedy adventure through space",
                "release_year": 1996,
                "original_language": "en",
            },
            {
                "movieId": 2,
                "title": "Quiet Courtroom",
                "genres_list": "['Drama']",
                "vote_average": 7.2,
                "overview": "A serious legal drama",
                "release_year": 1995,
                "original_language": "en",
            },
        ]
    )
    corpus = [clean_text(text) for text in movies["overview"]]
    vectorizer = TfidfVectorizer().fit(corpus)
    matrix = vectorizer.transform(corpus)
    monkeypatch.setattr(chatbot_flow, "_TRANSLATION_AVAILABLE", False)

    trace = chatbot_flow.run_pipeline(
        "a funny comedy space movie from the 90s",
        chatbot_flow.initialize_conversation_state(),
        movies,
        vectorizer,
        matrix,
        top_n=2,
    )

    assert trace.status == "ok"
    assert trace.recommendations[0]["title"] == "Galaxy Laughs"
    assert trace.filters["year"] == 1990
    assert trace.vocab_hits
