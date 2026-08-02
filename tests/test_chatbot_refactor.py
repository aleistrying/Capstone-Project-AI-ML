"""Regression tests for extracted chatbot internals.

Protected access is intentional: public integration tests cover composition,
while these unit tests lock down each small helper's behavior.
"""

# pylint: disable=protected-access

import pandas as pd

from src.chatbot import chatbot_flow


class VocabularyOnlyVectorizer:
    """Minimal vectorizer double for query-building tests."""

    vocabulary_ = {"comedy": 0, "family": 1, "bright": 2}


def test_build_recommender_filters_adapts_conversation_state():
    """The adapter preserves the recommender's established filter contract."""
    state = {
        "language": "es",
        "min_rating": 7.5,
        "rating": 6.0,
        "year_range": [1990, 1999],
    }

    assert chatbot_flow._build_recommender_filters(state) == {
        "language": "es",
        "rating": 7.5,
        "year": 1990,
    }


def test_build_query_details_deduplicates_and_reports_vocab(monkeypatch):
    """Query construction returns both focused text and auditable coverage."""
    monkeypatch.setattr(chatbot_flow, "_get_genre_vocab", lambda _movies: {"comedy"})
    monkeypatch.setattr(
        chatbot_flow,
        "build_query",
        lambda *_args, **_kwargs: {
            "entities": {"genres": ["comedy"]},
            "query": "comedy bright unknown",
        },
    )

    result = chatbot_flow._build_query_details(
        "a bright comedy",
        {"genres": ["comedy"], "mood": ["family"]},
        pd.DataFrame(),
        VocabularyOnlyVectorizer(),
    )

    _, query_text, cleaned_query, vocab_hits, vocab_total = result
    assert query_text == "comedy family bright unknown"
    assert cleaned_query == query_text
    assert vocab_hits == ["comedy", "family", "bright"]
    assert vocab_total == 4


def test_serialize_recommendations_keeps_response_schema(monkeypatch):
    """DataFrame rows become typed card data without changing the public schema."""
    monkeypatch.setattr(
        chatbot_flow,
        "generate_explanation",
        lambda movie, _state: f"Because you may like {movie['title']}",
    )
    recommendations = pd.DataFrame(
        [
            {
                "movieId": 42.0,
                "title": "Example Movie",
                "release_year": "2001",
                "vote_average": 8.2,
                "genres_list": "['Comedy', 'Family']",
                "original_language": "en",
                "overview": "  A warm story.  ",
                "similarity_score": 0.75,
            }
        ]
    )

    result = chatbot_flow._serialize_recommendations(recommendations, {})

    assert result == [
        {
            "movieId": 42,
            "title": "Example Movie",
            "year": 2001,
            "rating": 8.2,
            "genres": ["Comedy", "Family"],
            "language": "en",
            "overview": "A warm story.",
            "similarity": 0.75,
            "explanation": "Because you may like Example Movie",
        }
    ]


def test_genre_cache_is_invalidated_for_a_different_dataframe(monkeypatch):
    """A second dataset cannot inherit the first dataset's genre vocabulary."""
    calls = []

    def build_vocabulary(frame):
        calls.append(frame)
        return {frame.iloc[0]["genre"]}

    monkeypatch.setattr(chatbot_flow, "build_genre_vocabulary", build_vocabulary)
    cache = chatbot_flow._GenreVocabularyCache()
    first = pd.DataFrame([{"genre": "comedy"}])
    second = pd.DataFrame([{"genre": "drama"}])

    assert cache.get(first) == {"comedy"}
    assert cache.get(first) == {"comedy"}
    assert cache.get(second) == {"drama"}
    assert calls == [first, second]


def test_translation_failure_logs_and_falls_back(monkeypatch, caplog):
    """Expected translation failures preserve input and remain observable."""
    monkeypatch.setattr(chatbot_flow, "_TRANSLATION_AVAILABLE", True)

    def fail_translation(_text, _language):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(chatbot_flow, "_translate_to_english", fail_translation)

    assert chatbot_flow._to_english("hola", "es") == "hola"
    assert "model unavailable" in caplog.text


def test_pipeline_returns_greeting_for_an_empty_focused_query(monkeypatch):
    """An empty focused query exits before calling the recommender."""
    monkeypatch.setattr(chatbot_flow, "_safe_domain_normalize", lambda _text: {})
    monkeypatch.setattr(chatbot_flow, "_detect_ui_language", lambda *_args: "en")
    monkeypatch.setattr(chatbot_flow, "extract_preferences", lambda _text: {})
    monkeypatch.setattr(
        chatbot_flow,
        "_build_query_details",
        lambda *_args: chatbot_flow.QueryDetails({}, "", "", [], 0),
    )
    trace = chatbot_flow.run_pipeline(
        "please recommend something",
        chatbot_flow.initialize_conversation_state(),
        pd.DataFrame(),
        VocabularyOnlyVectorizer(),
    )

    assert trace.status == "empty_query"
    assert not trace.recommendations
    assert trace.intro_en.startswith("Hi!")
