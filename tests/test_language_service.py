"""Tests for lightweight language detection and vocabulary normalization."""

from backend.services.language_service import detect_language, normalize


def test_detect_spanish():
    assert detect_language("quiero una pelicula chistosa para familia") == "es"


def test_detect_english():
    assert detect_language("I want a funny comedy movie") == "en"


def test_detect_french():
    assert detect_language("je veux voir un film romantique") == "fr"


def test_normalize_spanish_comedy():
    result = normalize("quiero una pelicula chistosa para familia de los 2000")
    assert "comedy" in result["mapped_genres"]
    assert "family" in result["mapped_genres"]
    assert result["mapped_year_range"] == [2000, 2009]
    assert result["detected_language"] == "es"


def test_normalize_english_action():
    result = normalize("I want a thrilling action movie from the 90s")
    assert "action" in result["mapped_genres"]
    assert result["mapped_year_range"] == [1990, 1999]


def test_normalize_returns_normalized_text():
    result = normalize("quiero una pelicula de terror")
    assert isinstance(result["normalized_text"], str)
    assert len(result["normalized_text"]) > 0
