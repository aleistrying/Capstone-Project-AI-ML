"""Tests for lightweight language detection and vocabulary normalization."""

from backend.services import language_service
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


def test_language_detection_accepts_exactly_two_stopword_hits():
    """The documented detection threshold is inclusive at two unique hits."""
    assert detect_language("quiero pelicula") == "es"


def test_normalize_text_lowercases_and_removes_accents():
    """Canonical matching text is lowercase and accent-independent."""
    # pylint: disable=protected-access
    assert language_service._normalize_text("ÁCCIÓN Française") == "accion francaise"


def test_mapped_labels_only_returns_matching_canonical_values():
    """Synonym mapping neither drops matches nor adds unrelated labels."""
    # pylint: disable=protected-access
    vocabulary = {"comedy": ["funny"], "drama": ["serious"]}
    assert language_service._mapped_labels("a funny film", vocabulary) == ["comedy"]


def test_normalize_explicit_year_mood_and_movie_language():
    """Exact years, moods, and requested movie language remain observable."""
    result = normalize("I want a dark movie from 1997 in Spanish")

    assert result["mapped_year_range"] == [1997, 1997]
    assert result["mapped_moods"] == ["dark"]
    assert result["mapped_language"] == "es"
    assert result["normalized_text"] == (
        "dark from 1997 I want a dark movie from 1997 in Spanish"
    )


def test_normalize_has_exact_stable_response_keys():
    """Consumers receive every documented key even when nothing is mapped."""
    result = normalize("hello there")

    assert result == {
        "detected_language": "en",
        "normalized_text": "hello there",
        "mapped_genres": [],
        "mapped_moods": [],
        "mapped_year_range": None,
        "mapped_language": None,
    }


def test_decade_normalized_text_uses_start_year_label():
    """Decade normalization uses the start year and ordinary spaces."""
    result = normalize("I want a comedy from the 2000s")

    assert result["normalized_text"] == (
        "comedy from the 2000s I want a comedy from the 2000s"
    )
