import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.nlp.nlp_preferences import extract_preferences


def test_genre_extraction():
    prefs = extract_preferences("I want a funny family comedy from the 90s")
    assert "comedy" in prefs["genres"]
    assert "family" in prefs["genres"]


def test_decade_year_range():
    prefs = extract_preferences("something from the 90s")
    assert prefs["year_range"] == [1990, 1999]


def test_explicit_year():
    prefs = extract_preferences("a movie from 2001")
    assert prefs["year_range"] == [2001, 2001]


def test_language_extraction():
    prefs = extract_preferences("recommend me a movie in Spanish")
    assert prefs["language"] == "es"


def test_rating_extraction():
    prefs = extract_preferences("movies with rating above 8")
    assert prefs["min_rating"] == 8.0


def test_similar_to():
    prefs = extract_preferences("something similar to Inception")
    assert prefs["similar_to"] == "Inception"


def test_mood_extraction():
    prefs = extract_preferences("I want something dark and intense")
    assert "dark" in prefs["mood"]
    assert "intense" in prefs["mood"]
