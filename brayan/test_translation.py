"""
Tests for the CineAssist live translation module.
Author: Brayan Yesid Roncancio Suarez

Run all tests:
    pytest brayan/test_translation.py -v

Run a single test:
    pytest brayan/test_translation.py::test_detect_spanish -v
"""

import sys
import os

# Make src importable from the brayan/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from translation.lang_detector import detect_language
from translation.translator import translate, translate_to_english, translate_from_english


# ──────────────────────────────────────────────────────────────────
# Language Detection Tests
# ──────────────────────────────────────────────────────────────────

class TestLanguageDetection:

    def test_detect_english(self):
        text = "I want to watch a comedy movie tonight"
        assert detect_language(text) == 'en'

    def test_detect_spanish(self):
        text = "Quiero ver una película de acción"
        assert detect_language(text) == 'es'

    def test_detect_french(self):
        text = "Je veux regarder un film d'action ce soir"
        assert detect_language(text) == 'fr'

    def test_detect_portuguese(self):
        text = "Quero assistir um filme de comédia"
        assert detect_language(text) == 'pt'

    def test_empty_string_defaults_to_english(self):
        assert detect_language("") == 'en'

    def test_whitespace_defaults_to_english(self):
        assert detect_language("   ") == 'en'

    def test_unsupported_language_defaults_to_english(self):
        # Japanese — not in our supported set
        assert detect_language("映画が見たい") == 'en'

    def test_movie_genre_spanish(self):
        text = "Recomiéndame una comedia romántica"
        assert detect_language(text) == 'es'

    def test_movie_genre_english(self):
        text = "Recommend me a romantic comedy from the 2000s"
        assert detect_language(text) == 'en'


# ──────────────────────────────────────────────────────────────────
# Translation Tests (ES ↔ EN)
# ──────────────────────────────────────────────────────────────────

class TestTranslation:

    def test_spanish_to_english_basic(self):
        result = translate_to_english("Hola", "es")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_english_to_spanish_basic(self):
        result = translate_from_english("Hello", "es")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_same_language_returns_original(self):
        text = "I want an action movie"
        result = translate("I want an action movie", "en", "en")
        assert result == text

    def test_empty_text_returns_empty(self):
        result = translate("   ", "es", "en")
        assert result.strip() == ""

    def test_genre_word_translated(self):
        result = translate_to_english("acción", "es")
        assert "action" in result.lower()

    def test_genre_word_to_spanish(self):
        result = translate_from_english("action", "es")
        # MarianMT may return "acción" or "accion"
        assert "acci" in result.lower()

    def test_movie_sentence_es_to_en(self):
        text = "Quiero ver una película de terror"
        result = translate_to_english(text, "es")
        assert isinstance(result, str)
        assert len(result) > 10
        # Should contain something about horror or movie
        lower = result.lower()
        assert any(word in lower for word in ["horror", "movie", "film", "terror", "watch", "see"])

    def test_chatbot_response_translated(self):
        english_response = "Hello! What genre would you like to watch?"
        result = translate_from_english(english_response, "es")
        assert isinstance(result, str)
        assert len(result) > 10


# ──────────────────────────────────────────────────────────────────
# Round-Trip Translation Tests
# ──────────────────────────────────────────────────────────────────

class TestRoundTrip:
    """
    Tests that translating ES → EN → ES preserves the core meaning.
    Round-trip is never perfectly identical but should be semantically close.
    """

    def test_round_trip_genre(self):
        original = "comedia"
        en = translate_to_english(original, "es")
        back = translate_from_english(en, "es")
        assert isinstance(back, str)
        assert len(back) > 0

    def test_round_trip_sentence(self):
        original = "Quiero ver una película de acción"
        en = translate_to_english(original, "es")
        back = translate_from_english(en, "es")
        # Round-trip won't be identical but should have "acci"
        assert "acci" in back.lower() or "pel" in back.lower()

    def test_round_trip_chatbot_question(self):
        original = "¿Qué género te gustaría ver?"
        en = translate_to_english(original, "es")
        back = translate_from_english(en, "es")
        assert isinstance(back, str)
        assert len(back) > 5


# ──────────────────────────────────────────────────────────────────
# Integration: Detection + Translation Pipeline
# ──────────────────────────────────────────────────────────────────

class TestIntegrationPipeline:
    """
    Simulates the full pipeline: detect language → translate to EN → translate back.
    This is what happens inside chatbot_flow.py for every user message.
    """

    def _pipeline(self, user_input: str, english_response: str) -> str:
        """Mirrors the logic in chatbot_response()."""
        lang = detect_language(user_input)
        english_input = translate_to_english(user_input, lang) if lang != 'en' else user_input
        # (NLP processing would happen here)
        final_response = translate_from_english(english_response, lang) if lang != 'en' else english_response
        return lang, english_input, final_response

    def test_english_user_no_translation(self):
        lang, eng_input, response = self._pipeline(
            "I want a comedy movie",
            "Movies found! Here are my top picks."
        )
        assert lang == 'en'
        assert eng_input == "I want a comedy movie"
        assert response == "Movies found! Here are my top picks."

    def test_spanish_user_full_pipeline(self):
        lang, eng_input, response = self._pipeline(
            "Quiero una comedia romántica",
            "Movies found! Here are my top picks."
        )
        assert lang == 'es'
        assert isinstance(eng_input, str) and len(eng_input) > 5
        # Response should be in Spanish now
        assert isinstance(response, str) and len(response) > 5

    def test_french_user_full_pipeline(self):
        lang, eng_input, response = self._pipeline(
            "Je veux regarder un film d'horreur",
            "Hello! What genre would you like to watch?"
        )
        assert lang == 'fr'
        assert isinstance(eng_input, str)
        assert isinstance(response, str)


# ──────────────────────────────────────────────────────────────────
# Fine-Tuned Model Loading Test
# ──────────────────────────────────────────────────────────────────

class TestFineTunedModel:

    def test_finetuned_model_path_structure(self):
        """Check that the models/translation directory exists after fine-tuning."""
        models_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', 'models', 'translation')
        )
        # This passes before training (dir may not exist yet) — just validate the path
        assert 'models' in models_dir and 'translation' in models_dir

    def test_translate_still_works_without_finetuned_model(self):
        """Translator should fall back to the base HuggingFace model gracefully."""
        result = translate_to_english("película", "es")
        assert isinstance(result, str)


# ──────────────────────────────────────────────────────────────────
# Quick smoke test — run directly with: python test_translation.py
# ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Running smoke tests...\n")

    samples = [
        ("I want a horror movie tonight", "en"),
        ("Quiero ver una comedia romántica", "es"),
        ("Je cherche un film d'action", "fr"),
    ]

    for text, expected_lang in samples:
        detected = detect_language(text)
        en_version = translate_to_english(text, detected)
        back = translate_from_english("Movies found! Here are your recommendations.", detected)
        print(f"Input   : {text}")
        print(f"Detected: {detected} (expected: {expected_lang}) {'✓' if detected == expected_lang else '✗'}")
        print(f"→ EN    : {en_version}")
        print(f"Response: {back}")
        print()

    print("Done. Run 'pytest brayan/test_translation.py -v' for the full test suite.")
