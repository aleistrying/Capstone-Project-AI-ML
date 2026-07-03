"""
Translation service — backend wrapper around Brayan's translation module.

HOW THIS FITS IN THE ARCHITECTURE
-----------------------------------
The CineAssist backend has two parallel strategies for handling non-English input:

  1. language_service.py  (fast, no ML)
     Detects language using Spanish/French/Portuguese stopwords and maps
     movie-domain vocabulary (genre names, moods, decades) to their canonical
     English equivalents. Works offline with zero startup time.
     Used by: backend/main.py for every /recommend request.

  2. translation_service.py  (accurate, ML-based)  ← THIS FILE
     Full sentence translation using Helsinki-NLP MarianMT neural models via
     HuggingFace Transformers. Handles free-form sentences that go beyond the
     domain vocabulary covered by language_service. Supports EN ↔ ES/FR/PT/DE/IT.
     Used by: /translate API endpoint (routes.py).

IMPORTANT — Import path
------------------------
This file imports from the `src.translation` PACKAGE (the __init__.py), NOT
directly from `src.translation.translator`. This is intentional:
  - detect_language lives in src/translation/lang_detector.py
  - translate_to_english/translate_from_english live in src/translation/translator.py
  - get_supported_languages is defined in src/translation/__init__.py
The __init__.py re-exports all of them under a single namespace.
"""

import sys
from pathlib import Path

# Ensure the project root is on the path so `src.translation` can be resolved
# from anywhere (CLI, uvicorn, notebooks, etc.)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Import from the PACKAGE (__init__.py), not the raw module file.
# This is the fix for the original stub which imported directly from translator.py
# and crashed because detect_language and get_supported_languages don't live there.
from src.translation import (
    detect_language,          # from src/translation/lang_detector.py
    get_supported_languages,  # defined in src/translation/__init__.py
    translate_to_english,     # from src/translation/translator.py
    translate_from_english,   # from src/translation/translator.py
)


def translate(text: str, source_language: str | None = None) -> dict:
    """
    Translate free-form text to English and return enriched metadata.

    This is the main function called by the /translate API endpoint.
    It auto-detects the language if source_language is not provided.

    Args:
        text: Raw user input in any supported language.
        source_language: Optional ISO 639-1 code (e.g. 'es', 'fr').
                         If None, the language is auto-detected using langdetect.

    Returns:
        A dict with:
          - original_text:    the text as received
          - translated_text:  English translation (same as original if already English)
          - detected_language: ISO 639-1 code of the detected/provided language
          - was_translated:   True if translation actually happened

    Example:
        translate("Quiero una pelicula de acción")
        → {
            "original_text": "Quiero una pelicula de acción",
            "translated_text": "I want an action movie",
            "detected_language": "es",
            "was_translated": True
          }
    """
    # Use provided language code or auto-detect with langdetect
    detected = source_language or detect_language(text)

    # translate_to_english is a no-op when src_lang == 'en'
    translated = translate_to_english(text, detected)

    return {
        "original_text": text,
        "translated_text": translated,
        "detected_language": detected,
        "was_translated": translated != text,
    }


def translate_response(text: str, target_language: str) -> dict:
    """
    Translate an English chatbot response to the user's language.

    Used when the chatbot generates a response in English and needs to
    send it back in the language the user originally wrote in.

    Args:
        text: English text to translate (chatbot response, explanation, etc.)
        target_language: ISO 639-1 code of the target language (e.g. 'es').

    Returns:
        A dict with:
          - original_text:    the English source text
          - translated_text:  text in target_language
          - target_language:  the ISO 639-1 code used
    """
    translated = translate_from_english(text, target_language)
    return {
        "original_text": text,
        "translated_text": translated,
        "target_language": target_language,
    }


def supported_languages() -> dict:
    """
    Return all language codes this translation module supports.

    Returns:
        Dict mapping ISO 639-1 codes to human-readable names.
        e.g. {'en': 'English', 'es': 'Spanish', 'fr': 'French', ...}
    """
    return get_supported_languages()
