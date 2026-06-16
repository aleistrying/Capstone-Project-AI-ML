"""
Translation service wrapper for the backend controller.

Currently a pass-through: delegates to the stub in src/translation/translator.py.
Wire a real translation backend there to activate full translation.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.translation.translator import (
    detect_language,
    get_supported_languages,
    translate_to_english,
)


def translate(text: str, source_language: str | None = None) -> dict:
    """
    Translate text to English and return enriched metadata.

    Args:
        text: Raw user input.
        source_language: ISO 639-1 code or None for auto-detect.

    Returns:
        {
          "original_text": str,
          "translated_text": str,
          "detected_language": str,
          "was_translated": bool,
        }
    """
    detected = source_language or detect_language(text)
    translated = translate_to_english(text, source_language=detected)
    return {
        "original_text": text,
        "translated_text": translated,
        "detected_language": detected,
        "was_translated": translated != text,
    }


def supported_languages() -> dict:
    return get_supported_languages()
