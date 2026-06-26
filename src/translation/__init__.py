"""
Translation package for CineAssist.
Author: Brayan Yesid Roncancio Suarez

This package provides multilingual support for the chatbot. It is split into
two modules that each handle one responsibility:

  lang_detector.py  — DETECT what language the user is writing in
                       Uses: Google langdetect (statistical, no ML model)

  translator.py     — TRANSLATE text between English and other languages
                       Uses: Helsinki-NLP MarianMT (neural, HuggingFace)

  fine_tune.py      — FINE-TUNE the MarianMT models on movie-domain data
                       Run once to adapt the base models to movie vocabulary.
                       Not imported here; run directly as a script.

HOW TO USE THIS PACKAGE FROM OTHER MODULES
-------------------------------------------
Always import from the package (src.translation), not from the sub-modules
directly. This __init__.py re-exports everything under one namespace:

    from src.translation import detect_language
    from src.translation import translate_to_english, translate_from_english
    from src.translation import get_supported_languages

WHERE THIS PACKAGE IS USED
----------------------------
  src/chatbot/chatbot_flow.py       — main integration point (Streamlit path)
  backend/services/translation_service.py — API wrapper (FastAPI path)
"""

from .lang_detector import detect_language, SUPPORTED, LANGUAGE_NAMES
from .translator import translate, translate_to_english, translate_from_english


def get_supported_languages() -> dict:
    """
    Return all languages this translation module supports.

    Returns:
        Dict mapping ISO 639-1 codes to human-readable names.
        e.g. {'en': 'English', 'es': 'Spanish', 'fr': 'French', ...}

    Used by:
        backend/services/translation_service.supported_languages()
        GET /languages endpoint in backend/api/routes.py
    """
    return dict(LANGUAGE_NAMES)


__all__ = [
    "detect_language",
    "translate",
    "translate_to_english",
    "translate_from_english",
    "get_supported_languages",
    "SUPPORTED",
    "LANGUAGE_NAMES",
]
