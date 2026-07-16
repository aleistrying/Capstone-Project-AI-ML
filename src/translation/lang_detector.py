"""
Language detection module for CineAssist.

WHAT THIS DOES

Identifies what language a user is writing in and returns a standard
ISO 639-1 language code (e.g. 'es' for Spanish, 'fr' for French).

This is the FIRST step in chatbot_flow.py. The detected language is
stored in the conversation state (state_dict['ui_language']) so that:
  - The user's message can be translated to English for NLP processing.
  - The chatbot's response can be translated back to the user's language.

TECHNOLOGY

Uses the 'langdetect' library, a Python port of Google's language detection
library. It works by analyzing character n-grams (sequences of characters)
and comparing them to statistical profiles of 55 languages.

DetectorFactory.seed = 42 makes detection deterministic — same input always
gives the same result, which is important for reproducibility.

FALLBACK BEHAVIOR
------------------
If detection fails (e.g. input is a single word, or an unsupported language
like Japanese), the function returns 'en' (English) as a safe default.
This ensures the rest of the pipeline always receives a valid language code.

SUPPORTED LANGUAGES
--------------------
  en — English
  es — Spanish
  fr — French
  pt — Portuguese
  de — German
  it — Italian

Any language langdetect identifies that is NOT in this set falls back to 'en'.
"""

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Fix the random seed so detection is reproducible across runs.
# Without this, langdetect uses a random seed and the same text can
# return different results on different calls.
DetectorFactory.seed = 42

# Languages we have translation models for (see translator.py SUPPORTED_PAIRS).
SUPPORTED = {'en', 'es', 'fr', 'pt', 'de', 'it'}

# Human-readable names for each supported language code.
# Also used by get_supported_languages() in __init__.py.
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'pt': 'Portuguese',
    'de': 'German',
    'it': 'Italian',
}


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Uses statistical n-gram analysis (Google langdetect) to identify the
    language. Falls back to 'en' for empty input, detection errors, or
    languages we don't have translation models for.

    Args:
        text: The raw user input to analyze. Can be a single word or
              a full sentence. Longer text gives more accurate results.

    Returns:
        ISO 639-1 language code string. Always one of: 'en', 'es', 'fr',
        'pt', 'de', 'it'. Never raises an exception.

    Examples:
        detect_language("I want a comedy movie")   → 'en'
        detect_language("Quiero una comedia")       → 'es'
        detect_language("Je veux un film d'action") → 'fr'
        detect_language("")                          → 'en'  (fallback)
        detect_language("映画が見たい")               → 'en'  (unsupported → fallback)
    """
    if not text or not text.strip():
        return 'en'
    try:
        lang = detect(text)
        # Only return the detected language if we have a model for it.
        # Everything else (e.g. Japanese 'ja', Arabic 'ar') falls back to English.
        return lang if lang in SUPPORTED else 'en'
    except LangDetectException:
        # Raised for very short or ambiguous text (e.g. a single number).
        return 'en'
