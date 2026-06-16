"""
Translation module for CineAssist.

CURRENT STATUS: Domain-specific stubs only.
The system handles multilingual input through language_service.py, which does
domain-specific normalization (genre/mood/decade keywords) without full translation.

TODO: Replace stubs with a real translation backend if needed:
  - Option A: Google Cloud Translation API (pip install google-cloud-translate)
  - Option B: DeepL API (pip install deepl)
  - Option C: HuggingFace transformers (pip install transformers sentencepiece)
    e.g. Helsinki-NLP/opus-mt-es-en, opus-mt-fr-en, opus-mt-pt-en

The NLP pipeline in nlp_preferences.py operates on English text.
For non-English input, language_service.normalize() maps known domain terms to
canonical English before passing text to the NLP extractor. Full sentence
translation would improve free-text queries beyond the domain vocabulary.
"""

# Supported language codes → human-readable names
_SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "pt": "Portuguese",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    STUB: Returns 'en' for all inputs.

    TODO: Integrate a real language detector, e.g.:
        from langdetect import detect
        return detect(text)
    or use language_service.normalize() which already does heuristic detection.

    Args:
        text: Raw input text from the user.

    Returns:
        ISO 639-1 language code (e.g. 'en', 'es', 'fr').
    """
    # TODO: implement real language detection
    return "en"


def translate_to_english(text: str, source_language: str | None = None) -> str:
    """
    Translate text from any language to English.

    STUB: Returns the input text unchanged.

    The current architecture uses domain-specific normalization in
    language_service.normalize() instead of full translation. This stub
    exists as the integration point for a real translation API.

    TODO: Implement using one of:
        # Google Cloud Translation:
        from google.cloud import translate_v2 as translate
        client = translate.Client()
        result = client.translate(text, target_language="en")
        return result["translatedText"]

        # DeepL:
        import deepl
        translator = deepl.Translator(api_key)
        return translator.translate_text(text, target_lang="EN-US").text

        # HuggingFace (offline, no API key needed):
        from transformers import pipeline
        pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
        return pipe(text)[0]["translation_text"]

    Args:
        text: Input text to translate.
        source_language: ISO 639-1 source language code. If None, auto-detect.

    Returns:
        English translation of the input text.
    """
    # TODO: implement real translation
    return text


def get_supported_languages() -> dict:
    """
    Return the set of languages that CineAssist can process.

    Note: The domain-specific normalization in language_service.py covers
    English, Spanish, French, and Portuguese for movie-domain vocabulary.
    Full translation (when implemented) would extend to all languages below.

    Returns:
        Dict mapping ISO 639-1 codes to human-readable language names.
    """
    return dict(_SUPPORTED_LANGUAGES)
