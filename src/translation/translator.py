"""
Neural machine translation module for CineAssist.

WHAT THIS  DOES

Translates text between English and five other languages using pre-trained
neural machine translation models. It is used in two places:

  1. chatbot_flow.py — translates the user's message INTO English so the
     NLP preference extractor can always work on clean English text.

  2. chatbot_flow.py — translates the chatbot's English response BACK into
     the user's original language.

TECHNOLOGY: Helsinki-NLP MarianMT via HuggingFace Transformers

MarianMT is a family of neural translation models trained on millions of
parallel sentence pairs. CineAssist uses the Helsinki-NLP variants hosted
on HuggingFace (https://huggingface.co/Helsinki-NLP).

Each language pair uses a separate model. Models are downloaded
automatically on first use and cached in ~/.cache/huggingface/. Subsequent
calls use the in-memory _model_cache dict and are fast.

SUPPORTED LANGUAGE PAIRS

  es ↔ en   (Spanish ↔ English)
  fr ↔ en   (French ↔ English)
  pt ↔ en   (Portuguese ↔ English)
  de ↔ en   (German ↔ English)
  it ↔ en   (Italian ↔ English)

FINE-TUNING

If a fine-tuned model exists at models/translation/<src>-<tgt>/ it is loaded
instead of the base HuggingFace model. Run fine_tune.py to create one.
The fine-tuned model is trained on movie-domain sentence pairs and produces
more accurate translations for phrases like "película de acción" or "film d'horreur".

PUBLIC API (functions you call from outside this module)

  translate(text, src_lang, tgt_lang)  → str
  translate_to_english(text, src_lang) → str
  translate_from_english(text, tgt_lang) → str
"""

import os
import torch
from transformers import MarianMTModel, MarianTokenizer

# Maps (source_lang, target_lang) tuples to HuggingFace model IDs.
# These are the official Helsinki-NLP pretrained models.
SUPPORTED_PAIRS = {
    ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
    ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
    ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en',
    ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
    # Portuguese: no dedicated pt-en model exists on HuggingFace.
    # opus-mt-ROMANCE-en handles all Romance languages (ES, FR, PT, IT) → EN.
    # opus-mt-en-ROMANCE handles EN → all Romance languages (requires >>pt<< prefix in input,
    # but for our use case the output is acceptable without it).
    ('pt', 'en'): 'Helsinki-NLP/opus-mt-ROMANCE-en',
    ('en', 'pt'): 'Helsinki-NLP/opus-mt-en-ROMANCE',
    ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
    ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
    ('it', 'en'): 'Helsinki-NLP/opus-mt-it-en',
    ('en', 'it'): 'Helsinki-NLP/opus-mt-en-it',
}

# Path where fine_tune.py saves domain-adapted models.
# Structure: models/translation/es-en/  models/translation/en-es/  etc.
_BASE_DIR = os.path.dirname(__file__)
FINETUNED_DIR = os.path.normpath(os.path.join(
    _BASE_DIR, '..', '..', 'models', 'translation'))

# In-memory cache: avoids reloading the same model twice in one session.
# Key: "es-en" string. Value: (tokenizer, model) tuple.
_model_cache: dict = {}


def _load_model(src: str, tgt: str):
    """
    Load the tokenizer and model for a language pair.

    Checks the fine-tuned directory first; falls back to the HuggingFace
    pretrained model if no fine-tuned version exists.

    Args:
        src: Source language ISO 639-1 code (e.g. 'es').
        tgt: Target language ISO 639-1 code (e.g. 'en').

    Returns:
        (tokenizer, model) tuple, with model in eval mode.

    Raises:
        ValueError: If the language pair is not in SUPPORTED_PAIRS.
    """
    key = f"{src}-{tgt}"
    if key in _model_cache:
        return _model_cache[key]

    # Use the fine-tuned model if it was produced by fine_tune.py
    finetuned_path = os.path.join(FINETUNED_DIR, key)
    if os.path.exists(finetuned_path) and os.listdir(finetuned_path):
        model_path = finetuned_path
        print(f"[Translator] Loading fine-tuned model: {key}")
    else:
        model_path = SUPPORTED_PAIRS.get((src, tgt))
        if not model_path:
            raise ValueError(
                f"No translation model available for {src} → {tgt}")
        print(
            f"[Translator] Loading base model from HuggingFace: {model_path}")

    tokenizer = MarianTokenizer.from_pretrained(model_path)
    model = MarianMTModel.from_pretrained(model_path)
    model.eval()  # disable dropout — we are doing inference, not training

    _model_cache[key] = (tokenizer, model)
    return _model_cache[key]


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Translate text from one language to another.

    Uses beam search (num_beams=4) for better quality output. Beam search
    explores 4 candidate translations at each step and picks the best one,
    unlike greedy search which only picks the single most likely word.

    Args:
        text:     The text to translate.
        src_lang: Source language code (e.g. 'es').
        tgt_lang: Target language code (e.g. 'en').

    Returns:
        Translated string. Returns the original text unchanged if:
          - src_lang == tgt_lang (no translation needed)
          - text is blank/whitespace only
    """
    if src_lang == tgt_lang or not text.strip():
        return text

    tokenizer, model = _load_model(src_lang, tgt_lang)

    # Tokenize: convert the text string into a tensor of token IDs.
    # max_length=512 matches the model's maximum context window.
    inputs = tokenizer(
        [text],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
    )

    # Generate translation without computing gradients (faster, less memory).
    with torch.no_grad():
        outputs = model.generate(**inputs, num_beams=4, max_length=512)

    # Decode the output token IDs back into a human-readable string.
    # skip_special_tokens=True removes <pad>, </s> and similar markers.
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_to_english(text: str, src_lang: str) -> str:
    """
    Translate text from a supported language to English.

    This is called at the START of chatbot_response() to normalize user
    input before passing it to the NLP preference extractor.

    Args:
        text:     User's message in their language.
        src_lang: ISO 639-1 code of the source language (e.g. 'es').

    Returns:
        The text in English. If src_lang is 'en', returns text unchanged.
    """
    return translate(text, src_lang, 'en')


def translate_from_english(text: str, tgt_lang: str) -> str:
    """
    Translate an English string into the target language.

    This is called at the END of chatbot_response() to convert the chatbot's
    English-language reply into the user's original language before displaying it.

    Args:
        text:     English text to translate (e.g. a chatbot response).
        tgt_lang: ISO 639-1 code of the target language (e.g. 'es').

    Returns:
        The text in tgt_lang. If tgt_lang is 'en', returns text unchanged.
    """
    return translate(text, 'en', tgt_lang)
