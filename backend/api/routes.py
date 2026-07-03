"""
FastAPI routes for CineAssist backend.

ENDPOINTS
----------
  GET  /health          → confirm the API is running
  POST /recommend       → main recommendation pipeline (Team)
  POST /translate       → translate text to English (Brayan)
  GET  /languages       → list supported languages (Brayan)

Run locally with:
  uvicorn backend.api.routes:app --reload

Or use the run script:
  ./run.sh api
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.main import handle_user_message
from backend.services import translation_service

app = FastAPI(
    title="CineAssist API",
    description="NLP-powered movie recommendation chatbot with multilingual support",
    version="0.2.0",
)


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models — define the shape of request and response bodies
# ─────────────────────────────────────────────────────────────────────────────

class FormData(BaseModel):
    """Optional structured filters the user can send alongside free text."""
    genre: str | None = None
    mood: str | None = None
    year_range: list[int] | None = None
    language: str | None = None
    min_rating: float | None = None
    similar_to: str | None = None


class RecommendRequest(BaseModel):
    """Body for POST /recommend."""
    raw_text: str
    form_data: FormData | None = None
    top_n: int = 5


class MovieResult(BaseModel):
    """One movie in the recommendation list."""
    title: str
    year: int | None
    genres: list[str] | str
    rating: float
    score: float
    poster_url: str | None
    explanation: str


class RecommendResponse(BaseModel):
    """Full response from POST /recommend."""
    detected_language: str
    normalized_query: str
    preferences: dict
    recommendations: list[MovieResult]
    metadata: dict


class TranslateRequest(BaseModel):
    """
    Body for POST /translate.

    Brayan's endpoint: accepts raw text in any supported language and
    returns the English translation along with metadata about what happened.
    """
    text: str
    source_language: str | None = None   # optional: ISO 639-1 code e.g. 'es'


class TranslateResponse(BaseModel):
    """
    Response from POST /translate.

    Fields:
      original_text:    the text you sent
      translated_text:  the English translation
      detected_language: what language we detected (or you provided)
      was_translated:   False if the input was already English
    """
    original_text: str
    translated_text: str
    detected_language: str
    was_translated: bool


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Confirm the API server is up and responding."""
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Main recommendation pipeline.

    Accepts free-text in any supported language, extracts movie preferences,
    runs TF-IDF cosine similarity, and returns ranked movie recommendations.

    Language handling here uses the fast stopword-based normalization from
    language_service.py (no ML, instant startup). For full neural translation,
    use the /translate endpoint first.

    Example body:
        {
            "raw_text": "quiero una pelicula chistosa para familia de los 2000",
            "top_n": 5
        }
    """
    form = request.form_data.model_dump() if request.form_data else None

    try:
        result = handle_user_message(
            raw_text=request.raw_text,
            form_data=form,
            top_n=request.top_n,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest):
    """
    Translate text from any supported language to English.

    This endpoint is Brayan's contribution. It uses Helsinki-NLP MarianMT
    neural models running locally (no external API key needed).

    Supported source languages: Spanish (es), French (fr), Portuguese (pt),
    German (de), Italian (it). English input is returned unchanged.

    The first call for a language pair downloads the model from HuggingFace
    (~300 MB). Subsequent calls use the in-memory cache and are fast.

    Example body:
        { "text": "Quiero ver una pelicula de terror" }

    Example response:
        {
            "original_text": "Quiero ver una pelicula de terror",
            "translated_text": "I want to watch a horror movie",
            "detected_language": "es",
            "was_translated": true
        }

    With explicit source language:
        { "text": "Je veux un film d'action", "source_language": "fr" }
    """
    try:
        result = translation_service.translate(
            text=request.text,
            source_language=request.source_language,
        )
    except ValueError as exc:
        # Raised when the language pair has no available model
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@app.get("/languages")
def languages():
    """
    List all languages supported by the translation module.

    Returns a dict mapping ISO 639-1 codes to human-readable names.
    These are the languages users can write in and the chatbot will understand.

    Example response:
        {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "pt": "Portuguese",
            "de": "German",
            "it": "Italian"
        }
    """
    return translation_service.supported_languages()
