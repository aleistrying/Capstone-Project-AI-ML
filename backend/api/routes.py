"""
FastAPI routes for CineAssist backend.

Run with:  uvicorn backend.api.routes:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.main import handle_user_message

app = FastAPI(
    title="CineAssist API",
    description="NLP-powered movie recommendation chatbot",
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class FormData(BaseModel):
    genre: str | None = None
    mood: str | None = None
    year_range: list[int] | None = None
    language: str | None = None
    min_rating: float | None = None
    similar_to: str | None = None


class RecommendRequest(BaseModel):
    raw_text: str
    form_data: FormData | None = None
    top_n: int = 5


class MovieResult(BaseModel):
    title: str
    year: int | None
    genres: list[str] | str
    rating: float
    score: float
    poster_url: str | None
    explanation: str


class RecommendResponse(BaseModel):
    detected_language: str
    normalized_query: str
    preferences: dict
    recommendations: list[MovieResult]
    metadata: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Accept a free-text movie request, return ranked recommendations.

    Example body:
        {
            "raw_text": "quiero una pelicula chistosa para familia de los 2000",
            "form_data": { "genre": null, "mood": null }
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
        raise HTTPException(status_code=500, detail=str(exc))

    return result
