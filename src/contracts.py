"""Shared typed dictionary contracts for the CineAssist pipeline."""

from typing import TypedDict

from typing_extensions import NotRequired


class Preferences(TypedDict):
    """Structured preferences extracted from a user request."""

    genres: list[str]
    language: str | None
    year_range: list[int] | None
    mood: list[str]
    min_rating: float | None
    similar_to: str | None
    free_text: str


class ConversationState(Preferences):
    """Preferences retained across a conversation."""

    ui_language: str | None
    rating: NotRequired[float | None]


class RecommenderFilters(TypedDict):
    """Optional constraints understood by the recommendation engine."""

    language: str | None
    rating: float | None
    year: int | None


class FormFilters(TypedDict, total=False):
    """Structured filters accepted alongside free-form text."""

    genre: str
    mood: str
    year_range: list[int]
    language: str
    min_rating: float
    similar_to: str


class RecommendationCard(TypedDict):
    """Stable recommendation data consumed by the UI and API adapters."""

    movieId: int | None
    title: str
    year: int | None
    rating: float | None
    genres: list[str]
    language: str | None
    overview: str
    similarity: float
    explanation: str
    poster_url: NotRequired[str | None]
