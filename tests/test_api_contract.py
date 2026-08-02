"""HTTP contract tests for the FastAPI adapter."""

from fastapi.testclient import TestClient

from backend.api import routes

client = TestClient(routes.app)


def test_health_endpoint():
    """Health checks expose a stable success payload."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend_endpoint_serializes_pipeline_result(monkeypatch):
    """Valid recommendation payloads preserve the API response contract."""
    monkeypatch.setattr(
        routes,
        "handle_user_message",
        lambda **_kwargs: {
            "detected_language": "en",
            "normalized_query": "space comedy",
            "preferences": {"genres": ["comedy"]},
            "recommendations": [],
            "metadata": {"status": "ok"},
        },
    )

    response = client.post(
        "/recommend",
        json={"raw_text": "space comedy", "top_n": 3},
    )

    assert response.status_code == 200
    assert response.json()["normalized_query"] == "space comedy"


def test_recommend_validation_error_becomes_422(monkeypatch):
    """Domain validation errors are client errors rather than opaque failures."""

    def reject_request(**_kwargs):
        raise ValueError("invalid model/data alignment")

    monkeypatch.setattr(routes, "handle_user_message", reject_request)

    response = client.post("/recommend", json={"raw_text": "comedy"})

    assert response.status_code == 422
    assert "alignment" in response.json()["detail"]


def test_framework_rejects_invalid_top_n():
    """Pydantic rejects a top_n value outside the declared request range."""
    response = client.post("/recommend", json={"raw_text": "comedy", "top_n": 0})

    assert response.status_code == 422


def test_translate_validation_error_becomes_422(monkeypatch):
    """Unsupported translation directions return a documented client error."""

    def reject_translation(**_kwargs):
        raise ValueError("unsupported language")

    monkeypatch.setattr(routes.translation_service, "translate", reject_translation)

    response = client.post(
        "/translate",
        json={"text": "bonjour", "source_language": "xx"},
    )

    assert response.status_code == 422
