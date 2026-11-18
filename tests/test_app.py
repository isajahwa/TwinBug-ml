from fastapi.testclient import TestClient
import os

os.environ.setdefault("ML_API_KEY", "test-secret-key")
os.environ.setdefault("MAX_CANDIDATES", "1000")

from app import app  # noqa: E402

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_detect_empty_candidates():
    headers = {"X-API-Key": "test-secret-key"}
    response = client.post(
        "/detect",
        json={
            "summary": "login fails",
            "description": "users cannot log in",
            "candidates": [],
            "top_k": 5,
        },
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert "threshold_used" in body


def test_detect_sorted_and_filtered():
    headers = {"X-API-Key": "test-secret-key"}
    response = client.post(
        "/detect",
        json={
            "summary": "login fails on mobile",
            "description": "users cannot log in on mobile app",
            "candidates": [
                {"id": 1, "summary": "login error", "description": "login screen returns error"},
                {"id": 2, "summary": "payment crash", "description": "checkout crashes on submit"},
                {"id": 3, "summary": "login fails on mobile", "description": "users cannot log in on mobile app"},
            ],
            "top_k": 2,
        },
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    results = body["results"]
    assert len(results) == 2
    scores = [r["text_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    ids = [r["id"] for r in results]
    assert 3 in ids


def test_detect_api_key_missing():
    response = client.post(
        "/detect",
        json={
            "summary": "login fails",
            "description": "x",
            "candidates": [{"id": 1, "summary": "login error", "description": "x"}],
        },
    )
    assert response.status_code == 401


def test_detect_api_key_wrong():
    response = client.post(
        "/detect",
        json={
            "summary": "login fails",
            "description": "x",
            "candidates": [{"id": 1, "summary": "login error", "description": "x"}],
        },
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


def test_detect_api_key_correct():
    response = client.post(
        "/detect",
        json={
            "summary": "login fails",
            "description": "x",
            "candidates": [{"id": 1, "summary": "login error", "description": "x"}],
        },
        headers={"X-API-Key": "test-secret-key"},
    )
    assert response.status_code == 200


def test_health_public_with_key():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_duplicate_api_key():
    response = client.post(
        "/predict-duplicate",
        json={
            "summary_a": "login fails",
            "description_a": "x",
            "summary_b": "login error",
            "description_b": "x",
        },
    )
    assert response.status_code == 401


def test_exceeding_max_candidates():
    headers = {"X-API-Key": "test-secret-key"}
    candidates = [{"id": i, "summary": "bug", "description": "x"} for i in range(1001)]
    response = client.post(
        "/detect",
        json={"summary": "login fails", "description": "x", "candidates": candidates, "top_k": 5},
        headers=headers,
    )
    assert response.status_code == 422
    assert "Too many candidates" in response.json()["detail"]
