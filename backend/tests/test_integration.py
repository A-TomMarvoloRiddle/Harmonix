import pytest
from httpx import AsyncClient, ASGITransport
from backend.main import app

@pytest.mark.asyncio
async def test_telemetry_ingestion():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "user_id": 1,
            "track_id": 42,
            "event_type": "skip",
            "duration_ms": 12450,
            "context": {
                "source": "vibe_radio",
                "device": "desktop",
                "sliders": {
                    "energy": 0.8,
                    "valence": 0.3,
                    "acousticness": 0.1,
                    "danceability": 0.7
                }
            },
            "timestamp": "2026-09-19T11:50:00Z"
        }
        response = await ac.post("/api/v1/telemetry", json=payload)
    assert response.status_code == 202
    assert response.json()["status"] == "queued"

@pytest.mark.asyncio
async def test_candidate_retrieval():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        payload = {
            "user_id": 1,
            "limit": 5,
            "apply_vibe_steering": True
        }
        response = await ac.post("/api/v1/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "tracks" in data

@pytest.mark.asyncio
async def test_gradcam_explainability():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/tracks/1/explain")
    assert response.status_code == 200
    assert "heatmap_matrix" in response.json()
