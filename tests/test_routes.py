"""
tests/test_routes.py - Integration tests for Flask web routes, Live Social Feed, and REST APIs.
"""

import io
import pytest
from app import app, auto_initialize_system


@pytest.fixture(scope="module")
def client():
    auto_initialize_system()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Sentiment" in response.data


def test_analyze_route(client):
    response = client.get("/analyze")
    assert response.status_code == 200
    assert b"Analyze Social Media Post" in response.data


def test_bulk_route(client):
    response = client.get("/bulk")
    assert response.status_code == 200
    assert b"Bulk CSV Upload" in response.data


def test_live_feed_route(client):
    response = client.get("/live-feed")
    assert response.status_code == 200
    assert b"Real-Time Social Media Data Collection" in response.data


def test_analytics_route(client):
    response = client.get("/analytics")
    assert response.status_code == 200
    assert b"Dataset Analytics" in response.data


def test_performance_route(client):
    response = client.get("/performance")
    assert response.status_code == 200
    assert b"Model Performance" in response.data


def test_about_route(client):
    response = client.get("/about")
    assert response.status_code == 200
    assert b"About Project" in response.data


def test_api_predict_valid(client):
    payload = {"text": "I really love this project, it is fantastic! 😍", "source": "pytest"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert "sentiment" in data
    assert "confidence" in data
    assert data["sentiment"] in ["Positive", "Negative", "Neutral"]


def test_api_predict_empty(client):
    payload = {"text": "   "}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400


def test_api_stats(client):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.get_json()
    assert "total" in data
    assert "positive_pct" in data


def test_api_history(client):
    response = client.get("/api/history")
    assert response.status_code == 200
    data = response.get_json()
    assert "records" in data
    assert "count" in data


def test_api_fetch_live_all(client):
    response = client.get("/api/fetch-live?source=all&limit=6")
    assert response.status_code == 200
    data = response.get_json()
    assert "posts" in data
    assert "count" in data
    assert "summary" in data
    assert len(data["posts"]) > 0
    assert "sentiment" in data["posts"][0]
    assert "confidence" in data["posts"][0]


def test_api_fetch_live_sources(client):
    for src in ["reddit", "mastodon", "youtube"]:
        response = client.get(f"/api/fetch-live?source={src}&limit=4")
        assert response.status_code == 200
        data = response.get_json()
        assert "posts" in data
        assert len(data["posts"]) > 0


def test_bulk_upload_csv(client):
    csv_content = "id,text\n1,I love this product!\n2,Horrible and broken\n3,The flight takes off at noon\n"
    data = {
        "csv_file": (io.BytesIO(csv_content.encode("utf-8")), "test_posts.csv"),
        "max_rows": "500",
    }
    response = client.post("/bulk-upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Processed: 3 rows" in response.data
