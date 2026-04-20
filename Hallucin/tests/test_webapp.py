from __future__ import annotations

from io import BytesIO

from hallucination_detector.webapp import create_app


def _client():
    app = create_app({"TESTING": True})
    return app.test_client(), app


def test_health_endpoint():
    client, _ = _client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"


def test_index_renders():
    client, _ = _client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hallucin Studio" in response.data


def test_default_upload_limit_is_large():
    _, app = _client()
    assert app.config["MAX_CONTENT_LENGTH"] >= 128 * 1024 * 1024


def test_analyze_json_payload():
    client, _ = _client()
    response = client.post(
        "/api/analyze",
        json={
            "context": "Paris is in France. The Eiffel Tower is 330 metres tall.",
            "response": "The Eiffel Tower is in Paris and is 330 metres tall.",
            "model_name": "local",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert 0.0 <= payload["score"] <= 1.0
    assert "claims" in payload


def test_analyze_file_upload():
    client, _ = _client()
    data = {
        "context_file": (BytesIO(b"The moon orbits Earth. Earth has one moon."), "context.txt"),
        "response_file": (BytesIO(b"The moon orbits Earth."), "response.txt"),
    }
    response = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    payload = response.get_json()
    assert "counts" in payload


def test_analyze_requires_inputs():
    client, _ = _client()
    response = client.post("/api/analyze", json={"context": "", "response": ""})
    assert response.status_code == 400
