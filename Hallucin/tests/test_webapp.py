from __future__ import annotations

from io import BytesIO

from hallucination_detector.webapp import create_app


def _client(config: dict | None = None):
    app_config = {"TESTING": True}
    if config:
        app_config.update(config)
    app = create_app(app_config)
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


def test_privacy_headers_enabled_by_default():
    client, _ = _client()
    response = client.get("/health")
    assert response.headers["Cache-Control"] == "no-store, no-cache, must-revalidate, max-age=0"
    assert response.headers["Pragma"] == "no-cache"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_privacy_headers_can_be_disabled():
    client, _ = _client({"ENABLE_PRIVACY_HEADERS": "0"})
    response = client.get("/health")
    assert response.headers.get("X-Frame-Options") is None
    assert response.headers.get("Referrer-Policy") is None


def test_analyze_rejects_disallowed_upload_extension():
    client, _ = _client()
    data = {
        "context_file": (BytesIO(b"context"), "context.exe"),
        "response_file": (BytesIO(b"response"), "response.txt"),
    }
    response = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Unsupported file type" in response.get_json()["error"]


def test_upload_extensions_config_accepts_string():
    client, _ = _client({"UPLOAD_ALLOWED_EXTENSIONS": ".txt"})
    data = {
        "context_file": (BytesIO(b"context"), "context.md"),
        "response_file": (BytesIO(b"response"), "response.txt"),
    }
    response = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "Unsupported file type '.md'" in response.get_json()["error"]


def test_analyze_rejects_binary_upload():
    client, _ = _client()
    data = {
        "context_file": (BytesIO(b"\x00\x01binary"), "context.txt"),
        "response_file": (BytesIO(b"plain text"), "response.txt"),
    }
    response = client.post("/api/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "UTF-8 text" in response.get_json()["error"]


def test_analyze_rate_limited_after_threshold():
    client, _ = _client({"RATE_LIMIT_REQUESTS": 2, "RATE_LIMIT_WINDOW_SECONDS": 60})
    headers = {"X-Forwarded-For": "198.51.100.42"}
    payload = {"context": "", "response": ""}

    first = client.post("/api/analyze", json=payload, headers=headers)
    second = client.post("/api/analyze", json=payload, headers=headers)
    third = client.post("/api/analyze", json=payload, headers=headers)

    assert first.status_code == 400
    assert second.status_code == 400
    assert third.status_code == 429

    body = third.get_json()
    assert body["error"] == "Rate limit exceeded. Try again later."
    assert body["limit"] == 2
    assert body["window_seconds"] == 60
    assert body["retry_after_seconds"] >= 1
    assert third.headers["Retry-After"] == str(body["retry_after_seconds"])


def test_analyze_rate_limited_per_client():
    client, _ = _client({"RATE_LIMIT_REQUESTS": 1, "RATE_LIMIT_WINDOW_SECONDS": 60})
    payload = {"context": "", "response": ""}

    first_client = {"X-Forwarded-For": "203.0.113.10"}
    second_client = {"X-Forwarded-For": "203.0.113.11"}

    first = client.post("/api/analyze", json=payload, headers=first_client)
    blocked = client.post("/api/analyze", json=payload, headers=first_client)
    second = client.post("/api/analyze", json=payload, headers=second_client)

    assert first.status_code == 400
    assert blocked.status_code == 429
    assert second.status_code == 400
