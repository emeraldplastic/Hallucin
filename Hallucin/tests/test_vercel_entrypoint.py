from __future__ import annotations


def test_vercel_entrypoint_health():
    from api.index import app

    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
