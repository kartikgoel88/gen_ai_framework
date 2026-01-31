"""Tests for framework API (FastAPI app)."""

import pytest
from fastapi.testclient import TestClient

from src.framework.api.app import create_app


@pytest.fixture
def client():
    """Test client for the framework app."""
    app = create_app(title="Test API", version="0.0.1")
    return TestClient(app)


def test_health(client: TestClient):
    """GET /health returns status ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_redirects_to_docs(client: TestClient):
    """GET / redirects to /docs."""
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/docs"


def test_info(client: TestClient):
    """GET /info returns service info."""
    r = client.get("/info")
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "Test API"
    assert data["version"] == "0.0.1"
    assert data["docs"] == "/docs"
