"""Tests for Confluence client and RAG ingest endpoint."""

import pytest
from unittest.mock import patch

from src.framework.confluence.html_to_text import html_to_text
from src.framework.confluence.client import ConfluenceClient


def test_html_to_text_basic():
    """HTML is stripped to plain text."""
    html = "<p>Hello <b>world</b></p>"
    out = html_to_text(html)
    assert "Hello" in out and "world" in out


def test_html_to_text_empty():
    """Empty or whitespace HTML returns empty string."""
    assert html_to_text("") == ""
    assert html_to_text("   ") == ""


def test_html_to_text_script_removed():
    """Script and style tags are removed."""
    html = "<p>Visible</p><script>alert(1)</script><style>.x{}</style>"
    out = html_to_text(html)
    assert "Visible" in out
    assert "alert" not in out and ".x" not in out


def test_confluence_client_build_auth():
    """Auth is built from email+token or user+password."""
    c1 = ConfluenceClient("https://site.atlassian.net/wiki", email="a@b.com", api_token="t")
    assert c1._auth == ("a@b.com", "t")
    c2 = ConfluenceClient("https://confluence.local", username="u", password="p")
    assert c2._auth == ("u", "p")
    c3 = ConfluenceClient("https://site.atlassian.net/wiki")
    assert c3._auth is None


def test_confluence_client_fetch_pages_empty():
    """fetch_pages_for_ingest with no space_key or page_ids returns []."""
    client = ConfluenceClient("https://site.atlassian.net/wiki", email="a@b.com", api_token="t")
    out = client.fetch_pages_for_ingest()
    assert out == []


def test_confluence_client_fetch_pages_by_ids_mocked():
    """fetch_pages_for_ingest with page_ids uses get_page_text per id."""
    client = ConfluenceClient("https://site.atlassian.net/wiki", email="a@b.com", api_token="t")
    with patch.object(client, "get_page_text") as m:
        m.return_value = ("Page content here", {"title": "Test", "confluence_id": "123"})
        out = client.fetch_pages_for_ingest(page_ids=["123"], limit=10)
    assert len(out) == 1
    assert out[0][0] == "Page content here"
    assert out[0][1]["title"] == "Test"
    m.assert_called_once_with("123")


def test_rag_ingest_confluence_not_configured():
    """POST /tasks/rag/ingest/confluence returns 503 when Confluence is not configured."""
    from fastapi.testclient import TestClient
    from src.main import app

    client = TestClient(app)
    # Without CONFLUENCE_BASE_URL, get_confluence_client returns None
    r = client.post(
        "/tasks/rag/ingest/confluence",
        json={"space_key": "DEMO"},
    )
    assert r.status_code == 503
    assert "not configured" in r.json()["detail"].lower()


def test_rag_ingest_confluence_bad_request():
    """POST /tasks/rag/ingest/confluence with no space_key or page_ids returns 400 when Confluence is configured."""
    from fastapi.testclient import TestClient
    from src.main import app
    from src.framework.api import deps

    fake_client = ConfluenceClient("https://site.atlassian.net/wiki", email="a@b.com", api_token="t")
    app.dependency_overrides[deps.get_confluence_client] = lambda: fake_client
    try:
        client = TestClient(app)
        r = client.post("/tasks/rag/ingest/confluence", json={})
        assert r.status_code == 400
        assert "space_key" in r.json()["detail"].lower() or "page_ids" in r.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(deps.get_confluence_client, None)
