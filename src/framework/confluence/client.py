"""Confluence REST API client: list pages, fetch body, convert to plain text for RAG."""

from urllib.parse import quote_plus
from typing import Any, Optional

import httpx

from .html_to_text import html_to_text


class ConfluenceClient:
    """
    Client for Confluence REST API (Cloud and Server/Data Center).
    Fetches pages by space key or page IDs and returns plain text + metadata for RAG ingest.
    """

    def __init__(
        self,
        base_url: str,
        *,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        base_url: Confluence base URL, e.g. https://your-site.atlassian.net/wiki (Cloud)
                  or https://confluence.company.com (Server/DC).
        Auth: for Cloud use email + api_token; for Server use username + password (basic auth).
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._auth = self._build_auth(email=email, api_token=api_token, username=username, password=password)

    def _build_auth(
        self,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> Optional[tuple[str, str]]:
        if email and api_token:
            return (email, api_token)
        if username and password:
            return (username, password)
        return None

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self.base_url}{path}"
        auth = kwargs.pop("auth", self._auth)
        return httpx.request(
            method,
            url,
            auth=auth,
            timeout=self.timeout,
            **kwargs,
        )

    def list_pages_in_space(
        self,
        space_key: str,
        *,
        limit: int = 100,
        start: int = 0,
    ) -> list[dict[str, Any]]:
        """
        List pages in a space using CQL. Returns list of {id, title, type, _links, ...}.
        """
        # CQL: space=KEY and type=page (no body expansion so we can get up to 1000)
        cql = f"space={space_key} and type=page"
        path = f"/rest/api/content/search?cql={quote_plus(cql)}&start={start}&limit={min(limit, 100)}"
        resp = self._request("GET", path)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])

    def get_page(self, page_id: str, expand: str = "body.storage") -> dict[str, Any]:
        """Get a single page by ID with body.storage (HTML) or body.view (rendered)."""
        path = f"/rest/api/content/{page_id}?expand={expand}"
        resp = self._request("GET", path)
        resp.raise_for_status()
        return resp.json()

    def get_page_text(self, page_id: str) -> tuple[str, dict[str, Any]]:
        """
        Fetch page and return (plain_text, metadata) for RAG.
        Confluence body.storage is HTML; we convert to plain text.
        """
        page = self.get_page(page_id, expand="body.storage,version,space")
        title = page.get("title", "")
        storage = (page.get("body") or {}).get("storage") or {}
        html = storage.get("value") or ""
        text = html_to_text(html)
        meta = {
            "source": "confluence",
            "confluence_id": page.get("id"),
            "title": title,
            "space_key": (page.get("space") or {}).get("key"),
            "version": (page.get("version") or {}).get("number"),
        }
        return (text, meta)

    def fetch_pages_for_ingest(
        self,
        *,
        space_key: Optional[str] = None,
        page_ids: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Fetch pages and return list of (text, metadata) for RAG add_documents.
        Either provide space_key (fetch all pages in space, up to limit) or page_ids.
        """
        if page_ids:
            ids = page_ids[: limit] if limit else page_ids
        elif space_key:
            results = self.list_pages_in_space(space_key, limit=limit)
            ids = []
            for r in results:
                # Search returns content objects: id at top level (v1) or under content
                pid = r.get("content", {}).get("id") or r.get("id")
                if pid:
                    ids.append(str(pid))
            ids = ids[:limit] if limit else ids
        else:
            return []

        out: list[tuple[str, dict[str, Any]]] = []
        for pid in ids:
            try:
                text, meta = self.get_page_text(pid)
                if (text or "").strip():
                    out.append((text, meta))
            except Exception:
                continue
        return out
