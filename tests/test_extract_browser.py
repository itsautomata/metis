"""tests for the browser-UA extraction fallback."""

import httpx

from metis.ingest import extract as ex


def _fake_response(status: int, text: str):
    return httpx.Response(status_code=status, text=text,
                          request=httpx.Request("GET", "https://blocked.example"))


def test_browser_fetch_recovers_after_metis_ua_403(monkeypatch):
    """L2 gets a 403, L3 fetches with a browser UA and extracts the article."""
    article = "<html><head><title>Real Title</title></head><body><article>" \
              + ("this is the genuine article body. " * 20) + "</article></body></html>"

    def fake_get(url, **kwargs):
        ua = kwargs.get("headers", {}).get("User-Agent", "")
        if ua == ex.BROWSER_UA:
            return _fake_response(200, article)
        return _fake_response(403, "blocked")

    monkeypatch.setattr("socket.getaddrinfo", lambda host, port=None, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    monkeypatch.setattr(ex.trafilatura, "fetch_url", lambda url: None)
    monkeypatch.setattr(ex.httpx, "get", fake_get)

    title, text = ex.extract_from_url("https://blocked.example/post")
    assert "genuine article body" in text


def test_browser_fetch_uses_browser_user_agent(monkeypatch):
    """the L3 request carries the browser UA, not metis/0.1.0."""
    seen = {}

    def fake_get(url, **kwargs):
        seen["ua"] = kwargs.get("headers", {}).get("User-Agent", "")
        return _fake_response(200, "<html><body>" + ("word " * 50) + "</body></html>")

    monkeypatch.setattr("socket.getaddrinfo", lambda host, port=None, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))])
    monkeypatch.setattr(ex.httpx, "get", fake_get)
    result = ex._extract_with_browser("https://blocked.example")
    assert seen["ua"] == ex.BROWSER_UA
    assert result is not None


def test_browser_fetch_returns_none_when_fetch_fails(monkeypatch):
    def boom(url, **kwargs):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(ex.httpx, "get", boom)
    assert ex._extract_with_browser("https://blocked.example") is None
