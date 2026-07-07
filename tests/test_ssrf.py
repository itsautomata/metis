"""tests for the ssrf guard on the ingest url fetchers."""

import pytest

from metis.ingest.extract import _reject_ssrf, _safe_get


def _resolves_to(ip):
    return lambda host, port=None, *a, **k: [(2, 1, 6, "", (ip, 0))]


def test_reject_ssrf_rejects_non_http():
    with pytest.raises(ValueError, match="non-http"):
        _reject_ssrf("file:///etc/passwd")


def test_reject_ssrf_blocks_loopback(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _resolves_to("127.0.0.1"))
    with pytest.raises(ValueError, match="non-public"):
        _reject_ssrf("http://attacker.test/")


def test_reject_ssrf_blocks_metadata(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _resolves_to("169.254.169.254"))
    with pytest.raises(ValueError, match="non-public"):
        _reject_ssrf("http://metadata.test/")


def test_reject_ssrf_blocks_private(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _resolves_to("10.0.0.5"))
    with pytest.raises(ValueError, match="non-public"):
        _reject_ssrf("http://internal.test/")


def test_reject_ssrf_allows_public(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", _resolves_to("93.184.216.34"))
    _reject_ssrf("https://example.com/")  # does not raise


def test_safe_get_blocks_redirect_to_internal(monkeypatch):
    """the attack: a public page 302-redirects the fetch to an internal host."""

    def _resolve(host, port=None, *a, **k):
        ip = "93.184.216.34" if host == "safe.test" else "127.0.0.1"
        return [(2, 1, 6, "", (ip, 0))]

    monkeypatch.setattr("socket.getaddrinfo", _resolve)

    class _Redirect:
        is_redirect = True
        headers = {"location": "http://internal.test/"}

    monkeypatch.setattr("httpx.get", lambda url, **k: _Redirect())

    with pytest.raises(ValueError, match="non-public"):
        _safe_get("http://safe.test/")
