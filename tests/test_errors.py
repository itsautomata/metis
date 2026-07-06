"""tests for friendly provider/model error handling."""

import httpx
import openai
import pytest
from typer.testing import CliRunner

from metis.cli import app
from metis.client import ProviderError
from metis.config import MetisConfig, OpenAIConfig
from metis.index import embed

runner = CliRunner()


def _not_found(*a, **k):
    raise openai.NotFoundError(
        "model 'moonshotai/kimi-k2.6' not found",
        response=httpx.Response(404, request=httpx.Request("GET", "http://x")),
        body=None,
    )


class _FakeEmbeddings:
    def create(self, *a, **k):
        _not_found()


class _FakeClient:
    embeddings = _FakeEmbeddings()


def test_embed_error_wraps_as_provider_error_with_hint(monkeypatch):
    monkeypatch.setattr(embed, "get_embedding_client", lambda c: _FakeClient())
    cfg = MetisConfig(openai=OpenAIConfig(base_url="https://openrouter.ai/api/v1"))
    with pytest.raises(ProviderError, match="vendor-prefixed"):
        embed.embed_texts(["hi"], cfg)


def test_embed_error_no_hint_without_base_url(monkeypatch):
    monkeypatch.setattr(embed, "get_embedding_client", lambda c: _FakeClient())
    with pytest.raises(ProviderError) as ei:
        embed.embed_texts(["hi"], MetisConfig())
    assert "vendor-prefixed" not in str(ei.value)


def test_provider_guard_reports_cleanly_no_traceback(monkeypatch):
    monkeypatch.setattr("metis.cli.load_config", lambda: MetisConfig())
    monkeypatch.setattr("metis.cli._ensure_index_model", lambda c: True)
    monkeypatch.setattr("metis.search.embed_texts", _not_found)

    result = runner.invoke(app, ["search", "hello"])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "moonshotai/kimi-k2.6" in result.output
    assert "check the model id" in result.output
