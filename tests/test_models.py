"""tests for the `metis models` command."""

from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig, OpenAIConfig

runner = CliRunner()


def test_models_shows_chat_and_embedding(monkeypatch, tmp_path):
    cfg = MetisConfig(
        openai=OpenAIConfig(
            base_url="https://openrouter.ai/api/v1",
            chat_model="moonshotai/kimi-k2.6",
            embedding_model="openai/text-embedding-3-small",
        ),
        chromadb_path=tmp_path / "cdb",
    )
    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)

    result = runner.invoke(app, ["models"])

    assert result.exit_code == 0
    assert "moonshotai/kimi-k2.6" in result.output
    assert "openai/text-embedding-3-small" in result.output
    assert "openrouter" in result.output
    assert "shared with chat" in result.output


def test_models_flags_index_mismatch(monkeypatch, tmp_path):
    class _FakeCollection:
        metadata = {"embedding_model": "old-model"}

        def count(self):
            return 5

    cfg = MetisConfig(
        openai=OpenAIConfig(embedding_model="new-model"),
        chromadb_path=tmp_path / "cdb",
    )
    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)
    monkeypatch.setattr("metis.index.store.get_collection", lambda c: _FakeCollection())

    result = runner.invoke(app, ["models"])

    assert result.exit_code == 0
    assert "old-model" in result.output
    assert "reindex" in result.output


def test_models_flags_wrong_provider_key(monkeypatch, tmp_path):
    cfg = MetisConfig(
        openai=OpenAIConfig(base_url="https://openrouter.ai/api/v1"),
        chromadb_path=tmp_path / "cdb",
    )
    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)
    monkeypatch.setattr("keyring.get_password", lambda service, name: "sk-proj-an-openai-key")

    result = runner.invoke(app, ["models"])

    assert result.exit_code == 0
    assert "likely the wrong key" in result.output
