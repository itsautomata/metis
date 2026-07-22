"""a guarded command with no provider key (or a provider fault) must exit clean, never a traceback."""

from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig, OpenAIConfig

runner = CliRunner()


def _cfg(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    return MetisConfig(openai=OpenAIConfig(base_url=""), vault_path=vault, chromadb_path=tmp_path / "cdb")


def test_chat_without_key_exits_clean(monkeypatch, tmp_path):
    """`metis chat` with no key surfaces the provider message via the guard, not a raw ValueError."""
    monkeypatch.setattr("metis.cli.load_config", lambda: _cfg(tmp_path))
    monkeypatch.setattr("metis.client.get_provider_key", lambda: "")

    result = runner.invoke(app, ["chat", "hello"])

    assert result.exit_code == 1
    assert "provider-key" in result.output
    assert not isinstance(result.exception, RuntimeError)  # ProviderError did not leak past the guard


def test_folders_edit_provider_fault_exits_clean(monkeypatch, tmp_path):
    """folders --edit is guarded now: a re-embed failure exits cleanly instead of a traceback."""
    import subprocess

    from metis.client import ProviderError

    vault = tmp_path / "vault"
    (vault / "research").mkdir(parents=True)
    cfg = MetisConfig(openai=OpenAIConfig(base_url=""), vault_path=vault, chromadb_path=tmp_path / "cdb")
    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)

    def _fake_editor(argv, *a, **k):
        with open(argv[1], "w") as f:
            f.write("research: a brand new description\n")
        return 0

    monkeypatch.setattr(subprocess, "call", _fake_editor)

    def _boom(config):
        raise ProviderError("embedding model failed: provider down")

    monkeypatch.setattr("metis.classify.get_folder_embeddings", _boom)

    result = runner.invoke(app, ["folders", "--edit"])

    assert result.exit_code == 1
    assert not isinstance(result.exception, RuntimeError)
