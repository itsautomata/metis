"""tests for the `metis doctor` command."""

from typer.testing import CliRunner

from metis.cli import app
from metis.config import MetisConfig, OpenAIConfig

runner = CliRunner()


def test_doctor_all_good(monkeypatch, tmp_path):
    cfg = MetisConfig(
        openai=OpenAIConfig(base_url="https://openrouter.ai/api/v1", chat_model="moonshotai/kimi-k2.6"),
        chromadb_path=tmp_path / "cdb",
    )
    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)
    monkeypatch.setattr("keyring.get_password", lambda service, name: "sk-or-v1-a-good-key")

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "metis is ready" in result.output


def test_doctor_flags_wrong_provider_key(monkeypatch, tmp_path):
    cfg = MetisConfig(
        openai=OpenAIConfig(base_url="https://openrouter.ai/api/v1"),
        chromadb_path=tmp_path / "cdb",
    )
    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)
    monkeypatch.setattr("keyring.get_password", lambda service, name: "sk-proj-an-openai-key")

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "likely the wrong key" in " ".join(result.output.split())


def test_doctor_flags_missing_key(monkeypatch, tmp_path):
    monkeypatch.setattr("metis.cli.load_config", lambda: MetisConfig(chromadb_path=tmp_path / "cdb"))
    monkeypatch.setattr("keyring.get_password", lambda service, name: None)
    monkeypatch.delenv("METIS_PROVIDER_KEY", raising=False)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "not set" in result.output


def test_doctor_flags_index_mismatch(monkeypatch, tmp_path):
    class _FakeCollection:
        metadata = {"embedding_model": "old-model"}

        def count(self):
            return 5

    cfg = MetisConfig(openai=OpenAIConfig(embedding_model="new-model"), chromadb_path=tmp_path / "cdb")
    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)
    monkeypatch.setattr("keyring.get_password", lambda service, name: "sk-an-openai-key")
    monkeypatch.setattr("metis.index.store.get_collection", lambda c: _FakeCollection())

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "reindex" in result.output


def test_doctor_survives_keyring_error(monkeypatch, tmp_path):
    """a keyring backend error must not crash metis doctor (the setup-diagnostic command)."""
    monkeypatch.setattr("metis.cli.load_config", lambda: MetisConfig(chromadb_path=tmp_path / "cdb"))

    def _boom(service, name):
        raise Exception("keychain locked")

    monkeypatch.setattr("keyring.get_password", _boom)

    result = runner.invoke(app, ["doctor"])

    assert "Traceback" not in result.output
