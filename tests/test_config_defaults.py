"""a present-but-null config value must fall back to its default, not become None."""

from metis import config
from metis.client import get_embedding_model


def test_null_values_fall_back_to_defaults(monkeypatch, tmp_path):
    """keys present with no value use the defaults instead of None."""
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    p.write_text(
        "openai:\n"
        "  base_url:\n"
        "  embedding_model:\n"
        "  chat_model:\n"
        "vault_path:\n"
        "output_folder:\n",
        encoding="utf-8",
    )
    cfg = config.load_config()
    assert cfg.openai.embedding_model == "text-embedding-3-small"
    assert cfg.openai.chat_model == "gpt-4o"
    assert cfg.openai.base_url == ""
    assert str(cfg.vault_path)                    # not Path(None)
    assert cfg.output_folder == "metis-ingested"


def test_null_sections_fall_back(monkeypatch, tmp_path):
    """a null `openai:` / `chromadb:` section becomes {} rather than crashing on .get()."""
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    p.write_text("openai:\nchromadb:\n", encoding="utf-8")
    cfg = config.load_config()
    assert cfg.openai.embedding_model == "text-embedding-3-small"
    assert str(cfg.chromadb_path)


def test_null_embedding_model_survives_openrouter(monkeypatch, tmp_path):
    """null embedding_model + openrouter base_url must not raise on None.startswith."""
    p = tmp_path / "config.yaml"
    monkeypatch.setattr(config, "CONFIG_PATH", p)
    p.write_text(
        "openai:\n"
        "  base_url: https://openrouter.ai/api/v1\n"
        "  embedding_model:\n",
        encoding="utf-8",
    )
    cfg = config.load_config()
    model = get_embedding_model(cfg)              # must not raise AttributeError
    assert model == "openai/text-embedding-3-small"
