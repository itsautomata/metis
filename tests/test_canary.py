"""the embedding-drift canary must baseline once and correctly classify stable/drift/variance."""

from typer.testing import CliRunner

from metis.cli import app
from metis.client import ProviderError
from metis.config import MetisConfig, OpenAIConfig
from metis.index import canary

runner = CliRunner()

BASE = [[1.0, 0.0], [0.0, 1.0]]      # one vector per canary text
DRIFTED = [[0.0, 1.0], [1.0, 0.0]]   # orthogonal to BASE -> cosine 0


def _cfg(tmp_path, model="mymodel"):
    return MetisConfig(openai=OpenAIConfig(embedding_model=model), chromadb_path=tmp_path / "cdb")


def _fixed(batch):
    return lambda texts, config: [list(v) for v in batch]


def _seq(*batches):
    calls = {"n": 0}

    def _e(texts, config):
        b = batches[min(calls["n"], len(batches) - 1)]
        calls["n"] += 1
        return [list(v) for v in b]

    return _e


def _use_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(canary, "CANARY_PATH", tmp_path / "canary.json")


def test_ensure_baseline_writes_and_is_idempotent(tmp_path, monkeypatch):
    """the first call captures a baseline under the model key; a second call re-embeds nothing."""
    _use_tmp(monkeypatch, tmp_path)
    calls = {"n": 0}

    def _counting(texts, config):
        calls["n"] += 1
        return [list(v) for v in BASE]

    monkeypatch.setattr("metis.index.embed.embed_texts",_counting)
    cfg = _cfg(tmp_path)

    canary.ensure_baseline(cfg)
    canary.ensure_baseline(cfg)

    assert calls["n"] == 1                       # embedded once, second call is a sidecar read
    assert canary.baselined_model(cfg) == "mymodel"


def test_check_drift_stable(tmp_path, monkeypatch):
    """the same output as the baseline is stable."""
    _use_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr("metis.index.embed.embed_texts",_fixed(BASE))
    cfg = _cfg(tmp_path)
    canary.ensure_baseline(cfg)

    assert canary.check_drift(cfg).status == "stable"


def test_check_drift_detects_drift(tmp_path, monkeypatch):
    """self-consistent output that differs from the baseline is a one-time drift (reindex)."""
    _use_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr("metis.index.embed.embed_texts",_fixed(BASE))
    cfg = _cfg(tmp_path)
    canary.ensure_baseline(cfg)

    monkeypatch.setattr("metis.index.embed.embed_texts",_fixed(DRIFTED))   # both probes agree, differ from base
    assert canary.check_drift(cfg).status == "drift"


def test_check_drift_detects_variance(tmp_path, monkeypatch):
    """two back-to-back probes disagreeing is persistent provider variance (pin the provider)."""
    _use_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr("metis.index.embed.embed_texts",_fixed(BASE))
    cfg = _cfg(tmp_path)
    canary.ensure_baseline(cfg)

    monkeypatch.setattr("metis.index.embed.embed_texts",_seq(BASE, DRIFTED))  # probe1 != probe2
    assert canary.check_drift(cfg).status == "variance"


def test_check_drift_not_baselined(tmp_path, monkeypatch):
    """no baseline yet is a neutral verdict, not a failure, and embeds nothing."""
    _use_tmp(monkeypatch, tmp_path)

    def _boom(texts, config):
        raise AssertionError("must not embed when there is no baseline")

    monkeypatch.setattr("metis.index.embed.embed_texts",_boom)
    assert canary.check_drift(_cfg(tmp_path)).status == "not_baselined"


def test_check_drift_unavailable_on_provider_error(tmp_path, monkeypatch):
    """a provider failure during the probe is 'unavailable', not a false drift verdict."""
    _use_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr("metis.index.embed.embed_texts",_fixed(BASE))
    cfg = _cfg(tmp_path)
    canary.ensure_baseline(cfg)

    def _fail(texts, config):
        raise ProviderError("provider down")

    monkeypatch.setattr("metis.index.embed.embed_texts",_fail)
    assert canary.check_drift(cfg).status == "unavailable"


def test_reset_unlinks_baseline(tmp_path, monkeypatch):
    """reset drops the baseline so a reindex re-captures against the new model."""
    _use_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr("metis.index.embed.embed_texts",_fixed(BASE))
    cfg = _cfg(tmp_path)
    canary.ensure_baseline(cfg)
    assert canary.baselined_model(cfg) == "mymodel"

    canary.reset()
    assert canary.baselined_model(cfg) is None


class _FakeCol:
    metadata = {"embedding_model": "mymodel", "hnsw:space": "cosine"}

    def count(self):
        return 5


def _patch_doctor(monkeypatch, tmp_path, verdict):
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("metis.cli.load_config", lambda: cfg)
    monkeypatch.setattr("metis.cli._keychain_key", lambda: None)
    monkeypatch.setattr("metis.secrets.get_provider_key", lambda: "sk-test")
    monkeypatch.setattr("metis.index.store.get_collection", lambda config: _FakeCol())
    monkeypatch.setattr("metis.index.canary.check_drift", lambda config: verdict)


def test_doctor_reports_drift(tmp_path, monkeypatch):
    """`metis doctor` on a drifted index fails loud and points at reindex."""
    _patch_doctor(monkeypatch, tmp_path, canary.DriftVerdict("drift", "x"))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "reindex" in result.output


def test_doctor_reports_variance(tmp_path, monkeypatch):
    """variance points at pinning the provider, not reindex."""
    _patch_doctor(monkeypatch, tmp_path, canary.DriftVerdict("variance", "x"))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "pin the provider" in result.output
