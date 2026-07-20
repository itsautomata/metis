"""embedding-drift canary: fingerprint the embedding model's actual output to catch silent drift.

metis's model-name / dimension guard (store.check_embedding_model) misses same-name, same-
dimension weight drift, which corrupts the index (old vectors in one space, new queries in
another). this module baselines a fixed reference embedding at index-build time and re-checks it,
distinguishing a one-time model change (reindex) from persistent per-request provider variance
(pin the provider). provider-agnostic: it trusts behaviour, not the model id.
"""

import json
from dataclasses import dataclass

import numpy as np

from metis.client import get_embedding_model
from metis.config import CONFIG_DIR, MetisConfig
from metis.index import embed
from metis.index.store import _canonical_embedding_model

CANARY_PATH = CONFIG_DIR / "canary.json"

# fixed reference inputs: one ascii, one mixed-script/unicode (non-ascii surfaces the
# quantization/encoding drift OpenRouter routing is known for). versioned so the set can evolve.
CANARY_TEXTS = [
    "metis drift canary v1: the quick brown fox jumps over the lazy dog 1234567890.",
    "metis drift canary v1: cafe naive facade, 日本語 Ω π, 42.",
]

# same model+quantization re-embeds a string to a near-identical vector; below this cosine the
# output is treated as changed. may need empirical tuning per provider.
STABLE_SIM = 0.999


@dataclass
class DriftVerdict:
    status: str  # stable | drift | variance | not_baselined | unavailable
    detail: str
    similarity: float | None = None


def _load() -> dict:
    if not CANARY_PATH.exists():
        return {}
    try:
        data = json.loads(CANARY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save(data: dict) -> None:
    CANARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CANARY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(CANARY_PATH)


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _key(config: MetisConfig) -> str:
    """canonical model id (matches the index stamp), so an openai/openrouter alias switch
    does not force a spurious re-baseline."""
    return _canonical_embedding_model(get_embedding_model(config))


def ensure_baseline(config: MetisConfig) -> None:
    """capture the canary reference for the current model once, at index-build time.

    idempotent: a pure sidecar read once a baseline exists for this model. best-effort; a
    provider failure leaves no baseline and the next write/check retries.
    """
    key = _key(config)
    data = _load()
    if key in data:
        return
    try:
        vectors = embed.embed_texts(CANARY_TEXTS, config)
    except Exception as e:  # best-effort: a provider/key error must not break the ingest
        from rich.console import Console
        Console().print(f"[dim]note: drift canary not baselined ({e})[/dim]")
        return
    data[key] = vectors
    _save(data)


def check_drift(config: MetisConfig) -> DriftVerdict:
    """compare the model's current canary output to the stored baseline.

    two live probes per canary: if they disagree with each other, the provider is returning
    unstable output per request (pin the provider); if they agree but differ from the baseline,
    the model drifted since build (reindex).
    """
    key = _key(config)
    baseline = _load().get(key)
    if not baseline:
        return DriftVerdict("not_baselined", "no drift baseline for this model yet")
    try:
        probe1 = embed.embed_texts(CANARY_TEXTS, config)
        probe2 = embed.embed_texts(CANARY_TEXTS, config)
    except Exception as e:  # best-effort: never crash sync/doctor on a provider/key error
        return DriftVerdict("unavailable", f"embedding provider unavailable: {e}")

    # self-consistency first: unstable across two back-to-back calls means per-request variance
    worst_self = min(_cosine(a, b) for a, b in zip(probe1, probe2))
    if worst_self < STABLE_SIM:
        return DriftVerdict("variance", "embeddings differ across back-to-back calls", worst_self)

    worst_ref = min(_cosine(a, b) for a, b in zip(probe1, baseline))
    if worst_ref < STABLE_SIM:
        return DriftVerdict("drift", "embedding output changed since the index was built", worst_ref)

    return DriftVerdict("stable", "embedding output matches the baseline", worst_ref)


def baselined_model(config: MetisConfig) -> str | None:
    """the model id a baseline exists for (offline, sidecar-only), or None."""
    key = _key(config)
    return key if key in _load() else None


def reset() -> None:
    """drop the baseline (used on reindex; the next build re-captures against the new model)."""
    CANARY_PATH.unlink(missing_ok=True)
