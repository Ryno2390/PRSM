"""Sprint 310 — PRSM content-layer DataLoader.

Sprint 309 shipped the pluggable `DataLoader` Protocol
with `synthetic_data_loader` as the default. This sprint
wires a real PRSM-content-aware DataLoader so workers can
fetch their assigned (encrypted) training shards directly
from the PRSM content layer + decrypt inside the TEE +
deserialize to torch tensors — no operator-side glue
required.

The flow:
  1. Enterprise serializes (features, labels) → bytes via
     `serialize_training_data`
  2. Enterprise encrypts the bytes to the worker fleet's
     X25519 recipient pubkeys via sprint 304's
     `encrypt_for_recipients` (OR-decrypt)
  3. Enterprise uploads the encrypted bundle to PRSM via
     `/content/upload` — gets back a CID
  4. Orchestrator includes the CID in the FederatedJob
     `dataset_cids` list
  5. Worker receives a round assignment with one CID;
     `prsm_content_data_loader(content_provider,
     recipient_privkey_b64)` fetches + decrypts +
     deserializes
  6. Worker calls sprint 309's `pytorch_train_fn` with the
     loaded tensors

OR-decrypt only in v1 (the typical FL pattern is one
worker per shard, not multi-party reconstruction).
Threshold mode (sprint 307) doesn't fit the FL DataLoader
model and is deliberately out of scope.

Data envelope: JSON-wrapped {features, labels} where each
is {shape, dtype, data_b64}. Self-describing, no pickle,
no torch.save round-tripping.
"""
from __future__ import annotations

import base64
import json
from typing import Optional

import pytest

torch = pytest.importorskip("torch")

from prsm.compute.prsm_data_loader import (
    DataDeserializationError,
    deserialize_training_data,
    prsm_content_data_loader,
    prsm_content_data_loader_async,
    serialize_training_data,
)
from prsm.enterprise.recipient_encryption import (
    EnterpriseRecipient,
    encrypt_for_recipients,
    generate_recipient_keypair,
)


# ── serialize / deserialize round trip ──────────────


def test_serialize_round_trip_float32():
    features = torch.randn(16, 4)
    labels = torch.randint(0, 3, (16,))
    blob = serialize_training_data(features, labels)
    x, y = deserialize_training_data(blob)
    assert torch.equal(x, features)
    assert torch.equal(y, labels)


def test_serialize_round_trip_float64():
    features = torch.randn(8, 2, dtype=torch.float64)
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    blob = serialize_training_data(features, labels)
    x, y = deserialize_training_data(blob)
    assert x.dtype == torch.float64
    assert torch.equal(x, features)
    assert torch.equal(y, labels)


def test_serialize_2d_labels():
    features = torch.randn(4, 3)
    labels = torch.randn(4, 5)  # regression-style targets
    blob = serialize_training_data(features, labels)
    x, y = deserialize_training_data(blob)
    assert x.shape == (4, 3)
    assert y.shape == (4, 5)
    assert torch.equal(x, features)
    assert torch.equal(y, labels)


def test_serialize_rejects_non_tensor():
    with pytest.raises(TypeError):
        serialize_training_data([1, 2, 3], torch.zeros(3))
    with pytest.raises(TypeError):
        serialize_training_data(torch.zeros(3), [1, 2, 3])


def test_serialize_rejects_unsupported_dtype():
    """Complex / quantized tensors aren't supported in v1
    — refuse loud."""
    features = torch.zeros(4, dtype=torch.complex64)
    with pytest.raises(ValueError, match="dtype"):
        serialize_training_data(features, torch.zeros(4))


def test_deserialize_rejects_malformed_json():
    with pytest.raises(DataDeserializationError):
        deserialize_training_data(b"{not json")


def test_deserialize_rejects_missing_fields():
    blob = json.dumps({"features": {}}).encode()
    with pytest.raises(DataDeserializationError):
        deserialize_training_data(blob)


def test_deserialize_rejects_bad_shape_or_dtype():
    blob = json.dumps({
        "features": {
            "shape": "bad",  # not a list
            "dtype": "float32",
            "data_b64": "",
        },
        "labels": {
            "shape": [1], "dtype": "int64", "data_b64": "",
        },
    }).encode()
    with pytest.raises(DataDeserializationError):
        deserialize_training_data(blob)


# ── DataLoader fake content_provider ────────────────


class _FakeContentProvider:
    """Mimics PRSM's content_provider.request_content
    surface. Pre-populate with cid → ciphertext bytes."""

    def __init__(self, blobs=None, raise_on=None):
        self._blobs = blobs or {}
        self._raise_on = raise_on or set()

    async def request_content(
        self, *, cid: str,
        timeout: Optional[float] = None,
        verify_hash: bool = True,
    ) -> bytes:
        if cid in self._raise_on:
            raise RuntimeError(f"simulated fetch failure for {cid}")
        if cid not in self._blobs:
            raise FileNotFoundError(cid)
        return self._blobs[cid]


def _encrypt_for_one_recipient(plaintext_bytes, recipient_pub):
    payload = encrypt_for_recipients(
        plaintext_bytes,
        [EnterpriseRecipient(
            identifier="worker-1",
            x25519_pubkey_b64=recipient_pub,
        )],
    )
    return json.dumps(payload.to_dict()).encode("utf-8")


# ── DataLoader happy path ───────────────────────────


@pytest.mark.asyncio
async def test_data_loader_fetches_decrypts_deserializes():
    """End-to-end: enterprise encrypts shard → ciphertext
    in content layer → worker DataLoader fetches +
    decrypts + deserializes → torch tensors out."""
    priv, pub = generate_recipient_keypair()
    features = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8,))

    plaintext_bytes = serialize_training_data(
        features, labels,
    )
    ciphertext_bundle = _encrypt_for_one_recipient(
        plaintext_bytes, pub,
    )
    provider = _FakeContentProvider(
        blobs={"Qm-shard-1": ciphertext_bundle},
    )

    loader = prsm_content_data_loader_async(
        content_provider=provider,
        recipient_privkey_b64=priv,
    )
    x, y = await loader("Qm-shard-1")
    assert torch.equal(x, features)
    assert torch.equal(y, labels)


@pytest.mark.asyncio
async def test_data_loader_unauthorized_recipient_fails():
    """A worker whose privkey isn't in the recipient set
    can't decrypt the shard — the DataLoader surfaces the
    decryption failure clearly."""
    _, authorized_pub = generate_recipient_keypair()
    outsider_priv, _ = generate_recipient_keypair()

    plaintext_bytes = serialize_training_data(
        torch.zeros(4, 2), torch.zeros(4, dtype=torch.long),
    )
    ciphertext_bundle = _encrypt_for_one_recipient(
        plaintext_bytes, authorized_pub,
    )
    provider = _FakeContentProvider(
        blobs={"Qm-shard-1": ciphertext_bundle},
    )

    loader = prsm_content_data_loader_async(
        content_provider=provider,
        recipient_privkey_b64=outsider_priv,
    )
    with pytest.raises(ValueError, match="no entry"):
        await loader("Qm-shard-1")


@pytest.mark.asyncio
async def test_data_loader_unknown_cid_fails():
    priv, _ = generate_recipient_keypair()
    provider = _FakeContentProvider(blobs={})
    loader = prsm_content_data_loader_async(
        content_provider=provider,
        recipient_privkey_b64=priv,
    )
    with pytest.raises(FileNotFoundError):
        await loader("Qm-missing")


@pytest.mark.asyncio
async def test_data_loader_fetch_error_propagates():
    priv, _ = generate_recipient_keypair()
    provider = _FakeContentProvider(
        raise_on={"Qm-flaky"},
    )
    loader = prsm_content_data_loader_async(
        content_provider=provider,
        recipient_privkey_b64=priv,
    )
    with pytest.raises(RuntimeError, match="simulated"):
        await loader("Qm-flaky")


@pytest.mark.asyncio
async def test_data_loader_corrupt_payload_surfaces_clearly():
    """If the content layer returns bytes that aren't a
    valid encrypted bundle, the DataLoader fails with a
    clear error — not a confusing torch traceback."""
    priv, _ = generate_recipient_keypair()
    provider = _FakeContentProvider(
        blobs={"Qm-bad": b"not an encrypted bundle"},
    )
    loader = prsm_content_data_loader_async(
        content_provider=provider,
        recipient_privkey_b64=priv,
    )
    with pytest.raises(Exception):
        await loader("Qm-bad")


@pytest.mark.asyncio
async def test_data_loader_decrypted_but_malformed_data_fails():
    """Encryption succeeds, decryption succeeds, but the
    plaintext is malformed. The DataLoader surfaces the
    deserialization error clearly."""
    priv, pub = generate_recipient_keypair()
    bogus_plaintext = b"{i am not training data}"
    ciphertext_bundle = _encrypt_for_one_recipient(
        bogus_plaintext, pub,
    )
    provider = _FakeContentProvider(
        blobs={"Qm-malformed": ciphertext_bundle},
    )
    loader = prsm_content_data_loader_async(
        content_provider=provider,
        recipient_privkey_b64=priv,
    )
    with pytest.raises(DataDeserializationError):
        await loader("Qm-malformed")


# ── Integration with sprint 309 PyTorch backend ─────


def test_end_to_end_pytorch_train_on_prsm_shard():
    """The whole loop: shard exists encrypted on PRSM →
    worker DataLoader fetches + decrypts → PyTorch train
    fn consumes the tensors + produces a real gradient.

    Uses the SYNC `prsm_content_data_loader` wrapper since
    sprint-309's TrainingFn calls data_loader
    synchronously."""
    from prsm.compute.pytorch_train import (
        TorchTrainConfig, pytorch_train_fn,
    )

    priv, pub = generate_recipient_keypair()
    n_samples = 16
    features = torch.randn(n_samples, 4)
    labels = torch.randint(0, 2, (n_samples,))

    plaintext_bytes = serialize_training_data(
        features, labels,
    )
    ciphertext_bundle = _encrypt_for_one_recipient(
        plaintext_bytes, pub,
    )
    provider = _FakeContentProvider(
        blobs={"Qm-shard-1": ciphertext_bundle},
    )

    sync_loader = prsm_content_data_loader(
        content_provider=provider,
        recipient_privkey_b64=priv,
    )

    cfg = TorchTrainConfig(
        model_factory=lambda: torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        ),
        optimizer_factory=lambda p: torch.optim.SGD(
            p, lr=0.01,
        ),
        loss_fn=torch.nn.CrossEntropyLoss(),
        epochs=2,
        batch_size=4,
    )
    fn = pytorch_train_fn(
        config=cfg, data_loader=sync_loader,
    )
    grad = fn(
        job_id="j1", round_index=0,
        dataset_cid="Qm-shard-1",
        sample_count=n_samples,
    )
    # Gradient has the expected param count
    expected_params = sum(
        p.numel() for p in cfg.model_factory().parameters()
    )
    assert len(grad) == expected_params
    # Real training produces non-zero gradient
    nonzero = sum(1 for v in grad if v != 0.0)
    assert nonzero > expected_params * 0.5
