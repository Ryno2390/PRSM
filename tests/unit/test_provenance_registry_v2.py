"""PRSM-PROV-1 Item 7 T7.4 + T7.6 — ProvenanceRegistryV2 client tests.

Covers:
  - compute_embedding_commitment() canonical formula round-trip,
    determinism, and rejection of bad inputs.
  - compute_kind_tag() agreement with the Hardhat
    keccak256("text-vector") used in
    contracts/test/ProvenanceRegistryV2.test.js.
  - Cross-model isolation (same vector under different model_id /
    different dim → different commitment).
  - Length validation on the on-chain reads/writes (32-byte hashes
    enforced before tx build).
  - Royalty rate cap (mirrors contract MAX_ROYALTY_RATE_BPS = 9800).
  - dispute_provenance() round-trip via mocked contract:
    * matching vector → True
    * mismatched vector → False
    * unregistered content → False
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("web3")

from prsm.economy.web3.provenance_registry_v2 import (
    MAX_ROYALTY_RATE_BPS,
    ZERO_BYTES32,
    ProvenanceRegistryV2Client,
    compute_embedding_commitment,
    compute_kind_tag,
)


# ---- compute_embedding_commitment ---------------------------------


def test_commitment_is_32_bytes():
    out = compute_embedding_commitment(
        "openai-text-embedding-ada-002", 1536, b"\x01\x02\x03",
    )
    assert isinstance(out, bytes)
    assert len(out) == 32


def test_commitment_is_deterministic():
    args = ("test-model", 8, b"\xaa\xbb\xcc\xdd")
    assert compute_embedding_commitment(*args) == compute_embedding_commitment(
        *args,
    )


def test_commitment_changes_with_model_id():
    a = compute_embedding_commitment("model-a", 8, b"\x01" * 16)
    b = compute_embedding_commitment("model-b", 8, b"\x01" * 16)
    assert a != b


def test_commitment_changes_with_dim():
    a = compute_embedding_commitment("m", 8, b"\x01" * 16)
    b = compute_embedding_commitment("m", 16, b"\x01" * 16)
    assert a != b


def test_commitment_changes_with_vector():
    a = compute_embedding_commitment("m", 8, b"\x01" * 16)
    b = compute_embedding_commitment("m", 8, b"\x02" * 16)
    assert a != b


def test_commitment_rejects_empty_model_id():
    with pytest.raises(ValueError, match="model_id"):
        compute_embedding_commitment("", 8, b"\x01" * 16)


def test_commitment_rejects_zero_dim():
    with pytest.raises(ValueError, match="dim"):
        compute_embedding_commitment("m", 0, b"\x01" * 16)


def test_commitment_rejects_negative_dim():
    with pytest.raises(ValueError, match="dim"):
        compute_embedding_commitment("m", -5, b"\x01" * 16)


def test_commitment_rejects_huge_dim():
    with pytest.raises(ValueError, match="dim"):
        compute_embedding_commitment("m", 2**31, b"\x01" * 16)


def test_commitment_rejects_empty_vector():
    with pytest.raises(ValueError, match="vector_bytes"):
        compute_embedding_commitment("m", 8, b"")


def test_commitment_rejects_non_bytes_vector():
    with pytest.raises(ValueError, match="vector_bytes"):
        compute_embedding_commitment("m", 8, "not-bytes")  # type: ignore


def test_commitment_accepts_bytearray():
    a = compute_embedding_commitment("m", 8, b"\x01" * 16)
    b = compute_embedding_commitment("m", 8, bytearray(b"\x01" * 16))
    assert a == b


def test_commitment_canonical_format_matches_hardhat():
    """Cross-check: this Python helper MUST agree with the
    `makeCommitment` JS helper in
    contracts/test/ProvenanceRegistryV2.test.js so a record registered
    via the wrapper verifies on-chain. Spot-check with one of the
    fixed inputs from that test (model_id="test-model", dim=8,
    vector=[1..8]).

    The expected hash is keccak256(
        b"test-model" + struct.pack(">I", 8) + bytes(range(1, 9))
    )
    """
    from web3 import Web3
    expected = bytes(
        Web3.keccak(
            b"test-model" + (8).to_bytes(4, "big") + bytes([1, 2, 3, 4, 5, 6, 7, 8]),
        ),
    )
    actual = compute_embedding_commitment(
        "test-model", 8, bytes([1, 2, 3, 4, 5, 6, 7, 8]),
    )
    assert actual == expected


# ---- compute_kind_tag ---------------------------------------------


def test_kind_tag_is_32_bytes():
    assert len(compute_kind_tag("text-vector")) == 32


def test_kind_tag_deterministic():
    assert compute_kind_tag("text-vector") == compute_kind_tag("text-vector")


def test_kind_tag_distinguishes_kinds():
    assert compute_kind_tag("text-vector") != compute_kind_tag("image-phash")


def test_kind_tag_rejects_empty_label():
    with pytest.raises(ValueError, match="kind_label"):
        compute_kind_tag("")


def test_kind_tag_canonical_value_text_vector():
    """The tag for text-vector MUST equal the keccak256 the Hardhat
    test uses (KIND_TEXT in ProvenanceRegistryV2.test.js); otherwise
    on-chain getEmbeddingCommitment().fingerprintKind comparisons
    won't agree."""
    from web3 import Web3
    expected = bytes(Web3.keccak(b"text-vector"))
    assert compute_kind_tag("text-vector") == expected


def test_kind_tag_canonical_value_image_phash():
    from web3 import Web3
    expected = bytes(Web3.keccak(b"image-phash"))
    assert compute_kind_tag("image-phash") == expected


# ---- ZERO_BYTES32 -------------------------------------------------


def test_zero_bytes32_is_32_zeros():
    assert ZERO_BYTES32 == b"\x00" * 32
    assert len(ZERO_BYTES32) == 32


# ---- Client validation (no live RPC) ------------------------------


def _make_client_with_mock_contract():
    """Build a client with all Web3 internals replaced by mocks so we
    can exercise the wrapper logic without a live RPC."""
    client = ProvenanceRegistryV2Client.__new__(ProvenanceRegistryV2Client)
    client.web3 = MagicMock()
    client.contract_address = "0x" + "0" * 40
    client.contract = MagicMock()
    client._account = None
    client._tx_lock = MagicMock()
    client._tx_lock.__enter__ = MagicMock(return_value=None)
    client._tx_lock.__exit__ = MagicMock(return_value=False)
    return client


def test_register_rejects_short_content_hash():
    client = _make_client_with_mock_contract()
    client._account = MagicMock()
    with pytest.raises(ValueError, match="content_hash"):
        client.register_content_v2(
            b"\x00" * 16, 800, "ipfs://x", ZERO_BYTES32, ZERO_BYTES32,
        )


def test_register_rejects_short_commitment():
    client = _make_client_with_mock_contract()
    client._account = MagicMock()
    with pytest.raises(ValueError, match="embedding_commitment"):
        client.register_content_v2(
            b"\x00" * 32, 800, "ipfs://x", b"\x00" * 16, ZERO_BYTES32,
        )


def test_register_rejects_short_kind():
    client = _make_client_with_mock_contract()
    client._account = MagicMock()
    with pytest.raises(ValueError, match="fingerprint_kind"):
        client.register_content_v2(
            b"\x00" * 32, 800, "ipfs://x", ZERO_BYTES32, b"\x00" * 16,
        )


def test_register_rejects_high_royalty():
    client = _make_client_with_mock_contract()
    client._account = MagicMock()
    with pytest.raises(ValueError, match="royalty_rate_bps"):
        client.register_content_v2(
            b"\x00" * 32, MAX_ROYALTY_RATE_BPS + 1, "ipfs://x",
            ZERO_BYTES32, ZERO_BYTES32,
        )


def test_register_rejects_negative_royalty():
    client = _make_client_with_mock_contract()
    client._account = MagicMock()
    with pytest.raises(ValueError, match="royalty_rate_bps"):
        client.register_content_v2(
            b"\x00" * 32, -1, "ipfs://x", ZERO_BYTES32, ZERO_BYTES32,
        )


def test_register_requires_private_key():
    client = _make_client_with_mock_contract()  # _account is None
    with pytest.raises(RuntimeError, match="private_key"):
        client.register_content_v2(
            b"\x00" * 32, 800, "ipfs://x", ZERO_BYTES32, ZERO_BYTES32,
        )


def test_max_royalty_rate_constant_matches_contract():
    """If the contract's MAX_ROYALTY_RATE_BPS ever moves, this test
    catches the Python mirror going out of sync. 9800 = 100% - 2%
    network fee."""
    assert MAX_ROYALTY_RATE_BPS == 9800


def test_verify_rejects_short_hash():
    client = _make_client_with_mock_contract()
    with pytest.raises(ValueError, match="content_hash"):
        client.verify_embedding_commitment(b"\x00" * 16, ZERO_BYTES32)


def test_verify_rejects_short_claim():
    client = _make_client_with_mock_contract()
    with pytest.raises(ValueError, match="claimed"):
        client.verify_embedding_commitment(b"\x00" * 32, b"\x00" * 16)


# ---- dispute_provenance() round-trip via mocked contract ----------


def test_dispute_returns_true_when_vector_matches():
    client = _make_client_with_mock_contract()
    fn = MagicMock()
    # Mock the contract's view function to "return" True only for the
    # specific commitment derived from the test inputs.
    expected_commitment = compute_embedding_commitment(
        "test-model", 8, bytes(range(1, 9)),
    )

    def _verify_view(content_hash, claimed):
        call_obj = MagicMock()
        call_obj.call.return_value = (claimed == expected_commitment)
        return call_obj

    client.contract.functions.verifyEmbeddingCommitment = MagicMock(
        side_effect=_verify_view,
    )

    content_hash = b"\xaa" * 32
    out = client.dispute_provenance(
        content_hash, "test-model", 8, bytes(range(1, 9)),
    )
    assert out is True


def test_dispute_returns_false_when_vector_mismatches():
    client = _make_client_with_mock_contract()
    expected_commitment = compute_embedding_commitment(
        "test-model", 8, bytes(range(1, 9)),
    )

    def _verify_view(content_hash, claimed):
        call_obj = MagicMock()
        call_obj.call.return_value = (claimed == expected_commitment)
        return call_obj

    client.contract.functions.verifyEmbeddingCommitment = MagicMock(
        side_effect=_verify_view,
    )

    content_hash = b"\xaa" * 32
    out = client.dispute_provenance(
        content_hash, "test-model", 8, b"\x99" * 8,  # wrong vector
    )
    assert out is False


def test_dispute_returns_false_for_unregistered_content():
    """Contract returns False for any claim against unregistered or
    zero-commitment content. The wrapper must propagate that."""
    client = _make_client_with_mock_contract()

    def _verify_view(content_hash, claimed):
        call_obj = MagicMock()
        call_obj.call.return_value = False
        return call_obj

    client.contract.functions.verifyEmbeddingCommitment = MagicMock(
        side_effect=_verify_view,
    )

    out = client.dispute_provenance(
        b"\x00" * 32, "test-model", 8, bytes(range(1, 9)),
    )
    assert out is False


def test_dispute_changes_with_model_id():
    """An attacker who has the right vector but tries the wrong
    model_id MUST fail dispute — the commitment binds the model."""
    client = _make_client_with_mock_contract()
    truthful_commitment = compute_embedding_commitment(
        "real-model", 8, bytes(range(1, 9)),
    )

    def _verify_view(content_hash, claimed):
        call_obj = MagicMock()
        call_obj.call.return_value = (claimed == truthful_commitment)
        return call_obj

    client.contract.functions.verifyEmbeddingCommitment = MagicMock(
        side_effect=_verify_view,
    )

    content_hash = b"\xaa" * 32
    # Attacker claims same content with a different model_id.
    out = client.dispute_provenance(
        content_hash, "spoofed-model", 8, bytes(range(1, 9)),
    )
    assert out is False
