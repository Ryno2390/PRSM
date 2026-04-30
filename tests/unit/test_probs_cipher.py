"""Phase 3.x.11.q.y Task 2 — ProbsCipher AES-GCM unit tests.

Covers the cipher used by Phase 3.x.11.q.y to encrypt
``proposed_token_probs`` for the per-stage wire under Tier C
speculation. Tests target:
  - round-trip encrypt/decrypt produces identical plaintext
  - wrong-key tail rejects (ProbsDecryptionError)
  - AAD mismatch (replayed across (request, stage) pairs) rejects
  - ciphertext is non-deterministic across calls with same plaintext
  - construction validates key length (AES-256 = 32 bytes)
  - prob range validation on encrypt + decrypt
  - HKDF helper smoke
"""

from __future__ import annotations

import pytest

from prsm.compute.chain_rpc.probs_cipher import (
    ProbsCipher,
    ProbsCipherError,
    ProbsDecryptionError,
    derive_key_from_psk,
)


_TEST_KEY = b"\x00" * 32
_TEST_PROBS = [0.5, 0.3, 0.15, 0.05]


class TestProbsCipherRoundTrip:
    def test_round_trip_recovers_plaintext(self):
        c = ProbsCipher(_TEST_KEY)
        ct = c.encrypt(
            probs=_TEST_PROBS, request_id="req-1", stage_index=0,
        )
        out = c.decrypt(
            ciphertext=ct, request_id="req-1", stage_index=0,
            expected_k=len(_TEST_PROBS),
        )
        assert out == pytest.approx(_TEST_PROBS, abs=1e-12)

    def test_ciphertext_is_non_deterministic(self):
        # Same plaintext + key + AAD — different ciphertexts
        # because each encrypt uses a fresh random nonce.
        c = ProbsCipher(_TEST_KEY)
        ct1 = c.encrypt(
            probs=_TEST_PROBS, request_id="req-1", stage_index=0,
        )
        ct2 = c.encrypt(
            probs=_TEST_PROBS, request_id="req-1", stage_index=0,
        )
        assert ct1 != ct2
        # Both decrypt to the same plaintext.
        for ct in (ct1, ct2):
            out = c.decrypt(
                ciphertext=ct, request_id="req-1", stage_index=0,
                expected_k=len(_TEST_PROBS),
            )
            assert out == pytest.approx(_TEST_PROBS, abs=1e-12)

    def test_K1_round_trip(self):
        # Edge: K=1 (smallest valid speculation depth).
        c = ProbsCipher(_TEST_KEY)
        ct = c.encrypt(
            probs=[0.7], request_id="r", stage_index=1,
        )
        out = c.decrypt(
            ciphertext=ct, request_id="r", stage_index=1, expected_k=1,
        )
        assert out == pytest.approx([0.7], abs=1e-12)


class TestProbsCipherFailures:
    def test_wrong_key_rejects(self):
        c1 = ProbsCipher(_TEST_KEY)
        c2 = ProbsCipher(b"\x01" * 32)
        ct = c1.encrypt(
            probs=_TEST_PROBS, request_id="req-1", stage_index=0,
        )
        with pytest.raises(ProbsDecryptionError, match="tag mismatch"):
            c2.decrypt(
                ciphertext=ct, request_id="req-1", stage_index=0,
                expected_k=len(_TEST_PROBS),
            )

    def test_aad_request_id_mismatch_rejects(self):
        c = ProbsCipher(_TEST_KEY)
        ct = c.encrypt(
            probs=_TEST_PROBS, request_id="req-A", stage_index=0,
        )
        # Replay attack: same ciphertext on different request_id.
        with pytest.raises(ProbsDecryptionError):
            c.decrypt(
                ciphertext=ct, request_id="req-B", stage_index=0,
                expected_k=len(_TEST_PROBS),
            )

    def test_aad_stage_index_mismatch_rejects(self):
        c = ProbsCipher(_TEST_KEY)
        ct = c.encrypt(
            probs=_TEST_PROBS, request_id="req-1", stage_index=0,
        )
        # Replay attack: same ciphertext on different stage.
        with pytest.raises(ProbsDecryptionError):
            c.decrypt(
                ciphertext=ct, request_id="req-1", stage_index=1,
                expected_k=len(_TEST_PROBS),
            )

    def test_tampered_ciphertext_rejects(self):
        c = ProbsCipher(_TEST_KEY)
        ct = c.encrypt(
            probs=_TEST_PROBS, request_id="req-1", stage_index=0,
        )
        # Flip a byte in the ciphertext payload (after the nonce).
        ct_tampered = ct[:13] + bytes([ct[13] ^ 0x01]) + ct[14:]
        with pytest.raises(ProbsDecryptionError):
            c.decrypt(
                ciphertext=ct_tampered, request_id="req-1",
                stage_index=0, expected_k=len(_TEST_PROBS),
            )

    def test_ciphertext_too_short_rejects(self):
        c = ProbsCipher(_TEST_KEY)
        with pytest.raises(ProbsCipherError, match="too short"):
            c.decrypt(
                ciphertext=b"\x00" * 10, request_id="r",
                stage_index=0, expected_k=1,
            )

    def test_expected_k_mismatch_rejects(self):
        # Decrypt succeeds (correct key + AAD) but the resulting
        # plaintext length doesn't match the caller's expected_k.
        # Defends against tampered ciphertext that would otherwise
        # pass tag validation on a wrong K.
        c = ProbsCipher(_TEST_KEY)
        ct = c.encrypt(
            probs=[0.5, 0.5, 0.5, 0.5], request_id="r", stage_index=0,
        )
        with pytest.raises(
            ProbsCipherError, match="does not match expected",
        ):
            c.decrypt(
                ciphertext=ct, request_id="r", stage_index=0,
                expected_k=2,  # plaintext is 4 floats, not 2
            )


class TestProbsCipherConstruction:
    def test_rejects_non_bytes_key(self):
        with pytest.raises(ProbsCipherError, match="must be bytes"):
            ProbsCipher("not-bytes")  # type: ignore[arg-type]

    def test_rejects_wrong_key_length(self):
        for bad in [b"", b"\x00" * 16, b"\x00" * 31, b"\x00" * 33]:
            with pytest.raises(ProbsCipherError, match="32 bytes"):
                ProbsCipher(bad)

    def test_accepts_bytearray_key(self):
        c = ProbsCipher(bytearray(b"\x00" * 32))
        # Smoke: round-trip works through the bytearray path.
        ct = c.encrypt(probs=[0.5], request_id="r", stage_index=0)
        out = c.decrypt(
            ciphertext=ct, request_id="r", stage_index=0, expected_k=1,
        )
        assert out == pytest.approx([0.5], abs=1e-12)


class TestProbsCipherProbsValidation:
    def test_encrypt_rejects_out_of_range_probs(self):
        c = ProbsCipher(_TEST_KEY)
        with pytest.raises(ProbsCipherError, match="must be in"):
            c.encrypt(
                probs=[0.5, 1.5], request_id="r", stage_index=0,
            )

    def test_encrypt_rejects_bool_probs(self):
        c = ProbsCipher(_TEST_KEY)
        with pytest.raises(ProbsCipherError, match="must be number"):
            c.encrypt(
                probs=[True, False],  # type: ignore[list-item]
                request_id="r", stage_index=0,
            )


class TestProbsCipherAAD:
    def test_rejects_empty_request_id(self):
        c = ProbsCipher(_TEST_KEY)
        with pytest.raises(ProbsCipherError, match="non-empty str"):
            c.encrypt(probs=[0.5], request_id="", stage_index=0)

    def test_rejects_negative_stage_index(self):
        c = ProbsCipher(_TEST_KEY)
        with pytest.raises(ProbsCipherError, match="\\[0, 255\\]"):
            c.encrypt(probs=[0.5], request_id="r", stage_index=-1)

    def test_rejects_oversized_stage_index(self):
        c = ProbsCipher(_TEST_KEY)
        with pytest.raises(ProbsCipherError, match="\\[0, 255\\]"):
            c.encrypt(probs=[0.5], request_id="r", stage_index=256)


class TestDeriveKeyFromPSK:
    def test_round_trip_via_derived_key(self):
        psk = b"my-secret-psk-bytes-32-or-more-long"
        k1 = derive_key_from_psk(psk)
        k2 = derive_key_from_psk(psk)
        # Deterministic: same PSK + same salt → same key.
        assert k1 == k2
        assert len(k1) == 32
        c = ProbsCipher(k1)
        ct = c.encrypt(probs=[0.5], request_id="r", stage_index=0)
        out = c.decrypt(
            ciphertext=ct, request_id="r", stage_index=0, expected_k=1,
        )
        assert out == pytest.approx([0.5], abs=1e-12)

    def test_different_psks_yield_different_keys(self):
        k1 = derive_key_from_psk(b"psk-A-must-be-long-enough-bytes")
        k2 = derive_key_from_psk(b"psk-B-must-be-long-enough-bytes")
        assert k1 != k2

    def test_rejects_short_psk(self):
        with pytest.raises(ProbsCipherError, match="too short"):
            derive_key_from_psk(b"too-short")
