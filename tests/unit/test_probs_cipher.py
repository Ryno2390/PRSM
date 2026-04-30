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
    X25519AnchoredCipher,
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


# ──────────────────────────────────────────────────────────────────────────
# X25519AnchoredCipher — Phase 3.x.11.q.y'
# ──────────────────────────────────────────────────────────────────────────


class TestX25519AnchoredCipher:
    """Phase 3.x.11.q.y' — per-request ECDH cipher.

    Validates the load-bearing surface invariants:
      - encrypt/decrypt round-trip with shared (server, executor)
      - per-request fresh keypair → forward secrecy across requests
      - ephemeral_pubkey substitution breaks decrypt (AAD bound
        via signing_payload, but the cipher itself authenticates
        via the AES-GCM tag once the derived key changes)
      - rollback prefix path uses distinct AAD from probs path
    """

    def _setup_pair(self):
        """Mint server long-term keypair + ephemeral keypair, set
        up server-side cipher + simulate executor-side derive."""
        # Server long-term keypair.
        srv_priv, srv_pub = X25519AnchoredCipher.generate_keypair()
        # Executor mints fresh ephemeral keypair per request.
        eph_priv, eph_pub = X25519AnchoredCipher.generate_keypair()
        srv = X25519AnchoredCipher(srv_priv)
        # Executor uses an X25519AnchoredCipher constructed from
        # its EPHEMERAL private key. The peer pubkey it passes
        # in is the SERVER's long-term public key.
        exec_cipher = X25519AnchoredCipher(eph_priv)
        return srv, exec_cipher, srv_pub, eph_pub

    def test_round_trip_probs(self):
        srv, exec_cipher, srv_pub, eph_pub = self._setup_pair()
        # Executor encrypts with the server's long-term pubkey
        # as the "peer" → derive shared secret(eph_priv, srv_pub).
        ct = exec_cipher.encrypt(
            probs=[0.5, 0.3],
            request_id="req-1",
            stage_index=0,
            ephemeral_pubkey=srv_pub,
            chain_total_stages=2,
        )
        # Server decrypts with executor's eph_pub as the "peer" →
        # derive shared secret(srv_priv, eph_pub). Equal by ECDH
        # symmetry.
        out = srv.decrypt(
            ciphertext=ct,
            request_id="req-1",
            stage_index=0,
            expected_k=2,
            ephemeral_pubkey=eph_pub,
            chain_total_stages=2,
        )
        assert out == pytest.approx([0.5, 0.3], abs=1e-12)

    def test_round_trip_prefix(self):
        srv, exec_cipher, srv_pub, eph_pub = self._setup_pair()
        ct = exec_cipher.encrypt_prefix(
            prefix=[101, 202, 303],
            request_id="req-1",
            stage_index=0,
            ephemeral_pubkey=srv_pub,
            chain_total_stages=2,
        )
        out = srv.decrypt_prefix(
            ciphertext=ct,
            request_id="req-1",
            stage_index=0,
            expected_k=3,
            ephemeral_pubkey=eph_pub,
            chain_total_stages=2,
        )
        assert out == [101, 202, 303]

    def test_substituted_ephemeral_pubkey_fails(self):
        srv, exec_cipher, srv_pub, eph_pub = self._setup_pair()
        ct = exec_cipher.encrypt(
            probs=[0.5],
            request_id="req-1",
            stage_index=0,
            ephemeral_pubkey=srv_pub,
            chain_total_stages=2,
        )
        # Adversary mints a fresh keypair and substitutes the
        # ephemeral_pubkey on the wire. Server decrypt derives a
        # different shared secret → AES-GCM tag mismatch.
        _, fake_pub = X25519AnchoredCipher.generate_keypair()
        with pytest.raises(ProbsDecryptionError):
            srv.decrypt(
                ciphertext=ct,
                request_id="req-1",
                stage_index=0,
                expected_k=1,
                ephemeral_pubkey=fake_pub,  # substituted!
                chain_total_stages=2,
            )

    def test_per_request_forward_secrecy(self):
        # Two requests with DIFFERENT ephemeral keys yield
        # ciphertexts that cannot decrypt across requests.
        srv_priv, srv_pub = X25519AnchoredCipher.generate_keypair()
        srv = X25519AnchoredCipher(srv_priv)
        eph1_priv, eph1_pub = X25519AnchoredCipher.generate_keypair()
        eph2_priv, eph2_pub = X25519AnchoredCipher.generate_keypair()
        ec1 = X25519AnchoredCipher(eph1_priv)
        ec2 = X25519AnchoredCipher(eph2_priv)
        ct1 = ec1.encrypt(
            probs=[0.5], request_id="req-1", stage_index=0,
            ephemeral_pubkey=srv_pub, chain_total_stages=2,
        )
        # Server tries to decrypt req-1's ciphertext with req-2's
        # ephemeral pubkey → fails.
        with pytest.raises(ProbsDecryptionError):
            srv.decrypt(
                ciphertext=ct1,
                request_id="req-1", stage_index=0, expected_k=1,
                ephemeral_pubkey=eph2_pub,
                chain_total_stages=2,
            )

    def test_distinct_aad_probs_vs_rollback(self):
        # Cross-AAD replay defense: a probs ciphertext cannot be
        # decrypted by decrypt_prefix (AAD spaces are disjoint).
        srv, exec_cipher, srv_pub, eph_pub = self._setup_pair()
        ct_probs = exec_cipher.encrypt(
            probs=[0.5, 0.3],
            request_id="req-1",
            stage_index=0,
            ephemeral_pubkey=srv_pub,
            chain_total_stages=2,
        )
        with pytest.raises(ProbsDecryptionError):
            srv.decrypt_prefix(
                ciphertext=ct_probs,
                request_id="req-1",
                stage_index=0,
                expected_k=2,
                ephemeral_pubkey=eph_pub,
                chain_total_stages=2,
            )

    def test_chain_total_stages_binding(self):
        # info input includes chain_total_stages: a relay that
        # lifts the envelope from a 2-stage chain into a 3-stage
        # chain at the same stage_index gets a different derived
        # key → decrypt fails.
        srv, exec_cipher, srv_pub, eph_pub = self._setup_pair()
        ct = exec_cipher.encrypt(
            probs=[0.5],
            request_id="req-1",
            stage_index=0,
            ephemeral_pubkey=srv_pub,
            chain_total_stages=2,
        )
        with pytest.raises(ProbsDecryptionError):
            srv.decrypt(
                ciphertext=ct,
                request_id="req-1", stage_index=0, expected_k=1,
                ephemeral_pubkey=eph_pub,
                chain_total_stages=3,  # tampered!
            )

    def test_evict_request_drops_keys(self):
        srv, exec_cipher, srv_pub, eph_pub = self._setup_pair()
        # Force a key derivation by encrypting once.
        ct = exec_cipher.encrypt(
            probs=[0.5], request_id="req-1", stage_index=0,
            ephemeral_pubkey=srv_pub, chain_total_stages=2,
        )
        srv.decrypt(
            ciphertext=ct,
            request_id="req-1", stage_index=0, expected_k=1,
            ephemeral_pubkey=eph_pub, chain_total_stages=2,
        )
        # One cache entry expected.
        n = srv.evict_request("req-1")
        assert n == 1
        # Idempotent: second evict returns 0.
        assert srv.evict_request("req-1") == 0

    def test_rejects_wrong_length_privkey(self):
        with pytest.raises(ProbsCipherError, match="32 bytes"):
            X25519AnchoredCipher(b"\x00" * 16)

    def test_rejects_non_bytes_privkey(self):
        with pytest.raises(ProbsCipherError, match="must be bytes"):
            X25519AnchoredCipher("not-bytes")  # type: ignore[arg-type]
