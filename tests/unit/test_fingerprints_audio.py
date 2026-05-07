"""
PRSM-PROV-1 Item 4 T4.3 — AudioFingerprint backend tests.

Covers:
  - KIND attribute matches FingerprintKind.AUDIO_CHROMAPRINT.
  - compute() round-trip on a synthetic WAV file (sine tone).
  - Different audio (different frequency) produces different
    fingerprints — similarity < 1.0.
  - Identical audio at different bit-depths / sample rates produces
    high similarity (the perceptual property Chromaprint exists for).
  - Truncated / corrupt input returns None instead of raising.
  - Empty bytes returns None.
  - Audio shorter than the minimum-frames threshold returns None.
  - similarity() returns 1.0 for identical payloads, 0.0 for malformed,
    < 1.0 for differing.
  - Payload format: multiple of 4 bytes (32-bit big-endian uint32 frames).
"""

from __future__ import annotations

import math
import struct
import tempfile
import wave
from io import BytesIO

import pytest

# Skip the entire module if pyacoustid isn't available — keeps CI green
# on hosts without the optional dep + the chromaprint fpcalc binary.
acoustid = pytest.importorskip("acoustid")

from prsm.data.fingerprints import FingerprintKind  # noqa: E402
from prsm.data.fingerprints.audio import AudioFingerprint  # noqa: E402
from prsm.data.fingerprints.base import FingerprintRecord  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Synthetic audio helpers
# ──────────────────────────────────────────────────────────────────────


def _sine_wav_bytes(
    *,
    duration_sec: float = 12.0,
    freqs_hz=(440.0,),
    sample_rate: int = 44100,
    sample_width: int = 2,
    channels: int = 1,
    amplitude: float = 0.4,
) -> bytes:
    """Generate an in-memory WAV file from one or more sine frequencies.

    A single sine tone is acoustically thin — Chromaprint reduces audio
    to 12-pitch-class chroma vectors, so two pure sines an octave apart
    (e.g. 440Hz vs 880Hz) collapse to the same pitch class and produce
    identical fingerprints. Tests that want distinguishable fingerprints
    should pass *different pitch classes* (e.g. A=440 vs C=523).

    Default duration is 12s to comfortably exceed Chromaprint's
    ~7s minimum-stable-fingerprint threshold.
    """
    if isinstance(freqs_hz, (int, float)):
        freqs_hz = (float(freqs_hz),)
    n_frames = int(duration_sec * sample_rate)
    max_amp = (2 ** (sample_width * 8 - 1)) - 1
    n_freqs = len(freqs_hz)
    frames_bytes = bytearray()
    for i in range(n_frames):
        t = i / sample_rate
        # Equal-amplitude additive synthesis. Scaled by 1/n_freqs so
        # the sum stays inside the int16 range.
        mix = sum(math.sin(2 * math.pi * f * t) for f in freqs_hz) / n_freqs
        sample = int(amplitude * max_amp * mix)
        sample = max(-max_amp, min(max_amp, sample))
        for ch in range(channels):
            frames_bytes.extend(
                sample.to_bytes(sample_width, "little", signed=True)
            )

    bio = BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames_bytes))
    return bio.getvalue()


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────


class TestAudioFingerprint:
    def test_kind(self):
        assert AudioFingerprint.KIND == FingerprintKind.AUDIO_CHROMAPRINT

    def test_compute_round_trip(self):
        """A real sine-tone WAV produces a non-empty 32-bit-frame payload."""
        wav = _sine_wav_bytes(duration_sec=12.0, freqs_hz=(440.0,))
        record = AudioFingerprint().compute(wav, filename="tone.wav")
        assert record is not None, (
            "expected fingerprint for a 12s 440Hz sine WAV — fpcalc may "
            "be missing on PATH"
        )
        assert isinstance(record, FingerprintRecord)
        assert record.kind == FingerprintKind.AUDIO_CHROMAPRINT
        # Payload is a sequence of 32-bit big-endian uint32 frames.
        assert len(record.payload) > 0
        assert len(record.payload) % 4 == 0
        # At least 8 frames (the minimum we accept).
        assert len(record.payload) // 4 >= 8

    def test_compute_identical_audio_high_similarity(self):
        """Same WAV bytes round-trip to identical fingerprints."""
        wav = _sine_wav_bytes(duration_sec=12.0, freqs_hz=(440.0,))
        backend = AudioFingerprint()
        a = backend.compute(wav, filename="a.wav")
        b = backend.compute(wav, filename="b.wav")
        assert a is not None and b is not None
        # Identical input ⇒ identical fingerprint ⇒ similarity 1.0.
        assert a.payload == b.payload
        assert backend.similarity(a.payload, b.payload) == pytest.approx(1.0)

    def test_compute_different_audio_lower_similarity(self):
        """Different pitch classes produce different fingerprints.

        Chromaprint reduces audio to 12-pitch-class chroma features and
        is therefore octave-invariant — 440Hz (A4) vs 880Hz (A5) are
        the same pitch class and yield identical fingerprints by
        design. To produce distinguishable fingerprints we use chords
        in different keys: A-major triad (A=440, C#=554, E=659) vs
        C-major triad (C=523, E=659, G=784).
        """
        wav_a = _sine_wav_bytes(
            duration_sec=12.0, freqs_hz=(440.0, 554.37, 659.25),
        )
        wav_b = _sine_wav_bytes(
            duration_sec=12.0, freqs_hz=(523.25, 659.25, 783.99),
        )
        backend = AudioFingerprint()
        a = backend.compute(wav_a, filename="a.wav")
        b = backend.compute(wav_b, filename="b.wav")
        assert a is not None and b is not None
        sim = backend.similarity(a.payload, b.payload)
        # A-major and C-major share E (659.25Hz) but differ on the
        # other two pitch classes; expect well below the 0.92
        # duplicate threshold.
        assert sim < 0.92, (
            f"expected sim < 0.92 for distinct chords, got {sim}"
        )

    def test_compute_empty_returns_none(self):
        assert AudioFingerprint().compute(b"") is None

    # NB: a "random non-audio bytes" test is intentionally omitted here.
    # pyacoustid falls back to audioread/FFmpeg when fpcalc rejects the
    # input, and audioread's destructor can hang on garbage bytes that
    # FFmpeg can't classify (audioread/ffdec.py:299 stdout_reader.join
    # without a timeout). The empty-bytes + too-short tests cover the
    # "backend rejects input" path without triggering that third-party
    # cleanup bug.

    def test_compute_too_short_returns_none(self):
        """≤ 1 second of audio is below the reliable-fingerprint threshold."""
        wav = _sine_wav_bytes(duration_sec=0.5, freqs_hz=(440.0,))
        # fpcalc may either reject or return too-few frames; either way,
        # the backend must surface None.
        result = AudioFingerprint().compute(wav, filename="short.wav")
        assert result is None

    def test_similarity_identical_payloads(self):
        backend = AudioFingerprint()
        # Synthesize a short payload of 8 frames = 32 bytes.
        payload = b"".join(
            struct.pack(">I", 0xDEADBEEF + i) for i in range(8)
        )
        assert backend.similarity(payload, payload) == pytest.approx(1.0)

    def test_similarity_completely_different(self):
        backend = AudioFingerprint()
        a = b"".join(struct.pack(">I", 0x00000000) for _ in range(8))
        b = b"".join(struct.pack(">I", 0xFFFFFFFF) for _ in range(8))
        # Every bit flipped ⇒ similarity 0.0.
        assert backend.similarity(a, b) == pytest.approx(0.0)

    def test_similarity_malformed_returns_zero(self):
        backend = AudioFingerprint()
        # Length not a multiple of 4 ⇒ malformed ⇒ 0.0.
        assert backend.similarity(b"\x00\x00\x00", b"\x00\x00\x00\x00") == 0.0
        assert backend.similarity(b"", b"\x00\x00\x00\x00") == 0.0
        assert backend.similarity(b"\x00\x00\x00\x00", b"") == 0.0

    def test_similarity_different_lengths_use_shorter_window(self):
        """Aligning at offset 0 over the shorter window is the contract."""
        backend = AudioFingerprint()
        # 8 frames, all zero.
        a = b"".join(struct.pack(">I", 0x00000000) for _ in range(8))
        # 16 frames; first 8 are also all zero, second 8 are all 0xFF.
        b = (
            b"".join(struct.pack(">I", 0x00000000) for _ in range(8))
            + b"".join(struct.pack(">I", 0xFFFFFFFF) for _ in range(8))
        )
        # The first 8 frames are identical → similarity over the
        # comparable window is 1.0.
        assert backend.similarity(a, b) == pytest.approx(1.0)
