"""Sprint 804 — `prsm compute verify-receipt` pure offline verifier.

Sprint 802 shipped `compute infer --verify-receipt` which
verifies the receipt RETURNED by a fresh inference call.
Sprint 804 adds a pure-offline counterpart: verify a SAVED
receipt JSON file against a known pubkey. No daemon required.

Use cases:
- Auditing a receipt months later (long after the daemon's
  identity may have rotated or the daemon may be offline).
- CI gate: assert a receipt produced in an earlier pipeline
  step verifies against the expected operator pubkey.
- Independent third-party verification: anyone holding the
  operator's published pubkey + a receipt JSON can confirm
  the chain of custody.

  prsm compute verify-receipt --file <path>
                              --pubkey-b64 <base64>
                              [--format text|json]

Pin tests:
- Command registered under `compute` group.
- Valid receipt + correct pubkey → exit 0 + "verified".
- Valid receipt + wrong pubkey → exit 1 + "verification failed".
- Missing file → exit 1 + path in message.
- Malformed JSON → exit 1.
- Missing receipt fields → exit 1 + "parse" or "field" in
  message (operator-readable diagnostic).
- JSON mode returns parseable {ok, verified, ...}.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from click.testing import CliRunner


def _invoke(args):
    from prsm.cli import compute as _compute_group
    return CliRunner().invoke(
        _compute_group, ["verify-receipt"] + list(args),
    )


def _build_signed_receipt():
    """Real EIP-191-style Ed25519-signed receipt + matching
    public_key_b64."""
    from prsm.compute.inference.models import (
        InferenceReceipt, ContentTier,
    )
    from prsm.compute.tee.models import PrivacyLevel, TEEType
    from prsm.compute.inference.receipt import sign_receipt
    from prsm.node.identity import generate_node_identity

    identity = generate_node_identity("verify-file-test")
    receipt = InferenceReceipt(
        job_id="j1", request_id="r1", model_id="gpt2",
        content_tier=ContentTier.A,
        privacy_tier=PrivacyLevel.NONE,
        epsilon_spent=0.0,
        tee_type=TEEType.SOFTWARE,
        tee_attestation=b"att", output_hash=b"\xde\xad",
        duration_seconds=1.0, cost_ftns=Decimal("0.5"),
        settler_node_id="",
    )
    signed = sign_receipt(receipt, identity)
    return signed.to_dict(), identity.public_key_b64


# ---- Command registration --------------------------------------


def test_verify_receipt_command_registered():
    from prsm.cli import compute as _compute_group
    cmd_names = [c.name for c in _compute_group.commands.values()]
    assert "verify-receipt" in cmd_names


# ---- Pass / fail paths ----------------------------------------


def test_valid_receipt_correct_pubkey_passes(tmp_path: Path):
    receipt_dict, pubkey_b64 = _build_signed_receipt()
    rfile = tmp_path / "r.json"
    rfile.write_text(json.dumps(receipt_dict))

    result = _invoke([
        "--file", str(rfile),
        "--pubkey-b64", pubkey_b64,
        "--format", "text",
    ])
    assert result.exit_code == 0, result.output
    assert "verified" in result.output.lower()


def test_wrong_pubkey_fails(tmp_path: Path):
    from prsm.node.identity import generate_node_identity
    receipt_dict, _ = _build_signed_receipt()
    wrong_pubkey = generate_node_identity("other").public_key_b64

    rfile = tmp_path / "r.json"
    rfile.write_text(json.dumps(receipt_dict))

    result = _invoke([
        "--file", str(rfile),
        "--pubkey-b64", wrong_pubkey,
        "--format", "text",
    ])
    assert result.exit_code == 1
    assert (
        "verification failed" in result.output.lower()
        or "verify failed" in result.output.lower()
    )


# ---- Error paths ----------------------------------------------


def test_missing_file_exits_1(tmp_path: Path):
    nonexistent = tmp_path / "nope.json"
    result = _invoke([
        "--file", str(nonexistent),
        "--pubkey-b64", "doesntmatter",
        "--format", "text",
    ])
    assert result.exit_code != 0
    # Operator sees the path in the error
    assert "nope.json" in result.output


def test_malformed_json_exits_1(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json ]")
    result = _invoke([
        "--file", str(bad),
        "--pubkey-b64", "anything",
        "--format", "text",
    ])
    assert result.exit_code == 1
    assert (
        "json" in result.output.lower()
        or "parse" in result.output.lower()
    )


def test_missing_receipt_fields_exits_1(tmp_path: Path):
    """Truncated receipt missing required fields → exit 1 + a
    parse-error-style message so the operator can self-diagnose."""
    # An object missing nearly everything except job_id
    rfile = tmp_path / "incomplete.json"
    rfile.write_text(json.dumps({"job_id": "j1"}))
    result = _invoke([
        "--file", str(rfile),
        "--pubkey-b64", "anything",
        "--format", "text",
    ])
    assert result.exit_code == 1
    out = result.output.lower()
    assert (
        "parse" in out or "field" in out or "invalid" in out
        or "missing" in out
    )


# ---- JSON output ----------------------------------------------


def test_json_mode_returns_payload(tmp_path: Path):
    receipt_dict, pubkey_b64 = _build_signed_receipt()
    rfile = tmp_path / "r.json"
    rfile.write_text(json.dumps(receipt_dict))

    result = _invoke([
        "--file", str(rfile),
        "--pubkey-b64", pubkey_b64,
        "--format", "json",
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["ok"] is True
    assert data["verified"] is True
    # Include job_id so callers can correlate
    assert "job_id" in data
