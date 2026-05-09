"""CI assertion for the R-2026-05-08-1 composer-only invariant.

This test encodes the council-ratified regression-discipline rule
from `docs/governance/PRSM-CR-2026-05-08.md` §3 RESOLVED 6 as a
runtime/static assertion. The rule:

  > Introducing an actual-execution path on
  > `coinbase_offramp_initiate` (e.g., adding an `execute=true`
  > argument that triggers on-chain swap or fiat off-ramp)
  > without explicit council ratification of the commission
  > completion + audit-trail mechanism = P0 audit finding.

PRSM-CR-2026-05-08 §3 RESOLVED 6 puts the rule on the formal
council record. This test makes the rule technically binding —
a future engineer who silently introduces an execute path
breaks the test, surfacing the change in code review as "why
are you removing the composer-only assertion?"

Per audit-prep §7.27 honest-scope, CI enforcement was flagged
as a future hardening item. This file closes that gap.

When CDP commissioning completes and a future council resolution
ratifies the execute path, the supersedes-this-test mechanism
is: (a) a new resolution document committed under
`docs/governance/PRSM-CR-2026-MM-DD.md` superseding R-2026-05-08-1
explicitly, (b) THIS test file deleted or amended in the SAME
change-set as the supersession resolution, (c) commit message
references both. Without that explicit supersession, the
invariant stands.
"""
from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from prsm.mcp_server import (
    TOOLS,
    handle_coinbase_offramp_initiate,
)
from prsm.node.api import create_api_app


# ──────────────────────────────────────────────────────────────────────
# Tier 1: schema-level assertions
# ──────────────────────────────────────────────────────────────────────


# Tokens that, if present in the tool schema or Pydantic model,
# would indicate an execute-class capability. Defensive against
# obvious renaming (execute → confirm → submit etc.).
EXECUTE_CLASS_TOKENS = {
    "execute", "confirm", "submit", "send",
    "broadcast", "sign", "authorize",
}


class TestSchemaInvariant:
    """Tool definition must not expose any execute-class parameter."""

    def test_tool_schema_properties_excludes_execute_class_tokens(self):
        tool = next(
            t for t in TOOLS if t.name == "coinbase_offramp_initiate"
        )
        properties = tool.inputSchema.get("properties", {})
        property_names = set(properties.keys())

        for forbidden in EXECUTE_CLASS_TOKENS:
            matching = {p for p in property_names if forbidden in p.lower()}
            assert not matching, (
                f"R-2026-05-08-1 violation: coinbase_offramp_initiate "
                f"schema property {matching!r} contains execute-class "
                f"token {forbidden!r}. The tool is composer-only per "
                f"PRSM-CR-2026-05-08 §3 RESOLVED 6 — introducing "
                f"execute-class parameters requires a superseding "
                f"council resolution that explicitly amends or revokes "
                f"R-2026-05-08-1."
            )

    def test_tool_schema_required_is_only_usd_amount(self):
        # The composer-only contract: usd_amount is the only
        # required input. Adding more required fields could indicate
        # smuggling-in of execute-state (e.g., signature, approval
        # tx-hash, etc.).
        tool = next(
            t for t in TOOLS if t.name == "coinbase_offramp_initiate"
        )
        required = set(tool.inputSchema.get("required", []))
        assert required == {"usd_amount"}, (
            f"R-2026-05-08-1: coinbase_offramp_initiate required-fields "
            f"set is {required!r} but the composer-only contract "
            f"specifies exactly {{'usd_amount'}}. Additional required "
            f"fields could smuggle in execute-state (signatures, "
            f"approval tx-hashes, etc.). Adding new required fields "
            f"requires superseding R-2026-05-08-1."
        )

    def test_tool_description_mentions_pending_commission(self):
        # Description must signal composer-only status to MCP-client
        # consumers reading the tool list.
        tool = next(
            t for t in TOOLS if t.name == "coinbase_offramp_initiate"
        )
        description = tool.description.lower()
        assert "pending_commission" in description.upper().lower() or "pending" in description, (
            f"R-2026-05-08-1: coinbase_offramp_initiate description "
            f"must signal composer-only status (e.g., reference "
            f"PENDING_COMMISSION). MCP-client consumers reading the "
            f"tool list must see the composer-only framing."
        )


# ──────────────────────────────────────────────────────────────────────
# Tier 2: Pydantic model invariant
# ──────────────────────────────────────────────────────────────────────


class TestPydanticModelInvariant:
    """The `_OfframpQuoteRequest` Pydantic model is defined as a nested
    class inside `create_api_app`. Use source introspection to
    verify its field set."""

    def test_offramp_quote_request_fields_excluded_from_execute_tokens(self):
        source = inspect.getsource(create_api_app)
        # Find the _OfframpQuoteRequest class block.
        marker_start = "class _OfframpQuoteRequest(BaseModel):"
        marker_end_candidates = [
            "@app.post(",  # next route definition
        ]
        start_idx = source.find(marker_start)
        assert start_idx >= 0, (
            "_OfframpQuoteRequest class missing from create_api_app "
            "source. R-2026-05-08-1 enforcement requires the model to "
            "exist for field-set introspection."
        )

        end_idx = min(
            (source.find(m, start_idx + len(marker_start))
             for m in marker_end_candidates
             if source.find(m, start_idx + len(marker_start)) > 0),
            default=len(source),
        )
        class_block = source[start_idx:end_idx]

        # Search class body for execute-class field-name tokens.
        for forbidden in EXECUTE_CLASS_TOKENS:
            # Match `<token>:` or `<token> :` as a Pydantic field
            # annotation. Word boundary to avoid false-positives like
            # "submit_proof_failure" (which we don't ban as a
            # standalone word).
            import re
            pattern = re.compile(
                rf"\b\w*{re.escape(forbidden)}\w*\s*:", re.IGNORECASE,
            )
            matches = pattern.findall(class_block)
            # Allow purely substring-only matches that aren't
            # field-name-shaped (e.g., a comment containing the word).
            matches = [m for m in matches if not m.startswith("#")]
            assert not matches, (
                f"R-2026-05-08-1 violation: _OfframpQuoteRequest "
                f"contains field annotation {matches!r} matching "
                f"execute-class token {forbidden!r}. The composer-"
                f"only model accepts only `usd_amount` + "
                f"`bank_account_alias`. Adding execute-class fields "
                f"requires superseding R-2026-05-08-1 council "
                f"commitment."
            )


# ──────────────────────────────────────────────────────────────────────
# Tier 3: endpoint response invariant
# ──────────────────────────────────────────────────────────────────────


def _node_with_ftns(*, balance_ftns: float = 100.0):
    node = MagicMock()
    node.identity = MagicMock()
    node.identity.node_id = "test-node"
    ftns_ledger = MagicMock()
    ftns_ledger._is_initialized = True
    ftns_ledger._connected_address = "0x" + "11" * 20
    ftns_ledger._decimals = 18
    ftns_ledger.get_balance = AsyncMock(return_value=balance_ftns)
    node.ftns_ledger = ftns_ledger
    return node


class TestEndpointResponseInvariant:
    """The `/wallet/offramp/quote` endpoint MUST return
    `status: "PENDING_COMMISSION"` on the happy path. This is the
    user-visible composer-only signal."""

    def test_happy_path_returns_pending_commission(self):
        node = _node_with_ftns(balance_ftns=4200.0)
        client = TestClient(create_api_app(node, enable_security=False))
        response = client.post(
            "/wallet/offramp/quote",
            json={"usd_amount": 500.0},
        )
        assert response.status_code == 200
        body = response.json()
        assert body.get("status") == "PENDING_COMMISSION", (
            f"R-2026-05-08-1 violation: /wallet/offramp/quote happy-"
            f"path response status is {body.get('status')!r}, but "
            f"the composer-only contract pins it at "
            f"'PENDING_COMMISSION'. Changing the status field "
            f"(e.g., to 'IN_FLIGHT' or 'EXECUTED') without "
            f"superseding R-2026-05-08-1 = P0 audit finding per "
            f"PRSM-CR-2026-05-08 §3 RESOLVED 6."
        )

    def test_happy_path_response_carries_commission_gate_note(self):
        # The commission_gate_note field is the user-facing
        # explanation of why status is PENDING. Removing it would
        # silently strip the composer-only signal from end-user
        # output.
        node = _node_with_ftns(balance_ftns=4200.0)
        client = TestClient(create_api_app(node, enable_security=False))
        response = client.post(
            "/wallet/offramp/quote",
            json={"usd_amount": 500.0},
        )
        body = response.json()
        assert "commission_gate_note" in body, (
            "R-2026-05-08-1: /wallet/offramp/quote response must "
            "include `commission_gate_note` field — the user-facing "
            "explanation of PENDING_COMMISSION status. Removing this "
            "field would silently strip the composer-only signal "
            "from the end-user output."
        )
        note = body["commission_gate_note"].lower()
        # Defensive against re-wording: the note must reference the
        # composer-only constraint in some form.
        assert (
            "does not initiate" in note
            or "composer" in note
            or "pending" in note
        ), (
            f"R-2026-05-08-1: commission_gate_note must explicitly "
            f"signal composer-only constraint. Current text: "
            f"{body['commission_gate_note']!r}"
        )


# ──────────────────────────────────────────────────────────────────────
# Tier 4: handler source introspection
# ──────────────────────────────────────────────────────────────────────


# Forbidden tokens in the handler source — direct calls or
# imports that would constitute on-chain or fiat-side action.
HANDLER_FORBIDDEN_TOKENS = {
    "sign_transaction",
    "send_raw_transaction",
    "release_escrow",
    "pull_and_distribute",
    "deposit_key",
    "release(",  # specific function name on KeyDistributionClient
    "submit_proof_failure",
    "record_heartbeat",
    "record_funding",
    "Web3(",
    "eth.send",
    "wallet_sign",
    "private_key",
}


class TestHandlerSourceInvariant:
    """`handle_coinbase_offramp_initiate` source must not reference
    blockchain-write or signing primitives. The handler is intended
    to delegate to `_call_node_api` only; any direct contract or
    Web3 call constitutes an execute-path."""

    def test_handler_source_excludes_blockchain_write_tokens(self):
        handler_source = inspect.getsource(handle_coinbase_offramp_initiate)

        for forbidden in HANDLER_FORBIDDEN_TOKENS:
            assert forbidden not in handler_source, (
                f"R-2026-05-08-1 violation: handle_coinbase_offramp_"
                f"initiate source contains forbidden token "
                f"{forbidden!r}. The composer-only handler must "
                f"delegate to `_call_node_api` and string formatting "
                f"only — direct blockchain-write or signing primitives "
                f"would constitute an execute-path. Adding such a "
                f"call requires superseding R-2026-05-08-1 council "
                f"commitment."
            )

    def test_handler_source_uses_only_call_node_api_and_formatting(self):
        # Positive assertion: the handler MUST call `_call_node_api`
        # (otherwise it's a no-op or worse). Confirms the
        # delegation-only architecture.
        handler_source = inspect.getsource(handle_coinbase_offramp_initiate)
        assert "_call_node_api" in handler_source, (
            "R-2026-05-08-1: handle_coinbase_offramp_initiate must "
            "invoke `_call_node_api` to compose the quote. A handler "
            "that bypasses the API delegation pattern would be "
            "structurally distinct from the composer-only architecture."
        )


# ──────────────────────────────────────────────────────────────────────
# Tier 5: tool count drift signal
# ──────────────────────────────────────────────────────────────────────


class TestNoSiblingExecuteTool:
    """A common bypass for the composer-only invariant would be to
    add a SIBLING tool (e.g., `coinbase_offramp_execute`) that
    triggers the actual on-chain action. Defensive check: no such
    sibling tool exists today."""

    def test_no_offramp_execute_sibling_tool(self):
        execute_class_tool_names = [
            t.name for t in TOOLS
            if "offramp" in t.name.lower()
            and any(token in t.name.lower() for token in EXECUTE_CLASS_TOKENS)
        ]
        # The composer initiate tool itself contains 'initiate' — not
        # an execute-class token, so it's allowed. We only forbid
        # SIBLINGS that contain the explicit execute-class tokens.
        # Filter out the canonical composer name.
        sibling_violations = [
            n for n in execute_class_tool_names
            if n != "coinbase_offramp_initiate"
        ]
        assert not sibling_violations, (
            f"R-2026-05-08-1: sibling execute-class tool(s) detected: "
            f"{sibling_violations!r}. Adding an `offramp_execute` "
            f"or `offramp_submit` sibling tool would bypass the "
            f"composer-only invariant on coinbase_offramp_initiate. "
            f"Both must be ratified together via a superseding "
            f"council resolution."
        )
