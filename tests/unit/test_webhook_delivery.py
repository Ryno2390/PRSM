"""WebhookDeliverer + signature primitives."""
from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from prsm.node.webhook_delivery import (
    DeliveryResult,
    WebhookDeliverer,
    compute_signature,
)


# ──────────────────────────────────────────────────────────────────────
# Signature primitive
# ──────────────────────────────────────────────────────────────────────


class TestSignature:
    def test_compute_signature_deterministic(self):
        body = b'{"x":1}'
        sig1 = compute_signature("secret", body)
        sig2 = compute_signature("secret", body)
        assert sig1 == sig2
        assert sig1.startswith("sha256=")

    def test_signature_changes_with_body(self):
        s1 = compute_signature("secret", b'{"x":1}')
        s2 = compute_signature("secret", b'{"x":2}')
        assert s1 != s2

    def test_signature_changes_with_secret(self):
        s1 = compute_signature("a", b'{"x":1}')
        s2 = compute_signature("b", b'{"x":1}')
        assert s1 != s2

    def test_signature_verifiable_by_receiver(self):
        """Independent recompute matches our signature."""
        body = b'{"event":"daemon.crashed"}'
        sig = compute_signature("ops-shared-secret", body)
        # Receiver-side recompute.
        expected = hmac.new(
            b"ops-shared-secret", body, hashlib.sha256,
        ).hexdigest()
        assert sig == f"sha256={expected}"


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_max_attempts_must_be_positive(self):
        with pytest.raises(ValueError):
            WebhookDeliverer(max_attempts=0)

    def test_timeout_must_be_positive(self):
        with pytest.raises(ValueError):
            WebhookDeliverer(timeout_seconds=0)


# ──────────────────────────────────────────────────────────────────────
# Happy path
# ──────────────────────────────────────────────────────────────────────


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_2xx_returns_success_first_attempt(self):
        captured = {}

        async def fake_post(url, body, headers):
            captured["url"] = url
            captured["body"] = body
            captured["headers"] = headers
            return 200, "ok"

        deliverer = WebhookDeliverer(max_attempts=3)
        result = await deliverer.deliver(
            url="https://hook.example.com/in",
            event="test.event",
            payload={"hello": "world"},
            post_fn=fake_post,
        )
        assert result.success is True
        assert result.status_code == 200
        assert result.attempts == 1
        # URL passed through.
        assert captured["url"] == "https://hook.example.com/in"
        # Body is canonical JSON (sorted keys).
        assert captured["body"] == b'{"hello": "world"}'
        # Standard headers present.
        assert captured["headers"]["X-PRSM-Event"] == "test.event"

    @pytest.mark.asyncio
    async def test_signed_payload_includes_signature_header(self):
        captured = {}

        async def fake_post(url, body, headers):
            captured["headers"] = headers
            return 200, "ok"

        deliverer = WebhookDeliverer()
        await deliverer.deliver(
            url="https://hook.example.com/in",
            event="test.event",
            payload={"x": 1},
            secret="ops-secret",
            post_fn=fake_post,
        )
        assert "X-PRSM-Signature" in captured["headers"]
        # Verify the signature.
        body = b'{"x": 1}'
        expected = compute_signature("ops-secret", body)
        assert captured["headers"]["X-PRSM-Signature"] == expected

    @pytest.mark.asyncio
    async def test_no_secret_omits_signature_header(self):
        captured = {}

        async def fake_post(url, body, headers):
            captured["headers"] = headers
            return 200, "ok"

        deliverer = WebhookDeliverer()
        await deliverer.deliver(
            url="https://hook.example.com/in",
            event="test.event",
            payload={"x": 1},
            post_fn=fake_post,
        )
        assert "X-PRSM-Signature" not in captured["headers"]


# ──────────────────────────────────────────────────────────────────────
# Retry behavior
# ──────────────────────────────────────────────────────────────────────


class TestRetry:
    @pytest.mark.asyncio
    async def test_5xx_retries_and_eventually_succeeds(self):
        attempts = []

        async def fake_post(url, body, headers):
            attempts.append(len(attempts) + 1)
            if len(attempts) < 3:
                return 503, "service unavailable"
            return 200, "ok"

        deliverer = WebhookDeliverer(
            max_attempts=3,
            sleep_fn=lambda s: None,  # no real sleep in tests
        )

        # sleep_fn must be awaitable.
        async def no_sleep(s):
            return None

        deliverer._sleep = no_sleep

        result = await deliverer.deliver(
            url="https://hook.example.com/in",
            event="test.event",
            payload={"x": 1},
            post_fn=fake_post,
        )
        assert result.success is True
        assert result.status_code == 200
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_5xx_exhausts_retries_returns_failure(self):
        async def fake_post(url, body, headers):
            return 503, "down"

        async def no_sleep(s):
            return None

        deliverer = WebhookDeliverer(max_attempts=3)
        deliverer._sleep = no_sleep

        result = await deliverer.deliver(
            url="https://hook.example.com/in",
            event="test.event",
            payload={"x": 1},
            post_fn=fake_post,
        )
        assert result.success is False
        assert result.attempts == 3
        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_4xx_non_retryable_gives_up_immediately(self):
        attempts = [0]

        async def fake_post(url, body, headers):
            attempts[0] += 1
            return 400, "bad request"

        async def no_sleep(s):
            return None

        deliverer = WebhookDeliverer(max_attempts=5)
        deliverer._sleep = no_sleep

        result = await deliverer.deliver(
            url="https://hook.example.com/in",
            event="test.event",
            payload={"x": 1},
            post_fn=fake_post,
        )
        assert result.success is False
        assert attempts[0] == 1  # only one attempt made
        assert result.status_code == 400
        assert "non-retryable" in result.error

    @pytest.mark.asyncio
    async def test_429_is_retryable(self):
        attempts = [0]

        async def fake_post(url, body, headers):
            attempts[0] += 1
            if attempts[0] < 2:
                return 429, "too many"
            return 200, "ok"

        async def no_sleep(s):
            return None

        deliverer = WebhookDeliverer(max_attempts=3)
        deliverer._sleep = no_sleep

        result = await deliverer.deliver(
            url="https://hook.example.com/in",
            event="test.event",
            payload={"x": 1},
            post_fn=fake_post,
        )
        assert result.success is True
        assert attempts[0] == 2

    @pytest.mark.asyncio
    async def test_exception_in_post_retries(self):
        attempts = [0]

        async def fake_post(url, body, headers):
            attempts[0] += 1
            if attempts[0] < 2:
                raise ConnectionError("transient")
            return 200, "ok"

        async def no_sleep(s):
            return None

        deliverer = WebhookDeliverer(max_attempts=3)
        deliverer._sleep = no_sleep

        result = await deliverer.deliver(
            url="https://hook.example.com/in",
            event="test.event",
            payload={"x": 1},
            post_fn=fake_post,
        )
        assert result.success is True
        assert attempts[0] == 2
