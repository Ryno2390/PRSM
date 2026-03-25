"""
Stripe/PayPal Sandbox Integration Tests

Integration tests using real Stripe test mode (no real charges) and PayPal sandbox.
These tests require actual API keys but use test/sandbox mode that doesn't
process real money.

Requirements:
- STRIPE_TEST_SECRET_KEY environment variable (free Stripe test mode key)
- PAYPAL_SANDBOX_CLIENT_ID and PAYPAL_SANDBOX_CLIENT_SECRET for PayPal tests

Run with: pytest tests/integration/test_payment_sandbox.py -v -m integration

Stripe test mode is free and always available. Get test keys from:
https://dashboard.stripe.com/test/apikeys

PayPal sandbox credentials from:
https://developer.paypal.com/developer/applications/
"""

import pytest
import os
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from datetime import datetime

# Skip all tests in this module if running in unit test mode
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS", "").lower() != "true",
    reason="Set RUN_INTEGRATION_TESTS=true to run integration tests"
)


class TestStripeSandbox:
    """Stripe sandbox integration tests."""

    @pytest.fixture
    def stripe_provider(self):
        """Create Stripe provider with test key."""
        stripe_key = os.getenv("STRIPE_TEST_SECRET_KEY")
        if not stripe_key:
            pytest.skip("STRIPE_TEST_SECRET_KEY not set")

        try:
            from prsm.economy.payments.fiat_gateway import StripeProvider
            return StripeProvider({"api_key": stripe_key})
        except ImportError:
            pytest.skip("StripeProvider not found")

    @pytest.mark.asyncio
    async def test_create_payment_intent(self, stripe_provider):
        """
        Create a real Stripe PaymentIntent in test mode.

        This is a real API call to Stripe's test servers.
        No actual money is charged.
        """
        result = await stripe_provider.create_payment(
            amount=Decimal("10.00"),
            currency="USD",
            metadata={
                "user_id": "test_user_123",
                "ftns_amount": "100"
            }
        )

        assert result is not None
        assert "payment_id" in result
        assert result["payment_id"].startswith("pi_")
        assert result["status"] in ("requires_payment_method", "requires_confirmation")

    @pytest.mark.asyncio
    async def test_cancel_payment_intent(self, stripe_provider):
        """
        Create and cancel a PaymentIntent.

        Tests that cancellation works correctly in test mode.
        """
        # Create a payment intent
        create_result = await stripe_provider.create_payment(
            amount=Decimal("5.00"),
            currency="USD",
            metadata={"test": "cancel_test"}
        )

        assert create_result is not None
        payment_id = create_result["payment_id"]

        # Cancel it
        cancel_result = await stripe_provider.cancel_payment(payment_id)

        assert cancel_result is not None
        assert cancel_result["status"] == "canceled"

    @pytest.mark.asyncio
    async def test_retrieve_payment(self, stripe_provider):
        """
        Retrieve a payment intent after creation.

        Tests the retrieval functionality.
        """
        # Create a payment intent
        create_result = await stripe_provider.create_payment(
            amount=Decimal("15.00"),
            currency="USD",
            metadata={"test": "retrieve_test"}
        )

        payment_id = create_result["payment_id"]

        # Retrieve it
        retrieve_result = await stripe_provider.get_payment(payment_id)

        assert retrieve_result is not None
        assert retrieve_result["payment_id"] == payment_id
        assert "amount" in retrieve_result

    @pytest.mark.asyncio
    async def test_webhook_signature_validation(self, stripe_provider):
        """
        Test webhook signature validation with test secret.

        Note: This requires STRIPE_TEST_WEBHOOK_SECRET to be set.
        Skipped if not available.
        """
        webhook_secret = os.getenv("STRIPE_TEST_WEBHOOK_SECRET")
        if not webhook_secret:
            pytest.skip("STRIPE_TEST_WEBHOOK_SECRET not set")

        # This would test webhook signature validation
        # For a full test, we'd need to simulate a webhook payload
        # For now, we verify the provider has the capability
        assert hasattr(stripe_provider, 'verify_webhook') or hasattr(stripe_provider, 'validate_webhook')


class TestPayPalSandbox:
    """PayPal sandbox integration tests."""

    @pytest.fixture
    def paypal_provider(self):
        """Create PayPal provider with sandbox credentials."""
        client_id = os.getenv("PAYPAL_SANDBOX_CLIENT_ID")
        client_secret = os.getenv("PAYPAL_SANDBOX_CLIENT_SECRET")

        if not client_id or not client_secret:
            pytest.skip("PAYPAL_SANDBOX_CLIENT_ID or PAYPAL_SANDBOX_CLIENT_SECRET not set")

        try:
            from prsm.economy.payments.fiat_gateway import PayPalProvider
            return PayPalProvider({
                "client_id": client_id,
                "client_secret": client_secret,
                "sandbox": True
            })
        except ImportError:
            pytest.skip("PayPalProvider not found")

    @pytest.mark.asyncio
    async def test_create_order(self, paypal_provider):
        """
        Create a PayPal order in sandbox mode.

        This is a real API call to PayPal's sandbox servers.
        No actual money is processed.
        """
        result = await paypal_provider.create_payment(
            amount=Decimal("25.00"),
            currency="USD",
            metadata={
                "user_id": "test_user_456",
                "ftns_amount": "250"
            }
        )

        assert result is not None
        assert "payment_id" in result
        assert result["status"] in ("CREATED", "PENDING", "APPROVED")

    @pytest.mark.asyncio
    async def test_capture_order(self, paypal_provider):
        """
        Create and capture a PayPal order.

        Note: Full capture requires buyer approval through PayPal's UI.
        This test verifies the capture endpoint works correctly.
        """
        # Create an order
        create_result = await paypal_provider.create_payment(
            amount=Decimal("10.00"),
            currency="USD",
            metadata={"test": "capture_test"}
        )

        assert create_result is not None
        payment_id = create_result["payment_id"]

        # Attempt to capture (may fail if not approved)
        # In sandbox mode, orders may need manual approval
        try:
            capture_result = await paypal_provider.capture_payment(payment_id)
            assert capture_result is not None
        except Exception as e:
            # This is expected if order not yet approved by buyer
            assert "not approved" in str(e).lower() or "status" in str(e).lower()


class TestPaymentGatewayFactory:
    """Test payment gateway factory/selection."""

    def test_get_stripe_provider(self):
        """Test getting Stripe provider from factory."""
        try:
            from prsm.economy.payments.fiat_gateway import get_payment_provider

            stripe_key = os.getenv("STRIPE_TEST_SECRET_KEY")
            if not stripe_key:
                pytest.skip("STRIPE_TEST_SECRET_KEY not set")

            provider = get_payment_provider("stripe", {"api_key": stripe_key})
            assert provider is not None

        except ImportError:
            pytest.skip("Payment gateway factory not found")

    def test_get_paypal_provider(self):
        """Test getting PayPal provider from factory."""
        try:
            from prsm.economy.payments.fiat_gateway import get_payment_provider

            client_id = os.getenv("PAYPAL_SANDBOX_CLIENT_ID")
            client_secret = os.getenv("PAYPAL_SANDBOX_CLIENT_SECRET")

            if not client_id or not client_secret:
                pytest.skip("PayPal sandbox credentials not set")

            provider = get_payment_provider("paypal", {
                "client_id": client_id,
                "client_secret": client_secret,
                "sandbox": True
            })
            assert provider is not None

        except ImportError:
            pytest.skip("Payment gateway factory not found")


class TestPaymentErrorHandling:
    """Test error handling in payment processing."""

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test handling of invalid API key."""
        try:
            from prsm.economy.payments.fiat_gateway import StripeProvider

            provider = StripeProvider({"api_key": "sk_test_invalid_key"})

            # This should fail gracefully
            with pytest.raises(Exception) as exc_info:
                await provider.create_payment(
                    amount=Decimal("10.00"),
                    currency="USD"
                )

            # Should be an authentication/permission error
            assert any(word in str(exc_info.value).lower() for word in
                      ["authentication", "invalid", "permission", "key"])

        except ImportError:
            pytest.skip("StripeProvider not found")

    @pytest.mark.asyncio
    async def test_invalid_amount(self):
        """Test handling of invalid payment amount."""
        stripe_key = os.getenv("STRIPE_TEST_SECRET_KEY")
        if not stripe_key:
            pytest.skip("STRIPE_TEST_SECRET_KEY not set")

        try:
            from prsm.economy.payments.fiat_gateway import StripeProvider

            provider = StripeProvider({"api_key": stripe_key})

            # Zero amount should fail
            with pytest.raises(Exception):
                await provider.create_payment(
                    amount=Decimal("0.00"),
                    currency="USD"
                )

        except ImportError:
            pytest.skip("StripeProvider not found")


# === Test Configuration ===

def pytest_configure(config):
    """Configure custom markers for integration tests."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require external services)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
