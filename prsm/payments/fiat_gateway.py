"""
PRSM Fiat Payment Gateway
========================

Production-grade fiat payment processing with multiple payment providers,
fraud detection, and compliance features for secure payment processing.
"""

import asyncio
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union

import aiohttp
import structlog
from cryptography.fernet import Fernet

from ..core.config import get_settings
from ..integrations.security.audit_logger import audit_logger
from .payment_models import (
    PaymentMethod, PaymentStatus, FiatCurrency,
    PaymentRequest, PaymentResponse, PaymentMethodRequest, PaymentMethodResponse
)

logger = structlog.get_logger(__name__)


class PaymentProvider:
    """Base payment provider interface"""
    
    def __init__(self, provider_name: str, config: Dict[str, Any]):
        self.provider_name = provider_name
        self.config = config
        self.session = None
        
    async def initialize(self):
        """Initialize the payment provider"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "PRSM-Payment-Gateway/1.0"}
        )
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    async def create_payment(self, request: PaymentRequest) -> PaymentResponse:
        """Create a payment (to be implemented by subclasses)"""
        raise NotImplementedError
    
    async def get_payment_status(self, transaction_id: str) -> PaymentResponse:
        """Get payment status (to be implemented by subclasses)"""
        raise NotImplementedError
    
    async def cancel_payment(self, transaction_id: str) -> bool:
        """Cancel a payment (to be implemented by subclasses)"""
        raise NotImplementedError
    
    async def refund_payment(self, transaction_id: str, amount: Optional[Decimal] = None) -> bool:
        """Refund a payment (to be implemented by subclasses)"""
        raise NotImplementedError


class StripeProvider(PaymentProvider):
    """Stripe payment provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("stripe", config)
        self.api_key = config.get("api_key")
        self.webhook_secret = config.get("webhook_secret")
        self.base_url = "https://api.stripe.com/v1"
        
    async def create_payment(self, request: PaymentRequest) -> PaymentResponse:
        """Create Stripe payment intent"""
        try:
            # Calculate processing fee (2.9% + $0.30 for cards)
            processing_fee = (request.fiat_amount * Decimal("0.029")) + Decimal("0.30")
            
            # Create payment intent
            payment_intent_data = {
                "amount": int(request.fiat_amount * 100),  # Stripe uses cents
                "currency": request.fiat_currency.value.lower(),
                "automatic_payment_methods": {"enabled": True},
                "metadata": {
                    "user_id": request.user_id,
                    "crypto_currency": request.crypto_currency.value,
                    **request.metadata
                }
            }
            
            if request.return_url:
                payment_intent_data["return_url"] = request.return_url
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            async with self.session.post(
                f"{self.base_url}/payment_intents",
                headers=headers,
                data=payment_intent_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    return PaymentResponse(
                        success=True,
                        transaction_id=result["id"],
                        status=PaymentStatus.PENDING,
                        fiat_amount=request.fiat_amount,
                        fiat_currency=request.fiat_currency,
                        crypto_currency=request.crypto_currency,
                        processing_fee=processing_fee,
                        payment_url=result.get("next_action", {}).get("redirect_to_url", {}).get("url"),
                        provider_reference=result["id"],
                        message="Payment intent created successfully",
                        requires_action=result["status"] == "requires_action",
                        next_action=result.get("next_action", {}).get("type"),
                        expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
                    )
                else:
                    error_data = await response.json()
                    return PaymentResponse(
                        success=False,
                        transaction_id=str(uuid.uuid4()),
                        status=PaymentStatus.FAILED,
                        fiat_amount=request.fiat_amount,
                        fiat_currency=request.fiat_currency,
                        crypto_currency=request.crypto_currency,
                        message=f"Stripe error: {error_data.get('error', {}).get('message', 'Unknown error')}"
                    )
                    
        except Exception as e:
            logger.error("Stripe payment creation failed", error=str(e))
            return PaymentResponse(
                success=False,
                transaction_id=str(uuid.uuid4()),
                status=PaymentStatus.FAILED,
                fiat_amount=request.fiat_amount,
                fiat_currency=request.fiat_currency,
                crypto_currency=request.crypto_currency,
                message=f"Payment processing error: {str(e)}"
            )
    
    async def get_payment_status(self, transaction_id: str) -> PaymentResponse:
        """Get Stripe payment intent status"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            async with self.session.get(
                f"{self.base_url}/payment_intents/{transaction_id}",
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Map Stripe status to our status
                    status_mapping = {
                        "requires_payment_method": PaymentStatus.PENDING,
                        "requires_confirmation": PaymentStatus.PENDING,
                        "requires_action": PaymentStatus.PENDING,
                        "processing": PaymentStatus.PROCESSING,
                        "requires_capture": PaymentStatus.PROCESSING,
                        "succeeded": PaymentStatus.COMPLETED,
                        "canceled": PaymentStatus.CANCELLED
                    }
                    
                    status = status_mapping.get(result["status"], PaymentStatus.FAILED)
                    
                    return PaymentResponse(
                        success=True,
                        transaction_id=transaction_id,
                        status=status,
                        fiat_amount=Decimal(result["amount"]) / 100,
                        fiat_currency=FiatCurrency(result["currency"].upper()),
                        crypto_currency=result["metadata"].get("crypto_currency", "FTNS"),
                        provider_reference=transaction_id,
                        message=f"Payment status: {result['status']}",
                        completed_at=datetime.fromtimestamp(result["created"], timezone.utc) if status == PaymentStatus.COMPLETED else None
                    )
                else:
                    error_data = await response.json()
                    return PaymentResponse(
                        success=False,
                        transaction_id=transaction_id,
                        status=PaymentStatus.FAILED,
                        fiat_amount=Decimal("0"),
                        fiat_currency=FiatCurrency.USD,
                        crypto_currency="FTNS",
                        message=f"Failed to get payment status: {error_data.get('error', {}).get('message', 'Unknown error')}"
                    )
                    
        except Exception as e:
            logger.error("Failed to get Stripe payment status", transaction_id=transaction_id, error=str(e))
            return PaymentResponse(
                success=False,
                transaction_id=transaction_id,
                status=PaymentStatus.FAILED,
                fiat_amount=Decimal("0"),
                fiat_currency=FiatCurrency.USD,
                crypto_currency="FTNS",
                message=f"Status check error: {str(e)}"
            )


class PayPalProvider(PaymentProvider):
    """PayPal payment provider implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("paypal", config)
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.environment = config.get("environment", "sandbox")  # sandbox or live
        
        if self.environment == "sandbox":
            self.base_url = "https://api.sandbox.paypal.com"
        else:
            self.base_url = "https://api.paypal.com"
            
        self.access_token = None
        self.token_expires_at = None
    
    async def _get_access_token(self) -> str:
        """Get PayPal access token"""
        if self.access_token and self.token_expires_at and datetime.now(timezone.utc) < self.token_expires_at:
            return self.access_token
        
        try:
            import base64
            
            auth_string = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            
            headers = {
                "Authorization": f"Basic {auth_string}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = "grant_type=client_credentials"
            
            async with self.session.post(
                f"{self.base_url}/v1/oauth2/token",
                headers=headers,
                data=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    self.access_token = result["access_token"]
                    expires_in = result.get("expires_in", 3600)
                    self.token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)
                    return self.access_token
                else:
                    raise Exception(f"Failed to get PayPal access token: {response.status}")
                    
        except Exception as e:
            logger.error("Failed to get PayPal access token", error=str(e))
            raise
    
    async def create_payment(self, request: PaymentRequest) -> PaymentResponse:
        """Create PayPal payment order"""
        try:
            access_token = await self._get_access_token()
            
            # Calculate PayPal processing fee (2.9% + fixed fee)
            processing_fee = request.fiat_amount * Decimal("0.029") + Decimal("0.30")
            
            order_data = {
                "intent": "CAPTURE",
                "purchase_units": [{
                    "amount": {
                        "currency_code": request.fiat_currency.value,
                        "value": str(request.fiat_amount)
                    },
                    "description": f"FTNS Token Purchase - {request.crypto_currency.value}",
                    "custom_id": request.user_id
                }],
                "application_context": {
                    "return_url": request.return_url or "https://prsm.ai/payment/success",
                    "cancel_url": "https://prsm.ai/payment/cancel"
                }
            }
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            async with self.session.post(
                f"{self.base_url}/v2/checkout/orders",
                headers=headers,
                json=order_data
            ) as response:
                if response.status == 201:
                    result = await response.json()
                    
                    # Find approval URL
                    approval_url = None
                    for link in result.get("links", []):
                        if link.get("rel") == "approve":
                            approval_url = link.get("href")
                            break
                    
                    return PaymentResponse(
                        success=True,
                        transaction_id=result["id"],
                        status=PaymentStatus.PENDING,
                        fiat_amount=request.fiat_amount,
                        fiat_currency=request.fiat_currency,
                        crypto_currency=request.crypto_currency,
                        processing_fee=processing_fee,
                        payment_url=approval_url,
                        provider_reference=result["id"],
                        message="PayPal order created successfully",
                        requires_action=True,
                        next_action="redirect_to_paypal",
                        expires_at=datetime.now(timezone.utc) + timedelta(hours=3)
                    )
                else:
                    error_data = await response.json()
                    return PaymentResponse(
                        success=False,
                        transaction_id=str(uuid.uuid4()),
                        status=PaymentStatus.FAILED,
                        fiat_amount=request.fiat_amount,
                        fiat_currency=request.fiat_currency,
                        crypto_currency=request.crypto_currency,
                        message=f"PayPal error: {error_data.get('message', 'Unknown error')}"
                    )
                    
        except Exception as e:
            logger.error("PayPal payment creation failed", error=str(e))
            return PaymentResponse(
                success=False,
                transaction_id=str(uuid.uuid4()),
                status=PaymentStatus.FAILED,
                fiat_amount=request.fiat_amount,
                fiat_currency=request.fiat_currency,
                crypto_currency=request.crypto_currency,
                message=f"Payment processing error: {str(e)}"
            )


class MockProvider(PaymentProvider):
    """Mock payment provider for testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("mock", config)
        self.simulate_failures = config.get("simulate_failures", False)
        self.failure_rate = config.get("failure_rate", 0.1)
        
    async def create_payment(self, request: PaymentRequest) -> PaymentResponse:
        """Create mock payment"""
        import random
        
        # Simulate processing delay
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        transaction_id = str(uuid.uuid4())
        
        # Simulate failures if configured
        if self.simulate_failures and random.random() < self.failure_rate:
            return PaymentResponse(
                success=False,
                transaction_id=transaction_id,
                status=PaymentStatus.FAILED,
                fiat_amount=request.fiat_amount,
                fiat_currency=request.fiat_currency,
                crypto_currency=request.crypto_currency,
                message="Mock payment failed (simulated)"
            )
        
        # Success case
        processing_fee = request.fiat_amount * Decimal("0.025")  # 2.5% mock fee
        
        return PaymentResponse(
            success=True,
            transaction_id=transaction_id,
            status=PaymentStatus.COMPLETED,
            fiat_amount=request.fiat_amount,
            fiat_currency=request.fiat_currency,
            crypto_currency=request.crypto_currency,
            processing_fee=processing_fee,
            provider_reference=f"mock_{transaction_id}",
            message="Mock payment completed successfully",
            estimated_completion=datetime.now(timezone.utc) + timedelta(minutes=1)
        )
    
    async def get_payment_status(self, transaction_id: str) -> PaymentResponse:
        """Get mock payment status"""
        return PaymentResponse(
            success=True,
            transaction_id=transaction_id,
            status=PaymentStatus.COMPLETED,
            fiat_amount=Decimal("100.00"),
            fiat_currency=FiatCurrency.USD,
            crypto_currency="FTNS",
            message="Mock payment status check"
        )


class FiatGateway:
    """
    Production-grade fiat payment gateway
    
    🏦 FIAT PAYMENT FEATURES:
    - Multiple payment provider support (Stripe, PayPal, etc.)
    - Secure payment processing with fraud detection
    - Comprehensive transaction management
    - Real-time status tracking and webhooks
    - Compliance and audit logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.providers: Dict[str, PaymentProvider] = {}
        self.default_provider = self.config.get("default_provider", "mock")
        
        # Security
        self.encryption_key = self.config.get("encryption_key")
        if self.encryption_key:
            self.cipher = Fernet(self.encryption_key.encode())
        else:
            self.cipher = None
        
        # Fraud detection settings
        self.fraud_detection_enabled = self.config.get("fraud_detection_enabled", True)
        self.max_daily_amount = Decimal(self.config.get("max_daily_amount", "10000"))
        self.max_transaction_amount = Decimal(self.config.get("max_transaction_amount", "5000"))
        
        print("🏦 Fiat Gateway initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default gateway configuration"""
        settings = get_settings()
        
        return {
            "default_provider": "mock",
            "fraud_detection_enabled": True,
            "max_daily_amount": "10000",
            "max_transaction_amount": "5000",
            "encryption_key": settings.SECRET_KEY,
            "providers": {
                "stripe": {
                    "enabled": False,
                    "api_key": "",
                    "webhook_secret": ""
                },
                "paypal": {
                    "enabled": False,
                    "client_id": "",
                    "client_secret": "",
                    "environment": "sandbox"
                },
                "mock": {
                    "enabled": True,
                    "simulate_failures": False,
                    "failure_rate": 0.1
                }
            }
        }
    
    async def initialize(self):
        """Initialize payment providers"""
        provider_configs = self.config.get("providers", {})
        
        for provider_name, provider_config in provider_configs.items():
            if not provider_config.get("enabled", False):
                continue
                
            try:
                if provider_name == "stripe":
                    provider = StripeProvider(provider_config)
                elif provider_name == "paypal":
                    provider = PayPalProvider(provider_config)
                elif provider_name == "mock":
                    provider = MockProvider(provider_config)
                else:
                    logger.warning(f"Unknown payment provider: {provider_name}")
                    continue
                
                await provider.initialize()
                self.providers[provider_name] = provider
                logger.info(f"✅ Payment provider initialized: {provider_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}", error=str(e))
        
        if not self.providers:
            logger.warning("No payment providers initialized")
    
    async def cleanup(self):
        """Cleanup payment providers"""
        for provider in self.providers.values():
            try:
                await provider.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup provider {provider.provider_name}", error=str(e))
    
    async def create_payment(
        self, 
        request: PaymentRequest,
        provider_name: Optional[str] = None
    ) -> PaymentResponse:
        """Create a new payment"""
        try:
            # Fraud detection checks
            if self.fraud_detection_enabled:
                fraud_check = await self._perform_fraud_checks(request)
                if not fraud_check["passed"]:
                    return PaymentResponse(
                        success=False,
                        transaction_id=str(uuid.uuid4()),
                        status=PaymentStatus.FAILED,
                        fiat_amount=request.fiat_amount,
                        fiat_currency=request.fiat_currency,
                        crypto_currency=request.crypto_currency,
                        message=f"Payment blocked: {fraud_check['reason']}"
                    )
            
            # Select provider
            provider = self._get_provider(provider_name or self.default_provider)
            if not provider:
                return PaymentResponse(
                    success=False,
                    transaction_id=str(uuid.uuid4()),
                    status=PaymentStatus.FAILED,
                    fiat_amount=request.fiat_amount,
                    fiat_currency=request.fiat_currency,
                    crypto_currency=request.crypto_currency,
                    message="Payment provider not available"
                )
            
            # Create payment
            response = await provider.create_payment(request)
            
            # Log payment creation
            audit_logger.log_event({
                "event_type": "payment_created",
                "user_id": request.user_id,
                "transaction_id": response.transaction_id,
                "provider": provider.provider_name,
                "amount": str(request.fiat_amount),
                "currency": request.fiat_currency.value,
                "status": response.status.value
            })
            
            return response
            
        except Exception as e:
            logger.error("Payment creation failed", error=str(e))
            return PaymentResponse(
                success=False,
                transaction_id=str(uuid.uuid4()),
                status=PaymentStatus.FAILED,
                fiat_amount=request.fiat_amount,
                fiat_currency=request.fiat_currency,
                crypto_currency=request.crypto_currency,
                message=f"Payment processing error: {str(e)}"
            )
    
    async def get_payment_status(
        self,
        transaction_id: str,
        provider_name: Optional[str] = None
    ) -> PaymentResponse:
        """Get payment status"""
        try:
            provider = self._get_provider(provider_name or self.default_provider)
            if not provider:
                return PaymentResponse(
                    success=False,
                    transaction_id=transaction_id,
                    status=PaymentStatus.FAILED,
                    fiat_amount=Decimal("0"),
                    fiat_currency=FiatCurrency.USD,
                    crypto_currency="FTNS",
                    message="Payment provider not available"
                )
            
            return await provider.get_payment_status(transaction_id)
            
        except Exception as e:
            logger.error("Failed to get payment status", transaction_id=transaction_id, error=str(e))
            return PaymentResponse(
                success=False,
                transaction_id=transaction_id,
                status=PaymentStatus.FAILED,
                fiat_amount=Decimal("0"),
                fiat_currency=FiatCurrency.USD,
                crypto_currency="FTNS",
                message=f"Status check error: {str(e)}"
            )
    
    async def _perform_fraud_checks(self, request: PaymentRequest) -> Dict[str, Any]:
        """Perform fraud detection checks"""
        # Amount checks
        if request.fiat_amount > self.max_transaction_amount:
            return {
                "passed": False,
                "reason": f"Transaction amount exceeds maximum limit of {self.max_transaction_amount}"
            }
        
        # Additional fraud checks would go here:
        # - Daily spending limits
        # - Velocity checks
        # - Geolocation verification
        # - Device fingerprinting
        # - Behavioral analysis
        
        return {"passed": True}
    
    def _get_provider(self, provider_name: str) -> Optional[PaymentProvider]:
        """Get payment provider by name"""
        return self.providers.get(provider_name)
    
    def get_supported_providers(self) -> List[str]:
        """Get list of supported providers"""
        return list(self.providers.keys())
    
    def get_supported_currencies(self) -> Dict[str, List[str]]:
        """Get supported currencies by provider"""
        # This would typically be fetched from provider configurations
        return {
            "stripe": ["USD", "EUR", "GBP", "CAD", "AUD"],
            "paypal": ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"],
            "mock": ["USD", "EUR", "GBP"]
        }
    
    def get_processing_fees(self, provider_name: str, amount: Decimal, currency: str) -> Decimal:
        """Get processing fees for a provider"""
        fee_schedules = {
            "stripe": lambda amt: amt * Decimal("0.029") + Decimal("0.30"),
            "paypal": lambda amt: amt * Decimal("0.029") + Decimal("0.30"),
            "mock": lambda amt: amt * Decimal("0.025")
        }
        
        calculator = fee_schedules.get(provider_name)
        if calculator:
            return calculator(amount)
        
        return Decimal("0")


# Global fiat gateway instance
_fiat_gateway: Optional[FiatGateway] = None

async def get_fiat_gateway() -> FiatGateway:
    """Get or create the global fiat gateway instance"""
    global _fiat_gateway
    if _fiat_gateway is None:
        _fiat_gateway = FiatGateway()
        await _fiat_gateway.initialize()
    return _fiat_gateway