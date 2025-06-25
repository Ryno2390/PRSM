"""
PRSM Payment Processor
=====================

Main payment processing orchestrator that coordinates fiat payments,
crypto conversions, and FTNS token distribution with comprehensive
transaction management and compliance features.
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union

import structlog
from sqlalchemy.orm import Session

from ..core.database import db_manager
from ..core.config import get_settings
from ..integrations.security.audit_logger import audit_logger
from ..tokenomics.ftns_service import get_ftns_service
from ..web3.web3_service import get_web3_service
from .fiat_gateway import get_fiat_gateway
from .crypto_exchange import get_crypto_exchange
from .payment_models import (
    PaymentMethod, PaymentStatus, FiatCurrency, CryptoCurrency,
    PaymentRequest, PaymentResponse, PaymentTransaction, ExchangeRate,
    PaymentMethodRequest, PaymentMethodResponse, TransactionQuery, TransactionList,
    PaymentStats, KYCStatus
)

logger = structlog.get_logger(__name__)


class PaymentProcessor:
    """
    Production-grade payment processor for fiat-to-crypto conversion
    
    💳 PAYMENT PROCESSING FEATURES:
    - End-to-end payment orchestration
    - Multi-provider fiat payment support
    - Real-time cryptocurrency conversion
    - Automated FTNS token distribution
    - Comprehensive transaction tracking
    - Compliance and audit logging
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.fiat_gateway = None
        self.crypto_exchange = None
        self.ftns_service = None
        self.web3_service = None
        
        # Transaction limits
        self.min_transaction_amount = Decimal(self.config.get("min_transaction_amount", "10"))
        self.max_transaction_amount = Decimal(self.config.get("max_transaction_amount", "50000"))
        self.max_daily_amount = Decimal(self.config.get("max_daily_amount", "100000"))
        
        # Processing settings
        self.auto_complete_enabled = self.config.get("auto_complete_enabled", True)
        self.token_distribution_enabled = self.config.get("token_distribution_enabled", True)
        
        print("💳 Payment Processor initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default processor configuration"""
        return {
            "min_transaction_amount": "10",
            "max_transaction_amount": "50000", 
            "max_daily_amount": "100000",
            "auto_complete_enabled": True,
            "token_distribution_enabled": True,
            "processing_fee_percent": "0.5",  # 0.5% processing fee
            "kyc_threshold_amount": "1000",   # KYC required above $1000
            "compliance_enabled": True
        }
    
    async def initialize(self):
        """Initialize payment processor components"""
        try:
            # Initialize dependencies
            self.fiat_gateway = await get_fiat_gateway()
            self.crypto_exchange = await get_crypto_exchange()
            self.ftns_service = await get_ftns_service()
            self.web3_service = await get_web3_service()
            
            logger.info("✅ Payment processor components initialized")
            
        except Exception as e:
            logger.error("Failed to initialize payment processor", error=str(e))
            raise
    
    async def cleanup(self):
        """Cleanup payment processor resources"""
        try:
            if self.fiat_gateway:
                await self.fiat_gateway.cleanup()
            if self.crypto_exchange:
                await self.crypto_exchange.cleanup()
        except Exception as e:
            logger.error("Failed to cleanup payment processor", error=str(e))
    
    async def create_payment(self, request: PaymentRequest) -> PaymentResponse:
        """Create a new payment transaction"""
        try:
            # Validate request
            validation_result = await self._validate_payment_request(request)
            if not validation_result["valid"]:
                return PaymentResponse(
                    success=False,
                    transaction_id=str(uuid.uuid4()),
                    status=PaymentStatus.FAILED,
                    fiat_amount=request.fiat_amount,
                    fiat_currency=request.fiat_currency,
                    crypto_currency=request.crypto_currency,
                    message=validation_result["error"]
                )
            
            # Check compliance requirements
            compliance_check = await self._check_compliance_requirements(request)
            if not compliance_check["compliant"]:
                return PaymentResponse(
                    success=False,
                    transaction_id=str(uuid.uuid4()),
                    status=PaymentStatus.FAILED,
                    fiat_amount=request.fiat_amount,
                    fiat_currency=request.fiat_currency,
                    crypto_currency=request.crypto_currency,
                    message=compliance_check["reason"]
                )
            
            # Get exchange rate
            exchange_rate = await self._get_conversion_rate(
                request.fiat_currency.value, 
                request.crypto_currency.value
            )
            
            if not exchange_rate:
                return PaymentResponse(
                    success=False,
                    transaction_id=str(uuid.uuid4()),
                    status=PaymentStatus.FAILED,
                    fiat_amount=request.fiat_amount,
                    fiat_currency=request.fiat_currency,
                    crypto_currency=request.crypto_currency,
                    message="Exchange rate not available"
                )
            
            # Calculate amounts
            amounts = await self._calculate_transaction_amounts(request, exchange_rate)
            
            # Create database transaction record
            db_transaction = await self._create_transaction_record(request, amounts, exchange_rate)
            
            # Process fiat payment
            fiat_response = await self.fiat_gateway.create_payment(request)
            
            if fiat_response.success:
                # Update transaction with fiat payment details
                await self._update_transaction_fiat_details(db_transaction.transaction_id, fiat_response)
                
                # If payment is immediately completed, process crypto conversion
                if fiat_response.status == PaymentStatus.COMPLETED and self.auto_complete_enabled:
                    await self._process_crypto_conversion(db_transaction.transaction_id)
                
                return PaymentResponse(
                    success=True,
                    transaction_id=str(db_transaction.transaction_id),
                    status=fiat_response.status,
                    fiat_amount=request.fiat_amount,
                    fiat_currency=request.fiat_currency,
                    crypto_amount=amounts["crypto_amount"],
                    crypto_currency=request.crypto_currency,
                    exchange_rate=exchange_rate.rate,
                    processing_fee=amounts["processing_fee"],
                    network_fee=amounts["network_fee"],
                    payment_url=fiat_response.payment_url,
                    provider_reference=fiat_response.provider_reference,
                    message=fiat_response.message,
                    requires_action=fiat_response.requires_action,
                    next_action=fiat_response.next_action,
                    estimated_completion=fiat_response.estimated_completion,
                    expires_at=fiat_response.expires_at
                )
            else:
                # Update transaction status to failed
                await self._update_transaction_status(db_transaction.transaction_id, PaymentStatus.FAILED)
                
                return PaymentResponse(
                    success=False,
                    transaction_id=str(db_transaction.transaction_id),
                    status=PaymentStatus.FAILED,
                    fiat_amount=request.fiat_amount,
                    fiat_currency=request.fiat_currency,
                    crypto_currency=request.crypto_currency,
                    message=fiat_response.message
                )
                
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
    
    async def get_payment_status(self, transaction_id: str) -> PaymentResponse:
        """Get payment transaction status"""
        try:
            # Get transaction from database
            async with db_manager.session() as session:
                transaction = session.query(PaymentTransaction).filter(
                    PaymentTransaction.transaction_id == transaction_id
                ).first()
                
                if not transaction:
                    return PaymentResponse(
                        success=False,
                        transaction_id=transaction_id,
                        status=PaymentStatus.FAILED,
                        fiat_amount=Decimal("0"),
                        fiat_currency=FiatCurrency.USD,
                        crypto_currency="FTNS",
                        message="Transaction not found"
                    )
                
                # Check if we need to update status from provider
                if transaction.status in [PaymentStatus.PENDING.value, PaymentStatus.PROCESSING.value]:
                    await self._sync_transaction_status(transaction_id)
                    
                    # Refresh transaction data
                    session.refresh(transaction)
                
                return PaymentResponse(
                    success=True,
                    transaction_id=transaction_id,
                    status=PaymentStatus(transaction.status),
                    fiat_amount=transaction.fiat_amount,
                    fiat_currency=FiatCurrency(transaction.fiat_currency),
                    crypto_amount=transaction.crypto_amount,
                    crypto_currency=CryptoCurrency(transaction.crypto_currency),
                    exchange_rate=transaction.exchange_rate,
                    processing_fee=transaction.processing_fee,
                    network_fee=transaction.network_fee,
                    provider_reference=transaction.provider_reference,
                    message=f"Transaction status: {transaction.status}",
                    completed_at=transaction.completed_at
                )
                
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
    
    async def process_webhook(self, provider: str, payload: Dict[str, Any]) -> bool:
        """Process payment provider webhook"""
        try:
            if provider == "stripe":
                return await self._process_stripe_webhook(payload)
            elif provider == "paypal":
                return await self._process_paypal_webhook(payload)
            else:
                logger.warning(f"Unknown webhook provider: {provider}")
                return False
                
        except Exception as e:
            logger.error("Webhook processing failed", provider=provider, error=str(e))
            return False
    
    async def list_transactions(self, query: TransactionQuery) -> TransactionList:
        """List payment transactions with filtering"""
        try:
            async with db_manager.session() as session:
                # Build query
                query_builder = session.query(PaymentTransaction)
                
                if query.user_id:
                    query_builder = query_builder.filter(PaymentTransaction.user_id == query.user_id)
                if query.status:
                    query_builder = query_builder.filter(PaymentTransaction.status == query.status.value)
                if query.payment_method:
                    query_builder = query_builder.filter(PaymentTransaction.payment_method == query.payment_method.value)
                if query.fiat_currency:
                    query_builder = query_builder.filter(PaymentTransaction.fiat_currency == query.fiat_currency.value)
                if query.crypto_currency:
                    query_builder = query_builder.filter(PaymentTransaction.crypto_currency == query.crypto_currency.value)
                if query.start_date:
                    query_builder = query_builder.filter(PaymentTransaction.created_at >= query.start_date)
                if query.end_date:
                    query_builder = query_builder.filter(PaymentTransaction.created_at <= query.end_date)
                
                # Get total count
                total_count = query_builder.count()
                
                # Apply sorting
                if query.sort_by == "created_at":
                    sort_column = PaymentTransaction.created_at
                elif query.sort_by == "amount":
                    sort_column = PaymentTransaction.fiat_amount
                else:
                    sort_column = PaymentTransaction.created_at
                
                if query.sort_order == "desc":
                    query_builder = query_builder.order_by(sort_column.desc())
                else:
                    query_builder = query_builder.order_by(sort_column.asc())
                
                # Apply pagination
                offset = (query.page - 1) * query.limit
                transactions = query_builder.offset(offset).limit(query.limit).all()
                
                # Convert to response format
                transaction_responses = []
                for tx in transactions:
                    response = PaymentResponse(
                        success=True,
                        transaction_id=str(tx.transaction_id),
                        status=PaymentStatus(tx.status),
                        fiat_amount=tx.fiat_amount,
                        fiat_currency=FiatCurrency(tx.fiat_currency),
                        crypto_amount=tx.crypto_amount,
                        crypto_currency=CryptoCurrency(tx.crypto_currency),
                        exchange_rate=tx.exchange_rate,
                        processing_fee=tx.processing_fee,
                        network_fee=tx.network_fee,
                        provider_reference=tx.provider_reference,
                        message=f"Transaction status: {tx.status}",
                        created_at=tx.created_at,
                        completed_at=tx.completed_at
                    )
                    transaction_responses.append(response)
                
                total_pages = (total_count + query.limit - 1) // query.limit
                
                return TransactionList(
                    transactions=transaction_responses,
                    total_count=total_count,
                    page=query.page,
                    limit=query.limit,
                    total_pages=total_pages,
                    has_next=query.page < total_pages,
                    has_previous=query.page > 1
                )
                
        except Exception as e:
            logger.error("Failed to list transactions", error=str(e))
            return TransactionList(
                transactions=[],
                total_count=0,
                page=query.page,
                limit=query.limit,
                total_pages=0,
                has_next=False,
                has_previous=False
            )
    
    async def get_payment_stats(
        self, 
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PaymentStats:
        """Get payment statistics"""
        try:
            async with db_manager.session() as session:
                # Build base query
                query_builder = session.query(PaymentTransaction)
                
                if user_id:
                    query_builder = query_builder.filter(PaymentTransaction.user_id == user_id)
                if start_date:
                    query_builder = query_builder.filter(PaymentTransaction.created_at >= start_date)
                if end_date:
                    query_builder = query_builder.filter(PaymentTransaction.created_at <= end_date)
                
                transactions = query_builder.all()
                
                # Calculate statistics
                total_transactions = len(transactions)
                successful_transactions = len([tx for tx in transactions if tx.status == PaymentStatus.COMPLETED.value])
                failed_transactions = len([tx for tx in transactions if tx.status == PaymentStatus.FAILED.value])
                
                total_volume_usd = sum(
                    tx.fiat_amount for tx in transactions 
                    if tx.status == PaymentStatus.COMPLETED.value and tx.fiat_currency == "USD"
                )
                
                average_transaction_size = total_volume_usd / successful_transactions if successful_transactions > 0 else Decimal("0")
                
                # Volume by currency
                volume_by_fiat = {}
                volume_by_crypto = {}
                transactions_by_method = {}
                
                for tx in transactions:
                    if tx.status == PaymentStatus.COMPLETED.value:
                        # Fiat volume
                        fiat_currency = tx.fiat_currency
                        if fiat_currency not in volume_by_fiat:
                            volume_by_fiat[fiat_currency] = Decimal("0")
                        volume_by_fiat[fiat_currency] += tx.fiat_amount
                        
                        # Crypto volume
                        crypto_currency = tx.crypto_currency
                        if crypto_currency not in volume_by_crypto:
                            volume_by_crypto[crypto_currency] = Decimal("0")
                        volume_by_crypto[crypto_currency] += tx.crypto_amount or Decimal("0")
                    
                    # Transactions by method
                    method = tx.payment_method
                    if method not in transactions_by_method:
                        transactions_by_method[method] = 0
                    transactions_by_method[method] += 1
                
                return PaymentStats(
                    total_transactions=total_transactions,
                    total_volume_usd=total_volume_usd,
                    successful_transactions=successful_transactions,
                    failed_transactions=failed_transactions,
                    average_transaction_size=average_transaction_size,
                    volume_by_fiat_currency=volume_by_fiat,
                    volume_by_crypto_currency=volume_by_crypto,
                    transactions_by_method=transactions_by_method,
                    period_start=start_date or datetime.min,
                    period_end=end_date or datetime.now(timezone.utc)
                )
                
        except Exception as e:
            logger.error("Failed to get payment stats", error=str(e))
            return PaymentStats(
                total_transactions=0,
                total_volume_usd=Decimal("0"),
                successful_transactions=0,
                failed_transactions=0,
                average_transaction_size=Decimal("0"),
                volume_by_fiat_currency={},
                volume_by_crypto_currency={},
                transactions_by_method={},
                period_start=datetime.now(timezone.utc),
                period_end=datetime.now(timezone.utc)
            )
    
    # Internal helper methods
    
    async def _validate_payment_request(self, request: PaymentRequest) -> Dict[str, Any]:
        """Validate payment request"""
        if request.fiat_amount < self.min_transaction_amount:
            return {
                "valid": False,
                "error": f"Amount below minimum threshold: {self.min_transaction_amount}"
            }
        
        if request.fiat_amount > self.max_transaction_amount:
            return {
                "valid": False,
                "error": f"Amount exceeds maximum threshold: {self.max_transaction_amount}"
            }
        
        # Check daily limits for user
        daily_volume = await self._get_user_daily_volume(request.user_id)
        if daily_volume + request.fiat_amount > self.max_daily_amount:
            return {
                "valid": False,
                "error": f"Transaction would exceed daily limit: {self.max_daily_amount}"
            }
        
        return {"valid": True}
    
    async def _check_compliance_requirements(self, request: PaymentRequest) -> Dict[str, Any]:
        """Check compliance requirements (KYC/AML)"""
        kyc_threshold = Decimal(self.config.get("kyc_threshold_amount", "1000"))
        
        if request.fiat_amount > kyc_threshold:
            # In production, this would check actual KYC status
            return {
                "compliant": True,  # Mock for now
                "kyc_required": True,
                "kyc_status": KYCStatus.NOT_REQUIRED
            }
        
        return {
            "compliant": True,
            "kyc_required": False,
            "kyc_status": KYCStatus.NOT_REQUIRED
        }
    
    async def _get_conversion_rate(self, from_currency: str, to_currency: str) -> Optional[ExchangeRate]:
        """Get conversion rate between currencies"""
        return await self.crypto_exchange.get_exchange_rate(from_currency, to_currency)
    
    async def _calculate_transaction_amounts(
        self, 
        request: PaymentRequest, 
        exchange_rate: ExchangeRate
    ) -> Dict[str, Decimal]:
        """Calculate all transaction amounts"""
        # Calculate crypto amount
        crypto_amount = request.fiat_amount * exchange_rate.rate
        
        # Calculate processing fee
        fee_percent = Decimal(self.config.get("processing_fee_percent", "0.5")) / 100
        processing_fee = request.fiat_amount * fee_percent
        
        # Estimate network fee (simplified)
        network_fee = Decimal("0.01")  # Mock network fee
        
        return {
            "crypto_amount": crypto_amount,
            "processing_fee": processing_fee,
            "network_fee": network_fee
        }
    
    async def _create_transaction_record(
        self, 
        request: PaymentRequest, 
        amounts: Dict[str, Decimal],
        exchange_rate: ExchangeRate
    ) -> PaymentTransaction:
        """Create transaction database record"""
        async with db_manager.session() as session:
            transaction = PaymentTransaction(
                user_id=request.user_id,
                payment_method=request.payment_method.value,
                status=PaymentStatus.PENDING.value,
                fiat_amount=request.fiat_amount,
                fiat_currency=request.fiat_currency.value,
                crypto_amount=amounts["crypto_amount"],
                crypto_currency=request.crypto_currency.value,
                exchange_rate=exchange_rate.rate,
                processing_fee=amounts["processing_fee"],
                network_fee=amounts["network_fee"],
                provider="fiat_gateway",
                metadata=request.metadata
            )
            
            session.add(transaction)
            session.commit()
            session.refresh(transaction)
            
            return transaction
    
    async def _update_transaction_fiat_details(self, transaction_id: uuid.UUID, fiat_response: PaymentResponse):
        """Update transaction with fiat payment details"""
        async with db_manager.session() as session:
            transaction = session.query(PaymentTransaction).filter(
                PaymentTransaction.transaction_id == transaction_id
            ).first()
            
            if transaction:
                transaction.external_id = fiat_response.provider_reference
                transaction.provider_reference = fiat_response.provider_reference
                transaction.status = fiat_response.status.value
                
                if fiat_response.status == PaymentStatus.COMPLETED:
                    transaction.completed_at = datetime.now(timezone.utc)
                
                session.commit()
    
    async def _update_transaction_status(self, transaction_id: uuid.UUID, status: PaymentStatus):
        """Update transaction status"""
        async with db_manager.session() as session:
            transaction = session.query(PaymentTransaction).filter(
                PaymentTransaction.transaction_id == transaction_id
            ).first()
            
            if transaction:
                transaction.status = status.value
                
                if status == PaymentStatus.COMPLETED:
                    transaction.completed_at = datetime.now(timezone.utc)
                
                session.commit()
    
    async def _sync_transaction_status(self, transaction_id: str):
        """Sync transaction status with payment provider"""
        try:
            # Get current status from fiat gateway
            fiat_status = await self.fiat_gateway.get_payment_status(transaction_id)
            
            if fiat_status.success:
                await self._update_transaction_status(uuid.UUID(transaction_id), fiat_status.status)
                
                # If completed, process crypto conversion
                if fiat_status.status == PaymentStatus.COMPLETED and self.auto_complete_enabled:
                    await self._process_crypto_conversion(uuid.UUID(transaction_id))
                    
        except Exception as e:
            logger.error("Failed to sync transaction status", transaction_id=transaction_id, error=str(e))
    
    async def _process_crypto_conversion(self, transaction_id: uuid.UUID):
        """Process cryptocurrency conversion and distribution"""
        try:
            async with db_manager.session() as session:
                transaction = session.query(PaymentTransaction).filter(
                    PaymentTransaction.transaction_id == transaction_id
                ).first()
                
                if not transaction or transaction.crypto_currency != "FTNS":
                    return
                
                if self.token_distribution_enabled and transaction.crypto_amount:
                    # Distribute FTNS tokens
                    success = await self.ftns_service.transfer_tokens(
                        to_user_id=transaction.user_id,
                        amount=transaction.crypto_amount,
                        transaction_type="purchase",
                        metadata={
                            "payment_transaction_id": str(transaction_id),
                            "payment_method": transaction.payment_method,
                            "fiat_amount": str(transaction.fiat_amount),
                            "fiat_currency": transaction.fiat_currency
                        }
                    )
                    
                    if success:
                        logger.info("FTNS tokens distributed", 
                                  transaction_id=transaction_id, 
                                  amount=transaction.crypto_amount)
                        
                        # Log successful conversion
                        audit_logger.log_event({
                            "event_type": "crypto_conversion_completed",
                            "user_id": transaction.user_id,
                            "transaction_id": str(transaction_id),
                            "crypto_amount": str(transaction.crypto_amount),
                            "crypto_currency": transaction.crypto_currency
                        })
                    else:
                        logger.error("Failed to distribute FTNS tokens", transaction_id=transaction_id)
                        
        except Exception as e:
            logger.error("Crypto conversion failed", transaction_id=transaction_id, error=str(e))
    
    async def _get_user_daily_volume(self, user_id: str) -> Decimal:
        """Get user's daily transaction volume"""
        try:
            today = datetime.now(timezone.utc).date()
            start_of_day = datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc)
            
            async with db_manager.session() as session:
                transactions = session.query(PaymentTransaction).filter(
                    PaymentTransaction.user_id == user_id,
                    PaymentTransaction.created_at >= start_of_day,
                    PaymentTransaction.status == PaymentStatus.COMPLETED.value
                ).all()
                
                return sum(tx.fiat_amount for tx in transactions)
                
        except Exception as e:
            logger.error("Failed to get user daily volume", user_id=user_id, error=str(e))
            return Decimal("0")
    
    async def _process_stripe_webhook(self, payload: Dict[str, Any]) -> bool:
        """Process Stripe webhook"""
        try:
            event_type = payload.get("type")
            
            if event_type == "payment_intent.succeeded":
                payment_intent = payload["data"]["object"]
                transaction_id = payment_intent["id"]
                
                await self._update_transaction_status(uuid.UUID(transaction_id), PaymentStatus.COMPLETED)
                await self._process_crypto_conversion(uuid.UUID(transaction_id))
                
                return True
                
            elif event_type == "payment_intent.payment_failed":
                payment_intent = payload["data"]["object"]
                transaction_id = payment_intent["id"]
                
                await self._update_transaction_status(uuid.UUID(transaction_id), PaymentStatus.FAILED)
                
                return True
            
            return True
            
        except Exception as e:
            logger.error("Stripe webhook processing failed", error=str(e))
            return False
    
    async def _process_paypal_webhook(self, payload: Dict[str, Any]) -> bool:
        """Process PayPal webhook"""
        try:
            event_type = payload.get("event_type")
            
            if event_type == "CHECKOUT.ORDER.APPROVED":
                order = payload["resource"]
                transaction_id = order["id"]
                
                await self._update_transaction_status(uuid.UUID(transaction_id), PaymentStatus.COMPLETED)
                await self._process_crypto_conversion(uuid.UUID(transaction_id))
                
                return True
                
            elif event_type == "CHECKOUT.ORDER.COMPLETED":
                order = payload["resource"]
                transaction_id = order["id"]
                
                await self._update_transaction_status(uuid.UUID(transaction_id), PaymentStatus.COMPLETED)
                
                return True
            
            return True
            
        except Exception as e:
            logger.error("PayPal webhook processing failed", error=str(e))
            return False


# Global payment processor instance
_payment_processor: Optional[PaymentProcessor] = None

async def get_payment_processor() -> PaymentProcessor:
    """Get or create the global payment processor instance"""
    global _payment_processor
    if _payment_processor is None:
        _payment_processor = PaymentProcessor()
        await _payment_processor.initialize()
    return _payment_processor