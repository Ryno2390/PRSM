"""
CHRONOS Cashout API

Provides high-level API for FTNS cashout operations with multiple currency support.
Implements the hub-and-spoke architecture for optimal conversion paths.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime, timedelta
from fastapi import HTTPException
from pydantic import BaseModel, Field

from .models import AssetType, SwapType, TransactionStatus, CHRONOSStakingRequest
from .clearing_engine import ChronosEngine
from .hub_spoke_router import HubSpokeRouter, RoutingPath

logger = logging.getLogger(__name__)


class CashoutRequest(BaseModel):
    """Request for FTNS cashout operation."""
    
    user_id: str
    ftns_amount: Decimal
    target_currency: AssetType = AssetType.USD
    max_slippage: Decimal = Field(default=Decimal("0.015"))  # 1.5% for multi-hop
    preferred_speed: str = Field(default="balanced")  # fast, balanced, cheapest
    
    # Banking details for USD cashouts
    bank_account_info: Optional[Dict] = None
    compliance_verified: bool = False


class CashoutQuote(BaseModel):
    """Quote response for cashout operation."""
    
    quote_id: str
    from_asset: str = "FTNS"
    to_asset: str
    input_amount: str
    estimated_output: str
    total_fees: str
    exchange_rate: str
    
    # Routing details
    recommended_route: Dict
    alternative_routes: List[Dict]
    
    # Timing and reliability
    estimated_completion_minutes: int
    confidence_score: str
    quote_expires_at: datetime
    
    # Compliance info
    requires_kyc: bool = False
    estimated_tax_liability: Optional[str] = None


class CashoutExecution(BaseModel):
    """Result of cashout execution."""
    
    execution_id: str
    quote_id: str
    status: TransactionStatus
    
    # Execution details
    route_taken: List[str]
    actual_output: Optional[str] = None
    actual_fees: Optional[str] = None
    
    # Transaction tracking
    blockchain_transactions: List[Dict] = Field(default_factory=list)
    banking_reference: Optional[str] = None
    
    # Timing
    initiated_at: datetime
    estimated_completion: datetime
    completed_at: Optional[datetime] = None


class CHRONOSCashoutAPI:
    """High-level API for FTNS cashout operations."""
    
    def __init__(self, chronos_engine: ChronosEngine):
        self.chronos_engine = chronos_engine
        self.hub_spoke_router = chronos_engine.hub_spoke_router
        
        # Quote cache (5-minute expiry)
        self.quote_cache = {}
        self.quote_ttl = timedelta(minutes=5)
        
        # Active executions
        self.active_executions: Dict[str, CashoutExecution] = {}
        
        # Supported cashout currencies
        self.supported_currencies = [
            AssetType.USD,
            AssetType.USDC,
            AssetType.BTC,
            AssetType.ETH
        ]
        
        # KYC requirements by amount (USD equivalent)
        self.kyc_thresholds = {
            "basic": Decimal("1000"),     # $1K - basic verification
            "enhanced": Decimal("10000"), # $10K - enhanced verification
            "institutional": Decimal("100000")  # $100K - institutional verification
        }
    
    async def get_cashout_quote(self, request: CashoutRequest) -> CashoutQuote:
        """Get comprehensive quote for FTNS cashout."""
        
        if request.target_currency not in self.supported_currencies:
            raise HTTPException(
                status_code=400, 
                detail=f"Currency {request.target_currency} not supported for cashout"
            )
        
        # Check for cached quote
        cache_key = f"{request.user_id}_{request.ftns_amount}_{request.target_currency}"
        if cache_key in self.quote_cache:
            cached_quote, timestamp = self.quote_cache[cache_key]
            if datetime.utcnow() - timestamp < self.quote_ttl:
                return cached_quote
        
        try:
            # Get routing options from hub-spoke router
            if request.target_currency == AssetType.USD:
                quote_data = await self.hub_spoke_router.get_ftns_to_usd_quote(request.ftns_amount)
            else:
                routes = await self.hub_spoke_router.find_optimal_route(
                    AssetType.FTNS, 
                    request.target_currency, 
                    request.ftns_amount
                )\n                \n                if not routes:\n                    raise HTTPException(\n                        status_code=400,\n                        detail=f\"No route available for FTNS -> {request.target_currency}\"\n                    )\n                \n                # Build quote data from best route\n                best_route = routes[0]\n                quote_data = {\n                    \"recommended_route\": {\n                        \"path\": [asset.value for asset in best_route.path],\n                        \"estimated_output\": str(request.ftns_amount - best_route.total_fees),\n                        \"total_fees\": str(best_route.total_fees),\n                        \"estimated_time_minutes\": best_route.estimated_time_minutes,\n                        \"confidence_score\": str(best_route.confidence_score)\n                    },\n                    \"alternative_routes\": []\n                }\n            \n            if \"error\" in quote_data:\n                raise HTTPException(status_code=400, detail=quote_data[\"error\"])\n            \n            # Extract routing information\n            recommended = quote_data[\"recommended_route\"]\n            estimated_output = Decimal(recommended[\"estimated_output\"])\n            total_fees = Decimal(recommended[\"total_fees\"])\n            \n            # Calculate exchange rate\n            exchange_rate = estimated_output / request.ftns_amount\n            \n            # Determine KYC requirements\n            requires_kyc = await self._check_kyc_requirements(\n                request.user_id, \n                estimated_output, \n                request.target_currency\n            )\n            \n            # Apply speed preferences\n            timing_adjustment = self._apply_speed_preference(\n                recommended[\"estimated_time_minutes\"], \n                request.preferred_speed\n            )\n            \n            # Generate quote\n            quote = CashoutQuote(\n                quote_id=f\"quote_{request.user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}\",\n                to_asset=request.target_currency.value,\n                input_amount=str(request.ftns_amount),\n                estimated_output=str(estimated_output),\n                total_fees=str(total_fees),\n                exchange_rate=str(exchange_rate),\n                recommended_route=recommended,\n                alternative_routes=quote_data.get(\"alternative_routes\", []),\n                estimated_completion_minutes=timing_adjustment[\"time_minutes\"],\n                confidence_score=recommended[\"confidence_score\"],\n                quote_expires_at=datetime.utcnow() + self.quote_ttl,\n                requires_kyc=requires_kyc\n            )\n            \n            # Cache the quote\n            self.quote_cache[cache_key] = (quote, datetime.utcnow())\n            \n            return quote\n            \n        except Exception as e:\n            logger.error(f\"Failed to generate cashout quote: {e}\")\n            raise HTTPException(status_code=500, detail=f\"Quote generation failed: {str(e)}\")\n    \n    async def execute_cashout(self, quote_id: str, request: CashoutRequest) -> CashoutExecution:\n        \"\"\"Execute cashout based on previously generated quote.\"\"\"\n        \n        # Find the quote\n        quote = await self._find_quote_by_id(quote_id)\n        if not quote:\n            raise HTTPException(status_code=404, detail=\"Quote not found or expired\")\n        \n        # Validate quote hasn't expired\n        if datetime.utcnow() > quote.quote_expires_at:\n            raise HTTPException(status_code=400, detail=\"Quote has expired\")\n        \n        # Check KYC compliance for large amounts\n        if quote.requires_kyc and not request.compliance_verified:\n            raise HTTPException(\n                status_code=403, \n                detail=\"KYC verification required for this amount\"\n            )\n        \n        # Create execution record\n        execution = CashoutExecution(\n            execution_id=f\"exec_{request.user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}\",\n            quote_id=quote_id,\n            status=TransactionStatus.PENDING,\n            route_taken=quote.recommended_route[\"path\"],\n            initiated_at=datetime.utcnow(),\n            estimated_completion=datetime.utcnow() + timedelta(\n                minutes=quote.estimated_completion_minutes\n            )\n        )\n        \n        self.active_executions[execution.execution_id] = execution\n        \n        # Start async execution\n        asyncio.create_task(self._execute_cashout_async(execution, quote, request))\n        \n        return execution\n    \n    async def _execute_cashout_async(\n        self, \n        execution: CashoutExecution, \n        quote: CashoutQuote, \n        request: CashoutRequest\n    ):\n        \"\"\"Async execution of cashout operation.\"\"\"\n        \n        try:\n            execution.status = TransactionStatus.EXECUTING\n            \n            # Build routing path from quote\n            route_assets = [AssetType(asset) for asset in quote.recommended_route[\"path\"]]\n            \n            # Create routing path object\n            routing_path = RoutingPath(\n                path=route_assets,\n                total_fees=Decimal(quote.total_fees),\n                estimated_slippage=Decimal(\"0.01\"),  # Default slippage\n                estimated_time_minutes=quote.estimated_completion_minutes,\n                confidence_score=Decimal(quote.confidence_score)\n            )\n            \n            # Execute the route\n            result = await self.hub_spoke_router.execute_hub_spoke_route(\n                routing_path,\n                Decimal(quote.input_amount),\n                request.user_id\n            )\n            \n            if result[\"status\"] == \"completed\":\n                execution.status = TransactionStatus.COMPLETED\n                execution.actual_output = result[\"final_amount\"]\n                execution.actual_fees = result[\"total_fees\"]\n                execution.completed_at = datetime.utcnow()\n                \n                # Handle USD-specific completion (banking)\n                if request.target_currency == AssetType.USD:\n                    banking_ref = await self._process_usd_banking(\n                        execution, request.bank_account_info\n                    )\n                    execution.banking_reference = banking_ref\n                \n                logger.info(f\"Cashout completed: {execution.execution_id}\")\n                \n            else:\n                execution.status = TransactionStatus.FAILED\n                logger.error(f\"Cashout failed: {execution.execution_id} - {result.get('error')}\")\n                \n        except Exception as e:\n            execution.status = TransactionStatus.FAILED\n            logger.error(f\"Cashout execution error: {e}\")\n    \n    async def get_execution_status(self, execution_id: str) -> Optional[CashoutExecution]:\n        \"\"\"Get current status of cashout execution.\"\"\"\n        return self.active_executions.get(execution_id)\n    \n    async def get_supported_currencies(self) -> List[Dict]:\n        \"\"\"Get list of supported cashout currencies with details.\"\"\"\n        currencies = []\n        \n        for currency in self.supported_currencies:\n            # Get sample quote for $1000 equivalent\n            sample_ftns = Decimal(\"1000\")  # Assume 1 FTNS = $1 for calculation\n            \n            try:\n                if currency == AssetType.USD:\n                    quote_data = await self.hub_spoke_router.get_ftns_to_usd_quote(sample_ftns)\n                    if \"error\" not in quote_data:\n                        recommended = quote_data[\"recommended_route\"]\n                        currencies.append({\n                            \"currency\": currency.value,\n                            \"name\": self._get_currency_name(currency),\n                            \"typical_fee_percent\": str(\n                                (Decimal(recommended[\"total_fees\"]) / sample_ftns) * 100\n                            ),\n                            \"typical_time_minutes\": recommended[\"estimated_time_minutes\"],\n                            \"requires_banking\": currency == AssetType.USD,\n                            \"kyc_threshold_usd\": str(self.kyc_thresholds[\"basic\"])\n                        })\n                        \n            except Exception as e:\n                logger.warning(f\"Could not get info for {currency}: {e}\")\n                currencies.append({\n                    \"currency\": currency.value,\n                    \"name\": self._get_currency_name(currency),\n                    \"available\": False,\n                    \"error\": \"Temporarily unavailable\"\n                })\n        \n        return currencies\n    \n    async def _check_kyc_requirements(\n        self, \n        user_id: str, \n        amount_usd_equivalent: Decimal, \n        currency: AssetType\n    ) -> bool:\n        \"\"\"Check if KYC verification is required.\"\"\"\n        \n        # USD conversions above $1K require KYC\n        if currency == AssetType.USD and amount_usd_equivalent >= self.kyc_thresholds[\"basic\"]:\n            return True\n        \n        # Large crypto conversions also require KYC\n        if amount_usd_equivalent >= self.kyc_thresholds[\"enhanced\"]:\n            return True\n        \n        return False\n    \n    def _apply_speed_preference(self, base_time_minutes: int, speed: str) -> Dict:\n        \"\"\"Apply speed preference to timing estimates.\"\"\"\n        \n        multipliers = {\n            \"fast\": {\"time\": 0.7, \"fee_multiplier\": 1.5},\n            \"balanced\": {\"time\": 1.0, \"fee_multiplier\": 1.0},\n            \"cheapest\": {\"time\": 1.5, \"fee_multiplier\": 0.8}\n        }\n        \n        preference = multipliers.get(speed, multipliers[\"balanced\"])\n        \n        return {\n            \"time_minutes\": int(base_time_minutes * preference[\"time\"]),\n            \"fee_adjustment\": preference[\"fee_multiplier\"]\n        }\n    \n    async def _find_quote_by_id(self, quote_id: str) -> Optional[CashoutQuote]:\n        \"\"\"Find cached quote by ID.\"\"\"\n        for (cached_quote, timestamp) in self.quote_cache.values():\n            if cached_quote.quote_id == quote_id:\n                if datetime.utcnow() - timestamp < self.quote_ttl:\n                    return cached_quote\n        return None\n    \n    async def _process_usd_banking(\n        self, \n        execution: CashoutExecution, \n        bank_info: Optional[Dict]\n    ) -> str:\n        \"\"\"Process USD banking transfer (mock implementation).\"\"\"\n        \n        if not bank_info:\n            logger.warning(f\"No banking info for USD cashout: {execution.execution_id}\")\n            return \"BANKING_INFO_REQUIRED\"\n        \n        # In production, this would integrate with banking APIs\n        # For now, simulate banking reference generation\n        banking_ref = f\"BANK_TXN_{execution.execution_id}_{datetime.utcnow().strftime('%Y%m%d')}\"\n        \n        logger.info(f\"USD banking initiated: {banking_ref}\")\n        return banking_ref\n    \n    def _get_currency_name(self, currency: AssetType) -> str:\n        \"\"\"Get human-readable currency name.\"\"\"\n        names = {\n            AssetType.USD: \"US Dollar\",\n            AssetType.USDC: \"USD Coin (USDC)\",\n            AssetType.BTC: \"Bitcoin\",\n            AssetType.ETH: \"Ethereum\",\n            AssetType.FTNS: \"Fungible Tokens for Node Support\"\n        }\n        return names.get(currency, currency.value)\n    \n    async def get_cashout_analytics(self, user_id: str) -> Dict:\n        \"\"\"Get analytics for user's cashout history and options.\"\"\"\n        \n        # In production, this would query actual transaction history\n        return {\n            \"user_id\": user_id,\n            \"total_cashouts_30d\": 0,  # Would be calculated from DB\n            \"average_cashout_amount\": \"0\",\n            \"preferred_currency\": \"USD\",\n            \"kyc_status\": \"unverified\",  # Would be from user profile\n            \"available_for_cashout\": \"0\",  # User's FTNS balance\n            \"recommended_currencies\": [\n                currency.value for currency in self.supported_currencies\n            ]\n        }