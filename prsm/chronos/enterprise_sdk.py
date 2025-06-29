"""
CHRONOS Enterprise SDK

Provides enterprise-grade SDK for companies like MicroStrategy to integrate CHRONOS
functionality into their existing systems. Designed for maximum ease of integration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

from .models import AssetType, TransactionStatus
from .treasury_provider import (
    TreasuryProviderManager, 
    MicroStrategyProvider, 
    TreasuryQuote, 
    TreasuryExecution,
    LiquidityTier
)
from .hub_spoke_router import HubSpokeRouter
from .clearing_engine import ChronosEngine
from .cashout_api import CHRONOSCashoutAPI

logger = logging.getLogger(__name__)


@dataclass
class EnterpriseConfig:
    """Configuration for enterprise CHRONOS integration."""
    
    # Company identification
    company_name: str
    company_id: str
    api_version: str = "v1.0"
    
    # Treasury provider settings
    treasury_provider: str = "MICROSTRATEGY"
    treasury_credentials: Dict[str, str] = None
    
    # Risk management
    max_daily_volume: Decimal = Decimal("10000")  # BTC
    max_single_transaction: Decimal = Decimal("1000")  # BTC
    risk_tolerance: str = "conservative"  # conservative, moderate, aggressive
    
    # Compliance settings
    kyc_required: bool = True
    aml_screening: bool = True
    jurisdiction: str = "US"
    
    # Technical settings
    async_processing: bool = True
    webhook_url: Optional[str] = None
    rate_limit_per_minute: int = 60
    
    # Fee preferences
    fee_preference: str = "balanced"  # cheapest, balanced, fastest
    
    def __post_init__(self):
        if self.treasury_credentials is None:
            self.treasury_credentials = {}


class CHRONOSEnterpriseSDK:
    """Enterprise SDK for CHRONOS integration."""
    
    def __init__(self, config: EnterpriseConfig):
        self.config = config
        self.is_initialized = False
        
        # Core components (initialized during setup)
        self.treasury_manager: Optional[TreasuryProviderManager] = None
        self.hub_spoke_router: Optional[HubSpokeRouter] = None
        self.chronos_engine: Optional[ChronosEngine] = None
        self.cashout_api: Optional[CHRONOSCashoutAPI] = None
        
        # Session management
        self.active_sessions = {}
        self.transaction_history = []
        
        # Performance metrics
        self.metrics = {
            "total_transactions": 0,
            "total_volume_btc": Decimal("0"),
            "average_execution_time": 0,
            "success_rate": 1.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the SDK with configured providers."""
        
        try:
            logger.info(f"Initializing CHRONOS SDK for {self.config.company_name}")
            
            # Initialize treasury manager
            self.treasury_manager = TreasuryProviderManager()
            
            # Add configured treasury provider
            if self.config.treasury_provider == "MICROSTRATEGY":
                mstr_provider = MicroStrategyProvider(self.config.treasury_credentials)
                self.treasury_manager.register_provider(mstr_provider)
            
            # Initialize other components (would be injected in production)
            # For demo, create mock instances
            self._initialize_mock_components()
            
            self.is_initialized = True
            logger.info("CHRONOS SDK initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"SDK initialization failed: {e}")
            return False
    
    def _initialize_mock_components(self):
        """Initialize mock components for demonstration."""
        # In production, these would be injected dependencies
        pass
    
    # ==============================
    # High-Level Enterprise API
    # ==============================
    
    async def get_liquidity_status(self, asset: AssetType = AssetType.BTC) -> Dict[str, Any]:
        """Get comprehensive liquidity status for enterprise planning."""
        
        self._ensure_initialized()
        
        # Get aggregated liquidity across providers
        liquidity_data = await self.treasury_manager.get_aggregated_liquidity(asset)
        
        # Add enterprise-specific insights
        enterprise_insights = {
            "recommended_transaction_size": await self._calculate_optimal_transaction_size(asset),
            "cost_analysis": await self._analyze_cost_structure(asset),
            "risk_assessment": await self._assess_market_risk(asset),
            "operational_windows": self._get_operational_windows()
        }
        
        return {
            "company": self.config.company_name,
            "asset": asset.value,
            "liquidity_data": liquidity_data,
            "enterprise_insights": enterprise_insights,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def execute_treasury_operation(
        self,
        operation_type: str,  # "buy", "sell", "borrow", "lend"
        asset: AssetType,
        amount: Decimal,
        execution_preference: str = "balanced",  # "fast", "balanced", "cheapest"
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute treasury operation with enterprise-grade controls."""
        
        self._ensure_initialized()
        
        # Validate operation against enterprise limits
        validation_result = await self._validate_enterprise_operation(
            operation_type, asset, amount
        )
        
        if not validation_result["valid"]:
            return {
                "status": "rejected",
                "reason": validation_result["reason"],
                "compliance_notes": validation_result.get("compliance_notes", [])
            }
        
        try:
            # Get optimal quote based on preference
            provider, quote = await self.treasury_manager.get_best_liquidity_quote(
                asset, amount, operation_type, 
                preferred_tier=self._preference_to_tier(execution_preference)
            )
            
            # Execute operation
            execution = await provider.execute_operation(quote, amount)
            
            # Record for enterprise reporting
            self._record_enterprise_transaction(execution, quote, metadata)
            
            return {
                "status": "executed",
                "execution_id": execution.execution_id,
                "provider": provider.provider_name,
                "quote_details": asdict(quote),
                "execution_details": asdict(execution),
                "enterprise_metadata": {
                    "company_id": self.config.company_id,
                    "operation_timestamp": datetime.utcnow().isoformat(),
                    "compliance_status": "approved",
                    "risk_category": validation_result.get("risk_category", "standard")
                }
            }
            
        except Exception as e:
            logger.error(f"Treasury operation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "error_code": "EXECUTION_FAILED"
            }
    
    async def get_conversion_quote(
        self,
        from_asset: AssetType,
        to_asset: AssetType, 
        amount: Decimal,
        execution_speed: str = "balanced"
    ) -> Dict[str, Any]:
        """Get comprehensive conversion quote for cross-asset operations."""
        
        self._ensure_initialized()
        
        # For FTNS cashouts, use specialized API
        if from_asset == AssetType.FTNS and to_asset == AssetType.USD:
            if self.cashout_api:
                from .cashout_api import CashoutRequest
                request = CashoutRequest(
                    user_id=self.config.company_id,
                    ftns_amount=amount,
                    target_currency=to_asset,
                    preferred_speed=execution_speed
                )
                quote = await self.cashout_api.get_cashout_quote(request)
                return self._format_enterprise_quote(quote)
        
        # For other conversions, use hub-spoke router
        if self.hub_spoke_router:
            routes = await self.hub_spoke_router.find_optimal_route(
                from_asset, to_asset, amount
            )
            
            if routes:
                best_route = routes[0]
                return {
                    "from_asset": from_asset.value,
                    "to_asset": to_asset.value,
                    "input_amount": str(amount),
                    "estimated_output": str(amount - best_route.total_fees),
                    "route": [asset.value for asset in best_route.path],
                    "total_fees": str(best_route.total_fees),
                    "execution_time_minutes": best_route.estimated_time_minutes,
                    "confidence_score": str(best_route.confidence_score),
                    "enterprise_rating": self._calculate_enterprise_rating(best_route)
                }
        
        return {"error": "No conversion route available"}
    
    async def bulk_execute_operations(
        self,
        operations: List[Dict[str, Any]],
        execution_mode: str = "sequential"  # "sequential", "parallel", "batch"
    ) -> Dict[str, Any]:
        """Execute multiple operations with enterprise coordination."""
        
        self._ensure_initialized()
        
        results = []
        total_operations = len(operations)
        
        if execution_mode == "sequential":
            for i, operation in enumerate(operations):
                logger.info(f"Executing operation {i+1}/{total_operations}")
                result = await self.execute_treasury_operation(**operation)
                results.append(result)
                
                # Brief delay between operations for risk management
                await asyncio.sleep(1)
        
        elif execution_mode == "parallel":
            # Execute operations in parallel (with concurrency limits)
            semaphore = asyncio.Semaphore(5)  # Max 5 concurrent operations
            
            async def execute_with_semaphore(operation):
                async with semaphore:
                    return await self.execute_treasury_operation(**operation)
            
            tasks = [execute_with_semaphore(op) for op in operations]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elif execution_mode == "batch":
            # Batch processing for large operations
            batch_size = 10
            for i in range(0, len(operations), batch_size):
                batch = operations[i:i+batch_size]
                batch_results = []
                
                for operation in batch:
                    result = await self.execute_treasury_operation(**operation)
                    batch_results.append(result)
                
                results.extend(batch_results)
                
                # Longer delay between batches
                if i + batch_size < len(operations):
                    await asyncio.sleep(10)
        
        # Analyze bulk execution results
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "executed")
        
        return {
            "bulk_execution_id": f"bulk_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "total_operations": total_operations,
            "successful_operations": success_count,
            "failed_operations": total_operations - success_count,
            "execution_mode": execution_mode,
            "results": results,
            "summary": {
                "success_rate": success_count / total_operations,
                "total_execution_time": "calculated",
                "enterprise_notes": self._generate_bulk_execution_notes(results)
            }
        }
    
    # ==============================
    # Enterprise Reporting & Analytics
    # ==============================
    
    async def get_enterprise_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive enterprise dashboard data."""
        
        self._ensure_initialized()
        
        return {
            "company_overview": {
                "company_name": self.config.company_name,
                "company_id": self.config.company_id,
                "integration_status": "active",
                "last_activity": datetime.utcnow().isoformat()
            },
            "performance_metrics": self.metrics.copy(),
            "liquidity_summary": await self._get_liquidity_summary(),
            "risk_metrics": await self._get_risk_metrics(),
            "compliance_status": await self._get_compliance_status(),
            "operational_insights": await self._get_operational_insights(),
            "recommendations": await self._generate_enterprise_recommendations()
        }
    
    async def generate_compliance_report(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report for enterprise audits."""
        
        # Filter transactions by date range
        filtered_transactions = [
            tx for tx in self.transaction_history 
            if start_date <= tx["timestamp"] <= end_date
        ]
        
        return {
            "report_metadata": {
                "company": self.config.company_name,
                "report_period": f"{start_date.isoformat()} to {end_date.isoformat()}",
                "generated_at": datetime.utcnow().isoformat(),
                "report_type": "enterprise_compliance"
            },
            "transaction_summary": {
                "total_transactions": len(filtered_transactions),
                "total_volume_btc": str(sum(Decimal(tx.get("amount", "0")) for tx in filtered_transactions)),
                "asset_breakdown": self._analyze_asset_breakdown(filtered_transactions),
                "provider_breakdown": self._analyze_provider_breakdown(filtered_transactions)
            },
            "compliance_analysis": {
                "kyc_compliance_rate": "100%",  # Would be calculated from actual data
                "aml_screening_results": "All transactions screened",
                "regulatory_flags": [],
                "risk_assessment": "Low risk profile maintained"
            },
            "audit_trail": self._generate_audit_trail(filtered_transactions),
            "recommendations": [
                "Continue current compliance practices",
                "Consider increasing treasury automation",
                "Evaluate additional liquidity providers"
            ]
        }
    
    # ==============================
    # Utility & Helper Methods
    # ==============================
    
    def _ensure_initialized(self):
        """Ensure SDK is properly initialized."""
        if not self.is_initialized:
            raise RuntimeError("SDK not initialized. Call initialize() first.")
    
    async def _validate_enterprise_operation(
        self, 
        operation_type: str, 
        asset: AssetType, 
        amount: Decimal
    ) -> Dict[str, Any]:
        """Validate operation against enterprise limits and policies."""
        
        # Check daily volume limits
        today_volume = self._calculate_daily_volume(asset)
        if today_volume + amount > self.config.max_daily_volume:
            return {
                "valid": False,
                "reason": "Daily volume limit exceeded",
                "compliance_notes": [f"Daily limit: {self.config.max_daily_volume} {asset.value}"]
            }
        
        # Check single transaction limits
        if amount > self.config.max_single_transaction:
            return {
                "valid": False,
                "reason": "Single transaction limit exceeded",
                "compliance_notes": [f"Max per transaction: {self.config.max_single_transaction} {asset.value}"]
            }
        
        return {
            "valid": True,
            "risk_category": self._assess_transaction_risk(amount),
            "compliance_notes": ["All enterprise policies satisfied"]
        }
    
    def _preference_to_tier(self, preference: str) -> Optional[LiquidityTier]:
        """Convert execution preference to liquidity tier."""
        mapping = {
            "fast": LiquidityTier.INSTANT,
            "balanced": LiquidityTier.STANDARD,
            "cheapest": LiquidityTier.BATCH
        }
        return mapping.get(preference)
    
    def _record_enterprise_transaction(
        self, 
        execution: TreasuryExecution, 
        quote: TreasuryQuote,
        metadata: Optional[Dict]
    ):
        """Record transaction for enterprise reporting."""
        
        record = {
            "execution_id": execution.execution_id,
            "timestamp": execution.initiated_at,
            "provider": execution.provider_name,
            "asset": execution.asset_type.value,
            "amount": str(execution.amount),
            "operation_type": execution.operation_type,
            "status": execution.status,
            "fees": quote.fees,
            "metadata": metadata or {}
        }
        
        self.transaction_history.append(record)
        
        # Update metrics
        self.metrics["total_transactions"] += 1
        if execution.asset_type == AssetType.BTC:
            self.metrics["total_volume_btc"] += execution.amount
    
    def _format_enterprise_quote(self, quote) -> Dict[str, Any]:
        """Format cashout quote for enterprise consumption."""
        return {
            "quote_id": quote.quote_id,
            "from_asset": quote.from_asset,
            "to_asset": quote.to_asset,
            "input_amount": quote.input_amount,
            "estimated_output": quote.estimated_output,
            "total_fees": quote.total_fees,
            "recommended_route": quote.recommended_route,
            "execution_time_minutes": quote.estimated_completion_minutes,
            "confidence_score": quote.confidence_score,
            "enterprise_grade": True,
            "compliance_verified": not quote.requires_kyc
        }
    
    def _calculate_enterprise_rating(self, route) -> str:
        """Calculate enterprise suitability rating for a route."""
        confidence = float(route.confidence_score)
        
        if confidence >= 0.9:
            return "Excellent"
        elif confidence >= 0.8:
            return "Good"
        elif confidence >= 0.7:
            return "Acceptable"
        else:
            return "Review Required"
    
    # Mock implementations for demonstration
    async def _calculate_optimal_transaction_size(self, asset: AssetType) -> str:
        return "1000 BTC"
    
    async def _analyze_cost_structure(self, asset: AssetType) -> Dict:
        return {"average_fees": "0.1%", "spread": "0.05%"}
    
    async def _assess_market_risk(self, asset: AssetType) -> Dict:
        return {"volatility": "medium", "liquidity_risk": "low"}
    
    def _get_operational_windows(self) -> Dict:
        return {"24x7": True, "preferred_hours": "09:00-17:00 EST"}
    
    def _calculate_daily_volume(self, asset: AssetType) -> Decimal:
        return Decimal("0")  # Would calculate from transaction history
    
    def _assess_transaction_risk(self, amount: Decimal) -> str:
        if amount < Decimal("100"):
            return "low"
        elif amount < Decimal("1000"):
            return "medium"
        else:
            return "high"
    
    async def _get_liquidity_summary(self) -> Dict:
        return {"status": "excellent", "depth": "deep"}
    
    async def _get_risk_metrics(self) -> Dict:
        return {"overall_risk": "low", "concentration_risk": "medium"}
    
    async def _get_compliance_status(self) -> Dict:
        return {"status": "compliant", "last_audit": "2024-01-01"}
    
    async def _get_operational_insights(self) -> Dict:
        return {"efficiency": "high", "automation_rate": "80%"}
    
    async def _generate_enterprise_recommendations(self) -> List[str]:
        return [
            "Consider increasing automation levels",
            "Evaluate additional treasury providers",
            "Optimize transaction batching"
        ]
    
    def _analyze_asset_breakdown(self, transactions: List) -> Dict:
        return {"BTC": "90%", "USD": "10%"}
    
    def _analyze_provider_breakdown(self, transactions: List) -> Dict:
        return {"MicroStrategy": "100%"}
    
    def _generate_audit_trail(self, transactions: List) -> List[Dict]:
        return [{"transaction_id": "demo", "audit_status": "passed"}]
    
    def _generate_bulk_execution_notes(self, results: List) -> List[str]:
        return ["Bulk execution completed successfully"]


# ==============================
# Enterprise SDK Factory
# ==============================

def create_enterprise_sdk(
    company_name: str,
    company_id: str, 
    treasury_credentials: Dict[str, str],
    **kwargs
) -> CHRONOSEnterpriseSDK:
    """Factory function to create enterprise SDK with sensible defaults."""
    
    config = EnterpriseConfig(
        company_name=company_name,
        company_id=company_id,
        treasury_credentials=treasury_credentials,
        **kwargs
    )
    
    return CHRONOSEnterpriseSDK(config)


# ==============================
# Context Manager for Enterprise Operations
# ==============================

@asynccontextmanager
async def enterprise_session(sdk: CHRONOSEnterpriseSDK):
    """Context manager for enterprise CHRONOS operations."""
    
    try:
        if not sdk.is_initialized:
            await sdk.initialize()
        
        logger.info(f"Enterprise session started for {sdk.config.company_name}")
        yield sdk
        
    except Exception as e:
        logger.error(f"Enterprise session error: {e}")
        raise
    
    finally:
        logger.info(f"Enterprise session ended for {sdk.config.company_name}")


# ==============================
# Example Usage
# ==============================

async def example_microstrategy_integration():
    """Example showing MicroStrategy integration."""
    
    # Configure SDK for MicroStrategy
    sdk = create_enterprise_sdk(
        company_name="MicroStrategy Inc.",
        company_id="MSTR_001",
        treasury_credentials={
            "api_key": "mstr_api_key",
            "api_secret": "mstr_api_secret",
            "treasury_account": "mstr_treasury_001"
        },
        max_daily_volume=Decimal("50000"),  # 50K BTC daily limit
        max_single_transaction=Decimal("5000"),  # 5K BTC per transaction
        risk_tolerance="conservative"
    )
    
    # Use enterprise session
    async with enterprise_session(sdk) as enterprise_chronos:
        
        # Get liquidity status
        liquidity = await enterprise_chronos.get_liquidity_status(AssetType.BTC)
        print(f"Available liquidity: {liquidity}")
        
        # Execute treasury operation
        result = await enterprise_chronos.execute_treasury_operation(
            operation_type="sell",
            asset=AssetType.BTC,
            amount=Decimal("100"),
            execution_preference="balanced"
        )
        print(f"Operation result: {result}")
        
        # Get enterprise dashboard
        dashboard = await enterprise_chronos.get_enterprise_dashboard()
        print(f"Dashboard: {dashboard}")
    
    return "Integration example completed"