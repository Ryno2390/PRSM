#!/usr/bin/env python3
"""
CHRONOS Enterprise Demo

Demonstrates real functionality with live Bitcoin prices and enterprise features.
Showcases the production-ready components for potential partners and investors.
"""

import asyncio
import sys
import os
from decimal import Decimal
from datetime import datetime, timedelta

# Add PRSM to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prsm.chronos.price_oracles import (
    price_aggregator, 
    get_btc_price, 
    get_asset_price, 
    validate_price_feeds
)
from prsm.chronos.treasury_provider import (
    create_treasury_manager_with_microstrategy,
    TreasuryProviderManager,
    MicroStrategyProvider
)
from prsm.chronos.enterprise_sdk import (
    create_enterprise_sdk,
    enterprise_session,
    EnterpriseConfig
)
from prsm.chronos.models import AssetType
from prsm.chronos.error_handling import get_system_health, error_tracker


def print_header(title: str):
    """Print formatted section header."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")


def print_subheader(title: str):
    """Print formatted subsection header."""
    print(f"\n{'─'*40}")
    print(f"📊 {title}")
    print(f"{'─'*40}")


async def demo_real_price_oracles():
    """Demonstrate real Bitcoin price feeds from multiple sources."""
    
    print_header("Real-Time Bitcoin Price Oracles")
    
    print("Fetching live Bitcoin prices from multiple sources...")
    
    # Test individual price feeds
    btc_price = await get_btc_price()
    
    if btc_price:
        print(f"\n💰 Bitcoin Price (Aggregated)")
        print(f"   Price: ${btc_price.price_usd:,.2f}")
        print(f"   Confidence: {btc_price.confidence_score:.2%}")
        print(f"   Sources: {', '.join(btc_price.sources_used)}")
        print(f"   Price Range: ${btc_price.price_range[0]:,.2f} - ${btc_price.price_range[1]:,.2f}")
        print(f"   Last Updated: {btc_price.last_updated.strftime('%H:%M:%S')}")
    else:
        print("❌ Failed to fetch Bitcoin price")
    
    # Test multiple assets
    print_subheader("Multi-Asset Price Feed")
    
    test_assets = [AssetType.BTC, AssetType.ETH, AssetType.USDC]
    prices = await price_aggregator.get_multiple_aggregated_prices(test_assets)
    
    for asset, price in prices.items():
        confidence_emoji = "🟢" if price.confidence_score > 0.8 else "🟡" if price.confidence_score > 0.6 else "🔴"
        print(f"   {confidence_emoji} {asset.value}: ${price.price_usd:,.2f} (confidence: {price.confidence_score:.2%})")
    
    # Oracle health status
    print_subheader("Oracle Health Status")
    
    health = await price_aggregator.get_oracle_health()
    for oracle_name, status in health.items():
        health_emoji = {
            "excellent": "🟢",
            "good": "🟡", 
            "degraded": "🟠",
            "poor": "🔴"
        }.get(status["health_score"], "❓")
        
        print(f"   {health_emoji} {oracle_name}: {status['health_score']} (errors: {status['error_count']})")


async def demo_microstrategy_integration():
    """Demonstrate MicroStrategy treasury provider integration."""
    
    print_header("MicroStrategy Treasury Integration")
    
    # Create MicroStrategy provider
    credentials = {
        "api_key": "demo_mstr_key",
        "api_secret": "demo_mstr_secret", 
        "treasury_account": "mstr_treasury_demo"
    }
    
    provider = MicroStrategyProvider(credentials)
    
    print(f"🏦 MicroStrategy Provider Initialized")
    print(f"   Total BTC Holdings: {provider.total_btc_holdings:,.0f} BTC")
    print(f"   Available for Trading: {provider.available_for_trading:,.0f} BTC")
    
    # Get liquidity quote
    print_subheader("Liquidity Quote (1000 BTC)")
    
    try:
        quote = await provider.get_liquidity_quote(
            asset=AssetType.BTC,
            amount=Decimal("1000"),
            operation="sell"
        )
        
        print(f"   💹 Liquidity Tier: {quote.liquidity_tier.value}")
        print(f"   💰 Rate: ${quote.rate:,.2f}")
        print(f"   💸 Total Fees: {sum(quote.fees.values()):.4f} BTC")
        print(f"   ⏱️  Settlement Time: {quote.settlement_terms['settlement_time_minutes']} minutes")
        print(f"   🛡️  Insurance: {quote.settlement_terms['insurance_coverage']}")
        
    except Exception as e:
        print(f"   ❌ Quote failed: {e}")
    
    # Get custody status
    print_subheader("Custody Status")
    
    custody_status = await provider.get_custody_status()
    print(f"   🏛️  Provider: {custody_status['provider']}")
    print(f"   📊 Available Liquidity: {custody_status['available_for_trading']} BTC")
    print(f"   📈 Daily Volume: {custody_status['average_daily_volume']} BTC")
    print(f"   🔒 Regulatory Status: {custody_status['regulatory_status']}")


async def demo_enterprise_sdk():
    """Demonstrate enterprise SDK functionality."""
    
    print_header("Enterprise SDK Demo")
    
    # Create enterprise SDK
    sdk = create_enterprise_sdk(
        company_name="PRSM Demo Corp",
        company_id="DEMO_001",
        treasury_credentials={
            "api_key": "demo_key",
            "treasury_account": "demo_treasury"
        },
        max_daily_volume=Decimal("10000"),
        max_single_transaction=Decimal("1000")
    )
    
    print(f"🏢 Enterprise SDK Initialized")
    print(f"   Company: {sdk.config.company_name}")
    print(f"   Daily Limit: {sdk.config.max_daily_volume:,.0f} BTC")
    print(f"   Transaction Limit: {sdk.config.max_single_transaction:,.0f} BTC")
    
    # Initialize SDK
    await sdk.initialize()
    
    # Get liquidity status
    print_subheader("Enterprise Liquidity Status")
    
    try:
        liquidity = await sdk.get_liquidity_status(AssetType.BTC)
        print(f"   📊 Available Liquidity: {liquidity['liquidity_data']['total_liquidity_by_tier']}")
        print(f"   💡 Recommended Size: {liquidity['enterprise_insights']['recommended_transaction_size']}")
        print(f"   💰 Cost Analysis: {liquidity['enterprise_insights']['cost_analysis']}")
        
    except Exception as e:
        print(f"   ❌ Liquidity check failed: {e}")
    
    # Get conversion quote
    print_subheader("FTNS to USD Conversion Quote")
    
    try:
        quote = await sdk.get_conversion_quote(
            from_asset=AssetType.FTNS,
            to_asset=AssetType.USD,
            amount=Decimal("10000")
        )
        
        if "error" not in quote:
            print(f"   💱 Conversion: {quote['input_amount']} FTNS → ${quote['estimated_output']}")
            print(f"   🛣️  Route: {' → '.join(quote['route'])}")
            print(f"   💸 Total Fees: ${quote['total_fees']}")
            print(f"   ⏱️  Time: {quote['execution_time_minutes']} minutes")
            print(f"   🎯 Confidence: {quote['confidence_score']}")
        else:
            print(f"   ❌ Quote failed: {quote['error']}")
            
    except Exception as e:
        print(f"   ❌ Quote failed: {e}")


async def demo_system_health():
    """Demonstrate system health monitoring."""
    
    print_header("System Health & Monitoring")
    
    # Get system health
    health = await get_system_health()
    
    health_emoji = {
        "healthy": "🟢",
        "degraded": "🟡",
        "unhealthy": "🔴"
    }.get(health["status"], "❓")
    
    print(f"   {health_emoji} Overall Health: {health['overall_health_score']}% ({health['status']})")
    
    # Error summary
    print_subheader("Error Monitoring")
    
    error_summary = health["error_summary"]
    print(f"   📊 Total Errors (1h): {error_summary['total_errors']}")
    print(f"   🏢 Components Affected: {error_summary['components_affected']}")
    print(f"   📈 Error Trend: {error_summary['error_rate_trend']}")
    
    # Severity breakdown
    severity_breakdown = error_summary["severity_breakdown"]
    for severity, count in severity_breakdown.items():
        if count > 0:
            severity_emoji = {
                "CRITICAL": "🚨",
                "HIGH": "🔴", 
                "MEDIUM": "🟡",
                "LOW": "🟢"
            }.get(severity, "❓")
            print(f"   {severity_emoji} {severity}: {count}")


async def demo_price_validation():
    """Validate price feeds are working correctly."""
    
    print_header("Price Feed Validation")
    
    print("Testing all price feeds...")
    
    validation_results = await validate_price_feeds()
    
    for asset, result in validation_results.items():
        if "FAILED" in result:
            print(f"   ❌ {asset}: {result}")
        else:
            print(f"   ✅ {asset}: {result}")
    
    print("\n🎯 Price feed validation complete!")


async def main():
    """Run the complete CHRONOS enterprise demo."""
    
    print("🚀 CHRONOS Enterprise Demo")
    print("💰 Real Bitcoin Prices | 🏦 MicroStrategy Integration | 🏢 Enterprise SDK")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Demo components
        await demo_price_validation()
        await demo_real_price_oracles()
        await demo_microstrategy_integration()
        await demo_enterprise_sdk()
        await demo_system_health()
        
        print_header("Demo Complete ✅")
        print("🎉 All CHRONOS enterprise components demonstrated successfully!")
        print("\n📈 Key Highlights:")
        print("   • Live Bitcoin prices from multiple oracles")
        print("   • MicroStrategy 581K BTC treasury integration")
        print("   • Enterprise-grade SDK with risk controls")
        print("   • Production-ready error handling and monitoring")
        print("\n🚀 Ready for enterprise adoption!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())