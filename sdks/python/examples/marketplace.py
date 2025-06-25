#!/usr/bin/env python3
"""
PRSM Python SDK - Marketplace Examples

This example demonstrates marketplace functionality including model discovery,
FTNS token management, and marketplace interactions.
"""

import asyncio
import os
from prsm_sdk import PRSMClient, PRSMError


async def marketplace_discovery_example():
    """Demonstrate marketplace model discovery"""
    print("=== Marketplace Discovery ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        # Search for models
        search_results = await client.marketplace.search_models(
            query="text generation",
            filters={
                "max_cost_per_1k_tokens": 0.05,
                "min_quality_score": 0.8,
                "domains": ["general", "technical"]
            },
            limit=10
        )
        
        print(f"Found {len(search_results.models)} models:")
        
        for model in search_results.models[:5]:  # Show top 5
            print(f"\n📦 {model.name}")
            print(f"   Cost: ${model.cost_per_1k_tokens:.4f} per 1K tokens")
            print(f"   Quality: {model.quality_score:.2f}/1.0")
            print(f"   Provider: {model.provider}")
            print(f"   Description: {model.description[:80]}...")
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def ftns_token_management():
    """Demonstrate FTNS token operations"""
    print("\n=== FTNS Token Management ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        # Get current FTNS balance
        balance = await client.marketplace.get_ftns_balance()
        print(f"Current FTNS Balance: {balance.available:.2f}")
        print(f"Reserved: {balance.reserved:.2f}")
        print(f"Total: {balance.total:.2f}")
        
        # Get FTNS transaction history
        transactions = await client.marketplace.get_ftns_transactions(
            limit=5
        )
        
        print(f"\nRecent FTNS Transactions:")
        for tx in transactions.transactions:
            print(f"  {tx.timestamp} | {tx.type} | {tx.amount:+.2f} FTNS | {tx.description}")
        
        # Purchase FTNS tokens
        if balance.available < 100:  # Buy if balance is low
            print(f"\nPurchasing FTNS tokens...")
            purchase = await client.marketplace.purchase_ftns(
                amount=500.0,
                payment_method="credit_card"
            )
            
            print(f"Purchase initiated: {purchase.transaction_id}")
            print(f"Amount: {purchase.amount} FTNS")
            print(f"Cost: ${purchase.cost_usd:.2f}")
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def model_marketplace_interaction():
    """Demonstrate buying and using marketplace models"""
    print("\n=== Model Marketplace Interaction ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        # Browse available models
        models = await client.marketplace.list_models(
            category="code-generation",
            sort_by="popularity",
            limit=3
        )
        
        print("Popular Code Generation Models:")
        for model in models.models:
            print(f"\n🤖 {model.name}")
            print(f"   Price: {model.price_ftns} FTNS")
            print(f"   Rating: {model.rating:.1f}/5.0 ({model.review_count} reviews)")
            print(f"   Capabilities: {', '.join(model.capabilities)}")
        
        # Select and purchase a model
        if models.models:
            selected_model = models.models[0]
            print(f"\nPurchasing access to: {selected_model.name}")
            
            purchase = await client.marketplace.purchase_model_access(
                model_id=selected_model.id,
                access_type="unlimited",  # or "pay_per_use"
                duration="monthly"
            )
            
            print(f"Purchase successful: {purchase.access_token}")
            
            # Use the purchased model
            result = await client.models.infer(
                model=selected_model.id,
                prompt="Write a Python function to reverse a string",
                access_token=purchase.access_token
            )
            
            print(f"\nModel Response:")
            print(result.content)
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def marketplace_contribution():
    """Demonstrate contributing to the marketplace"""
    print("\n=== Marketplace Contribution ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        # List your contributed models
        my_models = await client.marketplace.get_my_models()
        
        print(f"Your Models in Marketplace: {len(my_models.models)}")
        
        for model in my_models.models:
            print(f"\n📊 {model.name}")
            print(f"   Status: {model.status}")
            print(f"   Downloads: {model.download_count}")
            print(f"   Revenue: {model.total_revenue_ftns:.2f} FTNS")
            print(f"   Rating: {model.average_rating:.1f}/5.0")
        
        # Get earnings summary
        earnings = await client.marketplace.get_earnings_summary()
        
        print(f"\nEarnings Summary:")
        print(f"  This Month: {earnings.current_month:.2f} FTNS")
        print(f"  Last Month: {earnings.previous_month:.2f} FTNS")
        print(f"  Total Earned: {earnings.total_earned:.2f} FTNS")
        print(f"  Pending Payouts: {earnings.pending_payout:.2f} FTNS")
        
        # Submit a new model (example)
        print(f"\nSubmitting new model to marketplace...")
        
        submission = await client.marketplace.submit_model(
            name="my-specialized-analyzer",
            description="A specialized text analyzer for technical documents",
            model_file_path="/path/to/model.pkl",  # Example path
            capabilities=["text-analysis", "technical-content"],
            pricing={
                "type": "pay_per_use",
                "cost_per_request": 5.0  # FTNS
            },
            metadata={
                "version": "1.0.0",
                "training_data": "technical-docs-v2",
                "accuracy_score": 0.94
            }
        )
        
        print(f"Model submitted for review: {submission.submission_id}")
        print(f"Expected review time: {submission.estimated_review_time}")
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def marketplace_analytics():
    """Demonstrate marketplace analytics and insights"""
    print("\n=== Marketplace Analytics ===")
    
    client = PRSMClient(api_key=os.getenv("PRSM_API_KEY"))
    
    try:
        # Get marketplace trends
        trends = await client.marketplace.get_market_trends(
            timeframe="7d"
        )
        
        print("Market Trends (Last 7 Days):")
        print(f"  Most Popular Category: {trends.top_category}")
        print(f"  Average Model Price: {trends.avg_price_ftns:.1f} FTNS")
        print(f"  New Models Added: {trends.new_models_count}")
        print(f"  Total Transactions: {trends.transaction_count}")
        
        # Get personalized recommendations
        recommendations = await client.marketplace.get_recommendations(
            based_on="usage_history",
            limit=5
        )
        
        print(f"\nRecommended Models for You:")
        for rec in recommendations.models:
            print(f"  🎯 {rec.name} (Match: {rec.match_score:.0%})")
            print(f"     {rec.reason}")
        
        # Market price analysis
        price_analysis = await client.marketplace.analyze_pricing(
            model_type="text-generation",
            timeframe="30d"
        )
        
        print(f"\nPricing Analysis - Text Generation Models:")
        print(f"  Price Range: {price_analysis.min_price:.1f} - {price_analysis.max_price:.1f} FTNS")
        print(f"  Average Price: {price_analysis.average_price:.1f} FTNS")
        print(f"  Price Trend: {price_analysis.trend} ({price_analysis.trend_percentage:+.1f}%)")
        
    except PRSMError as e:
        print(f"PRSM API Error: {e.message}")


async def main():
    """Run all marketplace examples"""
    if not os.getenv("PRSM_API_KEY"):
        print("Please set PRSM_API_KEY environment variable")
        return
    
    print("PRSM Python SDK - Marketplace Examples")
    print("=" * 50)
    
    await marketplace_discovery_example()
    await ftns_token_management()
    await model_marketplace_interaction()
    await marketplace_contribution()
    await marketplace_analytics()
    
    print("\n" + "=" * 50)
    print("Marketplace examples completed!")


if __name__ == "__main__":
    asyncio.run(main())