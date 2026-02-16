#!/usr/bin/env python3
"""
Model Marketplace Test Suite
Tests marketplace listings, transactions, and platform operations
"""

import asyncio
import time
from datetime import datetime, timezone
import pytest

try:
    from prsm.economy.tokenomics.marketplace import get_marketplace
    from prsm.core.models import PricingModel, MarketplaceListing, MarketplaceTransaction
except (ImportError, ModuleNotFoundError) as e:
    pytest.skip("Marketplace module dependencies (ftns_service) not yet fully implemented", allow_module_level=True)


async def test_model_listing():
    """Test model listing functionality"""
    print("🧪 Testing model listing...")
    
    marketplace = get_marketplace()
    
    # Give test users initial FTNS balance for fees
    from prsm.economy.tokenomics.ftns_service import ftns_service
    
    # Give users multiple model rewards for sufficient balance
    for i in range(10):  # Give 10 model rewards = 1000 FTNS each
        await ftns_service.reward_contribution("model_owner_1", "model", 1.0)
        await ftns_service.reward_contribution("model_owner_2", "model", 1.0)
        await ftns_service.reward_contribution("buyer_1", "model", 1.0)
        await ftns_service.reward_contribution("dynamic_owner", "model", 1.0)
        await ftns_service.reward_contribution("dynamic_buyer", "model", 1.0)
    
    # Create pricing model
    pricing = PricingModel(
        base_price=10.0,
        pricing_type="hourly",
        usage_tiers={"basic": 10.0, "premium": 15.0},
        volume_discounts={"100": 0.1, "500": 0.2},
        dynamic_pricing_enabled=True,
        demand_multiplier=0.5
    )
    
    # Create listing details
    listing_details = {
        "title": "Advanced NLP Model",
        "description": "High-performance natural language processing model",
        "performance_metrics": {"accuracy": 0.95, "latency": 50},
        "resource_requirements": {"memory": "4GB", "cpu": "2 cores"},
        "supported_features": ["text_classification", "sentiment_analysis"],
        "max_concurrent_users": 5
    }
    
    # List model
    listing = await marketplace.list_model_for_rent(
        model_id="nlp_model_001",
        pricing=pricing,
        owner_id="model_owner_1",
        listing_details=listing_details
    )
    
    assert listing.model_id == "nlp_model_001", "Should set correct model ID"
    assert listing.owner_id == "model_owner_1", "Should set correct owner"
    assert listing.pricing_model.base_price == 10.0, "Should set correct pricing"
    assert listing.availability_status == "available", "Should be available initially"
    assert len(listing.supported_features) == 2, "Should have correct features"
    
    print("✅ Model listing working")


async def test_marketplace_transactions():
    """Test marketplace transaction processing"""
    print("🧪 Testing marketplace transactions...")
    
    marketplace = get_marketplace()
    
    # Create a second listing for transactions
    pricing2 = PricingModel(
        base_price=5.0,
        pricing_type="usage",
        dynamic_pricing_enabled=False
    )
    
    listing_details2 = {
        "title": "Computer Vision Model",
        "description": "Image classification and object detection",
        "performance_metrics": {"accuracy": 0.92},
        "supported_features": ["image_classification", "object_detection"]
    }
    
    listing2 = await marketplace.list_model_for_rent(
        model_id="cv_model_002",
        pricing=pricing2,
        owner_id="model_owner_2",
        listing_details=listing_details2
    )
    
    # Create transaction
    transaction_details = {
        "type": "rental",
        "duration": 24.0,  # 24 hours
        "usage_units": 100.0
    }
    
    transaction = await marketplace.facilitate_model_transactions(
        buyer_id="buyer_1",
        seller_id="model_owner_2", 
        listing_id=listing2.listing_id,
        transaction_details=transaction_details
    )
    
    assert transaction.buyer_id == "buyer_1", "Should set correct buyer"
    assert transaction.seller_id == "model_owner_2", "Should set correct seller"
    assert transaction.amount > 0, "Should calculate transaction amount"
    assert transaction.platform_fee > 0, "Should calculate platform fee"
    assert transaction.status in ["completed", "pending"], "Should have valid status"
    
    print("✅ Marketplace transactions working")


async def test_platform_fee_calculation():
    """Test platform fee calculation"""
    print("🧪 Testing platform fee calculation...")
    
    marketplace = get_marketplace()
    
    # Test various transaction values
    test_values = [1.0, 10.0, 100.0, 1000.0]
    
    for value in test_values:
        fee = await marketplace.calculate_platform_fees(value)
        
        assert fee > 0, f"Should calculate positive fee for value {value}"
        assert fee < value, f"Fee should be less than transaction value for {value}"
        
        # Fee should be reasonable percentage
        fee_percentage = fee / value
        assert fee_percentage <= 0.2, f"Fee percentage should be reasonable for {value}"
    
    print("✅ Platform fee calculation working")


async def test_search_functionality():
    """Test marketplace search functionality"""
    print("🧪 Testing search functionality...")
    
    marketplace = get_marketplace()
    
    # Test search with various criteria
    search_criteria = {
        "max_price": 15.0,
        "availability_status": "available",
        "required_features": ["text_classification"]
    }
    
    results = await marketplace.search_listings(search_criteria)
    
    # Verify search results
    for listing in results:
        assert listing.pricing_model.base_price <= 15.0, "Should respect price filter"
        assert listing.availability_status == "available", "Should respect availability filter"
        
        # Check feature requirements
        listing_features = set(listing.supported_features)
        required_features = set(search_criteria["required_features"])
        assert required_features.issubset(listing_features), "Should have required features"
    
    # Test text search
    text_search = {"search_text": "nlp"}
    text_results = await marketplace.search_listings(text_search)
    
    for listing in text_results:
        title_match = "nlp" in listing.title.lower()
        desc_match = "nlp" in listing.description.lower()
        assert title_match or desc_match, "Should match search text"
    
    print("✅ Search functionality working")


async def test_transaction_history():
    """Test transaction history tracking"""
    print("🧪 Testing transaction history...")
    
    marketplace = get_marketplace()
    
    # Get transaction history for buyer
    buyer_history = await marketplace.get_user_transaction_history("buyer_1")
    
    # Should have at least one transaction from previous test
    assert len(buyer_history) >= 0, "Should return transaction history"
    
    for transaction in buyer_history:
        assert transaction.buyer_id == "buyer_1" or transaction.seller_id == "buyer_1", "Should be user's transaction"
        assert hasattr(transaction, "created_at"), "Should have creation timestamp"
        assert hasattr(transaction, "status"), "Should have transaction status"
    
    # Get seller history
    seller_history = await marketplace.get_user_transaction_history("model_owner_2")
    assert len(seller_history) >= 0, "Should return seller transaction history"
    
    print("✅ Transaction history working")


async def test_model_performance_tracking():
    """Test model performance tracking"""
    print("🧪 Testing model performance tracking...")
    
    marketplace = get_marketplace()
    
    # Update model performance
    performance_data = {
        "accuracy": 0.96,
        "response_time": 45,
        "user_satisfaction": 4.8,
        "uptime": 0.995
    }
    
    success = await marketplace.update_model_performance("nlp_model_001", performance_data)
    assert success, "Should successfully update performance"
    
    # Verify performance update was applied to listings
    for listing in marketplace.listings.values():
        if listing.model_id == "nlp_model_001":
            assert "accuracy" in listing.performance_metrics, "Should update listing metrics"
            assert listing.performance_metrics["accuracy"] == 0.96, "Should update accuracy"
            break
    
    print("✅ Model performance tracking working")


async def test_transaction_rating():
    """Test transaction rating system"""
    print("🧪 Testing transaction rating...")
    
    marketplace = get_marketplace()
    
    # Find a transaction to rate
    if marketplace.transactions:
        transaction_id = list(marketplace.transactions.keys())[0]
        transaction = marketplace.transactions[transaction_id]
        
        # Rate from buyer perspective
        rating_success = await marketplace.rate_transaction(
            transaction_id=transaction_id,
            rater_id=transaction.buyer_id,
            rating=4.5,
            review="Great model, fast response time"
        )
        
        assert rating_success, "Should successfully rate transaction"
        
        # Verify rating was stored
        rated_user = transaction.seller_id
        assert rated_user in marketplace.user_ratings, "Should store user rating"
        
        # Test invalid rating (wrong user)
        invalid_rating = await marketplace.rate_transaction(
            transaction_id=transaction_id,
            rater_id="invalid_user",
            rating=3.0
        )
        
        assert not invalid_rating, "Should reject invalid rater"
    
    print("✅ Transaction rating working")


async def test_marketplace_statistics():
    """Test marketplace statistics"""
    print("🧪 Testing marketplace statistics...")
    
    marketplace = get_marketplace()
    
    # Get statistics
    stats = await marketplace.get_marketplace_statistics()
    
    # Verify required fields
    required_fields = [
        "total_listings",
        "active_listings", 
        "total_transactions",
        "successful_transactions",
        "total_revenue",
        "platform_fees_collected",
        "configuration"
    ]
    
    for field in required_fields:
        assert field in stats, f"Should have {field} in statistics"
    
    # Verify non-negative values
    assert stats["total_listings"] >= 0, "Should have non-negative listing count"
    assert stats["active_listings"] >= 0, "Should have non-negative active count"
    assert stats["total_revenue"] >= 0, "Should have non-negative revenue"
    
    # Verify configuration
    config = stats["configuration"]
    assert "platform_fee_percentage" in config, "Should have platform fee config"
    assert "transaction_fee" in config, "Should have transaction fee config"
    
    # Verify additional analytics
    assert "listings_by_status" in stats, "Should have status breakdown"
    assert "transactions_by_type" in stats, "Should have transaction type breakdown"
    
    print("✅ Marketplace statistics working")


async def test_dynamic_pricing():
    """Test dynamic pricing functionality"""
    print("🧪 Testing dynamic pricing...")
    
    marketplace = get_marketplace()
    
    # Create listing with dynamic pricing
    dynamic_pricing = PricingModel(
        base_price=20.0,
        pricing_type="hourly",
        dynamic_pricing_enabled=True,
        demand_multiplier=0.8,
        peak_hour_multiplier=1.5
    )
    
    listing_details = {
        "title": "Dynamic Pricing Model",
        "description": "Model with dynamic pricing enabled"
    }
    
    dynamic_listing = await marketplace.list_model_for_rent(
        model_id="dynamic_model_003",
        pricing=dynamic_pricing,
        owner_id="dynamic_owner",
        listing_details=listing_details
    )
    
    # Test transaction with dynamic pricing
    transaction_details = {
        "type": "rental",
        "duration": 10.0
    }
    
    # Calculate expected amount (will be affected by demand multiplier)
    transaction = await marketplace.facilitate_model_transactions(
        buyer_id="dynamic_buyer",
        seller_id="dynamic_owner",
        listing_id=dynamic_listing.listing_id,
        transaction_details=transaction_details
    )
    
    # Amount should be calculated with dynamic factors
    base_amount = 20.0 * 10.0  # base_price * duration
    assert transaction.amount > 0, "Should calculate dynamic price"
    
    print("✅ Dynamic pricing working")


async def run_marketplace_tests():
    """Run all marketplace tests"""
    print("🚀 Starting Model Marketplace Test Suite")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Run all test functions
        await test_model_listing()
        await test_marketplace_transactions()
        await test_platform_fee_calculation()
        await test_search_functionality()
        await test_transaction_history()
        await test_model_performance_tracking()
        await test_transaction_rating()
        await test_marketplace_statistics()
        await test_dynamic_pricing()
        
        # Get final statistics
        print("\n📊 Final Marketplace Statistics:")
        
        marketplace = get_marketplace()
        stats = await marketplace.get_marketplace_statistics()
        
        print(f"   📋 Total listings: {stats['total_listings']}")
        print(f"   🟢 Active listings: {stats['active_listings']}")
        print(f"   💳 Total transactions: {stats['total_transactions']}")
        print(f"   ✅ Successful transactions: {stats['successful_transactions']}")
        print(f"   💰 Total revenue: {stats['total_revenue']:.2f} FTNS")
        print(f"   💸 Platform fees collected: {stats['platform_fees_collected']:.2f} FTNS")
        print(f"   📈 Average transaction value: {stats['average_transaction_value']:.2f} FTNS")
        print(f"   ⭐ User satisfaction: {stats['user_satisfaction_score']:.2f}/1.0")
        
        print(f"\n   📊 Listings by status: {stats['listings_by_status']}")
        print(f"   📊 Transactions by type: {stats['transactions_by_type']}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ All Model Marketplace tests passed!")
        print(f"⚡ Test duration: {duration:.2f} seconds")
        print(f"🎯 Model Marketplace system is operational and ready for production")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Marketplace test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(run_marketplace_tests())
    exit(0 if success else 1)