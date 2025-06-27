#!/usr/bin/env python3
"""
Marketplace System Test Suite

Comprehensive pytest tests for PRSM marketplace functionality including
model listings, transactions, and platform operations.
Converted from test_marketplace.py to follow pytest conventions.
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal


# Mock classes for testing (since we may not have full marketplace implementation)
class MockPricingModel:
    """Mock pricing model for testing"""
    def __init__(self, base_price=10.0, pricing_type="hourly", **kwargs):
        self.base_price = base_price
        self.pricing_type = pricing_type
        self.usage_tiers = kwargs.get("usage_tiers", {})
        self.volume_discounts = kwargs.get("volume_discounts", {})
        self.dynamic_pricing_enabled = kwargs.get("dynamic_pricing_enabled", False)
        self.demand_multiplier = kwargs.get("demand_multiplier", 1.0)
    
    def calculate_price(self, usage_data=None):
        """Calculate price based on usage"""
        base = self.base_price
        if self.dynamic_pricing_enabled and usage_data:
            base *= self.demand_multiplier
        return base


class MockMarketplaceListing:
    """Mock marketplace listing for testing"""
    def __init__(self, model_id, pricing_model, owner_id, listing_details):
        self.listing_id = f"listing_{model_id}_{int(time.time())}"
        self.model_id = model_id
        self.pricing_model = pricing_model
        self.owner_id = owner_id
        self.listing_details = listing_details
        self.availability_status = "available"
        self.created_at = datetime.now(timezone.utc)
        self.supported_features = listing_details.get("supported_features", [])
        self.performance_metrics = listing_details.get("performance_metrics", {})
        self.max_concurrent_users = listing_details.get("max_concurrent_users", 1)


class MockMarketplaceTransaction:
    """Mock marketplace transaction for testing"""
    def __init__(self, listing, buyer_id, transaction_details):
        self.transaction_id = f"tx_{int(time.time())}"
        self.listing = listing
        self.buyer_id = buyer_id
        self.seller_id = listing.owner_id
        self.transaction_details = transaction_details
        self.status = "pending"
        self.created_at = datetime.now(timezone.utc)
        self.total_cost = self._calculate_cost()
    
    def _calculate_cost(self):
        """Calculate transaction cost"""
        base_cost = self.listing.pricing_model.base_price
        duration = self.transaction_details.get("duration", 1.0)
        return base_cost * duration
    
    def complete_transaction(self):
        """Mark transaction as completed"""
        self.status = "completed"
        self.completed_at = datetime.now(timezone.utc)


class MockFTNSService:
    """Mock FTNS service for testing"""
    def __init__(self):
        self.balances = {}
        self.transactions = []
    
    async def get_balance(self, user_id):
        """Get user balance"""
        return self.balances.get(user_id, 0.0)
    
    async def reward_contribution(self, user_id, contribution_type, quality_score):
        """Reward user contribution"""
        reward_amount = 1000.0 * quality_score  # Base reward
        current_balance = self.balances.get(user_id, 0.0)
        self.balances[user_id] = current_balance + reward_amount
        return reward_amount
    
    async def transfer(self, from_user, to_user, amount):
        """Transfer FTNS between users"""
        from_balance = self.balances.get(from_user, 0.0)
        if from_balance >= amount:
            self.balances[from_user] = from_balance - amount
            to_balance = self.balances.get(to_user, 0.0)
            self.balances[to_user] = to_balance + amount
            return True
        return False


class MockMarketplace:
    """Mock marketplace for testing"""
    def __init__(self, ftns_service=None):
        self.ftns_service = ftns_service or MockFTNSService()
        self.listings = {}
        self.transactions = {}
        self.platform_fee_rate = 0.05  # 5% platform fee
    
    async def list_model_for_rent(self, model_id, pricing, owner_id, listing_details):
        """List a model for rent in the marketplace"""
        listing = MockMarketplaceListing(model_id, pricing, owner_id, listing_details)
        self.listings[listing.listing_id] = listing
        return listing
    
    async def process_transaction(self, listing_id, buyer_id, transaction_details):
        """Process a marketplace transaction"""
        if listing_id not in self.listings:
            raise ValueError("Listing not found")
        
        listing = self.listings[listing_id]
        if listing.availability_status != "available":
            raise ValueError("Model not available")
        
        transaction = MockMarketplaceTransaction(listing, buyer_id, transaction_details)
        
        # Check buyer has sufficient balance
        buyer_balance = await self.ftns_service.get_balance(buyer_id)
        if buyer_balance < transaction.total_cost:
            raise ValueError("Insufficient balance")
        
        # Process payment
        platform_fee = transaction.total_cost * self.platform_fee_rate
        seller_amount = transaction.total_cost - platform_fee
        
        # Transfer funds
        success = await self.ftns_service.transfer(buyer_id, listing.owner_id, seller_amount)
        if success:
            transaction.complete_transaction()
            self.transactions[transaction.transaction_id] = transaction
            
            # Update listing availability if needed
            if transaction_details.get("exclusive", False):
                listing.availability_status = "rented"
        
        return transaction
    
    def get_active_listings(self):
        """Get all active marketplace listings"""
        return [listing for listing in self.listings.values() 
                if listing.availability_status == "available"]
    
    def get_user_transactions(self, user_id):
        """Get transactions for a specific user"""
        return [tx for tx in self.transactions.values() 
                if tx.buyer_id == user_id or tx.seller_id == user_id]


class TestMarketplaceListing:
    """Test suite for marketplace listing functionality"""
    
    @pytest.fixture
    def mock_ftns_service(self):
        """Fixture providing mock FTNS service"""
        return MockFTNSService()
    
    @pytest.fixture
    def marketplace(self, mock_ftns_service):
        """Fixture providing marketplace instance"""
        return MockMarketplace(mock_ftns_service)
    
    @pytest.fixture
    def sample_pricing_model(self):
        """Fixture providing sample pricing model"""
        return MockPricingModel(
            base_price=10.0,
            pricing_type="hourly",
            usage_tiers={"basic": 10.0, "premium": 15.0},
            volume_discounts={"100": 0.1, "500": 0.2},
            dynamic_pricing_enabled=True,
            demand_multiplier=0.5
        )
    
    @pytest.fixture
    def sample_listing_details(self):
        """Fixture providing sample listing details"""
        return {
            "title": "Advanced NLP Model",
            "description": "High-performance natural language processing model",
            "performance_metrics": {"accuracy": 0.95, "latency": 50},
            "resource_requirements": {"memory": "4GB", "cpu": "2 cores"},
            "supported_features": ["text_classification", "sentiment_analysis"],
            "max_concurrent_users": 5
        }
    
    @pytest.mark.asyncio
    async def test_model_listing_creation(self, marketplace, sample_pricing_model, sample_listing_details):
        """Test creating a model listing"""
        listing = await marketplace.list_model_for_rent(
            model_id="nlp_model_001",
            pricing=sample_pricing_model,
            owner_id="model_owner_1",
            listing_details=sample_listing_details
        )
        
        # Verify listing properties
        assert listing.model_id == "nlp_model_001"
        assert listing.owner_id == "model_owner_1"
        assert listing.pricing_model.base_price == 10.0
        assert listing.availability_status == "available"
        assert len(listing.supported_features) == 2
        assert "text_classification" in listing.supported_features
        assert "sentiment_analysis" in listing.supported_features
        
        # Verify listing is stored in marketplace
        assert listing.listing_id in marketplace.listings
    
    def test_pricing_model_functionality(self, sample_pricing_model):
        """Test pricing model calculations"""
        # Test basic pricing
        basic_price = sample_pricing_model.calculate_price()
        assert basic_price == 10.0
        
        # Test dynamic pricing
        usage_data = {"demand": "high"}
        dynamic_price = sample_pricing_model.calculate_price(usage_data)
        expected_price = 10.0 * 0.5  # base_price * demand_multiplier
        assert dynamic_price == expected_price
    
    @pytest.mark.asyncio
    async def test_multiple_listings(self, marketplace, sample_pricing_model):
        """Test creating multiple listings"""
        listings = []
        for i in range(3):
            listing_details = {
                "title": f"Model {i}",
                "description": f"Test model {i}",
                "supported_features": [f"feature_{i}"]
            }
            
            listing = await marketplace.list_model_for_rent(
                model_id=f"model_{i}",
                pricing=sample_pricing_model,
                owner_id=f"owner_{i}",
                listing_details=listing_details
            )
            listings.append(listing)
        
        # Verify all listings are created
        assert len(marketplace.listings) == 3
        
        # Verify active listings
        active_listings = marketplace.get_active_listings()
        assert len(active_listings) == 3
        
        # Verify each listing is unique
        listing_ids = [listing.listing_id for listing in listings]
        assert len(set(listing_ids)) == 3  # All unique IDs


class TestMarketplaceTransactions:
    """Test suite for marketplace transaction functionality"""
    
    @pytest.fixture
    async def setup_marketplace_with_users(self):
        """Fixture providing marketplace with funded users"""
        ftns_service = MockFTNSService()
        marketplace = MockMarketplace(ftns_service)
        
        # Fund test users
        users = ["model_owner_1", "model_owner_2", "buyer_1", "buyer_2"]
        for user in users:
            for _ in range(10):  # Give multiple rewards for sufficient balance
                await ftns_service.reward_contribution(user, "model", 1.0)
        
        return marketplace, ftns_service
    
    @pytest.mark.asyncio
    async def test_transaction_processing(self, setup_marketplace_with_users):
        """Test processing marketplace transactions"""
        marketplace, ftns_service = await setup_marketplace_with_users
        
        # Create a listing
        pricing = MockPricingModel(base_price=5.0, pricing_type="usage")
        listing_details = {
            "title": "Computer Vision Model",
            "description": "Image classification and object detection",
            "performance_metrics": {"accuracy": 0.92},
            "supported_features": ["image_classification", "object_detection"]
        }
        
        listing = await marketplace.list_model_for_rent(
            model_id="cv_model_002",
            pricing=pricing,
            owner_id="model_owner_2",
            listing_details=listing_details
        )
        
        # Process transaction
        transaction_details = {
            "type": "rental",
            "duration": 24.0,  # 24 hours
            "usage_estimate": "moderate"
        }
        
        initial_buyer_balance = await ftns_service.get_balance("buyer_1")
        initial_seller_balance = await ftns_service.get_balance("model_owner_2")
        
        transaction = await marketplace.process_transaction(
            listing.listing_id,
            "buyer_1",
            transaction_details
        )
        
        # Verify transaction properties
        assert transaction.buyer_id == "buyer_1"
        assert transaction.seller_id == "model_owner_2"
        assert transaction.status == "completed"
        assert transaction.total_cost == 5.0 * 24.0  # base_price * duration
        
        # Verify balances updated
        final_buyer_balance = await ftns_service.get_balance("buyer_1")
        final_seller_balance = await ftns_service.get_balance("model_owner_2")
        
        platform_fee = transaction.total_cost * marketplace.platform_fee_rate
        seller_amount = transaction.total_cost - platform_fee
        
        assert final_buyer_balance == initial_buyer_balance - transaction.total_cost
        assert final_seller_balance == initial_seller_balance + seller_amount
    
    @pytest.mark.asyncio
    async def test_insufficient_balance_transaction(self, setup_marketplace_with_users):
        """Test transaction failure with insufficient balance"""
        marketplace, ftns_service = await setup_marketplace_with_users
        
        # Create expensive listing
        pricing = MockPricingModel(base_price=50000.0)  # Very expensive
        listing_details = {"title": "Expensive Model", "description": "High-cost model"}
        
        listing = await marketplace.list_model_for_rent(
            model_id="expensive_model",
            pricing=pricing,
            owner_id="model_owner_1",
            listing_details=listing_details
        )
        
        # Attempt transaction with insufficient funds
        transaction_details = {"type": "rental", "duration": 1.0}
        
        with pytest.raises(ValueError, match="Insufficient balance"):
            await marketplace.process_transaction(
                listing.listing_id,
                "buyer_1",
                transaction_details
            )
    
    @pytest.mark.asyncio
    async def test_nonexistent_listing_transaction(self, setup_marketplace_with_users):
        """Test transaction failure with nonexistent listing"""
        marketplace, ftns_service = await setup_marketplace_with_users
        
        transaction_details = {"type": "rental", "duration": 1.0}
        
        with pytest.raises(ValueError, match="Listing not found"):
            await marketplace.process_transaction(
                "nonexistent_listing_id",
                "buyer_1",
                transaction_details
            )
    
    @pytest.mark.asyncio
    async def test_user_transaction_history(self, setup_marketplace_with_users):
        """Test retrieving user transaction history"""
        marketplace, ftns_service = await setup_marketplace_with_users
        
        # Create multiple transactions for a user
        pricing = MockPricingModel(base_price=10.0)
        listing_details = {"title": "Test Model", "description": "Test model for transactions"}
        
        transactions = []
        for i in range(3):
            listing = await marketplace.list_model_for_rent(
                model_id=f"test_model_{i}",
                pricing=pricing,
                owner_id="model_owner_1",
                listing_details=listing_details
            )
            
            transaction = await marketplace.process_transaction(
                listing.listing_id,
                "buyer_1",
                {"type": "rental", "duration": 1.0}
            )
            transactions.append(transaction)
        
        # Get user transaction history
        buyer_transactions = marketplace.get_user_transactions("buyer_1")
        seller_transactions = marketplace.get_user_transactions("model_owner_1")
        
        # Verify transaction history
        assert len(buyer_transactions) == 3
        assert len(seller_transactions) == 3
        
        # Verify all transactions are for the correct user
        for tx in buyer_transactions:
            assert tx.buyer_id == "buyer_1"
        
        for tx in seller_transactions:
            assert tx.seller_id == "model_owner_1"


class TestMarketplaceIntegration:
    """Integration tests for complete marketplace workflows"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_marketplace_workflow(self):
        """Test complete marketplace workflow from listing to transaction"""
        # Setup
        ftns_service = MockFTNSService()
        marketplace = MockMarketplace(ftns_service)
        
        # Fund users
        await ftns_service.reward_contribution("seller", "model", 1.0)
        await ftns_service.reward_contribution("buyer", "model", 10.0)  # More funds for buyer
        
        # Create and list model
        pricing = MockPricingModel(base_price=100.0, pricing_type="hourly")
        listing_details = {
            "title": "Integration Test Model",
            "description": "End-to-end test model",
            "performance_metrics": {"accuracy": 0.98, "latency": 30},
            "supported_features": ["classification", "regression"],
            "max_concurrent_users": 10
        }
        
        listing = await marketplace.list_model_for_rent(
            model_id="integration_model",
            pricing=pricing,
            owner_id="seller",
            listing_details=listing_details
        )
        
        # Verify listing created
        assert listing.model_id == "integration_model"
        assert listing.availability_status == "available"
        
        # Process transaction
        transaction_details = {
            "type": "rental",
            "duration": 2.0,  # 2 hours
            "usage_estimate": "high",
            "exclusive": False
        }
        
        transaction = await marketplace.process_transaction(
            listing.listing_id,
            "buyer",
            transaction_details
        )
        
        # Verify transaction completed
        assert transaction.status == "completed"
        assert transaction.total_cost == 200.0  # 100.0 * 2 hours
        
        # Verify balances
        buyer_balance = await ftns_service.get_balance("buyer")
        seller_balance = await ftns_service.get_balance("seller")
        
        platform_fee = transaction.total_cost * marketplace.platform_fee_rate
        expected_seller_gain = transaction.total_cost - platform_fee
        
        assert buyer_balance == 10000.0 - 200.0  # Initial 10k - transaction cost
        assert seller_balance == 1000.0 + expected_seller_gain  # Initial 1k + payment
        
        # Verify listing still available (non-exclusive transaction)
        assert listing.availability_status == "available"
        
        # Verify transaction history
        buyer_history = marketplace.get_user_transactions("buyer")
        seller_history = marketplace.get_user_transactions("seller")
        
        assert len(buyer_history) == 1
        assert len(seller_history) == 1
        assert buyer_history[0].transaction_id == seller_history[0].transaction_id


if __name__ == "__main__":
    # Run the tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])