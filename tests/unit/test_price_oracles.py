"""
Unit tests for Price Oracle implementations.

Tests cover:
- CoinGecko oracle API key usage
- FTNS fallback when zero price returned
- Circuit breaker functionality
- FTNS oracle zero-price guards
- API key skip logic
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from decimal import Decimal


# =============================================================================
# Test 1: CoinGecko oracle uses API key header
# =============================================================================
@pytest.mark.asyncio
async def test_coingecko_oracle_uses_api_key_header():
    """Test that CoinGeckoOracle sends API key header when COINGECKO_API_KEY is set."""
    with patch.dict(os.environ, {"COINGECKO_API_KEY": "test-api-key-123"}):
        from prsm.compute.chronos.price_oracles import CoinGeckoOracle
        from prsm.compute.chronos.models import AssetType
        
        oracle = CoinGeckoOracle()
        
        # Verify API key is stored
        assert oracle.api_key == "test-api-key-123"
        
        # Verify headers include the API key
        headers = oracle._get_headers()
        assert headers == {"x-cg-demo-api-key": "test-api-key-123"}


# =============================================================================
# Test 2: CoinGecko oracle FTNS fallback when zero
# =============================================================================
@pytest.mark.asyncio
async def test_coingecko_oracle_ftns_fallback_when_zero():
    """Test that CoinGeckoOracle falls back to internal_oracle when FTNS returns zero price."""
    with patch.dict(os.environ, {"FTNS_USD_RATE": "0.10"}):
        from prsm.compute.chronos.price_oracles import CoinGeckoOracle
        from prsm.compute.chronos.models import AssetType
        
        oracle = CoinGeckoOracle()
        
        # Mock the session and response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"ftns-token": {"usd": 0}})  # Zero price
        
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        with patch.object(oracle, '_get_session', return_value=mock_session):
            result = await oracle.get_price(AssetType.FTNS)
        
        # Should fall back to internal oracle
        assert result is not None
        assert result.source == "internal_oracle"
        assert result.price_usd == Decimal("0.10")


# =============================================================================
# Test 3: CoinCap oracle FTNS in mapping
# =============================================================================
def test_coincap_oracle_ftns_in_mapping():
    """Test that AssetType.FTNS is in CoinCapOracle.asset_mapping."""
    from prsm.compute.chronos.price_oracles import CoinCapOracle
    from prsm.compute.chronos.models import AssetType
    
    oracle = CoinCapOracle()
    
    assert AssetType.FTNS in oracle.asset_mapping
    assert oracle.asset_mapping[AssetType.FTNS] == "ftns-token"


# =============================================================================
# Test 4: Price aggregator FTNS internal fallback
# =============================================================================
@pytest.mark.asyncio
async def test_price_aggregator_ftns_internal_fallback():
    """Test that PriceAggregator falls back to internal_oracle for FTNS when all oracles return None."""
    with patch.dict(os.environ, {"FTNS_USD_RATE": "0.10"}):
        from prsm.compute.chronos.price_oracles import PriceAggregator
        from prsm.compute.chronos.models import AssetType
        
        aggregator = PriceAggregator()
        
        # Mock all oracles to return None for FTNS
        for oracle in aggregator.oracles:
            oracle.get_price = AsyncMock(return_value=None)
        
        result = await aggregator.get_aggregated_price(AssetType.FTNS)
        
        # Should fall back to internal oracle
        assert result is not None
        assert result.price_usd == Decimal("0.10")
        assert result.sources_used == ["internal_oracle"]
        assert result.confidence_score == Decimal("0.5")


# =============================================================================
# Test 5: Circuit breaker disables oracle after errors
# =============================================================================
def test_circuit_breaker_disables_oracle_after_errors():
    """Test that oracle is disabled after MAX_ERRORS_BEFORE_DISABLE consecutive errors."""
    from prsm.compute.chronos.price_oracles import CoinGeckoOracle
    
    oracle = CoinGeckoOracle()
    
    # Initially active
    assert oracle.is_active == True
    assert oracle.error_count == 0
    
    # Record errors up to threshold
    for i in range(oracle.MAX_ERRORS_BEFORE_DISABLE):
        oracle._record_error()
    
    # Should now be disabled
    assert oracle.is_active == False
    assert oracle.error_count == oracle.MAX_ERRORS_BEFORE_DISABLE
    assert oracle.disabled_at is not None


# =============================================================================
# Test 6: Circuit breaker re-enables after recovery
# =============================================================================
def test_circuit_breaker_reenables_after_recovery():
    """Test that oracle is re-enabled after RECOVERY_WINDOW passes."""
    from prsm.compute.chronos.price_oracles import CoinGeckoOracle
    
    oracle = CoinGeckoOracle()
    
    # Disable the oracle
    oracle.is_active = False
    oracle.disabled_at = datetime.utcnow() - timedelta(minutes=6)  # 6 minutes ago
    
    # Try to re-enable
    oracle._maybe_reenable()
    
    # Should be re-enabled
    assert oracle.is_active == True
    assert oracle.error_count == 0
    assert oracle.disabled_at is None


# =============================================================================
# Test 7: FTNS oracle get_oracle_price zero guard
# =============================================================================
@pytest.mark.asyncio
async def test_ftns_oracle_get_oracle_price_zero_guard():
    """Test that FTNSOracle falls back to internal oracle when all sources return zero prices."""
    with patch.dict(os.environ, {"FTNS_USD_RATE": "0.10"}):
        from prsm.economy.blockchain.ftns_oracle import FTNSOracle
        
        oracle = FTNSOracle()
        
        # Mock all price fetchers to return None
        oracle._fetch_coingecko_price = AsyncMock(return_value=None)
        oracle._fetch_coinmarketcap_price = AsyncMock(return_value=None)
        oracle._fetch_dex_prices = AsyncMock(return_value=None)
        
        result = await oracle.get_oracle_price()
        
        # Should fall back to internal oracle
        assert result is not None
        assert result.source == "internal_oracle"
        assert result.price_usd == Decimal("0.10")


# =============================================================================
# Test 8: FTNS oracle CoinMarketCap skipped without key
# =============================================================================
@pytest.mark.asyncio
async def test_ftns_oracle_coinmarketcap_skipped_without_key():
    """Test that _fetch_coinmarketcap_price() returns None when CMC_API_KEY is empty."""
    with patch.dict(os.environ, {"CMC_API_KEY": ""}, clear=False):
        # Ensure CMC_API_KEY is not set
        os.environ.pop("CMC_API_KEY", None)
        
        from prsm.economy.blockchain.ftns_oracle import FTNSOracle
        
        oracle = FTNSOracle()
        
        result = await oracle._fetch_coinmarketcap_price()
        
        # Should return None (skipped)
        assert result is None


# =============================================================================
# Test 9: FTNS oracle DEX skipped without token address
# =============================================================================
@pytest.mark.asyncio
async def test_ftns_oracle_dex_skipped_without_token_address():
    """Test that _fetch_dex_prices() returns None when FTNS_TOKEN_ADDRESS is unset."""
    with patch.dict(os.environ, {}, clear=False):
        # Ensure FTNS_TOKEN_ADDRESS is not set
        os.environ.pop("FTNS_TOKEN_ADDRESS", None)
        
        from prsm.economy.blockchain.ftns_oracle import FTNSOracle
        
        oracle = FTNSOracle()
        
        result = await oracle._fetch_dex_prices()
        
        # Should return None (skipped)
        assert result is None
