"""
CHRONOS Staking Integration Example

Demonstrates how any company can use PRSM's staking infrastructure
with multi-currency support through CHRONOS.

This example shows the complete flow from README.md:
1. Company creates staking program
2. Users stake in their preferred currency
3. CHRONOS handles automatic conversion
4. Treasury-style auctions determine rates
5. Secondary market trading
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from prsm.chronos import (
    ChronosEngine, UniversalStakingPlatform, CHRONOSStakingInterface,
    AssetType
)
from prsm.chronos.models import StakingProgramStatus


async def main():
    """Demonstrate complete staking workflow."""
    
    print("🚀 CHRONOS Staking Integration Demo")
    print("=" * 50)
    
    # Initialize CHRONOS infrastructure
    # (In real implementation, these would be properly configured)
    chronos_engine = None  # Would be initialized with real components
    staking_platform = UniversalStakingPlatform(chronos_engine, None)
    chronos_interface = CHRONOSStakingInterface(staking_platform)
    
    # Example 1: University creates research funding program
    print("\n📚 Example 1: University Research Funding")
    print("-" * 40)
    
    university_program = await staking_platform.create_staking_program(
        issuer_id="duke_university",
        issuer_name="Duke University",
        terms={
            "program_name": "Quantum Computing Research Fund",
            "description": "3-year quantum computing research program",
            "target_raise": 1000000,  # 1M FTNS
            "min_stake": 1000,  # 1K FTNS minimum
            "max_stake": 100000,  # 100K FTNS maximum
            "duration_months": 36,  # 3 years
            "base_apy": 8.5,  # 8.5% base APY
            "risk_profile": "growth",
            "auction_start": datetime.utcnow() + timedelta(days=7),
            "auction_end": datetime.utcnow() + timedelta(days=14),
            "auction_reserve_apy": 6.0  # Won't accept below 6%
        },
        collateral={
            "amount": 500000,  # 500K FTNS collateral
            "asset": "FTNS",
            "insurance": 250000  # 250K insurance coverage
        }
    )
    
    print(f"✅ Created program: {university_program.program_name}")
    print(f"   Target raise: {university_program.target_raise} FTNS")
    print(f"   Duration: {university_program.duration_months} months")
    print(f"   Base APY: {university_program.base_apy}%")
    
    # Example 2: Multi-currency staking
    print("\n💰 Example 2: Multi-Currency Staking")
    print("-" * 40)
    
    # User stakes $10K USD (converted to FTNS)
    usd_staking_request = await chronos_interface.stake_in_preferred_currency(
        amount=Decimal("10000"),
        currency=AssetType.USD,
        program_id=university_program.id,
        staker_address="0x1234...investor1"
    )
    
    print(f"✅ USD staking request created: {usd_staking_request.id}")
    print(f"   Staking: ${usd_staking_request.stake_amount} USD")
    print(f"   Status: {usd_staking_request.status}")
    
    # User stakes 0.5 BTC (converted to FTNS)
    btc_staking_request = await chronos_interface.stake_in_preferred_currency(
        amount=Decimal("0.5"),
        currency=AssetType.BTC,
        program_id=university_program.id,
        staker_address="0x5678...investor2"
    )
    
    print(f"✅ BTC staking request created: {btc_staking_request.id}")
    print(f"   Staking: {btc_staking_request.stake_amount} BTC")
    print(f"   Status: {btc_staking_request.status}")
    
    # Example 3: Treasury-style auction
    print("\n🏛️ Example 3: Treasury-Style Auction")
    print("-" * 40)
    
    auction = await staking_platform.create_auction(
        program_id=university_program.id,
        auction_params={
            "start_time": datetime.utcnow(),
            "end_time": datetime.utcnow() + timedelta(hours=24),
            "min_bid_apy": 6.0,  # 6% minimum
            "max_bid_apy": 12.0  # 12% maximum
        }
    )
    
    print(f"✅ Auction created: {auction.id}")
    print(f"   APY range: {auction.min_bid_apy}% - {auction.max_bid_apy}%")
    print(f"   Duration: 24 hours")
    
    # Simulate auction bids
    bids = [
        {"address": "0xAAA1", "amount": 50000, "apy": 7.5},
        {"address": "0xBBB2", "amount": 75000, "apy": 8.0},
        {"address": "0xCCC3", "amount": 100000, "apy": 6.5},
        {"address": "0xDDD4", "amount": 150000, "apy": 9.0},
        {"address": "0xEEE5", "amount": 200000, "apy": 7.0},
    ]
    
    for bid_info in bids:
        bid = await staking_platform.submit_auction_bid(
            auction_id=auction.id,
            bidder_address=bid_info["address"],
            stake_amount=Decimal(str(bid_info["amount"])),
            bid_apy=Decimal(str(bid_info["apy"])),
            currency_preference=AssetType.FTNS
        )
        print(f"   📝 Bid: {bid_info['amount']} FTNS at {bid_info['apy']}% APY")
    
    # Simulate auction settlement (fast-forward time)
    print("\n⏰ Fast-forwarding 24 hours...")
    auction.end_time = datetime.utcnow() - timedelta(seconds=1)  # Mock ended
    
    settlement_result = await staking_platform.settle_auction(auction.id)
    
    print(f"✅ Auction settled!")
    print(f"   Winning APY: {settlement_result['winning_apy']}%")
    print(f"   Total funded: {settlement_result['total_funded']} FTNS")
    print(f"   Success: {settlement_result['is_successful']}")
    print(f"   Filled bids: {settlement_result['filled_bids']}/{settlement_result['total_bids']}")
    
    # Example 4: Secondary market trading
    print("\n🔄 Example 4: Secondary Market Trading")
    print("-" * 40)
    
    # Get first position for trading demo
    positions = [pos for pos in staking_platform.stake_positions.values() 
                if pos.is_transferable]
    
    if positions:
        position = positions[0]
        
        # Get current market value
        position_value = await staking_platform.get_position_value(position.id)
        
        print(f"📊 Position valuation:")
        print(f"   Position ID: {position.id}")
        print(f"   Principal: {position_value['principal_amount']} FTNS")
        print(f"   Accrued interest: {position_value['accrued_interest']} FTNS")
        print(f"   Current value: {position_value['current_value']} FTNS")
        print(f"   Days remaining: {position_value['days_remaining']}")
        
        # Transfer position (secondary market sale)
        transfer_success = await staking_platform.transfer_stake_position(
            position_id=position.id,
            from_address=position.current_owner,
            to_address="0x9999...buyer",
            price=Decimal(position_value['current_value'])
        )
        
        print(f"✅ Position transferred: {transfer_success}")
        print(f"   New owner: 0x9999...buyer")
    
    # Example 5: Clean Energy Company Program
    print("\n🌱 Example 5: Clean Energy Company")
    print("-" * 40)
    
    clean_energy_program = await staking_platform.create_staking_program(
        issuer_id="solar_innovations_inc",
        issuer_name="Solar Innovations Inc",
        terms={
            "program_name": "Next-Gen Solar Panel Development",
            "description": "Revolutionary perovskite-silicon tandem solar cells",
            "target_raise": 5000000,  # 5M FTNS
            "min_stake": 5000,  # 5K FTNS minimum
            "duration_months": 60,  # 5 years
            "base_apy": 12.0,  # 12% base APY (higher risk)
            "risk_profile": "moonshot",
            "auction_start": datetime.utcnow() + timedelta(days=30),
            "auction_end": datetime.utcnow() + timedelta(days=37),
            "auction_reserve_apy": 10.0
        },
        collateral={
            "amount": 2000000,  # 2M FTNS collateral
            "asset": "FTNS",
            "insurance": 1000000
        }
    )
    
    print(f"✅ Created program: {clean_energy_program.program_name}")
    print(f"   Target raise: {clean_energy_program.target_raise} FTNS")
    print(f"   Risk profile: {clean_energy_program.risk_profile}")
    print(f"   Base APY: {clean_energy_program.base_apy}%")
    
    print("\n🎉 Demo completed!")
    print("\nThis demonstrates how:")
    print("• Any company can create staking programs")
    print("• Users can stake in their preferred currency")
    print("• CHRONOS handles automatic conversions")
    print("• Treasury auctions determine competitive rates")
    print("• Secondary markets provide liquidity")
    print("• Multiple programs create diversified opportunities")


if __name__ == "__main__":
    # Note: This is a demonstration script
    # In real implementation, proper async setup would be needed
    print("CHRONOS Staking Integration Example")
    print("This script demonstrates the architecture described in README.md")
    print("In real implementation, this would connect to actual CHRONOS infrastructure")