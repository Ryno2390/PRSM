"""
CHRONOS Staking Integration

Enables multi-currency staking through CHRONOS clearing protocol.
Mirrors the architecture described in README.md.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import uuid

from .models import (
    CHRONOSStakingRequest, StakingProgram, StakePosition, StakingAuction,
    StakingBid, AssetType, TransactionStatus, SwapRequest, SwapType,
    StakingProgramStatus
)
from .clearing_engine import ChronosEngine
from ..tokenomics.ftns_service import FTNSService


logger = logging.getLogger(__name__)


class UniversalStakingPlatform:
    """
    Open-source staking framework available to any company on PRSM.
    Mirrors the UniversalStakingPlatform class from README.md.
    """
    
    def __init__(
        self,
        chronos_engine: ChronosEngine,
        ftns_service: FTNSService
    ):
        self.chronos_engine = chronos_engine
        self.ftns_service = ftns_service
        
        # Active programs and positions
        self.staking_programs: Dict[str, StakingProgram] = {}
        self.stake_positions: Dict[str, StakePosition] = {}
        self.active_auctions: Dict[str, StakingAuction] = {}
        self.auction_bids: Dict[str, List[StakingBid]] = {}
        
        # Processing queues
        self.pending_stakes: Dict[str, CHRONOSStakingRequest] = {}
    
    async def create_staking_program(
        self,
        issuer_id: str,
        issuer_name: str,
        terms: Dict,
        collateral: Dict
    ) -> StakingProgram:
        """
        Create a new staking program (available to any company).
        Mirrors the create_staking_program method from README.md.
        """
        program = StakingProgram(
            issuer_id=issuer_id,
            issuer_name=issuer_name,
            program_name=terms["program_name"],
            description=terms["description"],
            target_raise=Decimal(str(terms["target_raise"])),
            min_stake=Decimal(str(terms["min_stake"])),
            max_stake=Decimal(str(terms.get("max_stake", 0))) if terms.get("max_stake") else None,
            duration_months=terms["duration_months"],
            base_apy=Decimal(str(terms["base_apy"])),
            risk_profile=terms["risk_profile"],
            auction_start=terms["auction_start"],
            auction_end=terms["auction_end"],
            auction_reserve_apy=Decimal(str(terms["auction_reserve_apy"])),
            collateral_amount=Decimal(str(collateral["amount"])),
            collateral_asset=AssetType(collateral["asset"]),
            insurance_coverage=Decimal(str(collateral.get("insurance", 0))) if collateral.get("insurance") else None
        )
        
        # Validate issuer credentials and collateral
        if await self._validate_issuer_credentials(issuer_id):
            if await self._verify_collateral_deposit(program):
                program.status = StakingProgramStatus.ACTIVE
                logger.info(f"Staking program created: {program.id} by {issuer_name}")
            else:
                logger.error(f"Collateral verification failed for program: {program.id}")
                raise ValueError("Insufficient or invalid collateral")
        else:
            logger.error(f"Issuer credentials invalid: {issuer_id}")
            raise ValueError("Invalid issuer credentials")
        
        self.staking_programs[program.id] = program
        return program
    
    async def stake_in_preferred_currency(
        self,
        program_id: str,
        staker_address: str,
        amount: Decimal,
        currency: AssetType,
        max_slippage: Decimal = Decimal("0.01")
    ) -> CHRONOSStakingRequest:
        """
        Stake in user's preferred currency, converted via CHRONOS.
        Mirrors the CHRONOSStakingInterface from README.md.
        """
        program = self.staking_programs.get(program_id)
        if not program:
            raise ValueError(f"Staking program not found: {program_id}")
        
        if program.status != StakingProgramStatus.ACTIVE:
            raise ValueError(f"Program not accepting stakes: {program.status}")
        
        # Create staking request
        staking_request = CHRONOSStakingRequest(
            program_id=program_id,
            staker_address=staker_address,
            stake_amount=amount,
            stake_currency=currency,
            max_slippage=max_slippage
        )
        
        self.pending_stakes[staking_request.id] = staking_request
        
        # Process the multi-currency staking
        asyncio.create_task(self._process_multi_currency_stake(staking_request))
        
        return staking_request
    
    async def _process_multi_currency_stake(self, request: CHRONOSStakingRequest):
        """Process multi-currency staking through CHRONOS."""
        try:
            request.status = TransactionStatus.VERIFYING
            
            if request.stake_currency == AssetType.FTNS:
                # Direct FTNS staking
                ftns_amount = request.stake_amount
                request.final_ftns_amount = ftns_amount
            else:
                # Convert via CHRONOS
                logger.info(f"Converting {request.stake_amount} {request.stake_currency} to FTNS")
                
                # Get conversion quote
                quote = await self.chronos_engine.get_quote(
                    request.stake_currency,
                    AssetType.FTNS,
                    request.stake_amount
                )
                
                if "error" in quote:
                    raise Exception(f"Quote failed: {quote['error']}")
                
                request.conversion_quote = quote
                
                # Create swap request
                swap_request = SwapRequest(
                    user_id=request.staker_address,
                    from_asset=request.stake_currency,
                    to_asset=AssetType.FTNS,
                    from_amount=request.stake_amount,
                    swap_type=self._get_swap_type(request.stake_currency, AssetType.FTNS),
                    max_slippage=request.max_slippage,
                    expires_at=datetime.utcnow() + timedelta(minutes=30)
                )
                
                # Execute swap through CHRONOS
                transaction = await self.chronos_engine.submit_swap_request(swap_request)
                request.swap_transaction_id = transaction.id
                
                # Wait for swap completion
                while transaction.status in [TransactionStatus.PENDING, TransactionStatus.VERIFYING, TransactionStatus.EXECUTING]:
                    await asyncio.sleep(1)
                    transaction = await self.chronos_engine.get_transaction_status(transaction.id)
                
                if transaction.status != TransactionStatus.COMPLETED:
                    raise Exception(f"Swap failed: {transaction.error_message}")
                
                request.final_ftns_amount = transaction.settlement.net_amount
            
            # Create stake position
            program = self.staking_programs[request.program_id]
            maturity_date = datetime.utcnow() + timedelta(days=program.duration_months * 30)
            
            stake_position = StakePosition(
                program_id=request.program_id,
                staker_address=request.staker_address,
                principal_amount=request.final_ftns_amount,
                guaranteed_apy=program.current_apy or program.base_apy,
                currency_preference=request.stake_currency,
                maturity_timestamp=maturity_date,
                current_owner=request.staker_address
            )
            
            self.stake_positions[stake_position.id] = stake_position
            request.stake_position_id = stake_position.id
            
            # Update program totals
            program.total_staked += request.final_ftns_amount
            
            request.status = TransactionStatus.COMPLETED
            logger.info(f"Multi-currency stake completed: {request.id}")
            
        except Exception as e:
            logger.error(f"Multi-currency stake failed {request.id}: {str(e)}")
            request.status = TransactionStatus.FAILED
    
    async def create_auction(self, program_id: str, auction_params: Dict) -> StakingAuction:
        """Create Treasury-style auction for staking program."""
        program = self.staking_programs.get(program_id)
        if not program:
            raise ValueError(f"Program not found: {program_id}")
        
        auction = StakingAuction(
            program_id=program_id,
            start_time=auction_params["start_time"],
            end_time=auction_params["end_time"],
            min_bid_apy=Decimal(str(auction_params["min_bid_apy"])),
            max_bid_apy=Decimal(str(auction_params["max_bid_apy"]))
        )
        
        self.active_auctions[auction.id] = auction
        self.auction_bids[auction.id] = []
        
        # Update program status
        program.status = StakingProgramStatus.AUCTION_PHASE
        
        logger.info(f"Auction created for program {program_id}: {auction.id}")
        return auction
    
    async def submit_auction_bid(
        self,
        auction_id: str,
        bidder_address: str,
        stake_amount: Decimal,
        bid_apy: Decimal,
        currency_preference: AssetType = AssetType.FTNS
    ) -> StakingBid:
        """Submit bid in Treasury-style auction."""
        auction = self.active_auctions.get(auction_id)
        if not auction:
            raise ValueError(f"Auction not found: {auction_id}")
        
        if datetime.utcnow() > auction.end_time:
            raise ValueError("Auction has ended")
        
        if bid_apy < auction.min_bid_apy or bid_apy > auction.max_bid_apy:
            raise ValueError(f"Bid APY must be between {auction.min_bid_apy}% and {auction.max_bid_apy}%")
        
        bid = StakingBid(
            auction_id=auction_id,
            bidder_address=bidder_address,
            stake_amount=stake_amount,
            bid_apy=bid_apy,
            currency_preference=currency_preference
        )
        
        self.auction_bids[auction_id].append(bid)
        
        # Update auction statistics
        auction.total_bids += stake_amount
        auction.highest_apy = max(auction.highest_apy, bid_apy)
        auction.lowest_apy = min(auction.lowest_apy, bid_apy)
        
        logger.info(f"Auction bid submitted: {bid.id} for {stake_amount} FTNS at {bid_apy}% APY")
        return bid
    
    async def settle_auction(self, auction_id: str) -> Dict:
        """Settle Treasury-style auction (Dutch auction mechanism)."""
        auction = self.active_auctions.get(auction_id)
        if not auction:
            raise ValueError(f"Auction not found: {auction_id}")
        
        if datetime.utcnow() < auction.end_time:
            raise ValueError("Auction has not ended yet")
        
        bids = self.auction_bids[auction_id]
        program = self.staking_programs[auction.program_id]
        
        # Sort bids by APY (ascending - lowest APY wins)
        sorted_bids = sorted(bids, key=lambda b: b.bid_apy)
        
        total_filled = Decimal("0")
        winning_apy = None
        filled_bids = []
        
        # Fill bids starting from lowest APY
        for bid in sorted_bids:
            if total_filled >= program.target_raise:
                break
            
            remaining_capacity = program.target_raise - total_filled
            fill_amount = min(bid.stake_amount, remaining_capacity)
            
            bid.is_filled = True
            bid.fill_amount = fill_amount
            bid.is_winning = True
            total_filled += fill_amount
            winning_apy = bid.bid_apy
            filled_bids.append(bid)
            
            # Create stake positions for winning bids
            await self._create_position_from_bid(bid, winning_apy)
        
        # Update auction results
        auction.winning_apy = winning_apy
        auction.total_funded = total_filled
        auction.is_successful = total_filled >= program.target_raise * Decimal("0.8")  # 80% threshold
        
        # Update program
        if auction.is_successful:
            program.status = StakingProgramStatus.FUNDED
            program.current_apy = winning_apy
            program.funded_at = datetime.utcnow()
            program.maturity_date = datetime.utcnow() + timedelta(days=program.duration_months * 30)
        else:
            program.status = StakingProgramStatus.CANCELLED
        
        logger.info(f"Auction settled: {auction_id}, Winning APY: {winning_apy}%, Funded: {total_filled}")
        
        return {
            "auction_id": auction_id,
            "winning_apy": str(winning_apy) if winning_apy else None,
            "total_funded": str(total_filled),
            "is_successful": auction.is_successful,
            "filled_bids": len(filled_bids),
            "total_bids": len(bids)
        }
    
    async def _create_position_from_bid(self, bid: StakingBid, winning_apy: Decimal):
        """Create stake position from winning auction bid."""
        auction = self.active_auctions[bid.auction_id]
        program = self.staking_programs[auction.program_id]
        
        maturity_date = datetime.utcnow() + timedelta(days=program.duration_months * 30)
        
        position = StakePosition(
            program_id=auction.program_id,
            staker_address=bid.bidder_address,
            principal_amount=bid.fill_amount,
            guaranteed_apy=winning_apy,
            currency_preference=bid.currency_preference,
            maturity_timestamp=maturity_date,
            current_owner=bid.bidder_address
        )
        
        self.stake_positions[position.id] = position
        program.total_staked += bid.fill_amount
    
    async def transfer_stake_position(
        self,
        position_id: str,
        from_address: str,
        to_address: str,
        price: Optional[Decimal] = None
    ) -> bool:
        """Transfer stake position on secondary market."""
        position = self.stake_positions.get(position_id)
        if not position:
            raise ValueError(f"Position not found: {position_id}")
        
        if position.current_owner != from_address:
            raise ValueError("Only position owner can transfer")
        
        if not position.is_transferable:
            raise ValueError("Position is not transferable")
        
        # Update ownership
        position.current_owner = to_address
        
        # Record transfer (in real implementation, this would be on-chain)
        logger.info(f"Position transferred: {position_id} from {from_address} to {to_address}")
        
        return True
    
    async def get_position_value(self, position_id: str) -> Dict:
        """Get current market value of stake position."""
        position = self.stake_positions.get(position_id)
        if not position:
            raise ValueError(f"Position not found: {position_id}")
        
        # Calculate accrued interest
        days_elapsed = (datetime.utcnow() - position.staked_at).days
        annual_rate = position.guaranteed_apy / 100
        daily_rate = annual_rate / 365
        accrued = position.principal_amount * (daily_rate * days_elapsed)
        
        # Calculate time value (discount for remaining time)
        days_remaining = (position.maturity_timestamp - datetime.utcnow()).days
        time_discount = Decimal("0.99") ** (days_remaining / 30)  # Small monthly discount
        
        current_value = (position.principal_amount + accrued) * time_discount
        
        return {
            "position_id": position_id,
            "principal_amount": str(position.principal_amount),
            "accrued_interest": str(accrued),
            "current_value": str(current_value),
            "guaranteed_apy": str(position.guaranteed_apy),
            "days_remaining": days_remaining,
            "maturity_timestamp": position.maturity_timestamp.isoformat()
        }
    
    def _get_swap_type(self, from_asset: AssetType, to_asset: AssetType) -> SwapType:
        """Get appropriate swap type for asset pair."""
        if from_asset == AssetType.BTC and to_asset == AssetType.FTNS:
            return SwapType.BTC_TO_FTNS
        elif from_asset == AssetType.USD and to_asset == AssetType.FTNS:
            return SwapType.USD_TO_FTNS
        elif from_asset == AssetType.FTNS and to_asset == AssetType.BTC:
            return SwapType.FTNS_TO_BTC
        elif from_asset == AssetType.FTNS and to_asset == AssetType.USD:
            return SwapType.FTNS_TO_USD
        elif from_asset == AssetType.BTC and to_asset == AssetType.USD:
            return SwapType.BTC_TO_USD
        elif from_asset == AssetType.USD and to_asset == AssetType.BTC:
            return SwapType.USD_TO_BTC
        else:
            raise ValueError(f"Unsupported swap pair: {from_asset} -> {to_asset}")
    
    async def _validate_issuer_credentials(self, issuer_id: str) -> bool:
        """Validate that issuer is authorized to create staking programs."""
        # In real implementation, this would check:
        # - KYC/AML status
        # - Previous program performance
        # - Regulatory compliance
        # - Credit rating
        
        logger.info(f"Validating issuer credentials: {issuer_id}")
        return True  # Mock validation
    
    async def _verify_collateral_deposit(self, program: StakingProgram) -> bool:
        """Verify that required collateral has been deposited."""
        # In real implementation, this would check blockchain transactions
        logger.info(f"Verifying collateral deposit: {program.collateral_amount} {program.collateral_asset}")
        return True  # Mock verification


class CHRONOSStakingInterface:
    """
    Interface layer for CHRONOS-integrated staking.
    Mirrors the CHRONOSStakingInterface from README.md.
    """
    
    def __init__(self, staking_platform: UniversalStakingPlatform):
        self.staking_platform = staking_platform
    
    async def stake_in_preferred_currency(
        self,
        amount: Decimal,
        currency: AssetType,
        program_id: str,
        staker_address: str
    ) -> CHRONOSStakingRequest:
        """
        Stake in preferred currency with automatic CHRONOS conversion.
        Mirrors the stake_in_preferred_currency method from README.md.
        """
        if currency != AssetType.FTNS:
            # Convert to FTNS via CHRONOS
            logger.info(f"Converting {amount} {currency} to FTNS for staking")
            return await self.staking_platform.stake_in_preferred_currency(
                program_id, staker_address, amount, currency
            )
        else:
            # Direct FTNS staking
            return await self.staking_platform.stake_in_preferred_currency(
                program_id, staker_address, amount, currency
            )