"""
PRSM SDK FTNS Token Manager
Handles FTNS token operations including balance, transfers, and staking
"""

import structlog
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from .exceptions import InsufficientFundsError, PRSMError

logger = structlog.get_logger(__name__)


class FTNSBalance(BaseModel):
    """FTNS token balance information"""
    total_balance: float = Field(..., description="Total FTNS balance")
    available_balance: float = Field(..., description="Available for spending")
    reserved_balance: float = Field(..., description="Reserved for pending operations")
    staked_balance: float = Field(0.0, description="Staked for network participation")
    earned_today: float = Field(0.0, description="FTNS earned today")
    spent_today: float = Field(0.0, description="FTNS spent today")
    last_updated: datetime = Field(..., description="Last balance update")


class Transaction(BaseModel):
    """FTNS transaction record"""
    transaction_id: str = Field(..., description="Unique transaction ID")
    transaction_type: str = Field(..., description="Type: transfer, stake, unstake, reward")
    amount: float = Field(..., description="Transaction amount in FTNS")
    from_address: Optional[str] = Field(None, description="Sender address")
    to_address: Optional[str] = Field(None, description="Recipient address")
    status: str = Field(..., description="Status: pending, completed, failed")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    memo: Optional[str] = Field(None, description="Transaction memo")
    block_number: Optional[int] = Field(None, description="Block number")


class StakeInfo(BaseModel):
    """Staking information"""
    staked_amount: float = Field(..., description="Currently staked amount")
    rewards_earned: float = Field(..., description="Total rewards earned")
    lock_period: int = Field(..., description="Lock period in days")
    apy: float = Field(..., description="Annual percentage yield")
    unlock_date: Optional[datetime] = Field(None, description="When stake unlocks")


class TransferRequest(BaseModel):
    """Transfer request parameters"""
    to_address: str = Field(..., description="Recipient address")
    amount: float = Field(..., gt=0, description="Amount to transfer")
    memo: Optional[str] = Field(None, description="Optional memo")


class TransferResponse(BaseModel):
    """Transfer response"""
    transaction_id: str = Field(..., description="Transaction ID")
    status: str = Field(..., description="Transaction status")
    amount: float = Field(..., description="Transferred amount")
    fee: float = Field(..., description="Transaction fee")
    timestamp: datetime = Field(..., description="Transaction timestamp")


class FTNSManager:
    """
    Manages FTNS token operations
    
    Provides methods for:
    - Checking balances
    - Transferring tokens
    - Staking/unstaking
    - Transaction history
    """
    
    def __init__(self, client):
        """
        Initialize FTNS manager
        
        Args:
            client: PRSMClient instance for making API requests
        """
        self._client = client
    
    async def get_balance(self) -> FTNSBalance:
        """
        Get current FTNS balance
        
        Returns:
            FTNSBalance with current balance information
            
        Example:
            balance = await client.ftns.get_balance()
            print(f"Available: {balance.available_balance} FTNS")
        """
        response = await self._client._request("GET", "/ftns/balance")
        return FTNSBalance(**response)
    
    async def transfer(
        self,
        to_address: str,
        amount: float,
        memo: Optional[str] = None
    ) -> TransferResponse:
        """
        Transfer FTNS tokens to another address
        
        Args:
            to_address: Recipient address
            amount: Amount to transfer
            memo: Optional memo for the transaction
            
        Returns:
            TransferResponse with transaction details
            
        Raises:
            InsufficientFundsError: If balance is insufficient
            
        Example:
            result = await client.ftns.transfer(
                to_address="0x123...",
                amount=10.5,
                memo="Payment for services"
            )
        """
        # Check balance first
        balance = await self.get_balance()
        if balance.available_balance < amount:
            raise InsufficientFundsError(amount, balance.available_balance)
        
        request = TransferRequest(
            to_address=to_address,
            amount=amount,
            memo=memo
        )
        
        response = await self._client._request(
            "POST",
            "/ftns/transfer",
            json_data=request.model_dump()
        )
        
        return TransferResponse(**response)
    
    async def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        transaction_type: Optional[str] = None
    ) -> List[Transaction]:
        """
        Get transaction history
        
        Args:
            limit: Maximum number of transactions to return
            offset: Offset for pagination
            transaction_type: Filter by type (transfer, stake, unstake, reward)
            
        Returns:
            List of Transaction objects
            
        Example:
            history = await client.ftns.get_history(limit=10)
            for tx in history:
                print(f"{tx.transaction_type}: {tx.amount} FTNS")
        """
        params = {"limit": limit, "offset": offset}
        if transaction_type:
            params["type"] = transaction_type
        
        response = await self._client._request(
            "GET",
            "/ftns/history",
            params=params
        )
        
        return [Transaction(**tx) for tx in response.get("transactions", [])]
    
    async def stake(self, amount: float, lock_period: int = 30) -> StakeInfo:
        """
        Stake FTNS tokens for network participation
        
        Args:
            amount: Amount to stake
            lock_period: Lock period in days (default: 30)
            
        Returns:
            StakeInfo with staking details
            
        Raises:
            InsufficientFundsError: If balance is insufficient
            
        Example:
            stake_info = await client.ftns.stake(100, lock_period=60)
            print(f"Staked {amount} FTNS at {stake_info.apy}% APY")
        """
        balance = await self.get_balance()
        if balance.available_balance < amount:
            raise InsufficientFundsError(amount, balance.available_balance)
        
        response = await self._client._request(
            "POST",
            "/ftns/stake",
            json_data={"amount": amount, "lock_period": lock_period}
        )
        
        return StakeInfo(**response)
    
    async def unstake(self, amount: Optional[float] = None) -> StakeInfo:
        """
        Unstake FTNS tokens
        
        Args:
            amount: Amount to unstake (if None, unstake all)
            
        Returns:
            StakeInfo with updated staking details
            
        Example:
            stake_info = await client.ftns.unstake(50)
        """
        json_data = {}
        if amount is not None:
            json_data["amount"] = amount
        
        response = await self._client._request(
            "POST",
            "/ftns/unstake",
            json_data=json_data
        )
        
        return StakeInfo(**response)
    
    async def get_stake_info(self) -> StakeInfo:
        """
        Get current staking information
        
        Returns:
            StakeInfo with current staking details
        """
        response = await self._client._request("GET", "/ftns/stake/info")
        return StakeInfo(**response)
    
    async def estimate_transfer_fee(self, amount: float) -> float:
        """
        Estimate the fee for a transfer
        
        Args:
            amount: Transfer amount
            
        Returns:
            Estimated fee in FTNS
        """
        response = await self._client._request(
            "POST",
            "/ftns/estimate-fee",
            json_data={"amount": amount}
        )
        return response.get("fee", 0.0)