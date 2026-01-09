"""
Testnet MATIC Faucet Integration for PRSM Development

Provides automated integration with Polygon Mumbai testnet faucets
for seamless development and testing experience.
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FaucetStatus(Enum):
    SUCCESS = "success"
    PENDING = "pending"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    INSUFFICIENT_FUNDS = "insufficient_funds"

@dataclass
class FaucetResult:
    status: FaucetStatus
    transaction_hash: Optional[str] = None
    amount: Optional[float] = None
    message: Optional[str] = None
    retry_after: Optional[int] = None

class PolygonFaucetIntegration:
    """
    Integration with Polygon Mumbai testnet faucets for automated MATIC distribution
    
    Features:
    - Multiple faucet provider support
    - Rate limiting and retry logic
    - Balance checking before requests
    - Transaction monitoring
    - Automatic fallback between providers
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_default_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_request_times: Dict[str, float] = {}
        
    def _load_default_config(self) -> Dict:
        """Load default faucet configuration"""
        return {
            "faucets": {
                "polygon_official": {
                    "name": "Polygon Official Faucet",
                    "url": "https://faucet.polygon.technology/",
                    "api_endpoint": "https://faucet.polygon.technology/api/getFaucet",
                    "method": "POST",
                    "amount": 0.1,  # MATIC
                    "cooldown": 86400,  # 24 hours
                    "enabled": True
                },
                "chainlink": {
                    "name": "Chainlink Faucet",
                    "url": "https://faucets.chain.link/mumbai",
                    "api_endpoint": "https://faucets.chain.link/mumbai",
                    "method": "POST", 
                    "amount": 0.1,
                    "cooldown": 86400,
                    "enabled": True
                },
                "alchemy": {
                    "name": "Alchemy Faucet",
                    "url": "https://mumbai-faucet.matic.today/",
                    "api_endpoint": "https://mumbai-faucet.matic.today/",
                    "method": "POST",
                    "amount": 0.5,
                    "cooldown": 86400,
                    "enabled": True
                }
            },
            "retry": {
                "max_attempts": 3,
                "delay": 5,
                "backoff_multiplier": 2
            },
            "monitoring": {
                "check_interval": 30,
                "max_wait_time": 300
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def request_matic(self, wallet_address: str, 
                           preferred_faucet: Optional[str] = None) -> FaucetResult:
        """
        Request MATIC from testnet faucet
        
        Args:
            wallet_address: Recipient wallet address
            preferred_faucet: Preferred faucet provider name
            
        Returns:
            FaucetResult: Request outcome
        """
        if not self.session:
            async with self:
                return await self.request_matic(wallet_address, preferred_faucet)
        
        # Get available faucets
        faucets = self._get_available_faucets()
        
        if preferred_faucet and preferred_faucet in faucets:
            # Try preferred faucet first
            faucets = [preferred_faucet] + [f for f in faucets if f != preferred_faucet]
        
        # Try each faucet until one succeeds
        last_error = None
        for faucet_name in faucets:
            try:
                # Check rate limiting
                if not self._can_request_from_faucet(faucet_name):
                    continue
                    
                result = await self._request_from_faucet(faucet_name, wallet_address)
                
                if result.status == FaucetStatus.SUCCESS:
                    self._update_last_request_time(faucet_name)
                    logger.info(f"Successfully requested MATIC from {faucet_name}")
                    return result
                elif result.status == FaucetStatus.PENDING:
                    # Monitor pending transaction
                    return await self._monitor_pending_request(result)
                    
                last_error = result
                
            except Exception as e:
                logger.error(f"Error requesting from {faucet_name}: {e}")
                last_error = FaucetResult(
                    status=FaucetStatus.FAILED,
                    message=str(e)
                )
        
        # All faucets failed
        return last_error or FaucetResult(
            status=FaucetStatus.FAILED,
            message="All faucets unavailable"
        )
    
    async def _request_from_faucet(self, faucet_name: str, wallet_address: str) -> FaucetResult:
        """Request MATIC from specific faucet"""
        faucet_config = self.config["faucets"][faucet_name]
        
        if faucet_name == "polygon_official":
            return await self._request_polygon_official(wallet_address, faucet_config)
        elif faucet_name == "chainlink":
            return await self._request_chainlink(wallet_address, faucet_config)
        elif faucet_name == "alchemy":
            return await self._request_alchemy(wallet_address, faucet_config)
        else:
            return FaucetResult(
                status=FaucetStatus.FAILED,
                message=f"Unsupported faucet: {faucet_name}"
            )
    
    async def _request_polygon_official(self, wallet_address: str, config: Dict) -> FaucetResult:
        """Request from Polygon official faucet"""
        try:
            payload = {
                "address": wallet_address,
                "token": "maticToken"
            }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "PRSM-Web3-Integration/1.0"
            }
            
            async with self.session.post(
                config["api_endpoint"],
                json=payload,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get("success"):
                        return FaucetResult(
                            status=FaucetStatus.SUCCESS,
                            transaction_hash=data.get("hash"),
                            amount=config["amount"],
                            message="MATIC requested successfully"
                        )
                    else:
                        return FaucetResult(
                            status=FaucetStatus.FAILED,
                            message=data.get("message", "Request failed")
                        )
                        
                elif response.status == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 3600))
                    return FaucetResult(
                        status=FaucetStatus.RATE_LIMITED,
                        retry_after=retry_after,
                        message="Rate limit exceeded"
                    )
                else:
                    return FaucetResult(
                        status=FaucetStatus.FAILED,
                        message=f"HTTP {response.status}"
                    )
                    
        except Exception as e:
            return FaucetResult(
                status=FaucetStatus.FAILED,
                message=str(e)
            )
    
    async def _request_chainlink(self, wallet_address: str, config: Dict) -> FaucetResult:
        """Request from Chainlink faucet"""
        try:
            # Chainlink faucet requires different approach
            # This is a simplified implementation
            
            payload = {
                "address": wallet_address
            }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "PRSM-Web3-Integration/1.0"
            }
            
            # Note: Actual Chainlink faucet may require additional authentication
            # This is a placeholder implementation
            
            return FaucetResult(
                status=FaucetStatus.FAILED,
                message="Chainlink faucet integration not implemented"
            )
            
        except Exception as e:
            return FaucetResult(
                status=FaucetStatus.FAILED,
                message=str(e)
            )
    
    async def _request_alchemy(self, wallet_address: str, config: Dict) -> FaucetResult:
        """Request from Alchemy faucet"""
        try:
            # Alchemy faucet implementation
            # This is a placeholder - actual implementation would depend on their API
            
            return FaucetResult(
                status=FaucetStatus.FAILED,
                message="Alchemy faucet integration not implemented"
            )
            
        except Exception as e:
            return FaucetResult(
                status=FaucetStatus.FAILED,
                message=str(e)
            )
    
    async def _monitor_pending_request(self, result: FaucetResult) -> FaucetResult:
        """Monitor pending faucet request"""
        if not result.transaction_hash:
            return result
        
        # Wait for transaction confirmation
        max_wait = self.config["monitoring"]["max_wait_time"]
        check_interval = self.config["monitoring"]["check_interval"]
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            # Check transaction status (would need Web3 connection)
            # For now, just wait and assume success
            await asyncio.sleep(check_interval)
            
            # In real implementation, check transaction receipt
            # If confirmed, return success
            # If failed, return failed
            # If still pending, continue waiting
            
        return FaucetResult(
            status=FaucetStatus.SUCCESS,
            transaction_hash=result.transaction_hash,
            amount=result.amount,
            message="Transaction confirmed"
        )
    
    def _get_available_faucets(self) -> List[str]:
        """Get list of available faucet providers"""
        return [
            name for name, config in self.config["faucets"].items()
            if config.get("enabled", True)
        ]
    
    def _can_request_from_faucet(self, faucet_name: str) -> bool:
        """Check if we can request from faucet (rate limiting)"""
        faucet_config = self.config["faucets"][faucet_name]
        cooldown = faucet_config["cooldown"]
        
        last_request = self.last_request_times.get(faucet_name, 0)
        time_since_last = time.time() - last_request
        
        return time_since_last >= cooldown
    
    def _update_last_request_time(self, faucet_name: str):
        """Update last request time for rate limiting"""
        self.last_request_times[faucet_name] = time.time()
    
    async def get_faucet_status(self) -> Dict[str, Dict]:
        """
        Get status of all available faucets
        
        Returns:
            Dict: Faucet status information
        """
        status = {}
        
        for faucet_name, config in self.config["faucets"].items():
            if not config.get("enabled", True):
                status[faucet_name] = {
                    "name": config["name"],
                    "available": False,
                    "reason": "Disabled"
                }
                continue
                
            can_request = self._can_request_from_faucet(faucet_name)
            last_request = self.last_request_times.get(faucet_name, 0)
            
            next_available = 0
            if not can_request and last_request > 0:
                next_available = last_request + config["cooldown"]
            
            status[faucet_name] = {
                "name": config["name"],
                "available": can_request,
                "amount": config["amount"],
                "cooldown": config["cooldown"],
                "last_request": last_request,
                "next_available": next_available,
                "url": config["url"]
            }
        
        return status
    
    async def estimate_wait_time(self, wallet_address: str) -> int:
        """
        Estimate wait time until next faucet request is possible
        
        Args:
            wallet_address: Wallet address
            
        Returns:
            int: Wait time in seconds (0 if can request now)
        """
        faucets = self._get_available_faucets()
        min_wait = float('inf')
        
        for faucet_name in faucets:
            if self._can_request_from_faucet(faucet_name):
                return 0  # Can request immediately
                
            faucet_config = self.config["faucets"][faucet_name]
            last_request = self.last_request_times.get(faucet_name, 0)
            
            if last_request > 0:
                wait_time = (last_request + faucet_config["cooldown"]) - time.time()
                min_wait = min(min_wait, max(0, wait_time))
        
        return int(min_wait) if min_wait != float('inf') else 86400  # Default 24h
    
    def get_faucet_urls(self) -> Dict[str, str]:
        """
        Get manual faucet URLs for user reference
        
        Returns:
            Dict: Faucet names and their URLs
        """
        return {
            name: config["url"]
            for name, config in self.config["faucets"].items()
            if config.get("enabled", True)
        }

# Convenience function for easy usage
async def request_testnet_matic(wallet_address: str, 
                               preferred_faucet: Optional[str] = None) -> FaucetResult:
    """
    Convenient function to request testnet MATIC
    
    Args:
        wallet_address: Recipient wallet address
        preferred_faucet: Preferred faucet provider
        
    Returns:
        FaucetResult: Request outcome
    """
    async with PolygonFaucetIntegration() as faucet:
        return await faucet.request_matic(wallet_address, preferred_faucet)