"""Bandwidth limiting using token bucket algorithm.

This module provides rate limiting for bandwidth control in PRSM nodes.
The token bucket algorithm allows for burst traffic up to the bucket size
while maintaining an average rate limit over time.
"""

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter for bandwidth control.
    
    The token bucket algorithm allows for burst traffic up to the bucket size
    while maintaining an average rate limit over time.
    
    How it works:
    - Tokens are added at a constant rate (rate_mbps)
    - Each byte of data consumes one token
    - If tokens are available, data passes immediately
    - If not enough tokens, the caller waits until enough accumulate
    - A rate of 0 means unlimited (no throttling)
    
    Example:
        >>> bucket = TokenBucket(rate_mbps=10.0)  # 10 Mbps limit
        >>> await bucket.consume(1024 * 1024)  # Consume 1MB, may wait
    """
    
    def __init__(self, rate_mbps: float, bucket_size_mb: Optional[float] = None):
        """Initialize token bucket.
        
        Args:
            rate_mbps: Maximum rate in megabits per second (0 = unlimited)
            bucket_size_mb: Bucket size in megabits (defaults to 1 second of traffic)
        
        Raises:
            ValueError: If rate is negative
        """
        if rate_mbps < 0:
            raise ValueError("Rate must be non-negative")
        
        self._rate_mbps = rate_mbps
        self._rate_bytes_per_sec = rate_mbps * 1024 * 1024 / 8  # Convert Mbps to bytes/sec
        
        # Bucket size defaults to 1 second of traffic at max rate
        # This allows for short bursts while maintaining average rate
        if rate_mbps > 0:
            bucket_size_bytes = (bucket_size_mb or rate_mbps) * 1024 * 1024 / 8
            self._max_tokens = bucket_size_bytes
            self._tokens = self._max_tokens  # Start with full bucket
        else:
            # Unlimited rate - use infinity
            self._max_tokens = float('inf')
            self._tokens = float('inf')
        
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
    
    @property
    def rate_mbps(self) -> float:
        """Current rate limit in Mbps."""
        return self._rate_mbps
    
    @property
    def max_tokens(self) -> float:
        """Maximum bucket capacity in bytes."""
        return self._max_tokens
    
    async def update_rate(self, rate_mbps: float) -> None:
        """Update the rate limit.
        
        This can be called at runtime to adjust the bandwidth limit.
        Existing tokens are preserved but capped to the new maximum.
        
        Args:
            rate_mbps: New rate limit in Mbps (0 = unlimited)
        
        Raises:
            ValueError: If rate is negative
        """
        if rate_mbps < 0:
            raise ValueError("Rate must be non-negative")
        
        async with self._lock:
            old_rate = self._rate_mbps
            self._rate_mbps = rate_mbps
            self._rate_bytes_per_sec = rate_mbps * 1024 * 1024 / 8
            
            if rate_mbps > 0:
                new_max = rate_mbps * 1024 * 1024 / 8
                self._max_tokens = new_max
                # Cap existing tokens to new max
                self._tokens = min(self._tokens, self._max_tokens)
            else:
                # Unlimited - set to infinity
                self._max_tokens = float('inf')
                self._tokens = float('inf')
            
            if old_rate != rate_mbps:
                logger.debug(
                    f"TokenBucket rate updated: {old_rate} Mbps -> {rate_mbps} Mbps"
                )
    
    async def consume(self, bytes_count: int) -> None:
        """Wait until we have enough tokens, then consume them.
        
        This method will block until enough tokens are available or
        the rate limit is disabled (rate = 0).
        
        Args:
            bytes_count: Number of bytes to consume
        """
        # Unlimited rate - no throttling needed
        if self._rate_mbps == 0:
            return
        
        if bytes_count <= 0:
            return
        
        async with self._lock:
            while True:
                # Refill tokens based on elapsed time
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(
                    self._max_tokens,
                    self._tokens + elapsed * self._rate_bytes_per_sec
                )
                self._last_refill = now
                
                # Check if we have enough tokens
                if self._tokens >= bytes_count:
                    self._tokens -= bytes_count
                    logger.debug(
                        f"Consumed {bytes_count} bytes, {self._tokens:.0f} tokens remaining"
                    )
                    return
                
                # Calculate wait time for enough tokens
                tokens_needed = bytes_count - self._tokens
                wait_time = tokens_needed / self._rate_bytes_per_sec
                
                logger.debug(
                    f"Throttling: need {tokens_needed:.0f} more bytes, "
                    f"waiting {wait_time:.3f}s"
                )
                
                # Release lock while waiting to allow other operations
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()
    
    @property
    def available_tokens(self) -> float:
        """Current available tokens in bytes.
        
        Note: This is a snapshot and may change immediately.
        """
        return self._tokens
    
    def try_consume(self, bytes_count: int) -> bool:
        """Non-blocking attempt to consume tokens.
        
        This is a best-effort method that checks if tokens are available
        without waiting. Useful for checking if throttling would occur.
        
        Args:
            bytes_count: Number of bytes to consume
        
        Returns:
            True if tokens were consumed, False if not enough available
        """
        if self._rate_mbps == 0:
            return True
        
        # Refill tokens (synchronous approximation)
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._max_tokens,
            self._tokens + elapsed * self._rate_bytes_per_sec
        )
        self._last_refill = now
        
        if self._tokens >= bytes_count:
            self._tokens -= bytes_count
            return True
        return False


class BandwidthLimiter:
    """Manages upload and download bandwidth limits.
    
    This class provides a unified interface for controlling both
    upload and download bandwidth using separate token buckets.
    
    Example:
        >>> limiter = BandwidthLimiter(upload_mbps=10.0, download_mbps=50.0)
        >>> await limiter.throttle_upload(1024 * 1024)  # Throttle 1MB upload
        >>> await limiter.update_limits(upload_mbps=20.0, download_mbps=100.0)
    """
    
    def __init__(self, upload_mbps: float = 0.0, download_mbps: float = 0.0):
        """Initialize bandwidth limiter.
        
        Args:
            upload_mbps: Upload limit in Mbps (0 = unlimited)
            download_mbps: Download limit in Mbps (0 = unlimited)
        """
        self._upload_bucket = TokenBucket(upload_mbps)
        self._download_bucket = TokenBucket(download_mbps)
        
        logger.info(
            f"BandwidthLimiter initialized: upload={upload_mbps} Mbps, "
            f"download={download_mbps} Mbps"
        )
    
    @property
    def upload_limit_mbps(self) -> float:
        """Current upload rate limit in Mbps."""
        return self._upload_bucket.rate_mbps
    
    @property
    def download_limit_mbps(self) -> float:
        """Current download rate limit in Mbps."""
        return self._download_bucket.rate_mbps
    
    async def update_limits(self, upload_mbps: float, download_mbps: float) -> None:
        """Update both upload and download limits.
        
        Args:
            upload_mbps: New upload limit in Mbps (0 = unlimited)
            download_mbps: New download limit in Mbps (0 = unlimited)
        """
        await self._upload_bucket.update_rate(upload_mbps)
        await self._download_bucket.update_rate(download_mbps)
        
        logger.info(
            f"Bandwidth limits updated: upload={upload_mbps} Mbps, "
            f"download={download_mbps} Mbps"
        )
    
    async def throttle_upload(self, bytes_count: int) -> None:
        """Throttle an upload operation.
        
        This method will block until the upload is allowed to proceed
        based on the current rate limit.
        
        Args:
            bytes_count: Number of bytes being uploaded
        """
        if self._upload_bucket.rate_mbps > 0:
            logger.debug(f"Throttling upload of {bytes_count} bytes")
        await self._upload_bucket.consume(bytes_count)
    
    async def throttle_download(self, bytes_count: int) -> None:
        """Throttle a download operation.
        
        This method will block until the download is allowed to proceed
        based on the current rate limit.
        
        Args:
            bytes_count: Number of bytes being downloaded
        """
        if self._download_bucket.rate_mbps > 0:
            logger.debug(f"Throttling download of {bytes_count} bytes")
        await self._download_bucket.consume(bytes_count)
    
    @property
    def upload_tokens_available(self) -> float:
        """Current available upload tokens in bytes."""
        return self._upload_bucket.available_tokens
    
    @property
    def download_tokens_available(self) -> float:
        """Current available download tokens in bytes."""
        return self._download_bucket.available_tokens
    
    def get_stats(self) -> dict:
        """Get current bandwidth limiter statistics.
        
        Returns:
            Dict with current limits and available tokens
        """
        return {
            "upload_limit_mbps": self.upload_limit_mbps,
            "download_limit_mbps": self.download_limit_mbps,
            "upload_tokens_available": self.upload_tokens_available,
            "download_tokens_available": self.download_tokens_available,
            "upload_bucket_max": self._upload_bucket.max_tokens,
            "download_bucket_max": self._download_bucket.max_tokens,
        }
