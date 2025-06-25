"""
Enhanced Anthropic Claude Client for PRSM
Production-ready implementation similar to enhanced_openai_client.py

🎯 PURPOSE IN PRSM:
Dedicated Claude client with PRSM-specific optimizations including:
- Cost tracking and budget management for Claude API
- Claude-specific features (system prompts, tool use, streaming)
- Comprehensive error handling and retry logic
- Integration with PRSM's safety monitoring and FTNS system
"""

import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, Any, Optional, List, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)

class ClaudeModel(Enum):
    """Available Claude models"""
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229" 
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_2_1 = "claude-2.1"
    CLAUDE_2_0 = "claude-2.0"
    CLAUDE_INSTANT = "claude-instant-1.2"

@dataclass
class ClaudeUsageStats:
    """Track Claude API usage for cost management"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_requests: int = 0
    total_cost: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    
    def add_request(self, input_tokens: int, output_tokens: int, 
                   cost: float, response_time: float, success: bool):
        """Add usage data from a request"""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_requests += 1
        self.total_cost += cost
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            
        # Update rolling average response time
        if self.total_requests == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_requests - 1) + response_time) 
                / self.total_requests
            )

@dataclass
class ClaudeRequest:
    """Claude API request configuration"""
    messages: List[Dict[str, str]]
    model: ClaudeModel = ClaudeModel.CLAUDE_3_SONNET
    max_tokens: int = 1000
    temperature: float = 0.7
    system: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    stream: bool = False
    stop_sequences: Optional[List[str]] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None

@dataclass 
class ClaudeResponse:
    """Claude API response"""
    content: str
    model: str
    usage: Dict[str, int]
    stop_reason: str
    success: bool
    error: Optional[str] = None
    response_time: float = 0.0
    cost: float = 0.0
    tool_calls: Optional[List[Dict[str, Any]]] = None

class EnhancedAnthropicClient:
    """
    Enhanced Anthropic Claude client with PRSM optimizations
    
    🚀 FEATURES:
    - Automatic cost tracking and budget management
    - Advanced retry logic with exponential backoff  
    - Streaming response support for real-time applications
    - Tool use integration for Claude's function calling
    - Rate limiting and quota management
    - Comprehensive error handling and logging
    - Integration with PRSM safety monitoring
    """
    
    # Claude pricing per 1K tokens (as of 2024)
    PRICING = {
        ClaudeModel.CLAUDE_3_OPUS: {"input": 0.015, "output": 0.075},
        ClaudeModel.CLAUDE_3_SONNET: {"input": 0.003, "output": 0.015},
        ClaudeModel.CLAUDE_3_HAIKU: {"input": 0.00025, "output": 0.00125},
        ClaudeModel.CLAUDE_2_1: {"input": 0.008, "output": 0.024},
        ClaudeModel.CLAUDE_2_0: {"input": 0.008, "output": 0.024},
        ClaudeModel.CLAUDE_INSTANT: {"input": 0.0008, "output": 0.0024}
    }
    
    def __init__(self, 
                 api_key: str,
                 budget_limit: Optional[float] = None,
                 requests_per_minute: int = 60,
                 max_retries: int = 3,
                 timeout: int = 60):
        """
        Initialize enhanced Claude client
        
        Args:
            api_key: Anthropic API key
            budget_limit: Maximum spending limit in USD
            requests_per_minute: Rate limiting
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1"
        self.budget_limit = budget_limit
        self.requests_per_minute = requests_per_minute
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Usage tracking
        self.usage_stats = ClaudeUsageStats()
        self.request_times: List[float] = []
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the client session"""
        if self._initialized:
            return
            
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers=self._get_headers()
        )
        
        # Test connectivity
        await self._test_connection()
        self._initialized = True
        logger.info("Enhanced Anthropic client initialized successfully")
    
    async def close(self):
        """Close the client session"""
        if self.session:
            await self.session.close()
        self._initialized = False
        
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "User-Agent": "PRSM-Enhanced-Client/1.0"
        }
    
    async def _test_connection(self):
        """Test API connectivity"""
        try:
            # Simple test request
            test_request = ClaudeRequest(
                messages=[{"role": "user", "content": "Hello"}],
                model=ClaudeModel.CLAUDE_3_HAIKU,
                max_tokens=10
            )
            await self._make_request(test_request)
            logger.info("Claude API connection test successful")
        except Exception as e:
            logger.warning(f"Claude API connection test failed: {e}")
    
    def _calculate_cost(self, usage: Dict[str, int], model: ClaudeModel) -> float:
        """Calculate request cost based on token usage"""
        if model not in self.PRICING:
            return 0.0
            
        pricing = self.PRICING[model]
        input_cost = (usage.get("input_tokens", 0) / 1000) * pricing["input"]
        output_cost = (usage.get("output_tokens", 0) / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _check_budget(self, estimated_cost: float) -> bool:
        """Check if request would exceed budget"""
        if self.budget_limit is None:
            return True
        return (self.usage_stats.total_cost + estimated_cost) <= self.budget_limit
    
    async def _rate_limit_check(self):
        """Implement rate limiting"""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [
            t for t in self.request_times 
            if current_time - t < 60
        ]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(current_time)
    
    async def _make_request(self, request: ClaudeRequest) -> ClaudeResponse:
        """Make request to Claude API with retry logic"""
        if not self._initialized:
            await self.initialize()
        
        # Rate limiting
        await self._rate_limit_check()
        
        # Estimate cost for budget check
        estimated_tokens = len(str(request.messages)) * 0.75  # Rough estimate
        estimated_cost = estimated_tokens / 1000 * self.PRICING[request.model]["input"]
        
        if not self._check_budget(estimated_cost):
            raise RuntimeError(f"Request would exceed budget limit of ${self.budget_limit}")
        
        # Prepare request payload
        payload = {
            "model": request.model.value,
            "max_tokens": request.max_tokens,
            "messages": request.messages,
            "temperature": request.temperature
        }
        
        if request.system:
            payload["system"] = request.system
        if request.tools:
            payload["tools"] = request.tools
        if request.stream:
            payload["stream"] = True
        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.top_k is not None:
            payload["top_k"] = request.top_k
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                async with self.session.post(
                    f"{self.base_url}/messages",
                    json=payload
                ) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Calculate actual cost
                        usage = data.get("usage", {})
                        cost = self._calculate_cost(usage, request.model)
                        
                        # Extract content
                        content = ""
                        tool_calls = None
                        if data.get("content"):
                            for item in data["content"]:
                                if item["type"] == "text":
                                    content += item["text"]
                                elif item["type"] == "tool_use":
                                    if tool_calls is None:
                                        tool_calls = []
                                    tool_calls.append(item)
                        
                        # Update usage stats
                        self.usage_stats.add_request(
                            usage.get("input_tokens", 0),
                            usage.get("output_tokens", 0),
                            cost,
                            response_time,
                            True
                        )
                        
                        return ClaudeResponse(
                            content=content,
                            model=data.get("model", request.model.value),
                            usage=usage,
                            stop_reason=data.get("stop_reason", ""),
                            success=True,
                            response_time=response_time,
                            cost=cost,
                            tool_calls=tool_calls
                        )
                    else:
                        error_data = await response.json()
                        error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status}")
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=error_msg
                        )
                        
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    logger.warning(f"Claude request failed, retrying in {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    # Update failed request stats
                    self.usage_stats.add_request(0, 0, 0, 0, False)
                    logger.error(f"Claude request failed after {self.max_retries} retries: {e}")
        
        return ClaudeResponse(
            content="",
            model=request.model.value,
            usage={},
            stop_reason="error",
            success=False,
            error=str(last_error)
        )
    
    async def complete(self, 
                      messages: List[Dict[str, str]],
                      model: ClaudeModel = ClaudeModel.CLAUDE_3_SONNET,
                      **kwargs) -> ClaudeResponse:
        """
        Simple completion interface
        
        Args:
            messages: List of conversation messages
            model: Claude model to use
            **kwargs: Additional request parameters
        """
        request = ClaudeRequest(
            messages=messages,
            model=model,
            **kwargs
        )
        return await self._make_request(request)
    
    async def stream_complete(self, 
                             messages: List[Dict[str, str]],
                             model: ClaudeModel = ClaudeModel.CLAUDE_3_SONNET,
                             **kwargs) -> AsyncIterator[str]:
        """
        Streaming completion interface
        
        Args:
            messages: List of conversation messages
            model: Claude model to use
            **kwargs: Additional request parameters
        
        Yields:
            Streaming response content chunks
        """
        request = ClaudeRequest(
            messages=messages,
            model=model,
            stream=True,
            **kwargs
        )
        
        # Note: Streaming implementation would require SSE parsing
        # For now, fall back to regular completion
        response = await self._make_request(request)
        if response.success:
            yield response.content
        else:
            raise RuntimeError(f"Streaming request failed: {response.error}")
    
    def get_usage_stats(self) -> ClaudeUsageStats:
        """Get current usage statistics"""
        return self.usage_stats
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get budget utilization status"""
        if self.budget_limit is None:
            return {
                "budget_limit": None,
                "total_spent": self.usage_stats.total_cost,
                "remaining": "unlimited",
                "utilization": 0.0
            }
        
        remaining = self.budget_limit - self.usage_stats.total_cost
        utilization = (self.usage_stats.total_cost / self.budget_limit) * 100
        
        return {
            "budget_limit": self.budget_limit,
            "total_spent": self.usage_stats.total_cost,
            "remaining": remaining,
            "utilization": utilization
        }
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

# Integration with existing PRSM systems
class PRSMClaudeIntegration:
    """
    Integration layer for Claude client with PRSM systems
    
    🔗 INTEGRATIONS:
    - FTNS cost tracking and token rewards
    - Safety monitoring and content filtering
    - Performance metrics and optimization
    - Multi-provider routing and fallback
    """
    
    def __init__(self, claude_client: EnhancedAnthropicClient):
        self.claude = claude_client
        
    async def safe_complete(self, 
                           messages: List[Dict[str, str]], 
                           **kwargs) -> ClaudeResponse:
        """
        Complete with safety monitoring integration
        """
        # Pre-processing: Safety check inputs
        # (Integration point for PRSM safety monitoring)
        
        response = await self.claude.complete(messages, **kwargs)
        
        # Post-processing: Safety check outputs
        # (Integration point for content filtering)
        
        # FTNS integration: Track costs and rewards
        # (Integration point for token economics)
        
        return response

# Example usage for PRSM integration
async def example_usage():
    """Example of enhanced Claude client usage in PRSM"""
    
    # Initialize with budget management
    async with EnhancedAnthropicClient(
        api_key="your-api-key",
        budget_limit=100.0,  # $100 budget limit
        requests_per_minute=30  # Conservative rate limiting
    ) as claude:
        
        # Simple completion
        response = await claude.complete([
            {"role": "user", "content": "Explain quantum computing in simple terms"}
        ], model=ClaudeModel.CLAUDE_3_SONNET)
        
        if response.success:
            print(f"Response: {response.content}")
            print(f"Cost: ${response.cost:.4f}")
        
        # Check usage
        stats = claude.get_usage_stats()
        print(f"Total requests: {stats.total_requests}")
        print(f"Total cost: ${stats.total_cost:.4f}")
        print(f"Success rate: {stats.successful_requests/stats.total_requests*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(example_usage())