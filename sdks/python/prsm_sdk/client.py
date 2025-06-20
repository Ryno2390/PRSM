"""
PRSM Python SDK Client
Main client for interacting with PRSM API
"""

import asyncio
import aiohttp
import structlog
from typing import Dict, Any, Optional, List, Union, AsyncIterator
from contextlib import asynccontextmanager

from .auth import AuthManager
from .ftns import FTNSManager
from .marketplace import ModelMarketplace
from .tools import ToolExecutor
from .models import (
    PRSMResponse, QueryRequest, FTNSBalance, ModelInfo, 
    ToolSpec, SafetyStatus, WebSocketMessage
)
from .exceptions import PRSMError, NetworkError, AuthenticationError
from .websocket import WebSocketClient

logger = structlog.get_logger(__name__)


class PRSMClient:
    """
    Main PRSM client for AI queries, token management, and marketplace access
    
    🎯 PURPOSE:
    Provides a simple, intuitive interface to PRSM's distributed AI infrastructure
    while handling authentication, cost management, and safety monitoring.
    
    🚀 BASIC USAGE:
        client = PRSMClient(api_key="your_key")
        response = await client.query("Explain machine learning")
        print(response.content)
    
    🔧 ADVANCED FEATURES:
        # Stream responses for real-time output
        async for chunk in client.stream("Write a story"):
            print(chunk.content, end="")
            
        # Check FTNS balance and costs
        balance = await client.ftns.get_balance()
        cost = await client.estimate_cost("Complex query")
        
        # Use marketplace models
        models = await client.marketplace.search_models("gpt")
        response = await client.query("Hello", model_id=models[0].id)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.prsm.ai/v1",
        websocket_url: str = "wss://ws.prsm.ai/v1",
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize PRSM client
        
        Args:
            api_key: PRSM API key (can also be set via PRSM_API_KEY env var)
            base_url: Base URL for PRSM API 
            websocket_url: WebSocket URL for streaming
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            **kwargs: Additional configuration options
        """
        self.base_url = base_url.rstrip("/")
        self.websocket_url = websocket_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.config = kwargs
        
        # Initialize managers
        self.auth = AuthManager(api_key)
        self.ftns = FTNSManager(self)
        self.marketplace = ModelMarketplace(self)
        self.tools = ToolExecutor(self)
        
        # HTTP session will be created on first use
        self._session: Optional[aiohttp.ClientSession] = None
        self._websocket: Optional[WebSocketClient] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = await self.auth.get_headers()
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                connector=aiohttp.TCPConnector(limit=100)
            )
        return self._session
    
    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries + 1):
            try:
                async with session.request(
                    method, url, json=json_data, params=params, **kwargs
                ) as response:
                    
                    if response.status == 401:
                        raise AuthenticationError("Invalid API key or expired token")
                    elif response.status >= 400:
                        error_data = await response.json() if response.content_type == 'application/json' else {}
                        raise PRSMError(f"API error {response.status}: {error_data.get('message', 'Unknown error')}")
                    
                    return await response.json()
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.max_retries:
                    raise NetworkError(f"Failed after {self.max_retries + 1} attempts: {e}")
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                logger.warning(f"Request failed, retrying (attempt {attempt + 1}): {e}")
    
    async def query(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> PRSMResponse:
        """
        Execute AI query with PRSM
        
        Args:
            prompt: The query/prompt to process
            model_id: Specific model to use (defaults to best available)
            max_tokens: Maximum tokens in response
            temperature: Randomness in response (0.0-1.0)
            system_prompt: Optional system instruction
            context: Additional context for the query
            **kwargs: Additional query parameters
        
        Returns:
            PRSMResponse with content, metadata, and usage info
        """
        request_data = QueryRequest(
            prompt=prompt,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            context=context or {},
            **kwargs
        )
        
        logger.info(f"Executing query with model {model_id or 'auto'}")
        
        response_data = await self._request(
            "POST", 
            "/nwtn/query",
            json_data=request_data.model_dump()
        )
        
        return PRSMResponse(**response_data)
    
    async def stream(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[PRSMResponse]:
        """
        Stream AI response in real-time
        
        Args:
            prompt: The query to process
            model_id: Specific model to use
            **kwargs: Additional query parameters
            
        Yields:
            PRSMResponse chunks with partial content
        """
        if not self._websocket:
            self._websocket = WebSocketClient(self.websocket_url, self.auth)
        
        request = QueryRequest(prompt=prompt, model_id=model_id, **kwargs)
        
        async for message in self._websocket.stream_query(request):
            if message.type == "response_chunk":
                yield PRSMResponse(**message.data)
            elif message.type == "error":
                raise PRSMError(message.data.get("message", "Stream error"))
    
    async def estimate_cost(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> float:
        """
        Estimate FTNS cost for a query without executing it
        
        Args:
            prompt: The query to estimate
            model_id: Specific model to use
            **kwargs: Additional query parameters
            
        Returns:
            Estimated cost in FTNS tokens
        """
        request_data = {
            "prompt": prompt,
            "model_id": model_id,
            **kwargs
        }
        
        response_data = await self._request(
            "POST",
            "/nwtn/estimate-cost",
            json_data=request_data
        )
        
        return response_data["estimated_cost"]
    
    async def get_safety_status(self) -> SafetyStatus:
        """Get current safety monitoring status"""
        response_data = await self._request("GET", "/safety/status")
        return SafetyStatus(**response_data)
    
    async def list_available_models(self) -> List[ModelInfo]:
        """List all available models in the network"""
        response_data = await self._request("GET", "/models")
        return [ModelInfo(**model) for model in response_data["models"]]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health and connectivity"""
        return await self._request("GET", "/health")
    
    @asynccontextmanager
    async def session(self):
        """Context manager for client session"""
        try:
            yield self
        finally:
            await self.close()
    
    async def close(self):
        """Close client connections and cleanup resources"""
        if self._session:
            await self._session.close()
            self._session = None
            
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            
        logger.info("PRSM client closed")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()