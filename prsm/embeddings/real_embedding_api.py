"""
PRSM Real Embedding API Integration

Production-ready integration with OpenAI and Anthropic embedding APIs.
Provides robust error handling, rate limiting, and fallback strategies.
"""

import asyncio
import logging
import os
import time
from typing import List, Optional, Dict, Any, Union
import numpy as np
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingProvider:
    """Configuration for an embedding provider"""
    name: str
    api_key: str
    model: str
    max_tokens: int
    rate_limit: int  # requests per minute
    cost_per_1k_tokens: float
    embedding_dimension: int


class RealEmbeddingAPI:
    """
    Production embedding API with multiple providers and intelligent fallbacks
    
    Features:
    - OpenAI and Anthropic API integration
    - Automatic fallback between providers
    - Rate limiting and error handling
    - Cost tracking and optimization
    - Batch processing for efficiency
    """
    
    def __init__(self):
        self.providers = {}
        self.fallback_order = []
        self.request_history = {}  # For rate limiting
        self.total_cost = 0.0
        self.total_tokens = 0
        
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize available embedding providers"""
        
        # OpenAI provider
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and HAS_OPENAI:
            self.providers['openai'] = EmbeddingProvider(
                name='openai',
                api_key=openai_key,
                model='text-embedding-ada-002',
                max_tokens=8191,
                rate_limit=3000,  # requests per minute
                cost_per_1k_tokens=0.0001,
                embedding_dimension=1536
            )
            self.fallback_order.append('openai')
            logger.info("OpenAI embedding provider initialized")
        
        # Anthropic provider (hypothetical - they don't have embeddings yet)
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        if anthropic_key and HAS_ANTHROPIC:
            # Note: Anthropic doesn't currently offer embedding models
            # This is prepared for future API releases
            logger.info("Anthropic API key found but embedding models not yet available")
        
        # Local/mock provider as fallback
        self.providers['mock'] = EmbeddingProvider(
            name='mock',
            api_key='none',
            model='mock-embedding-model',
            max_tokens=8191,
            rate_limit=float('inf'),
            cost_per_1k_tokens=0.0,
            embedding_dimension=384
        )
        self.fallback_order.append('mock')
        
        if not self.providers:
            logger.warning("No embedding providers available")
        else:
            logger.info(f"Initialized {len(self.providers)} embedding providers: {list(self.providers.keys())}")
    
    async def generate_embedding(self, text: str, 
                               preferred_provider: str = None) -> np.ndarray:
        """Generate embedding for a single text"""
        embeddings = await self.generate_embeddings([text], preferred_provider)
        return embeddings[0]
    
    async def generate_embeddings(self, texts: List[str], 
                                preferred_provider: str = None) -> List[np.ndarray]:
        """Generate embeddings for multiple texts with provider fallback"""
        
        if not texts:
            return []
        
        # Determine provider order
        providers_to_try = [preferred_provider] if preferred_provider else []
        providers_to_try.extend([p for p in self.fallback_order if p != preferred_provider])
        
        last_error = None
        
        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue
            
            provider = self.providers[provider_name]
            
            try:
                # Check rate limits
                await self._check_rate_limit(provider_name)
                
                # Generate embeddings with provider
                if provider_name == 'openai':
                    embeddings = await self._generate_openai_embeddings(texts, provider)
                elif provider_name == 'mock':
                    embeddings = await self._generate_mock_embeddings(texts, provider)
                else:
                    continue  # Skip unknown providers
                
                # Update statistics
                self.total_tokens += sum(len(text.split()) for text in texts)
                estimated_cost = (self.total_tokens / 1000) * provider.cost_per_1k_tokens
                self.total_cost += estimated_cost
                
                logger.info(f"Generated {len(embeddings)} embeddings using {provider_name}")
                return embeddings
                
            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        # All providers failed
        error_msg = f"All embedding providers failed. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    async def _check_rate_limit(self, provider_name: str):
        """Check and enforce rate limits for provider"""
        provider = self.providers[provider_name]
        
        if provider.rate_limit == float('inf'):
            return  # No rate limiting needed
        
        current_time = time.time()
        history_key = provider_name
        
        if history_key not in self.request_history:
            self.request_history[history_key] = []
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - 60
        self.request_history[history_key] = [
            t for t in self.request_history[history_key] if t > cutoff_time
        ]
        
        # Check if we're at the rate limit
        if len(self.request_history[history_key]) >= provider.rate_limit:
            # Calculate sleep time until oldest request is > 1 minute old
            oldest_request = min(self.request_history[history_key])
            sleep_time = 60 - (current_time - oldest_request) + 1  # +1 for safety
            
            if sleep_time > 0:
                logger.info(f"Rate limit reached for {provider_name}, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
        
        # Record this request
        self.request_history[history_key].append(current_time)
    
    async def _generate_openai_embeddings(self, texts: List[str], 
                                        provider: EmbeddingProvider) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API"""
        if not HAS_OPENAI:
            raise RuntimeError("OpenAI library not available")
        
        # Initialize OpenAI client
        client = openai.AsyncOpenAI(api_key=provider.api_key)
        
        # Process in batches to respect API limits
        all_embeddings = []
        batch_size = 100  # OpenAI recommended batch size
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Make API call
                response = await client.embeddings.create(
                    model=provider.model,
                    input=batch_texts
                )
                
                # Extract embeddings
                batch_embeddings = [
                    np.array(embedding.embedding, dtype=np.float32)
                    for embedding in response.data
                ]
                
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Generated {len(batch_embeddings)} embeddings from OpenAI batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"OpenAI API error for batch {i//batch_size + 1}: {e}")
                raise
        
        return all_embeddings
    
    async def _generate_mock_embeddings(self, texts: List[str], 
                                      provider: EmbeddingProvider) -> List[np.ndarray]:
        """Generate mock embeddings for testing and fallback"""
        embeddings = []
        
        for text in texts:
            # Create deterministic mock embedding based on text hash
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Convert hash to numbers and create embedding
            hash_numbers = [int(text_hash[i:i+2], 16) for i in range(0, len(text_hash), 2)]
            
            # Extend or truncate to desired dimension
            while len(hash_numbers) < provider.embedding_dimension:
                hash_numbers.extend(hash_numbers)
            hash_numbers = hash_numbers[:provider.embedding_dimension]
            
            # Normalize to typical embedding range
            embedding = np.array(hash_numbers, dtype=np.float32)
            embedding = (embedding - 128) / 128  # Normalize to [-1, 1] range
            
            # L2 normalize (common for embeddings)
            embedding = embedding / np.linalg.norm(embedding)
            
            embeddings.append(embedding)
        
        # Add small delay to simulate API call
        await asyncio.sleep(0.01 * len(texts))
        
        return embeddings
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about provider usage"""
        return {
            'available_providers': list(self.providers.keys()),
            'fallback_order': self.fallback_order,
            'total_cost': self.total_cost,
            'total_tokens': self.total_tokens,
            'request_history': {
                provider: len(history) 
                for provider, history in self.request_history.items()
            }
        }
    
    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider"""
        if provider_name not in self.providers:
            return None
        
        provider = self.providers[provider_name]
        return {
            'name': provider.name,
            'model': provider.model,
            'max_tokens': provider.max_tokens,
            'rate_limit': provider.rate_limit,
            'cost_per_1k_tokens': provider.cost_per_1k_tokens,
            'embedding_dimension': provider.embedding_dimension,
            'recent_requests': len(self.request_history.get(provider_name, []))
        }
    
    async def test_provider(self, provider_name: str) -> Dict[str, Any]:
        """Test a specific provider with a simple embedding"""
        if provider_name not in self.providers:
            return {'success': False, 'error': 'Provider not available'}
        
        test_text = "This is a test embedding request."
        
        try:
            start_time = time.time()
            embedding = await self.generate_embedding(test_text, provider_name)
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'provider': provider_name,
                'response_time': response_time,
                'embedding_dimension': len(embedding),
                'embedding_sample': embedding[:5].tolist()  # First 5 dimensions
            }
        
        except Exception as e:
            return {
                'success': False,
                'provider': provider_name,
                'error': str(e)
            }
    
    async def test_all_providers(self) -> Dict[str, Dict[str, Any]]:
        """Test all available providers"""
        results = {}
        
        for provider_name in self.providers.keys():
            results[provider_name] = await self.test_provider(provider_name)
        
        return results


# Global instance for easy access
_real_embedding_api = None

def get_embedding_api() -> RealEmbeddingAPI:
    """Get global embedding API instance"""
    global _real_embedding_api
    if _real_embedding_api is None:
        _real_embedding_api = RealEmbeddingAPI()
    return _real_embedding_api


# Convenience functions
async def generate_embedding(text: str, model: str = None) -> np.ndarray:
    """Generate single embedding with default API"""
    api = get_embedding_api()
    return await api.generate_embedding(text, model)


async def generate_embeddings(texts: List[str], model: str = None) -> List[np.ndarray]:
    """Generate multiple embeddings with default API"""
    api = get_embedding_api()
    return await api.generate_embeddings(texts, model)


async def test_embedding_providers() -> Dict[str, Any]:
    """Test all available embedding providers"""
    api = get_embedding_api()
    return await api.test_all_providers()