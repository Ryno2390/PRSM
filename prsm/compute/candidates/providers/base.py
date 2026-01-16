from abc import ABC, abstractmethod
from typing import Dict, Any, List

class AbstractReasoningProvider(ABC):
    """
    Hardware Abstraction Layer (HAL) for Reasoning Providers.
    Decouples reasoning logic from specific hardware backends (Cloud vs Edge).
    """
    
    @abstractmethod
    async def generate(self, prompt: str, context: str, **kwargs) -> Dict[str, Any]:
        """
        Generates a reasoning response.
        
        Args:
            prompt: The input query.
            context: Additional context.
            **kwargs: Additional parameters.
            
        Returns:
            Dict containing 'content', 'latency', and 'metadata'.
        """
        pass

    @abstractmethod
    def get_provider_type(self) -> str:
        """Returns the type of provider ('cloud' or 'edge')."""
        pass
