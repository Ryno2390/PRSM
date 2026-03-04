#!/usr/bin/env python3
"""
Mock Backend
============

Mock backend for testing. Returns deterministic responses without making
actual API calls. This preserves the current mock behavior of the NWTN
pipeline and is used for testing without requiring API keys.

Usage:
    backend = MockBackend(delay_seconds=0.1)
    await backend.initialize()
    
    result = await backend.generate("What is AI?")
    print(result.content)
    
    await backend.close()
"""

import asyncio
import hashlib
import time
from typing import Dict, List, Optional, Any

from .base import ModelBackend, BackendType, GenerateResult, EmbedResult, TokenUsage
from .exceptions import BackendError


class MockBackend(ModelBackend):
    """
    Mock backend for testing.
    
    This backend preserves the current mock behavior of the NWTN pipeline
    and is used for testing without requiring API keys or network access.
    
    The generated content is deterministic based on the input prompt hash,
    making tests reproducible.
    
    Attributes:
        DEFAULT_MODEL: Default model identifier
        PREDEFINED_RESPONSES: Dictionary of predefined responses for common prompts
        delay_seconds: Simulated processing delay
        call_count: Number of generate calls made
    
    Example:
        backend = MockBackend(delay_seconds=0.01)
        async with backend:
            result = await backend.generate("test prompt")
            assert result.content is not None
    """
    
    DEFAULT_MODEL = "mock-model"
    
    # Predefined responses for common test scenarios
    PREDEFINED_RESPONSES = {
        "research": """Based on comprehensive analysis, the research indicates several key findings:

1. **Primary Finding**: The data suggests a strong correlation between the variables under study, with statistical significance (p < 0.05).

2. **Methodology**: The methodology employed has been validated across multiple peer-reviewed studies and follows established protocols.

3. **Implications**: The implications of this research extend to practical applications in the field, including potential improvements in efficiency and accuracy.

4. **Recommendations**: Further investigation is recommended to explore additional variables and validate findings across different contexts.

5. **Limitations**: Current limitations include sample size constraints and potential confounding factors that should be addressed in future studies.""",

        "analysis": """The analysis reveals multiple patterns in the data:

**Key Observations:**
1. A consistent trend in the primary metrics with 15.3% improvement over baseline
2. Anomalies detected in 2.3% of samples, requiring further investigation
3. Statistical significance achieved with p-value < 0.05

**Data Summary:**
- Total samples processed: 10,000
- Success rate: 97.7%
- Average processing time: 0.45 seconds

**Recommendations:**
1. Implement automated anomaly detection for the 2.3% outlier cases
2. Scale processing capacity to handle increased load
3. Continue monitoring for pattern changes over time""",

        "coding": """Here is the implementation:

```python
def solution(input_data):
    \"\"\"
    Process the input data and return results.
    
    Args:
        input_data: The input to process
        
    Returns:
        The processed result
    \"\"\"
    # Validate input
    if not input_data:
        raise ValueError("Input cannot be empty")
    
    # Process the input
    result = process(input_data)
    
    return result


def process(data):
    \"\"\"Core processing logic.\"\"\"
    # Implementation details
    processed = transform(data)
    return validate(processed)
```

This solution has O(n) time complexity and handles edge cases appropriately. The modular design allows for easy testing and maintenance.""",

        "general": """I have processed your request and generated a comprehensive response.

**Summary:**
The analysis considers multiple factors and provides actionable insights based on the available information.

**Key Points:**
1. The request has been analyzed according to established criteria
2. Multiple approaches have been considered
3. The recommended solution balances efficiency and accuracy

**Next Steps:**
Please let me know if you need any clarification or additional details on any aspect of this response.""",

        "question": """Based on my analysis, here is the answer to your question:

**Direct Answer:**
The information you're seeking can be understood through several key concepts that I'll explain below.

**Detailed Explanation:**
1. **Context**: The background context is important for understanding the full picture.
2. **Main Points**: The primary factors to consider are clarity, accuracy, and relevance.
3. **Examples**: Practical examples help illustrate these concepts.

**Conclusion:**
The answer encompasses multiple perspectives and provides a comprehensive view of the topic at hand.""",

        "reasoning": """**Reasoning Process:**

**Step 1: Problem Analysis**
- Identify the core question
- Break down into component parts
- Establish evaluation criteria

**Step 2: Information Gathering**
- Review relevant data
- Consider multiple perspectives
- Identify key relationships

**Step 3: Logical Deduction**
- Apply logical principles
- Evaluate each possibility
- Consider edge cases

**Step 4: Synthesis**
- Combine findings
- Draw conclusions
- Validate against criteria

**Conclusion:**
Based on this systematic reasoning process, the conclusion follows logically from the premises and available evidence."""
    }
    
    def __init__(self, delay_seconds: float = 0.1, **kwargs):
        """
        Initialize the mock backend.
        
        Args:
            delay_seconds: Simulated processing delay in seconds (default: 0.1)
            **kwargs: Additional configuration options
        """
        super().__init__(kwargs)
        self.delay_seconds = delay_seconds
        self.call_count = 0
    
    @property
    def backend_type(self) -> BackendType:
        """Return the backend type identifier."""
        return BackendType.MOCK
    
    @property
    def models_supported(self) -> List[str]:
        """Return list of supported model IDs."""
        return ["mock-model", "mock-embedding"]
    
    async def initialize(self) -> None:
        """
        Initialize the backend.
        
        For the mock backend, this just sets the initialized flag.
        No external connections are needed.
        """
        self._initialized = True
    
    async def close(self) -> None:
        """
        Clean up resources.
        
        For the mock backend, this just resets the initialized flag.
        """
        self._initialized = False
    
    async def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> GenerateResult:
        """
        Generate text from a prompt.
        
        Returns deterministic mock responses based on prompt content.
        The response is selected from predefined responses matching
        keywords in the prompt.
        
        Args:
            prompt: The input prompt
            model_id: Specific model to use (ignored, uses default)
            max_tokens: Maximum tokens to generate (used to truncate response)
            temperature: Sampling temperature (ignored for deterministic output)
            system_prompt: Optional system prompt (affects response style)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            GenerateResult: The generated content and metadata
        """
        start_time = time.time()
        
        # Simulate processing delay
        await asyncio.sleep(self.delay_seconds)
        
        # Generate deterministic response based on prompt content
        content = self._generate_mock_response(prompt, system_prompt)
        
        # Truncate if needed (rough char-to-token ratio of ~4)
        if len(content) > max_tokens * 4:
            content = content[:max_tokens * 4]
            if not content.endswith('.'):
                content = content[:content.rfind('.')] + '.'
        
        self.call_count += 1
        
        execution_time = time.time() - start_time
        
        return GenerateResult(
            content=content,
            model_id=model_id or self.DEFAULT_MODEL,
            provider=self.backend_type,
            token_usage=TokenUsage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(content.split()),
                total_tokens=len(prompt.split()) + len(content.split())
            ),
            execution_time=execution_time,
            finish_reason="stop",
            metadata={
                "mock": True,
                "call_count": self.call_count,
                "prompt_hash": hashlib.md5(prompt.encode()).hexdigest()[:8],
                "delay_seconds": self.delay_seconds
            }
        )
    
    def _generate_mock_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a deterministic mock response based on prompt content.
        
        Matches keywords in the prompt to predefined response categories.
        
        Args:
            prompt: The input prompt
            system_prompt: Optional system prompt (affects response style)
            
        Returns:
            str: The generated mock response
        """
        prompt_lower = prompt.lower()
        
        # Check for system prompt influence
        if system_prompt:
            system_lower = system_prompt.lower()
            # Add system prompt context to matching
            if "code" in system_lower or "program" in system_lower:
                return self.PREDEFINED_RESPONSES["coding"]
            if "research" in system_lower or "study" in system_lower:
                return self.PREDEFINED_RESPONSES["research"]
        
        # Match prompt type to predefined response
        if any(kw in prompt_lower for kw in ["research", "study", "investigate", "paper", "academic"]):
            return self.PREDEFINED_RESPONSES["research"]
        elif any(kw in prompt_lower for kw in ["analyze", "analysis", "data", "metrics", "statistics"]):
            return self.PREDEFINED_RESPONSES["analysis"]
        elif any(kw in prompt_lower for kw in ["code", "implement", "function", "program", "python", "javascript"]):
            return self.PREDEFINED_RESPONSES["coding"]
        elif any(kw in prompt_lower for kw in ["why", "how", "what", "when", "where", "explain"]):
            return self.PREDEFINED_RESPONSES["question"]
        elif any(kw in prompt_lower for kw in ["reason", "logic", "think", "deduce", "infer"]):
            return self.PREDEFINED_RESPONSES["reasoning"]
        else:
            return self.PREDEFINED_RESPONSES["general"]
    
    async def embed(
        self,
        text: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> EmbedResult:
        """
        Generate embedding for text.
        
        Creates a deterministic embedding vector based on the text hash.
        The embedding is 1536-dimensional (same as OpenAI's text-embedding-3-small).
        The vector is normalized to unit length for cosine similarity.
        
        Args:
            text: Text to embed
            model_id: Specific embedding model to use (ignored)
            **kwargs: Additional parameters (dimensions - allows custom dimensionality)
            
        Returns:
            EmbedResult: The embedding vector and metadata
        """
        # Get desired dimensions (default 1536 to match OpenAI text-embedding-3-small)
        dimensions = kwargs.get("dimensions", 1536)
        
        # Generate deterministic embedding based on text hash
        text_hash = hashlib.sha256(text.encode()).digest()
        
        # Create embedding with requested dimensions
        embedding = []
        for i in range(dimensions):
            byte_val = text_hash[i % len(text_hash)]
            # Normalize to [-1, 1] range
            embedding.append((byte_val - 128) / 128.0)
        
        # Normalize to unit vector for cosine similarity
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return EmbedResult(
            embedding=embedding,
            model_id=model_id or "mock-embedding",
            provider=self.backend_type,
            token_count=len(text.split()),
            metadata={
                "mock": True,
                "dimensions": dimensions,
                "normalized": True,
                "text_hash": hashlib.md5(text.encode()).hexdigest()[:8]
            }
        )
    
    async def is_available(self) -> bool:
        """
        Check if the backend is available.
        
        The mock backend is always available.
        
        Returns:
            bool: Always True for mock backend
        """
        return True
    
    def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available models.
        
        Args:
            model_id: Specific model to get info for (all if None)
            
        Returns:
            Dict with model information
        """
        models = {
            "mock-model": {
                "model_id": "mock-model",
                "provider": "mock",
                "context_window": 4096,
                "supports_streaming": False,
                "supports_functions": False,
                "supports_vision": False,
                "pricing": {"input": 0, "output": 0},
                "description": "Mock model for testing"
            },
            "mock-embedding": {
                "model_id": "mock-embedding",
                "provider": "mock",
                "context_window": 8191,
                "dimensions": 1536,
                "supports_streaming": False,
                "supports_functions": False,
                "pricing": {"input": 0, "output": 0},
                "description": "Mock embedding model for testing"
            }
        }
        
        if model_id:
            return models.get(model_id, {})
        return models
    
    def __repr__(self) -> str:
        """String representation of the mock backend"""
        return f"MockBackend(delay={self.delay_seconds}s, calls={self.call_count})"