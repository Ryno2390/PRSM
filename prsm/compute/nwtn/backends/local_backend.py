#!/usr/bin/env python3
"""
Local Backend
=============

Local model backend using Ollama or HuggingFace transformers.
Supports local inference without external API calls.

Usage:
    # Using Ollama (default)
    backend = LocalBackend(use_ollama=True, ollama_host="http://localhost:11434")
    async with backend:
        result = await backend.generate("What is AI?")
        print(result.content)
    
    # Using transformers
    backend = LocalBackend(use_ollama=False, model_path="meta-llama/Llama-2-7b-chat-hf")
    async with backend:
        result = await backend.generate("What is AI?")

Requirements:
    - For Ollama: Ollama server running locally
    - For transformers: transformers and torch packages
"""

import time
from typing import Dict, List, Optional, Any

from .base import ModelBackend, BackendType, GenerateResult, EmbedResult, TokenUsage
from .exceptions import (
    BackendUnavailableError,
    BackendResponseError,
    ModelNotFoundError,
)


class LocalBackend(ModelBackend):
    """
    Local model backend using Ollama or HuggingFace transformers.
    
    Supports:
    - Ollama local inference server (default)
    - HuggingFace transformers models
    - PRSM distilled models
    
    Attributes:
        DEFAULT_MODEL: Default model for Ollama
        DEFAULT_EMBEDDING_MODEL: Default embedding model
        model_path: Path to local model (for transformers)
        ollama_host: Ollama server host URL
        use_ollama: Whether to use Ollama vs transformers
        _model: Loaded transformers model
        _tokenizer: Loaded tokenizer
        _session: HTTP session for Ollama
    """
    
    DEFAULT_MODEL = "llama3.2"
    DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
    
    # Common local models
    MODELS = {
        "llama3.2": {
            "model_id": "llama3.2",
            "provider": "local",
            "context_window": 128000,
            "supports_streaming": True,
            "supports_functions": False,
            "pricing": {"input": 0, "output": 0}
        },
        "llama3.1": {
            "model_id": "llama3.1",
            "provider": "local",
            "context_window": 128000,
            "supports_streaming": True,
            "supports_functions": False,
            "pricing": {"input": 0, "output": 0}
        },
        "mistral": {
            "model_id": "mistral",
            "provider": "local",
            "context_window": 32000,
            "supports_streaming": True,
            "supports_functions": False,
            "pricing": {"input": 0, "output": 0}
        },
        "codellama": {
            "model_id": "codellama",
            "provider": "local",
            "context_window": 16000,
            "supports_streaming": True,
            "supports_functions": False,
            "pricing": {"input": 0, "output": 0}
        },
        "nomic-embed-text": {
            "model_id": "nomic-embed-text",
            "provider": "local",
            "context_window": 8192,
            "dimensions": 768,
            "supports_streaming": False,
            "supports_functions": False,
            "pricing": {"input": 0, "output": 0}
        },
        "mxbai-embed-large": {
            "model_id": "mxbai-embed-large",
            "provider": "local",
            "context_window": 512,
            "dimensions": 1024,
            "supports_streaming": False,
            "supports_functions": False,
            "pricing": {"input": 0, "output": 0}
        }
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        ollama_host: str = "http://localhost:11434",
        use_ollama: bool = True,
        timeout: int = 300,
        **kwargs
    ):
        """
        Initialize the local backend.
        
        Args:
            model_path: Path to local model (for transformers mode)
            ollama_host: Ollama server host URL
            use_ollama: Whether to use Ollama (True) or transformers (False)
            timeout: Request timeout in seconds
            **kwargs: Additional configuration options
        """
        super().__init__(kwargs)
        self.model_path = model_path
        self.ollama_host = ollama_host
        self.use_ollama = use_ollama
        self.timeout = timeout
        self._model = None
        self._tokenizer = None
        self._session: Optional[Any] = None
    
    @property
    def backend_type(self) -> BackendType:
        """Return the backend type identifier."""
        return BackendType.LOCAL
    
    @property
    def models_supported(self) -> List[str]:
        """Return list of supported model IDs."""
        return list(self.MODELS.keys())
    
    async def initialize(self) -> None:
        """
        Initialize the backend.
        
        For Ollama: Connects to Ollama server and validates availability.
        For transformers: Loads the model into memory.
        
        Raises:
            BackendUnavailableError: If backend cannot be initialized
        """
        if self._initialized:
            return
        
        if self.use_ollama:
            await self._initialize_ollama()
        else:
            await self._initialize_transformers()
        
        self._initialized = True
    
    async def _initialize_ollama(self) -> None:
        """Initialize Ollama client."""
        try:
            import aiohttp
        except ImportError:
            raise BackendUnavailableError(
                "aiohttp package not installed. Install with: pip install aiohttp",
                backend_type=self.backend_type.value
            )
        
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
        # Check if Ollama is running
        try:
            async with self._session.get(f"{self.ollama_host}/api/tags") as response:
                if response.status != 200:
                    raise BackendUnavailableError(
                        f"Ollama server not available at {self.ollama_host}",
                        backend_type=self.backend_type.value
                    )
        except Exception as e:
            if self._session:
                await self._session.close()
                self._session = None
            raise BackendUnavailableError(
                f"Cannot connect to Ollama at {self.ollama_host}: {e}",
                backend_type=self.backend_type.value
            )
    
    async def _initialize_transformers(self) -> None:
        """Initialize local transformers model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise BackendUnavailableError(
                "transformers package not installed. Install with: pip install transformers torch",
                backend_type=self.backend_type.value
            )
        
        model_path = self.model_path or self.DEFAULT_MODEL
        
        try:
            # Load tokenizer and model
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Determine device and dtype
            device_map = "auto" if torch.cuda.is_available() else None
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map
            )
            
        except Exception as e:
            raise BackendUnavailableError(
                f"Failed to load local model '{model_path}': {e}",
                backend_type=self.backend_type.value
            )
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None
        
        # Free GPU memory if using transformers
        if self._model:
            del self._model
            self._model = None
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None
        
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
        
        Args:
            prompt: The input prompt
            model_id: Specific model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            GenerateResult: The generated content and metadata
            
        Raises:
            BackendUnavailableError: If backend is not initialized
        """
        if not self._initialized:
            await self.initialize()
        
        model = model_id or self.DEFAULT_MODEL
        start_time = time.time()
        
        if self.use_ollama:
            result = await self._generate_ollama(
                prompt, model, max_tokens, temperature, system_prompt, **kwargs
            )
        else:
            result = await self._generate_transformers(
                prompt, model, max_tokens, temperature, system_prompt, **kwargs
            )
        
        result.execution_time = time.time() - start_time
        return result
    
    async def _generate_ollama(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        **kwargs
    ) -> GenerateResult:
        """Generate using Ollama API."""
        import aiohttp
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            async with self._session.post(
                f"{self.ollama_host}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise BackendResponseError(
                        f"Ollama generation failed: {error}",
                        backend_type=self.backend_type.value,
                        status_code=response.status
                    )
                
                data = await response.json()
            
            return GenerateResult(
                content=data.get("response", ""),
                model_id=model,
                provider=self.backend_type,
                token_usage=TokenUsage(
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
                execution_time=0,  # Set by caller
                finish_reason="stop" if data.get("done") else "length",
                metadata={"raw_response": data}
            )
        
        except aiohttp.ClientError as e:
            raise BackendUnavailableError(
                f"Network error calling Ollama: {e}",
                backend_type=self.backend_type.value
            )
    
    async def _generate_transformers(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        **kwargs
    ) -> GenerateResult:
        """Generate using local transformers model."""
        import torch
        
        if not self._model or not self._tokenizer:
            raise BackendUnavailableError(
                "Transformers model not loaded",
                backend_type=self.backend_type.value
            )
        
        # Build full prompt
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        # Tokenize
        inputs = self._tokenizer(full_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return GenerateResult(
            content=generated_text,
            model_id=model,
            provider=self.backend_type,
            token_usage=TokenUsage(
                prompt_tokens=inputs["input_ids"].shape[1],
                completion_tokens=outputs.shape[1] - inputs["input_ids"].shape[1],
                total_tokens=outputs.shape[1]
            ),
            execution_time=0,  # Set by caller
            finish_reason="stop"
        )
    
    async def embed(
        self,
        text: str,
        model_id: Optional[str] = None,
        **kwargs
    ) -> EmbedResult:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            model_id: Specific embedding model to use
            **kwargs: Additional parameters
            
        Returns:
            EmbedResult: The embedding vector and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        model = model_id or self.DEFAULT_EMBEDDING_MODEL
        
        if self.use_ollama:
            return await self._embed_ollama(text, model)
        else:
            return await self._embed_transformers(text, model)
    
    async def _embed_ollama(self, text: str, model: str) -> EmbedResult:
        """Generate embedding using Ollama API."""
        import aiohttp
        
        try:
            async with self._session.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": model, "prompt": text}
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    raise BackendResponseError(
                        f"Ollama embedding failed: {error}",
                        backend_type=self.backend_type.value,
                        status_code=response.status
                    )
                
                data = await response.json()
            
            return EmbedResult(
                embedding=data.get("embedding", []),
                model_id=model,
                provider=self.backend_type,
                token_count=len(text.split()),  # Approximate
                metadata={"raw_response": data}
            )
        
        except aiohttp.ClientError as e:
            raise BackendUnavailableError(
                f"Network error calling Ollama: {e}",
                backend_type=self.backend_type.value
            )
    
    async def _embed_transformers(self, text: str, model: str) -> EmbedResult:
        """Generate embedding using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise BackendUnavailableError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers",
                backend_type=self.backend_type.value
            )
        
        encoder = SentenceTransformer(model)
        embedding = encoder.encode(text).tolist()
        
        return EmbedResult(
            embedding=embedding,
            model_id=model,
            provider=self.backend_type,
            token_count=len(text.split())
        )
    
    async def is_available(self) -> bool:
        """
        Check if the backend is available.
        
        Returns:
            bool: True if backend can process requests
        """
        if self.use_ollama:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.ollama_host}/api/tags") as response:
                        return response.status == 200
            except Exception:
                return False
        else:
            return self._model is not None and self._tokenizer is not None
    
    def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about available models.
        
        Args:
            model_id: Specific model to get info for (all if None)
            
        Returns:
            Dict with model information
        """
        if model_id:
            return self.MODELS.get(model_id, {})
        return self.MODELS.copy()
    
    def __repr__(self) -> str:
        """String representation of the backend."""
        mode = "ollama" if self.use_ollama else "transformers"
        return f"LocalBackend(mode={mode}, initialized={self._initialized})"