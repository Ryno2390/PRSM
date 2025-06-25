# OpenAI Integration Guide

Integrate PRSM with OpenAI's powerful language models and APIs for advanced AI capabilities including GPT models, embeddings, and function calling.

## 🎯 Overview

This guide covers integrating PRSM with OpenAI's API services, including GPT models, embeddings, DALL-E image generation, and advanced features like function calling and fine-tuning.

## 📋 Prerequisites

- PRSM instance configured
- OpenAI API key
- Python 3.8+ installed
- Basic knowledge of OpenAI API

## 🚀 Quick Start

### 1. Installation and Setup

```bash
# Install OpenAI SDK
pip install openai>=1.0.0
pip install tiktoken  # For token counting
pip install pillow   # For image processing

# Optional dependencies
pip install aiofiles  # For async file operations
pip install pydantic  # For data validation
```

### 2. Basic Configuration

```python
# prsm/integrations/openai/config.py
import os
from typing import Optional, Dict, Any, List
from pydantic import BaseSettings, Field
import openai

class OpenAIConfig(BaseSettings):
    """OpenAI integration configuration."""
    
    # API Configuration
    api_key: str = Field(..., env="OPENAI_API_KEY")
    organization: Optional[str] = Field(None, env="OPENAI_ORG_ID")
    base_url: str = "https://api.openai.com/v1"
    
    # Model Configurations
    default_chat_model: str = "gpt-3.5-turbo"
    default_completion_model: str = "gpt-3.5-turbo-instruct"
    default_embedding_model: str = "text-embedding-ada-002"
    default_image_model: str = "dall-e-3"
    
    # Chat Parameters
    chat_temperature: float = 0.7
    chat_max_tokens: int = 2000
    chat_top_p: float = 1.0
    chat_frequency_penalty: float = 0.0
    chat_presence_penalty: float = 0.0
    
    # Completion Parameters
    completion_temperature: float = 0.7
    completion_max_tokens: int = 1000
    completion_top_p: float = 1.0
    
    # Image Generation Parameters
    image_size: str = "1024x1024"
    image_quality: str = "standard"
    image_style: str = "natural"
    
    # Rate Limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 40000
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Fine-tuning
    fine_tune_model: str = "gpt-3.5-turbo"
    fine_tune_epochs: int = 3
    fine_tune_batch_size: int = 1
    fine_tune_learning_rate_multiplier: float = 2.0
    
    class Config:
        env_prefix = "PRSM_OPENAI_"

# Global configuration
openai_config = OpenAIConfig()

# Configure OpenAI client
openai.api_key = openai_config.api_key
if openai_config.organization:
    openai.organization = openai_config.organization
```

### 3. Core OpenAI Client

```python
# prsm/integrations/openai/client.py
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import openai
from openai import AsyncOpenAI
import tiktoken

from prsm.integrations.openai.config import openai_config
from prsm.models.query import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

class OpenAIRateLimiter:
    """Rate limiter for OpenAI API calls."""
    
    def __init__(self, max_requests: int, max_tokens: int, window: int = 60):
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.window = window
        
        self.requests = []
        self.tokens = []
    
    async def acquire(self, token_count: int = 0):
        """Acquire permission to make a request."""
        current_time = time.time()
        
        # Clean old entries
        self.requests = [t for t in self.requests if current_time - t < self.window]
        self.tokens = [t for t in self.tokens if current_time - t[0] < self.window]
        
        # Check limits
        if len(self.requests) >= self.max_requests:
            wait_time = self.window - (current_time - self.requests[0])
            await asyncio.sleep(wait_time)
            return await self.acquire(token_count)
        
        current_token_count = sum(t[1] for t in self.tokens)
        if current_token_count + token_count > self.max_tokens:
            wait_time = self.window - (current_time - self.tokens[0][0])
            await asyncio.sleep(wait_time)
            return await self.acquire(token_count)
        
        # Record this request
        self.requests.append(current_time)
        if token_count > 0:
            self.tokens.append((current_time, token_count))

class PRSMOpenAIClient:
    """Enhanced OpenAI client for PRSM integration."""
    
    def __init__(self):
        self.config = openai_config
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            organization=self.config.organization,
            base_url=self.config.base_url
        )
        
        # Rate limiter
        self.rate_limiter = OpenAIRateLimiter(
            max_requests=self.config.max_requests_per_minute,
            max_tokens=self.config.max_tokens_per_minute
        )
        
        # Token encoders
        self.encoders = {}
        
    def _get_encoder(self, model: str):
        """Get tiktoken encoder for model."""
        if model not in self.encoders:
            try:
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self.encoders[model]
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for given model."""
        encoder = self._get_encoder(model)
        return len(encoder.encode(text))
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Create chat completion."""
        try:
            model = model or self.config.default_chat_model
            
            # Count tokens for rate limiting
            message_text = " ".join([msg["content"] for msg in messages])
            token_count = self.count_tokens(message_text, model)
            
            # Apply rate limiting
            await self.rate_limiter.acquire(token_count)
            
            # Prepare parameters
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature or self.config.chat_temperature,
                "max_tokens": max_tokens or self.config.chat_max_tokens,
                "top_p": top_p or self.config.chat_top_p,
                "frequency_penalty": frequency_penalty or self.config.chat_frequency_penalty,
                "presence_penalty": presence_penalty or self.config.chat_presence_penalty
            }
            
            # Add function calling if provided
            if functions:
                kwargs["functions"] = functions
                if function_call:
                    kwargs["function_call"] = function_call
            
            # Add tools if provided (newer format)
            if tools:
                kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice
            
            # Make API call
            response = await self.client.chat.completions.create(**kwargs)
            
            return {
                "id": response.id,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                            "function_call": choice.message.function_call,
                            "tool_calls": choice.message.tool_calls
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "created": response.created
            }
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create streaming chat completion."""
        try:
            model = model or self.config.default_chat_model
            
            # Count tokens for rate limiting
            message_text = " ".join([msg["content"] for msg in messages])
            token_count = self.count_tokens(message_text, model)
            
            # Apply rate limiting
            await self.rate_limiter.acquire(token_count)
            
            # Prepare parameters
            params = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": kwargs.get("temperature", self.config.chat_temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.chat_max_tokens)
            }
            
            # Make streaming API call
            async for chunk in await self.client.chat.completions.create(**params):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield {
                        "id": chunk.id,
                        "model": chunk.model,
                        "content": chunk.choices[0].delta.content,
                        "finish_reason": chunk.choices[0].finish_reason
                    }
                    
        except Exception as e:
            logger.error(f"Streaming chat completion failed: {e}")
            raise
    
    async def create_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        dimensions: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create embeddings for text(s)."""
        try:
            model = model or self.config.default_embedding_model
            
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Count tokens for rate limiting
            total_tokens = sum(self.count_tokens(text, model) for text in texts)
            await self.rate_limiter.acquire(total_tokens)
            
            # Prepare parameters
            kwargs = {
                "model": model,
                "input": texts
            }
            
            if dimensions:
                kwargs["dimensions"] = dimensions
            
            # Make API call
            response = await self.client.embeddings.create(**kwargs)
            
            return {
                "model": response.model,
                "embeddings": [embedding.embedding for embedding in response.data],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise
    
    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        style: Optional[str] = None,
        n: int = 1,
        response_format: str = "url"
    ) -> Dict[str, Any]:
        """Generate image using DALL-E."""
        try:
            model = model or self.config.default_image_model
            
            # Apply rate limiting (images use different limits)
            await self.rate_limiter.acquire()
            
            # Prepare parameters
            kwargs = {
                "model": model,
                "prompt": prompt,
                "size": size or self.config.image_size,
                "quality": quality or self.config.image_quality,
                "n": n,
                "response_format": response_format
            }
            
            if model == "dall-e-3":
                kwargs["style"] = style or self.config.image_style
            
            # Make API call
            response = await self.client.images.generate(**kwargs)
            
            return {
                "model": model,
                "created": response.created,
                "data": [
                    {
                        "url": image.url,
                        "b64_json": image.b64_json,
                        "revised_prompt": getattr(image, "revised_prompt", None)
                    }
                    for image in response.data
                ]
            }
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    async def transcribe_audio(
        self,
        audio_file: bytes,
        filename: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Transcribe audio using Whisper."""
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Prepare file
            files = {"file": (filename, audio_file, "audio/mpeg")}
            
            # Prepare parameters
            data = {
                "model": model,
                "response_format": response_format,
                "temperature": temperature
            }
            
            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt
            
            # Make API call
            response = await self.client.audio.transcriptions.create(
                file=audio_file,
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature
            )
            
            return {"text": response.text}
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise
    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Process query using OpenAI models."""
        try:
            # Prepare messages
            messages = []
            
            # Add system message if context provided
            if request.context:
                messages.append({
                    "role": "system",
                    "content": request.context
                })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": request.prompt
            })
            
            # Create chat completion
            response = await self.chat_completion(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Extract response
            choice = response["choices"][0]
            content = choice["message"]["content"]
            
            # Create PRSM response
            prsm_response = QueryResponse(
                query_id=f"openai_{request.user_id}_{int(time.time())}",
                final_answer=content,
                user_id=request.user_id,
                model=response["model"],
                token_usage=response["usage"]["total_tokens"],
                metadata={
                    "openai_integration": True,
                    "finish_reason": choice["finish_reason"],
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "completion_tokens": response["usage"]["completion_tokens"]
                }
            )
            
            return prsm_response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise

# Global OpenAI client
openai_client = PRSMOpenAIClient()
```

## 🔧 Function Calling Integration

### 1. Function Registry

```python
# prsm/integrations/openai/functions.py
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable, get_type_hints
from pydantic import BaseModel, Field
import inspect
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

class FunctionRegistry:
    """Registry for OpenAI function calling."""
    
    def __init__(self):
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.handlers: Dict[str, Callable] = {}
    
    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ):
        """Register a function for OpenAI function calling."""
        self.functions[name] = {
            "name": name,
            "description": description,
            "parameters": parameters
        }
        self.handlers[name] = handler
        logger.info(f"Registered function: {name}")
    
    def register_from_callable(self, func: Callable, description: str = None):
        """Register function automatically from callable."""
        name = func.__name__
        description = description or func.__doc__ or f"Execute {name}"
        
        # Get function signature
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Build parameters schema
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, str)
            
            # Convert Python types to JSON schema types
            if param_type == int:
                prop_type = "integer"
            elif param_type == float:
                prop_type = "number"
            elif param_type == bool:
                prop_type = "boolean"
            elif param_type == list:
                prop_type = "array"
            elif param_type == dict:
                prop_type = "object"
            else:
                prop_type = "string"
            
            parameters["properties"][param_name] = {
                "type": prop_type,
                "description": f"Parameter {param_name}"
            }
            
            # Required parameters (no default value)
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        self.register_function(name, description, parameters, func)
    
    def get_function_definitions(self) -> List[Dict[str, Any]]:
        """Get all function definitions for OpenAI."""
        return list(self.functions.values())
    
    async def execute_function(
        self,
        name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Execute a registered function."""
        if name not in self.handlers:
            raise ValueError(f"Function {name} not registered")
        
        handler = self.handlers[name]
        
        try:
            if inspect.iscoroutinefunction(handler):
                return await handler(**arguments)
            else:
                return handler(**arguments)
        except Exception as e:
            logger.error(f"Function {name} execution failed: {e}")
            return f"Error executing {name}: {str(e)}"

# Global function registry
function_registry = FunctionRegistry()

# Built-in functions
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search_web(query: str) -> Dict[str, Any]:
    """Search the web for information."""
    try:
        # This is a simplified example - in production, use proper search APIs
        search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
        response = requests.get(search_url, timeout=10)
        data = response.json()
        
        results = []
        for result in data.get("RelatedTopics", [])[:3]:
            if "Text" in result and "FirstURL" in result:
                results.append({
                    "title": result.get("Text", "")[:100],
                    "url": result.get("FirstURL", ""),
                    "snippet": result.get("Text", "")[:200]
                })
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        return {"error": str(e)}

def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        # Safe evaluation of mathematical expressions
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

async def get_weather(city: str) -> Dict[str, Any]:
    """Get weather information for a city."""
    # This is a placeholder - integrate with a real weather API
    return {
        "city": city,
        "temperature": "22°C",
        "description": "Partly cloudy",
        "humidity": "65%",
        "note": "This is a placeholder response"
    }

# Register built-in functions
function_registry.register_from_callable(get_current_time, "Get current date and time")
function_registry.register_from_callable(search_web, "Search the web for information")
function_registry.register_from_callable(calculate, "Perform mathematical calculations")
function_registry.register_from_callable(get_weather, "Get weather information for a city")
```

### 2. Function Calling Handler

```python
# prsm/integrations/openai/function_calling.py
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

from prsm.integrations.openai.client import openai_client
from prsm.integrations.openai.functions import function_registry
from prsm.models.query import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

class OpenAIFunctionCaller:
    """Handle OpenAI function calling workflows."""
    
    def __init__(self):
        self.client = openai_client
        self.registry = function_registry
        self.max_iterations = 5
    
    async def process_with_functions(
        self,
        request: QueryRequest,
        available_functions: Optional[List[str]] = None
    ) -> QueryResponse:
        """Process query with function calling capability."""
        try:
            # Get available functions
            all_functions = self.registry.get_function_definitions()
            
            if available_functions:
                # Filter to specific functions
                functions = [
                    f for f in all_functions 
                    if f["name"] in available_functions
                ]
            else:
                functions = all_functions
            
            # Prepare initial messages
            messages = []
            
            if request.context:
                messages.append({
                    "role": "system",
                    "content": request.context
                })
            
            messages.append({
                "role": "user",
                "content": request.prompt
            })
            
            # Function calling loop
            total_tokens = 0
            function_calls = []
            
            for iteration in range(self.max_iterations):
                # Make chat completion with functions
                response = await self.client.chat_completion(
                    messages=messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    tools=[
                        {"type": "function", "function": func}
                        for func in functions
                    ],
                    tool_choice="auto"
                )
                
                total_tokens += response["usage"]["total_tokens"]
                choice = response["choices"][0]
                message = choice["message"]
                
                # Add assistant's response to messages
                messages.append({
                    "role": "assistant",
                    "content": message["content"],
                    "tool_calls": message["tool_calls"]
                })
                
                # Check if function calls were made
                if message["tool_calls"]:
                    # Execute function calls
                    for tool_call in message["tool_calls"]:
                        function_call = tool_call["function"]
                        function_name = function_call["name"]
                        
                        try:
                            # Parse arguments
                            arguments = json.loads(function_call["arguments"])
                            
                            # Execute function
                            result = await self.registry.execute_function(
                                function_name,
                                arguments
                            )
                            
                            # Convert result to string
                            if isinstance(result, dict):
                                result_str = json.dumps(result)
                            else:
                                result_str = str(result)
                            
                            # Add function result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": result_str
                            })
                            
                            # Track function call
                            function_calls.append({
                                "name": function_name,
                                "arguments": arguments,
                                "result": result
                            })
                            
                        except Exception as e:
                            logger.error(f"Function execution failed: {e}")
                            # Add error message
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call["id"],
                                "content": f"Error: {str(e)}"
                            })
                    
                    # Continue conversation after function calls
                    continue
                
                else:
                    # No function calls, we're done
                    final_answer = message["content"]
                    break
            
            else:
                # Max iterations reached
                final_answer = "Unable to complete the request within the maximum number of iterations."
            
            # Create response
            prsm_response = QueryResponse(
                query_id=f"openai_func_{request.user_id}_{int(asyncio.get_event_loop().time())}",
                final_answer=final_answer,
                user_id=request.user_id,
                model=request.model or "gpt-3.5-turbo",
                token_usage=total_tokens,
                metadata={
                    "openai_function_calling": True,
                    "function_calls": function_calls,
                    "iterations": iteration + 1,
                    "available_functions": [f["name"] for f in functions]
                }
            )
            
            return prsm_response
            
        except Exception as e:
            logger.error(f"Function calling failed: {e}")
            raise

# Global function caller
openai_function_caller = OpenAIFunctionCaller()
```

## 🎨 Image Generation and Vision

### 1. DALL-E Integration

```python
# prsm/integrations/openai/image_generation.py
import asyncio
import base64
import io
import logging
from typing import Dict, Any, List, Optional, Union
from PIL import Image
import requests

from prsm.integrations.openai.client import openai_client

logger = logging.getLogger(__name__)

class OpenAIImageGenerator:
    """DALL-E image generation integration."""
    
    def __init__(self):
        self.client = openai_client
    
    async def generate_image(
        self,
        prompt: str,
        model: str = "dall-e-3",
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "natural",
        n: int = 1,
        response_format: str = "url"
    ) -> Dict[str, Any]:
        """Generate image using DALL-E."""
        try:
            response = await self.client.generate_image(
                prompt=prompt,
                model=model,
                size=size,
                quality=quality,
                style=style,
                n=n,
                response_format=response_format
            )
            
            return {
                "prompt": prompt,
                "model": model,
                "images": response["data"],
                "created": response["created"]
            }
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            raise
    
    async def edit_image(
        self,
        image: Union[str, bytes],
        mask: Optional[Union[str, bytes]],
        prompt: str,
        model: str = "dall-e-2",
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url"
    ) -> Dict[str, Any]:
        """Edit image using DALL-E."""
        try:
            # Note: Image editing is only available with DALL-E 2
            # This is a placeholder for the actual implementation
            # as the OpenAI SDK structure may vary
            
            response = await self.client.images.edit(
                image=image,
                mask=mask,
                prompt=prompt,
                model=model,
                n=n,
                size=size,
                response_format=response_format
            )
            
            return {
                "prompt": prompt,
                "model": model,
                "images": [
                    {
                        "url": image.url,
                        "b64_json": image.b64_json
                    }
                    for image in response.data
                ]
            }
            
        except Exception as e:
            logger.error(f"Image editing failed: {e}")
            raise
    
    async def create_variations(
        self,
        image: Union[str, bytes],
        model: str = "dall-e-2",
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url"
    ) -> Dict[str, Any]:
        """Create variations of an image."""
        try:
            response = await self.client.images.create_variation(
                image=image,
                model=model,
                n=n,
                size=size,
                response_format=response_format
            )
            
            return {
                "model": model,
                "variations": [
                    {
                        "url": image.url,
                        "b64_json": image.b64_json
                    }
                    for image in response.data
                ]
            }
            
        except Exception as e:
            logger.error(f"Image variation creation failed: {e}")
            raise
    
    def download_image(self, url: str) -> bytes:
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Image download failed: {e}")
            raise
    
    def encode_image_base64(self, image_data: bytes) -> str:
        """Encode image data as base64."""
        return base64.b64encode(image_data).decode('utf-8')
    
    def decode_image_base64(self, base64_data: str) -> bytes:
        """Decode base64 image data."""
        return base64.b64decode(base64_data)

# Global image generator
openai_image_generator = OpenAIImageGenerator()
```

### 2. Vision Analysis

```python
# prsm/integrations/openai/vision.py
import asyncio
import base64
import logging
from typing import Dict, Any, List, Optional, Union
from PIL import Image
import io

from prsm.integrations.openai.client import openai_client

logger = logging.getLogger(__name__)

class OpenAIVision:
    """OpenAI Vision capabilities for image analysis."""
    
    def __init__(self):
        self.client = openai_client
    
    async def analyze_image(
        self,
        image: Union[str, bytes, Image.Image],
        prompt: str = "What's in this image?",
        model: str = "gpt-4-vision-preview",
        max_tokens: int = 1000,
        detail: str = "auto"
    ) -> Dict[str, Any]:
        """Analyze image using GPT-4 Vision."""
        try:
            # Prepare image data
            if isinstance(image, str):
                if image.startswith("http"):
                    # URL
                    image_data = {"url": image}
                else:
                    # Base64 encoded
                    image_data = {"url": f"data:image/jpeg;base64,{image}"}
            elif isinstance(image, bytes):
                # Convert bytes to base64
                base64_image = base64.b64encode(image).decode('utf-8')
                image_data = {"url": f"data:image/jpeg;base64,{base64_image}"}
            elif isinstance(image, Image.Image):
                # Convert PIL Image to base64
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_data = {"url": f"data:image/jpeg;base64,{base64_image}"}
            else:
                raise ValueError("Unsupported image format")
            
            # Add detail parameter
            image_data["detail"] = detail
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": image_data}
                    ]
                }
            ]
            
            # Make API call
            response = await self.client.chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens
            )
            
            return {
                "analysis": response["choices"][0]["message"]["content"],
                "model": response["model"],
                "usage": response["usage"]
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise
    
    async def describe_image(
        self,
        image: Union[str, bytes, Image.Image],
        model: str = "gpt-4-vision-preview"
    ) -> str:
        """Get a detailed description of an image."""
        result = await self.analyze_image(
            image=image,
            prompt="Provide a detailed description of this image, including objects, people, actions, colors, and setting.",
            model=model
        )
        return result["analysis"]
    
    async def extract_text_from_image(
        self,
        image: Union[str, bytes, Image.Image],
        model: str = "gpt-4-vision-preview"
    ) -> str:
        """Extract text from an image."""
        result = await self.analyze_image(
            image=image,
            prompt="Extract all text visible in this image. If there's no text, respond with 'No text found'.",
            model=model
        )
        return result["analysis"]
    
    async def answer_about_image(
        self,
        image: Union[str, bytes, Image.Image],
        question: str,
        model: str = "gpt-4-vision-preview"
    ) -> str:
        """Answer a specific question about an image."""
        result = await self.analyze_image(
            image=image,
            prompt=f"Question about this image: {question}",
            model=model
        )
        return result["analysis"]
    
    async def count_objects(
        self,
        image: Union[str, bytes, Image.Image],
        object_type: str,
        model: str = "gpt-4-vision-preview"
    ) -> Dict[str, Any]:
        """Count specific objects in an image."""
        result = await self.analyze_image(
            image=image,
            prompt=f"Count the number of {object_type} in this image. Provide your answer as a number followed by a brief explanation.",
            model=model
        )
        
        # Try to extract the number from the response
        analysis = result["analysis"]
        try:
            # Simple number extraction - you might want to make this more sophisticated
            import re
            numbers = re.findall(r'\d+', analysis)
            count = int(numbers[0]) if numbers else None
        except:
            count = None
        
        return {
            "object_type": object_type,
            "count": count,
            "analysis": analysis,
            "model": result["model"]
        }

# Global vision analyzer
openai_vision = OpenAIVision()
```

## 🎙️ Audio Processing

### 1. Whisper Integration

```python
# prsm/integrations/openai/audio.py
import asyncio
import logging
from typing import Dict, Any, Optional, Union
import aiofiles

from prsm.integrations.openai.client import openai_client

logger = logging.getLogger(__name__)

class OpenAIAudio:
    """OpenAI audio processing with Whisper."""
    
    def __init__(self):
        self.client = openai_client
    
    async def transcribe_audio(
        self,
        audio_file: Union[str, bytes],
        filename: Optional[str] = None,
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Transcribe audio to text."""
        try:
            # Handle different input types
            if isinstance(audio_file, str):
                # File path
                async with aiofiles.open(audio_file, 'rb') as f:
                    audio_data = await f.read()
                filename = filename or audio_file.split('/')[-1]
            else:
                # Bytes
                audio_data = audio_file
                filename = filename or "audio.mp3"
            
            # Transcribe
            response = await self.client.transcribe_audio(
                audio_file=audio_data,
                filename=filename,
                model=model,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature
            )
            
            return {
                "text": response["text"],
                "model": model,
                "language": language,
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise
    
    async def translate_audio(
        self,
        audio_file: Union[str, bytes],
        filename: Optional[str] = None,
        model: str = "whisper-1",
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Translate audio to English text."""
        try:
            # Handle different input types
            if isinstance(audio_file, str):
                async with aiofiles.open(audio_file, 'rb') as f:
                    audio_data = await f.read()
                filename = filename or audio_file.split('/')[-1]
            else:
                audio_data = audio_file
                filename = filename or "audio.mp3"
            
            # Translate using OpenAI API
            response = await self.client.audio.translations.create(
                file=audio_data,
                model=model,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature
            )
            
            return {
                "text": response.text,
                "model": model,
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"Audio translation failed: {e}")
            raise
    
    async def transcribe_with_timestamps(
        self,
        audio_file: Union[str, bytes],
        filename: Optional[str] = None,
        model: str = "whisper-1"
    ) -> Dict[str, Any]:
        """Transcribe audio with word-level timestamps."""
        try:
            # Note: This requires the response_format to be "verbose_json"
            result = await self.transcribe_audio(
                audio_file=audio_file,
                filename=filename,
                model=model,
                response_format="verbose_json"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Timestamped transcription failed: {e}")
            raise

# Global audio processor
openai_audio = OpenAIAudio()
```

## 📊 Fine-tuning and Training

### 1. Fine-tuning Manager

```python
# prsm/integrations/openai/fine_tuning.py
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import aiofiles

from prsm.integrations.openai.client import openai_client
from prsm.integrations.openai.config import openai_config

logger = logging.getLogger(__name__)

class OpenAIFineTuning:
    """Manage OpenAI fine-tuning operations."""
    
    def __init__(self):
        self.client = openai_client
        self.config = openai_config
    
    async def prepare_training_data(
        self,
        conversations: List[Dict[str, Any]],
        output_file: str = "training_data.jsonl"
    ) -> str:
        """Prepare training data in the correct format."""
        try:
            training_examples = []
            
            for conversation in conversations:
                if "messages" in conversation:
                    # Already in correct format
                    training_examples.append({
                        "messages": conversation["messages"]
                    })
                elif "prompt" in conversation and "completion" in conversation:
                    # Convert prompt/completion to messages format
                    training_examples.append({
                        "messages": [
                            {"role": "user", "content": conversation["prompt"]},
                            {"role": "assistant", "content": conversation["completion"]}
                        ]
                    })
                else:
                    logger.warning(f"Skipping invalid conversation format: {conversation}")
            
            # Write to JSONL file
            async with aiofiles.open(output_file, 'w') as f:
                for example in training_examples:
                    await f.write(json.dumps(example) + '\n')
            
            logger.info(f"Prepared {len(training_examples)} training examples in {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            raise
    
    async def upload_training_file(self, file_path: str) -> str:
        """Upload training file to OpenAI."""
        try:
            async with aiofiles.open(file_path, 'rb') as f:
                file_content = await f.read()
            
            response = await self.client.files.create(
                file=file_content,
                purpose="fine-tune"
            )
            
            file_id = response.id
            logger.info(f"Uploaded training file: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            raise
    
    async def create_fine_tune_job(
        self,
        training_file_id: str,
        model: Optional[str] = None,
        validation_file_id: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None
    ) -> str:
        """Create a fine-tuning job."""
        try:
            model = model or self.config.fine_tune_model
            
            # Prepare hyperparameters
            if not hyperparameters:
                hyperparameters = {
                    "n_epochs": self.config.fine_tune_epochs,
                    "batch_size": self.config.fine_tune_batch_size,
                    "learning_rate_multiplier": self.config.fine_tune_learning_rate_multiplier
                }
            
            # Create fine-tuning job
            response = await self.client.fine_tuning.jobs.create(
                training_file=training_file_id,
                model=model,
                validation_file=validation_file_id,
                hyperparameters=hyperparameters,
                suffix=suffix
            )
            
            job_id = response.id
            logger.info(f"Created fine-tuning job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Fine-tuning job creation failed: {e}")
            raise
    
    async def get_fine_tune_status(self, job_id: str) -> Dict[str, Any]:
        """Get fine-tuning job status."""
        try:
            response = await self.client.fine_tuning.jobs.retrieve(job_id)
            
            return {
                "id": response.id,
                "status": response.status,
                "model": response.model,
                "fine_tuned_model": response.fine_tuned_model,
                "created_at": response.created_at,
                "finished_at": response.finished_at,
                "training_file": response.training_file,
                "validation_file": response.validation_file,
                "hyperparameters": response.hyperparameters,
                "result_files": response.result_files,
                "trained_tokens": response.trained_tokens
            }
            
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            raise
    
    async def list_fine_tune_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List fine-tuning jobs."""
        try:
            response = await self.client.fine_tuning.jobs.list(limit=limit)
            
            return [
                {
                    "id": job.id,
                    "status": job.status,
                    "model": job.model,
                    "fine_tuned_model": job.fine_tuned_model,
                    "created_at": job.created_at
                }
                for job in response.data
            ]
            
        except Exception as e:
            logger.error(f"Job listing failed: {e}")
            raise
    
    async def cancel_fine_tune_job(self, job_id: str) -> bool:
        """Cancel a fine-tuning job."""
        try:
            await self.client.fine_tuning.jobs.cancel(job_id)
            logger.info(f"Cancelled fine-tuning job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Job cancellation failed: {e}")
            return False

# Global fine-tuning manager
openai_fine_tuning = OpenAIFineTuning()
```

## 📋 FastAPI Integration

### API Endpoints

```python
# prsm/api/openai_endpoints.py
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
import base64

from prsm.integrations.openai.client import openai_client
from prsm.integrations.openai.function_calling import openai_function_caller
from prsm.integrations.openai.image_generation import openai_image_generator
from prsm.integrations.openai.vision import openai_vision
from prsm.integrations.openai.audio import openai_audio
from prsm.models.query import QueryRequest, QueryResponse

router = APIRouter(prefix="/api/v1/openai", tags=["OpenAI"])

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False

class EmbeddingRequest(BaseModel):
    texts: Union[str, List[str]]
    model: Optional[str] = None
    dimensions: Optional[int] = None

class ImageGenerationRequest(BaseModel):
    prompt: str
    model: str = "dall-e-3"
    size: str = "1024x1024"
    quality: str = "standard"
    style: str = "natural"
    n: int = 1

class FunctionCallingRequest(BaseModel):
    prompt: str
    user_id: str
    available_functions: Optional[List[str]] = None
    context: Optional[str] = None

@router.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Create chat completion."""
    try:
        if request.stream:
            # Streaming response
            async def generate():
                async for chunk in openai_client.chat_completion_stream(
                    messages=request.messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = await openai_client.chat_completion(
                messages=request.messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """Create embeddings."""
    try:
        response = await openai_client.create_embeddings(
            texts=request.texts,
            model=request.model,
            dimensions=request.dimensions
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/images/generations")
async def generate_image(request: ImageGenerationRequest):
    """Generate image with DALL-E."""
    try:
        response = await openai_image_generator.generate_image(
            prompt=request.prompt,
            model=request.model,
            size=request.size,
            quality=request.quality,
            style=request.style,
            n=request.n
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vision/analyze")
async def analyze_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    model: str = Form("gpt-4-vision-preview")
):
    """Analyze image with GPT-4 Vision."""
    try:
        image_data = await image.read()
        
        response = await openai_vision.analyze_image(
            image=image_data,
            prompt=prompt,
            model=model
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/audio/transcriptions")
async def transcribe_audio(
    audio: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None)
):
    """Transcribe audio with Whisper."""
    try:
        audio_data = await audio.read()
        
        response = await openai_audio.transcribe_audio(
            audio_file=audio_data,
            filename=audio.filename,
            model=model,
            language=language
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/function-calling")
async def function_calling(request: FunctionCallingRequest):
    """Process query with function calling."""
    try:
        query_request = QueryRequest(
            prompt=request.prompt,
            user_id=request.user_id,
            context=request.context
        )
        
        response = await openai_function_caller.process_with_functions(
            query_request,
            available_functions=request.available_functions
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """List available OpenAI models."""
    try:
        models = await openai_client.client.models.list()
        return {"models": [model.id for model in models.data]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

**Need help with OpenAI integration?** Join our [Discord community](https://discord.gg/prsm) or [open an issue](https://github.com/prsm-org/prsm/issues).