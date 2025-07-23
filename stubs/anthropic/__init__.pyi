"""
Type stubs for anthropic
========================

Provides type annotations for the Anthropic API client library
to support static type checking in the PRSM codebase.
"""

from typing import Any, Dict, List, Optional, Union, AsyncIterator, Iterator
from typing_extensions import Literal
import httpx

# Message types
MessageRole = Literal["user", "assistant", "system"]

class ContentBlock:
    """Base class for content blocks"""
    type: str

class TextBlock(ContentBlock):
    """Text content block"""
    type: Literal["text"]
    text: str
    
class ImageBlock(ContentBlock):
    """Image content block"""
    type: Literal["image"]
    source: Dict[str, Any]

class ToolUseBlock(ContentBlock):
    """Tool use content block"""
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ToolResultBlock(ContentBlock):
    """Tool result content block"""
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[ContentBlock]]
    is_error: Optional[bool]

ContentBlockType = Union[TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock]

class Message:
    """Anthropic message object"""
    role: MessageRole
    content: Union[str, List[ContentBlockType]]
    
class MessageParam:
    """Message parameter for API calls"""
    role: MessageRole
    content: Union[str, List[Dict[str, Any]]]

class Usage:
    """Token usage information"""
    input_tokens: int
    output_tokens: int

class MessageResponse:
    """Response from messages API"""
    id: str
    type: Literal["message"]
    role: Literal["assistant"] 
    content: List[ContentBlockType]
    model: str
    stop_reason: Optional[str]
    stop_sequence: Optional[str]
    usage: Usage

class MessageStreamEvent:
    """Streaming message event"""
    type: str
    
class MessageStartEvent(MessageStreamEvent):
    type: Literal["message_start"]
    message: MessageResponse
    
class ContentBlockStartEvent(MessageStreamEvent):
    type: Literal["content_block_start"]
    index: int
    content_block: ContentBlockType
    
class ContentBlockDeltaEvent(MessageStreamEvent):
    type: Literal["content_block_delta"]
    index: int
    delta: Dict[str, Any]
    
class ContentBlockStopEvent(MessageStreamEvent):
    type: Literal["content_block_stop"]
    index: int
    
class MessageDeltaEvent(MessageStreamEvent):
    type: Literal["message_delta"]
    delta: Dict[str, Any]
    usage: Usage
    
class MessageStopEvent(MessageStreamEvent):
    type: Literal["message_stop"]

MessageStreamEventType = Union[
    MessageStartEvent,
    ContentBlockStartEvent, 
    ContentBlockDeltaEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageStopEvent
]

class Tool:
    """Tool definition for function calling"""
    name: str
    description: str
    input_schema: Dict[str, Any]

# API Resource classes
class Messages:
    """Messages API resource"""
    def create(
        self,
        *,
        model: str,
        messages: List[MessageParam],
        max_tokens: int,
        metadata: Optional[Dict[str, Any]] = ...,
        stop_sequences: Optional[List[str]] = ...,
        stream: Literal[False] = False,
        system: Optional[str] = ...,
        temperature: Optional[float] = ...,
        tools: Optional[List[Tool]] = ...,
        top_k: Optional[int] = ...,
        top_p: Optional[float] = ...
    ) -> MessageResponse: ...
    
    def create(
        self,
        *,
        model: str,
        messages: List[MessageParam],
        max_tokens: int,
        metadata: Optional[Dict[str, Any]] = ...,
        stop_sequences: Optional[List[str]] = ...,
        stream: Literal[True],
        system: Optional[str] = ...,
        temperature: Optional[float] = ...,
        tools: Optional[List[Tool]] = ...,
        top_k: Optional[int] = ...,
        top_p: Optional[float] = ...
    ) -> Iterator[MessageStreamEventType]: ...

class AsyncMessages:
    """Async Messages API resource"""  
    async def create(
        self,
        *,
        model: str,
        messages: List[MessageParam],
        max_tokens: int,
        metadata: Optional[Dict[str, Any]] = ...,
        stop_sequences: Optional[List[str]] = ...,
        stream: Literal[False] = False,
        system: Optional[str] = ...,
        temperature: Optional[float] = ...,
        tools: Optional[List[Tool]] = ...,
        top_k: Optional[int] = ...,
        top_p: Optional[float] = ...
    ) -> MessageResponse: ...
    
    async def create(
        self,
        *,
        model: str,
        messages: List[MessageParam],
        max_tokens: int,
        metadata: Optional[Dict[str, Any]] = ...,
        stop_sequences: Optional[List[str]] = ...,
        stream: Literal[True],
        system: Optional[str] = ...,
        temperature: Optional[float] = ...,
        tools: Optional[List[Tool]] = ...,
        top_k: Optional[int] = ...,
        top_p: Optional[float] = ...
    ) -> AsyncIterator[MessageStreamEventType]: ...

# Exception classes
class AnthropicError(Exception):
    """Base exception for Anthropic API errors"""
    pass

class APIError(AnthropicError):
    """API error"""
    pass

class APIConnectionError(AnthropicError):
    """API connection error"""
    pass

class APITimeoutError(AnthropicError):
    """API timeout error"""
    pass

class RateLimitError(APIError):
    """Rate limit exceeded error"""
    pass

class AuthenticationError(APIError):
    """Authentication error"""  
    pass

class BadRequestError(APIError):
    """Bad request error"""
    pass

class InternalServerError(APIError):
    """Internal server error"""
    pass

# Main client classes
class Anthropic:
    """Synchronous Anthropic API client"""
    messages: Messages
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = ...,
        base_url: Optional[str] = ...,
        timeout: Optional[float] = ...,
        max_retries: int = ...,
        default_headers: Optional[Dict[str, str]] = ...,
        default_query: Optional[Dict[str, Any]] = ...,
        http_client: Optional[httpx.Client] = ...
    ) -> None: ...

class AsyncAnthropic:
    """Asynchronous Anthropic API client"""
    messages: AsyncMessages
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = ...,
        base_url: Optional[str] = ...,
        timeout: Optional[float] = ...,
        max_retries: int = ...,
        default_headers: Optional[Dict[str, str]] = ...,
        default_query: Optional[Dict[str, Any]] = ...,
        http_client: Optional[httpx.AsyncClient] = ...
    ) -> None: ...
    
    async def close(self) -> None: ...
    async def __aenter__(self) -> AsyncAnthropic: ...
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...

# Model constants
CLAUDE_3_OPUS = "claude-3-opus-20240229"
CLAUDE_3_SONNET = "claude-3-sonnet-20240229"  
CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"