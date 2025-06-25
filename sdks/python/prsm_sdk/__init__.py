"""
PRSM Python SDK
Official Python client for the Protocol for Recursive Scientific Modeling

🎯 MAIN EXPORTS:
- PRSMClient: Main client for interacting with PRSM API
- PRSMResponse: Response object for PRSM queries  
- FTNSManager: Token balance and cost management
- ModelMarketplace: Access to PRSM model ecosystem
- ToolExecutor: MCP tool protocol integration

🚀 QUICK START:
    from prsm_sdk import PRSMClient
    
    client = PRSMClient(api_key="your_api_key")
    response = await client.query("Explain quantum computing")
    print(response.content)

📚 DOCUMENTATION: https://docs.prsm.ai/python-sdk
"""

from .__version__ import __version__
from .client import PRSMClient
from .models import (
    PRSMResponse,
    QueryRequest,
    FTNSBalance,
    ModelInfo,
    ToolSpec,
    SafetyStatus
)
from .auth import AuthManager
from .ftns import FTNSManager
from .marketplace import ModelMarketplace
from .tools import ToolExecutor
from .exceptions import (
    PRSMError,
    AuthenticationError,
    InsufficientFundsError,
    SafetyViolationError,
    NetworkError
)

__all__ = [
    "__version__",
    "PRSMClient",
    "PRSMResponse", 
    "QueryRequest",
    "FTNSBalance",
    "ModelInfo",
    "ToolSpec",
    "SafetyStatus",
    "AuthManager",
    "FTNSManager", 
    "ModelMarketplace",
    "ToolExecutor",
    "PRSMError",
    "AuthenticationError",
    "InsufficientFundsError", 
    "SafetyViolationError",
    "NetworkError",
]

# Package metadata
__author__ = "PRSM Development Team"
__email__ = "dev@prsm.ai"
__description__ = "Official Python SDK for PRSM (Protocol for Recursive Scientific Modeling)"
__url__ = "https://github.com/PRSM-AI/PRSM"
__license__ = "MIT"