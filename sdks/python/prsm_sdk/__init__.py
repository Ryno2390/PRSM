"""
PRSM Python SDK
Official Python client for the Protocol for Recursive Scientific Modeling

🎯 MAIN EXPORTS:
- PRSMClient: Main client for interacting with PRSM API
- PRSMResponse: Response object for PRSM queries  
- FTNSManager: Token balance and cost management
- ModelMarketplace: Access to PRSM model ecosystem
- ToolExecutor: MCP tool protocol integration
- ComputeClient: Submit and manage compute jobs
- StorageClient: IPFS storage operations
- GovernanceClient: Participate in governance

🚀 QUICK START:
    from prsm_sdk import PRSMClient
    
    async with PRSMClient(api_key="your_api_key") as client:
        # Execute a query
        response = await client.query("Explain quantum computing")
        print(response.content)
        
        # Check balance
        balance = await client.ftns.get_balance()
        print(f"Balance: {balance.available_balance} FTNS")
        
        # Submit compute job
        job = await client.compute.submit_job("Analyze data")
        result = await client.compute.wait_for_completion(job.job_id)
        
        # Upload to storage
        result = await client.storage.upload_bytes(b"content")
        print(f"Uploaded to: {result.cid}")
        
        # Vote on governance
        proposals = await client.governance.list_proposals()
        await client.governance.vote(proposals[0].proposal_id, VoteChoice.YES)

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
    SafetyStatus,
    SafetyLevel,
    ModelProvider,
    WebSocketMessage,
    MarketplaceQuery,
    ToolExecutionRequest,
    ToolExecutionResponse
)
from .auth import AuthManager, AuthConfig
from .ftns import FTNSManager, FTNSBalance, Transaction, StakeInfo, TransferRequest, TransferResponse
from .marketplace import ModelMarketplace, ModelInfo, ModelCategory, ModelSearchRequest, ModelSearchResult, ModelRental, ModelStats
from .tools import ToolExecutor, ToolInfo, ToolCategory, ToolExecutionRequest, ToolExecutionResult, ToolSearchResult
from .compute import ComputeClient, JobStatus, JobPriority, JobRequest, JobResponse, JobResult, JobInfo
from .storage import StorageClient, StorageStatus, ContentType, StorageInfo, StorageUploadResult, PinInfo
from .governance import GovernanceClient, ProposalStatus, ProposalType, VoteChoice, Proposal, Vote, GovernanceStats
from .exceptions import (
    PRSMError,
    AuthenticationError,
    InsufficientFundsError,
    SafetyViolationError,
    NetworkError,
    ModelNotFoundError,
    ToolExecutionError,
    RateLimitError,
    ValidationError
)

__all__ = [
    # Version
    "__version__",
    
    # Main client
    "PRSMClient",
    
    # Models
    "PRSMResponse", 
    "QueryRequest",
    "FTNSBalance",
    "ModelInfo",
    "ToolSpec",
    "SafetyStatus",
    "SafetyLevel",
    "ModelProvider",
    "WebSocketMessage",
    "MarketplaceQuery",
    "ToolExecutionRequest",
    "ToolExecutionResponse",
    
    # Auth
    "AuthManager",
    "AuthConfig",
    
    # FTNS
    "FTNSManager",
    "Transaction",
    "StakeInfo",
    "TransferRequest",
    "TransferResponse",
    
    # Marketplace
    "ModelMarketplace",
    "ModelCategory",
    "ModelSearchRequest",
    "ModelSearchResult",
    "ModelRental",
    "ModelStats",
    
    # Tools
    "ToolExecutor",
    "ToolInfo",
    "ToolCategory",
    "ToolExecutionResult",
    "ToolSearchResult",
    
    # Compute
    "ComputeClient",
    "JobStatus",
    "JobPriority",
    "JobRequest",
    "JobResponse",
    "JobResult",
    "JobInfo",
    
    # Storage
    "StorageClient",
    "StorageStatus",
    "ContentType",
    "StorageInfo",
    "StorageUploadResult",
    "PinInfo",
    
    # Governance
    "GovernanceClient",
    "ProposalStatus",
    "ProposalType",
    "VoteChoice",
    "Proposal",
    "Vote",
    "GovernanceStats",
    
    # Exceptions
    "PRSMError",
    "AuthenticationError",
    "InsufficientFundsError", 
    "SafetyViolationError",
    "NetworkError",
    "ModelNotFoundError",
    "ToolExecutionError",
    "RateLimitError",
    "ValidationError",
]

# Package metadata
__author__ = "PRSM Development Team"
__email__ = "dev@prsm.ai"
__description__ = "Official Python SDK for PRSM (Protocol for Recursive Scientific Modeling)"
__url__ = "https://github.com/PRSM-AI/PRSM"
__license__ = "MIT"