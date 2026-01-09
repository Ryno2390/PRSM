"""
PRSM Integration Layer
=====================

The integration layer enables seamless connectivity to major collaborative 
coding platforms while preserving PRSM's core principles of decentralization, 
safety, and economic incentives.

Key Features:
- One-click model import from GitHub, Hugging Face, Ollama, and more
- Automated FTNS provenance tracking for creator rewards
- Security sandbox for safe external content evaluation
- License compliance and vulnerability scanning
- Integration with PRSM's existing safety infrastructure

Components:
- core/: Base classes and coordination logic
- connectors/: Platform-specific integration implementations
- security/: Sandbox and compliance validation
- models/: Integration-specific data models
- api/: REST endpoints for integration management
"""

from .core.integration_manager import IntegrationManager
from .connectors.github_connector import GitHubConnector
from .connectors.huggingface_connector import HuggingFaceConnector
from .connectors.ollama_connector import OllamaConnector
from .models.integration_models import (
    IntegrationSource,
    IntegrationRecord,
    ConnectorConfig,
    ImportRequest,
    ImportResult
)

__version__ = "1.0.0"
__all__ = [
    "IntegrationManager",
    "GitHubConnector",
    "HuggingFaceConnector", 
    "OllamaConnector",
    "IntegrationSource", 
    "IntegrationRecord",
    "ConnectorConfig",
    "ImportRequest",
    "ImportResult"
]