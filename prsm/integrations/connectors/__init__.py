"""
Platform Connectors
===================

Platform-specific integration implementations for major collaborative
coding and AI model platforms.

Available Connectors:
- GitHub: Repository and codebase integration
- Hugging Face: Model Hub and dataset integration  
- Ollama: Local LLM model integration
- Meta/LLaMA: LLaMA model variants
- DeepSeek: Multilingual and code-oriented models
- Qwen: Alibaba's multilingual models
- Mistral: Mistral AI model integration
"""

from .github_connector import GitHubConnector
from .huggingface_connector import HuggingFaceConnector

__all__ = [
    "GitHubConnector",
    "HuggingFaceConnector"
]