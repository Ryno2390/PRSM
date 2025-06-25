"""
PRSM LangChain Integration
========================

LangChain provider for PRSM that enables seamless integration with the LangChain ecosystem.
This module provides PRSM-compatible implementations of LangChain interfaces including
LLMs, chat models, embeddings, and agent tools.

Key Components:
- PRSMLangChainLLM: LangChain LLM wrapper for PRSM
- PRSMChatModel: Chat model implementation
- PRSMEmbeddings: Embeddings provider
- PRSMAgentTools: Collection of PRSM-powered tools for agents
- PRSMRetriever: Document retriever using PRSM's knowledge base

Example Usage:
    from prsm.integrations.langchain import PRSMLangChainLLM, PRSMAgentTools
    
    # Use PRSM as LangChain LLM
    llm = PRSMLangChainLLM(base_url="http://localhost:8000", api_key="your-key")
    result = llm("What is machine learning?")
    
    # Use PRSM tools in LangChain agents
    tools = PRSMAgentTools(prsm_client=prsm_client)
    agent = initialize_agent(tools.get_tools(), llm, agent="zero-shot-react-description")
"""

from .llm import PRSMLangChainLLM
from .chat_model import PRSMChatModel
from .embeddings import PRSMEmbeddings
from .tools import PRSMAgentTools
from .retriever import PRSMRetriever
from .chains import PRSMChain, PRSMConversationChain
from .memory import PRSMChatMemory

__all__ = [
    "PRSMLangChainLLM",
    "PRSMChatModel",
    "PRSMEmbeddings",
    "PRSMAgentTools",
    "PRSMRetriever",
    "PRSMChain",
    "PRSMConversationChain",
    "PRSMChatMemory",
]

# Version info
__version__ = "0.1.0"
__author__ = "PRSM Team"
__description__ = "LangChain integration for PRSM (Protocol for Recursive Scientific Modeling)"
