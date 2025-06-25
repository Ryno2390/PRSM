"""
PRSM Knowledge Diffing System
=============================

The 13th subsystem providing automated epistemic alignment and knowledge base
integrity through continuous comparison with external data sources. Prevents
closed-loop drift and maintains scientific accuracy as PRSM scales.

Core Components:
- External data collection and crawling
- Semantic embedding comparison and analysis
- Divergence detection and gap identification
- Privacy-preserving diffing protocols
- FTNS-incentivized community curation
- Governance-directed resource allocation

Key Features:
- Automated external source monitoring
- Privacy-preserving comparison techniques
- Real-time divergence detection
- Community-driven knowledge curation
- Economic incentives for gap resolution
- Democratic prioritization of diffing efforts
"""

from .external_collector import external_data_collector
from .embedding_analyzer import semantic_embedding_analyzer
from .divergence_detector import knowledge_divergence_detector
from .privacy_diffing import privacy_preserving_diffing_engine
from .community_curator import community_curation_system
from .diffing_orchestrator import knowledge_diffing_orchestrator

__version__ = "1.0.0-alpha"

__all__ = [
    "external_data_collector",
    "semantic_embedding_analyzer", 
    "knowledge_divergence_detector",
    "privacy_preserving_diffing_engine",
    "community_curation_system",
    "knowledge_diffing_orchestrator"
]