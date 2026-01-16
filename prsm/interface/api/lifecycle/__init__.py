"""
Application Lifecycle Management
================================

Handles application startup and shutdown sequences.
"""

from .startup import startup_sequence
from .shutdown import shutdown_sequence

__all__ = ["startup_sequence", "shutdown_sequence"]
