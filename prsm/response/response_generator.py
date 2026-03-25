"""
Response Generator
==================

Generate responses for PRSM queries.
"""

from typing import Dict, Any, Optional


class ResponseGenerator:
    """Generate responses for queries."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    async def generate(self, query: str, context: Dict[str, Any] = None) -> str:
        """Generate a response."""
        return f"Response to: {query}"
