#!/usr/bin/env python3
"""
Simple data models for PRSM collaboration components
==================================================

Basic data models to support the collaboration integration tests.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class QueryRequest:
    """Simple query request model for testing"""
    user_id: str
    query_text: str
    context: Optional[Dict[str, Any]] = None