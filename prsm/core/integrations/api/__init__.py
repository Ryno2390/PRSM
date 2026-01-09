"""
Integration API Endpoints
=========================

REST API endpoints for managing platform integrations, extending
PRSM's existing API with integration-specific functionality.
"""

from .integration_api import integration_router

__all__ = ["integration_router"]