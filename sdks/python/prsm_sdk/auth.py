"""
PRSM SDK Authentication Manager
Handles API key authentication and session management
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class AuthConfig:
    """Authentication configuration"""
    api_key: Optional[str] = None
    api_key_env_var: str = "PRSM_API_KEY"
    session_timeout: int = 3600  # 1 hour


class AuthManager:
    """
    Manages authentication for PRSM API requests
    
    Handles:
    - API key management
    - Session token generation
    - Header construction for authenticated requests
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[AuthConfig] = None):
        """
        Initialize authentication manager
        
        Args:
            api_key: PRSM API key (can also be set via PRSM_API_KEY env var)
            config: Optional authentication configuration
        """
        self.config = config or AuthConfig()
        self._api_key = api_key or self._get_api_key_from_env()
        self._session_token: Optional[str] = None
        self._session_expires: Optional[datetime] = None
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable"""
        return os.environ.get(self.config.api_key_env_var)
    
    @property
    def api_key(self) -> Optional[str]:
        """Get the current API key"""
        return self._api_key
    
    @api_key.setter
    def api_key(self, value: str) -> None:
        """Set the API key"""
        self._api_key = value
        self._session_token = None  # Reset session when API key changes
        self._session_expires = None
    
    def is_authenticated(self) -> bool:
        """Check if we have valid authentication"""
        if self._api_key:
            return True
        if self._session_token and self._session_expires:
            return datetime.utcnow() < self._session_expires
        return False
    
    async def get_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for API requests
        
        Returns:
            Dictionary of headers to include in requests
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        elif self._session_token:
            headers["Authorization"] = f"Bearer {self._session_token}"
        
        return headers
    
    async def create_session(self) -> str:
        """
        Create a new authenticated session
        
        Returns:
            Session token
        """
        if not self._api_key:
            raise ValueError("API key required to create session")
        
        # Session token generation (in production, this would be server-side)
        import secrets
        self._session_token = secrets.token_urlsafe(32)
        self._session_expires = datetime.utcnow() + timedelta(seconds=self.config.session_timeout)
        
        return self._session_token
    
    async def refresh_session(self) -> str:
        """
        Refresh an existing session
        
        Returns:
            New session token
        """
        return await self.create_session()
    
    def clear_session(self) -> None:
        """Clear the current session"""
        self._session_token = None
        self._session_expires = None