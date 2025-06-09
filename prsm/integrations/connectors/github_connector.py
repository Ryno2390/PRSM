"""
GitHub Connector
===============

GitHub API integration connector for PRSM, enabling seamless repository
and codebase integration with OAuth authentication and comprehensive
metadata extraction.

Key Features:
- OAuth-based repository access and authentication
- Repository search and discovery
- Commit-level provenance tracking for FTNS rewards
- License compliance validation
- Content download and metadata extraction
- Rate limiting and error handling
"""

import asyncio
import base64
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from urllib.parse import quote

import aiohttp

from ..core.base_connector import BaseConnector, ConnectorStatus
from ..models.integration_models import (
    IntegrationPlatform, ConnectorConfig, IntegrationSource,
    ImportRequest, ImportResult, SecurityRisk, LicenseType
)
from ...core.config import settings


class GitHubConnector(BaseConnector):
    """
    GitHub API integration connector
    
    Provides comprehensive GitHub repository integration including:
    - OAuth authentication and token management
    - Repository search and metadata extraction
    - Content download with commit-level tracking
    - License compliance validation
    - Creator attribution for FTNS rewards
    """
    
    def __init__(self, config: ConnectorConfig):
        """
        Initialize GitHub connector
        
        Args:
            config: ConnectorConfig with GitHub OAuth credentials
        """
        super().__init__(config)
        
        # GitHub API configuration
        self.api_base_url = "https://api.github.com"
        self.github_base_url = "https://github.com"
        self.api_version = "2022-11-28"
        
        # Authentication
        self.access_token = config.oauth_credentials.get("access_token") if config.oauth_credentials else None
        self.client_id = getattr(settings, "PRSM_GITHUB_CLIENT_ID", None)
        self.client_secret = getattr(settings, "PRSM_GITHUB_CLIENT_SECRET", None)
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.rate_limit_remaining = 5000  # GitHub's default
        self.rate_limit_reset = None
        
        print(f"🐙 GitHub Connector initialized for user {self.user_id}")
    
    # === Authentication Methods ===
    
    async def authenticate(self) -> bool:
        """
        Authenticate with GitHub using OAuth token
        
        Returns:
            True if authentication successful
        """
        try:
            if not self.access_token:
                print("❌ No GitHub access token provided")
                return False
            
            # Create HTTP session with authentication
            await self._create_session()
            
            # Test authentication by getting user info
            user_info = await self._make_api_request("/user")
            
            if user_info:
                self.authenticated_user = user_info.get("login")
                print(f"✅ GitHub authentication successful for user: {self.authenticated_user}")
                return True
            else:
                print("❌ GitHub authentication failed")
                return False
                
        except Exception as e:
            print(f"❌ GitHub authentication error: {e}")
            return False
    
    async def _create_session(self) -> None:
        """Create authenticated HTTP session"""
        if self.session:
            await self.session.close()
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": self.api_version,
            "User-Agent": "PRSM-Integration/1.0"
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
    
    # === Content Discovery Methods ===
    
    async def search_content(self, query: str, content_type: str = "repository",
                           limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[IntegrationSource]:
        """
        Search GitHub for repositories and content
        
        Args:
            query: Search query string
            content_type: Type of content (repository, code, user)
            limit: Maximum results to return
            filters: Additional search filters (language, size, stars, etc.)
            
        Returns:
            List of IntegrationSource objects
        """
        try:
            print(f"🔍 Searching GitHub for '{query}' (type: {content_type})")
            
            # Build search query
            search_query = await self._build_search_query(query, content_type, filters)
            
            # Execute search based on content type
            if content_type == "repository":
                results = await self._search_repositories(search_query, limit)
            elif content_type == "code":
                results = await self._search_code(search_query, limit)
            else:
                print(f"⚠️ Unsupported content type: {content_type}")
                return []
            
            # Convert to IntegrationSource objects
            integration_sources = []
            for item in results:
                source = await self._convert_to_integration_source(item, content_type)
                if source:
                    integration_sources.append(source)
            
            print(f"📊 Found {len(integration_sources)} GitHub results")
            return integration_sources
            
        except Exception as e:
            print(f"❌ GitHub search error: {e}")
            return []
    
    async def _search_repositories(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search GitHub repositories"""
        endpoint = f"/search/repositories?q={quote(query)}&sort=stars&order=desc&per_page={min(limit, 100)}"
        response = await self._make_api_request(endpoint)
        
        if response and "items" in response:
            return response["items"]
        return []
    
    async def _search_code(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search GitHub code"""
        endpoint = f"/search/code?q={quote(query)}&sort=indexed&order=desc&per_page={min(limit, 100)}"
        response = await self._make_api_request(endpoint)
        
        if response and "items" in response:
            return response["items"]
        return []
    
    async def _build_search_query(self, query: str, content_type: str, 
                                filters: Optional[Dict[str, Any]]) -> str:
        """Build GitHub search query with filters"""
        search_parts = [query]
        
        if filters:
            # Language filter
            if "language" in filters:
                search_parts.append(f"language:{filters['language']}")
            
            # Repository size filter
            if "size" in filters:
                search_parts.append(f"size:{filters['size']}")
            
            # Stars filter
            if "stars" in filters:
                search_parts.append(f"stars:{filters['stars']}")
            
            # License filter
            if "license" in filters:
                search_parts.append(f"license:{filters['license']}")
            
            # Pushed date filter
            if "pushed" in filters:
                search_parts.append(f"pushed:{filters['pushed']}")
        
        return " ".join(search_parts)
    
    async def _convert_to_integration_source(self, item: Dict[str, Any], 
                                           content_type: str) -> Optional[IntegrationSource]:
        """Convert GitHub API response to IntegrationSource"""
        try:
            if content_type == "repository":
                return IntegrationSource(
                    platform=IntegrationPlatform.GITHUB,
                    external_id=item["full_name"],
                    display_name=item["name"],
                    description=item.get("description", ""),
                    owner_id=item["owner"]["login"],
                    url=item["html_url"],
                    metadata={
                        "stars": item.get("stargazers_count", 0),
                        "forks": item.get("forks_count", 0),
                        "language": item.get("language"),
                        "size": item.get("size", 0),
                        "created_at": item.get("created_at"),
                        "updated_at": item.get("updated_at"),
                        "pushed_at": item.get("pushed_at"),
                        "license": item.get("license"),
                        "topics": item.get("topics", []),
                        "default_branch": item.get("default_branch", "main")
                    }
                )
            elif content_type == "code":
                repo = item.get("repository", {})
                return IntegrationSource(
                    platform=IntegrationPlatform.GITHUB,
                    external_id=f"{repo.get('full_name', '')}/blob/{item.get('sha', '')}/{item.get('path', '')}",
                    display_name=item.get("name", ""),
                    description=f"Code file in {repo.get('full_name', '')}",
                    owner_id=repo.get("owner", {}).get("login", ""),
                    url=item.get("html_url", ""),
                    metadata={
                        "repository": repo.get("full_name"),
                        "path": item.get("path"),
                        "sha": item.get("sha"),
                        "size": item.get("size", 0),
                        "type": "file"
                    }
                )
        
        except Exception as e:
            print(f"⚠️ Error converting GitHub item to IntegrationSource: {e}")
            return None
    
    # === Content Metadata Methods ===
    
    async def get_content_metadata(self, external_id: str) -> Dict[str, Any]:
        """
        Get detailed metadata for GitHub repository or file
        
        Args:
            external_id: GitHub repository full name (owner/repo) or file path
            
        Returns:
            Comprehensive metadata dictionary
        """
        try:
            print(f"📋 Fetching GitHub metadata for: {external_id}")
            
            # Determine if this is a repository or file
            if "/blob/" in external_id:
                return await self._get_file_metadata(external_id)
            else:
                return await self._get_repository_metadata(external_id)
                
        except Exception as e:
            print(f"❌ Error fetching GitHub metadata: {e}")
            return {"error": str(e)}
    
    async def _get_repository_metadata(self, repo_full_name: str) -> Dict[str, Any]:
        """Get comprehensive repository metadata"""
        try:
            # Get basic repository info
            repo_info = await self._make_api_request(f"/repos/{repo_full_name}")
            if not repo_info:
                return {"error": "Repository not found"}
            
            # Get additional metadata in parallel
            tasks = [
                self._get_repository_languages(repo_full_name),
                self._get_repository_contributors(repo_full_name),
                self._get_repository_readme(repo_full_name),
                self._get_repository_releases(repo_full_name),
                self._get_repository_commits(repo_full_name, limit=5)
            ]
            
            languages, contributors, readme, releases, recent_commits = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            
            # Build comprehensive metadata
            metadata = {
                "type": "repository",
                "full_name": repo_info["full_name"],
                "name": repo_info["name"],
                "description": repo_info.get("description", ""),
                "owner": repo_info["owner"]["login"],
                "creator": repo_info["owner"]["login"],  # For provenance
                "private": repo_info.get("private", False),
                "html_url": repo_info["html_url"],
                "clone_url": repo_info["clone_url"],
                "git_url": repo_info["git_url"],
                "ssh_url": repo_info["ssh_url"],
                "size": repo_info.get("size", 0),
                "stargazers_count": repo_info.get("stargazers_count", 0),
                "watchers_count": repo_info.get("watchers_count", 0),
                "forks_count": repo_info.get("forks_count", 0),
                "open_issues_count": repo_info.get("open_issues_count", 0),
                "default_branch": repo_info.get("default_branch", "main"),
                "language": repo_info.get("language"),
                "license": repo_info.get("license"),
                "topics": repo_info.get("topics", []),
                "created_at": repo_info.get("created_at"),
                "updated_at": repo_info.get("updated_at"),
                "pushed_at": repo_info.get("pushed_at"),
                "archived": repo_info.get("archived", False),
                "disabled": repo_info.get("disabled", False),
                "has_issues": repo_info.get("has_issues", False),
                "has_projects": repo_info.get("has_projects", False),
                "has_wiki": repo_info.get("has_wiki", False),
                "has_pages": repo_info.get("has_pages", False),
                "has_downloads": repo_info.get("has_downloads", False)
            }
            
            # Add additional metadata if available
            if not isinstance(languages, Exception) and languages:
                metadata["languages"] = languages
            
            if not isinstance(contributors, Exception) and contributors:
                metadata["contributors"] = contributors[:10]  # Top 10 contributors
            
            if not isinstance(readme, Exception) and readme:
                metadata["readme"] = readme
            
            if not isinstance(releases, Exception) and releases:
                metadata["releases"] = releases[:5]  # Latest 5 releases
            
            if not isinstance(recent_commits, Exception) and recent_commits:
                metadata["recent_commits"] = recent_commits
            
            return metadata
            
        except Exception as e:
            print(f"❌ Error getting repository metadata: {e}")
            return {"error": str(e)}
    
    async def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata from GitHub"""
        try:
            # Parse file path: owner/repo/blob/sha/path
            parts = file_path.split("/blob/", 1)
            if len(parts) != 2:
                return {"error": "Invalid file path format"}
            
            repo_full_name = parts[0]
            blob_parts = parts[1].split("/", 1)
            if len(blob_parts) != 2:
                return {"error": "Invalid blob path format"}
            
            sha = blob_parts[0]
            file_path_in_repo = blob_parts[1]
            
            # Get file content and metadata
            file_info = await self._make_api_request(f"/repos/{repo_full_name}/contents/{file_path_in_repo}")
            
            if file_info:
                return {
                    "type": "file",
                    "repository": repo_full_name,
                    "path": file_path_in_repo,
                    "name": file_info.get("name"),
                    "sha": file_info.get("sha"),
                    "size": file_info.get("size", 0),
                    "download_url": file_info.get("download_url"),
                    "html_url": file_info.get("html_url"),
                    "encoding": file_info.get("encoding", "base64"),
                    "content_preview": file_info.get("content", "")[:1000] if file_info.get("content") else None
                }
            else:
                return {"error": "File not found"}
                
        except Exception as e:
            print(f"❌ Error getting file metadata: {e}")
            return {"error": str(e)}
    
    async def _get_repository_languages(self, repo_full_name: str) -> Dict[str, int]:
        """Get programming languages used in repository"""
        return await self._make_api_request(f"/repos/{repo_full_name}/languages") or {}
    
    async def _get_repository_contributors(self, repo_full_name: str) -> List[Dict[str, Any]]:
        """Get repository contributors"""
        return await self._make_api_request(f"/repos/{repo_full_name}/contributors?per_page=10") or []
    
    async def _get_repository_readme(self, repo_full_name: str) -> Optional[Dict[str, Any]]:
        """Get repository README file"""
        readme = await self._make_api_request(f"/repos/{repo_full_name}/readme")
        if readme:
            return {
                "name": readme.get("name"),
                "path": readme.get("path"),
                "sha": readme.get("sha"),
                "size": readme.get("size"),
                "download_url": readme.get("download_url"),
                "html_url": readme.get("html_url")
            }
        return None
    
    async def _get_repository_releases(self, repo_full_name: str) -> List[Dict[str, Any]]:
        """Get repository releases"""
        releases = await self._make_api_request(f"/repos/{repo_full_name}/releases?per_page=5") or []
        return [
            {
                "tag_name": r.get("tag_name"),
                "name": r.get("name"),
                "published_at": r.get("published_at"),
                "prerelease": r.get("prerelease", False),
                "draft": r.get("draft", False)
            }
            for r in releases
        ]
    
    async def _get_repository_commits(self, repo_full_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent repository commits"""
        commits = await self._make_api_request(f"/repos/{repo_full_name}/commits?per_page={limit}") or []
        return [
            {
                "sha": c.get("sha"),
                "message": c.get("commit", {}).get("message", ""),
                "author": c.get("commit", {}).get("author", {}),
                "date": c.get("commit", {}).get("author", {}).get("date"),
                "html_url": c.get("html_url")
            }
            for c in commits
        ]
    
    # === Content Download Methods ===
    
    async def download_content(self, external_id: str, target_path: str) -> bool:
        """
        Download GitHub repository or file content
        
        Args:
            external_id: GitHub repository full name or file path
            target_path: Local path to save content
            
        Returns:
            True if download successful
        """
        try:
            print(f"⬇️ Downloading GitHub content: {external_id}")
            
            # Determine content type and download accordingly
            if "/blob/" in external_id:
                return await self._download_file(external_id, target_path)
            else:
                return await self._download_repository(external_id, target_path)
                
        except Exception as e:
            print(f"❌ GitHub download error: {e}")
            return False
    
    async def _download_repository(self, repo_full_name: str, target_path: str) -> bool:
        """Download entire repository as ZIP archive"""
        try:
            # Get repository metadata to determine default branch
            repo_info = await self._make_api_request(f"/repos/{repo_full_name}")
            if not repo_info:
                return False
            
            default_branch = repo_info.get("default_branch", "main")
            
            # Download repository archive
            archive_url = f"{self.api_base_url}/repos/{repo_full_name}/zipball/{default_branch}"
            
            if not self.session:
                await self._create_session()
            
            async with self.session.get(archive_url) as response:
                if response.status == 200:
                    # Ensure target directory exists
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    # Write archive to file
                    with open(target_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    print(f"✅ Repository downloaded: {target_path}")
                    return True
                else:
                    print(f"❌ Download failed with status: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ Repository download error: {e}")
            return False
    
    async def _download_file(self, file_path: str, target_path: str) -> bool:
        """Download individual file from GitHub"""
        try:
            # Get file metadata
            file_metadata = await self._get_file_metadata(file_path)
            
            if "error" in file_metadata:
                return False
            
            download_url = file_metadata.get("download_url")
            if not download_url:
                return False
            
            # Download file content
            if not self.session:
                await self._create_session()
            
            async with self.session.get(download_url) as response:
                if response.status == 200:
                    # Ensure target directory exists
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    # Write file content
                    with open(target_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    print(f"✅ File downloaded: {target_path}")
                    return True
                else:
                    print(f"❌ File download failed with status: {response.status}")
                    return False
                    
        except Exception as e:
            print(f"❌ File download error: {e}")
            return False
    
    # === License Validation Methods ===
    
    async def validate_license(self, external_id: str) -> Dict[str, Any]:
        """
        Validate license compliance for GitHub repository
        
        Args:
            external_id: GitHub repository full name
            
        Returns:
            License validation results
        """
        try:
            print(f"📄 Validating GitHub license for: {external_id}")
            
            # Get repository metadata
            repo_metadata = await self._get_repository_metadata(external_id)
            
            if "error" in repo_metadata:
                return {
                    "type": "unknown",
                    "compliant": False,
                    "details": repo_metadata,
                    "issues": ["Failed to fetch repository metadata"]
                }
            
            license_info = repo_metadata.get("license")
            issues = []
            
            if not license_info:
                return {
                    "type": "unknown",
                    "compliant": False,
                    "details": {},
                    "issues": ["No license information found"]
                }
            
            license_key = license_info.get("key", "").lower()
            license_name = license_info.get("name", "Unknown")
            
            # Check against permissive licenses
            permissive_licenses = [
                "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause",
                "unlicense", "cc0-1.0", "isc", "zlib"
            ]
            
            if license_key in permissive_licenses:
                license_type = "permissive"
                compliant = True
            elif license_key in ["gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0", "agpl-3.0"]:
                license_type = "copyleft"
                compliant = False
                issues.append(f"Copyleft license not permitted: {license_name}")
            elif "proprietary" in license_key or "commercial" in license_key:
                license_type = "proprietary"
                compliant = False
                issues.append(f"Proprietary license not permitted: {license_name}")
            else:
                license_type = "unknown"
                compliant = False
                issues.append(f"Unknown or unrecognized license: {license_name}")
            
            return {
                "type": license_type,
                "compliant": compliant,
                "details": license_info,
                "issues": issues,
                "license_key": license_key,
                "license_name": license_name,
                "license_url": license_info.get("url")
            }
            
        except Exception as e:
            print(f"❌ License validation error: {e}")
            return {
                "type": "unknown",
                "compliant": False,
                "details": {"error": str(e)},
                "issues": [f"License validation error: {str(e)}"]
            }
    
    # === API Helper Methods ===
    
    async def _make_api_request(self, endpoint: str, method: str = "GET", 
                              data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Make authenticated request to GitHub API"""
        try:
            if not self.session:
                await self._create_session()
            
            url = f"{self.api_base_url}{endpoint}"
            
            # Make request
            if method == "GET":
                async with self.session.get(url) as response:
                    return await self._handle_api_response(response)
            elif method == "POST":
                async with self.session.post(url, json=data) as response:
                    return await self._handle_api_response(response)
            elif method == "PUT":
                async with self.session.put(url, json=data) as response:
                    return await self._handle_api_response(response)
            elif method == "DELETE":
                async with self.session.delete(url) as response:
                    return await self._handle_api_response(response)
            
            return None
            
        except Exception as e:
            print(f"❌ GitHub API request error: {e}")
            self.error_count += 1
            return None
    
    async def _handle_api_response(self, response: aiohttp.ClientResponse) -> Optional[Dict[str, Any]]:
        """Handle GitHub API response with rate limiting"""
        try:
            # Update rate limit info
            self.rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
            self.rate_limit_reset = response.headers.get("X-RateLimit-Reset")
            
            if response.status == 200:
                return await response.json()
            elif response.status == 403 and self.rate_limit_remaining == 0:
                print("⚠️ GitHub API rate limit exceeded")
                self.status = ConnectorStatus.RATE_LIMITED
                return None
            elif response.status == 401:
                print("❌ GitHub API authentication failed")
                self.status = ConnectorStatus.AUTH_FAILED
                return None
            elif response.status == 404:
                print("⚠️ GitHub resource not found")
                return None
            else:
                print(f"⚠️ GitHub API error: {response.status} - {await response.text()}")
                return None
                
        except Exception as e:
            print(f"❌ Error handling GitHub API response: {e}")
            return None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.session and not self.session.closed:
            try:
                asyncio.get_event_loop().run_until_complete(self.session.close())
            except Exception:
                pass