"""
Hugging Face Connector
=====================

Hugging Face Hub API integration connector for PRSM, enabling seamless
model and dataset discovery, download, and integration with comprehensive
metadata extraction and creator attribution.

Key Features:
- API token-based authentication
- Model and dataset search with advanced filtering
- Model card and metadata extraction
- Download with progress tracking
- License compliance validation
- Creator attribution for FTNS rewards
- Integration with transformers ecosystem
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from urllib.parse import quote

import aiohttp

from ..core.base_connector import BaseConnector, ConnectorStatus
from ..models.integration_models import (
    IntegrationPlatform, ConnectorConfig, IntegrationSource,
    ImportRequest, ImportResult, SecurityRisk, LicenseType
)
from ...core.config import settings


class HuggingFaceConnector(BaseConnector):
    """
    Hugging Face Hub API integration connector
    
    Provides comprehensive Hugging Face Hub integration including:
    - API token authentication
    - Model and dataset search and discovery
    - Model card and metadata extraction
    - Content download with progress tracking
    - License compliance validation
    - Creator attribution for FTNS rewards
    """
    
    def __init__(self, config: ConnectorConfig):
        """
        Initialize Hugging Face connector
        
        Args:
            config: ConnectorConfig with HF API token
        """
        super().__init__(config)
        
        # Hugging Face API configuration
        self.api_base_url = "https://huggingface.co/api"
        self.hub_base_url = "https://huggingface.co"
        
        # Authentication
        self.api_token = config.api_key or getattr(settings, "PRSM_HF_API_TOKEN", None)
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting (HF has generous limits)
        self.rate_limit_remaining = 1000
        self.rate_limit_reset = None
        
        print(f"🤗 Hugging Face Connector initialized for user {self.user_id}")
    
    # === Authentication Methods ===
    
    async def authenticate(self) -> bool:
        """
        Authenticate with Hugging Face using API token
        
        Returns:
            True if authentication successful
        """
        try:
            # Create HTTP session
            await self._create_session()
            
            if self.api_token:
                # Test authentication by getting user info
                user_info = await self._make_api_request("/whoami")
                
                if user_info and "name" in user_info:
                    self.authenticated_user = user_info["name"]
                    print(f"✅ Hugging Face authentication successful for user: {self.authenticated_user}")
                    return True
                else:
                    print("❌ Hugging Face authentication failed")
                    return False
            else:
                # Anonymous access (limited functionality)
                print("⚠️ Using Hugging Face in anonymous mode (limited functionality)")
                self.authenticated_user = "anonymous"
                return True
                
        except Exception as e:
            print(f"❌ Hugging Face authentication error: {e}")
            return False
    
    async def _create_session(self) -> None:
        """Create HTTP session with optional authentication"""
        if self.session:
            await self.session.close()
        
        headers = {
            "User-Agent": "PRSM-Integration/1.0"
        }
        
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self.session = aiohttp.ClientSession(headers=headers, timeout=timeout)
    
    # === Content Discovery Methods ===
    
    async def search_content(self, query: str, content_type: str = "model",
                           limit: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[IntegrationSource]:
        """
        Search Hugging Face Hub for models, datasets, or spaces
        
        Args:
            query: Search query string
            content_type: Type of content (model, dataset, space)
            limit: Maximum results to return
            filters: Additional search filters (task, library, language, etc.)
            
        Returns:
            List of IntegrationSource objects
        """
        try:
            print(f"🔍 Searching Hugging Face for '{query}' (type: {content_type})")
            
            # Execute search based on content type
            if content_type == "model":
                results = await self._search_models(query, limit, filters)
            elif content_type == "dataset":
                results = await self._search_datasets(query, limit, filters)
            elif content_type == "space":
                results = await self._search_spaces(query, limit, filters)
            else:
                print(f"⚠️ Unsupported content type: {content_type}")
                return []
            
            # Convert to IntegrationSource objects
            integration_sources = []
            for item in results:
                source = await self._convert_to_integration_source(item, content_type)
                if source:
                    integration_sources.append(source)
            
            print(f"📊 Found {len(integration_sources)} Hugging Face results")
            return integration_sources
            
        except Exception as e:
            print(f"❌ Hugging Face search error: {e}")
            return []
    
    async def _search_models(self, query: str, limit: int, 
                           filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search Hugging Face models"""
        params = {
            "search": query,
            "limit": min(limit, 100),
            "full": "true"
        }
        
        # Add filters
        if filters:
            if "task" in filters:
                params["pipeline_tag"] = filters["task"]
            if "library" in filters:
                params["library"] = filters["library"]
            if "language" in filters:
                params["language"] = filters["language"]
            if "sort" in filters:
                params["sort"] = filters["sort"]
            else:
                params["sort"] = "downloads"  # Default sort by downloads
        
        # Build query string
        query_params = "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
        endpoint = f"/models?{query_params}"
        
        response = await self._make_api_request(endpoint)
        return response if response else []
    
    async def _search_datasets(self, query: str, limit: int,
                             filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search Hugging Face datasets"""
        params = {
            "search": query,
            "limit": min(limit, 100),
            "full": "true"
        }
        
        # Add filters
        if filters:
            if "task" in filters:
                params["task_categories"] = filters["task"]
            if "language" in filters:
                params["language"] = filters["language"]
            if "size" in filters:
                params["size_categories"] = filters["size"]
            if "sort" in filters:
                params["sort"] = filters["sort"]
            else:
                params["sort"] = "downloads"
        
        query_params = "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
        endpoint = f"/datasets?{query_params}"
        
        response = await self._make_api_request(endpoint)
        return response if response else []
    
    async def _search_spaces(self, query: str, limit: int,
                           filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Search Hugging Face Spaces"""
        params = {
            "search": query,
            "limit": min(limit, 100),
            "full": "true"
        }
        
        if filters and "sort" in filters:
            params["sort"] = filters["sort"]
        else:
            params["sort"] = "likes"
        
        query_params = "&".join([f"{k}={quote(str(v))}" for k, v in params.items()])
        endpoint = f"/spaces?{query_params}"
        
        response = await self._make_api_request(endpoint)
        return response if response else []
    
    async def _convert_to_integration_source(self, item: Dict[str, Any], 
                                           content_type: str) -> Optional[IntegrationSource]:
        """Convert Hugging Face API response to IntegrationSource"""
        try:
            base_metadata = {
                "content_type": content_type,
                "downloads": item.get("downloads", 0),
                "likes": item.get("likes", 0),
                "created_at": item.get("createdAt"),
                "last_modified": item.get("lastModified"),
                "tags": item.get("tags", []),
                "private": item.get("private", False),
                "gated": item.get("gated", False)
            }
            
            if content_type == "model":
                return IntegrationSource(
                    platform=IntegrationPlatform.HUGGINGFACE,
                    external_id=item["id"],
                    display_name=item["id"].split("/")[-1],
                    description=item.get("description", ""),
                    owner_id=item["id"].split("/")[0] if "/" in item["id"] else "unknown",
                    url=f"{self.hub_base_url}/{item['id']}",
                    metadata={
                        **base_metadata,
                        "pipeline_tag": item.get("pipeline_tag"),
                        "library_name": item.get("library_name"),
                        "model_index": item.get("model-index"),
                        "config": item.get("config"),
                        "transformers_version": item.get("transformersVersion"),
                        "safetensors": item.get("safetensors")
                    }
                )
            elif content_type == "dataset":
                return IntegrationSource(
                    platform=IntegrationPlatform.HUGGINGFACE,
                    external_id=item["id"],
                    display_name=item["id"].split("/")[-1],
                    description=item.get("description", ""),
                    owner_id=item["id"].split("/")[0] if "/" in item["id"] else "unknown",
                    url=f"{self.hub_base_url}/datasets/{item['id']}",
                    metadata={
                        **base_metadata,
                        "task_categories": item.get("task_categories", []),
                        "language_creators": item.get("language_creators", []),
                        "languages": item.get("languages", []),
                        "multilinguality": item.get("multilinguality", []),
                        "size_categories": item.get("size_categories", []),
                        "source_datasets": item.get("source_datasets", [])
                    }
                )
            elif content_type == "space":
                return IntegrationSource(
                    platform=IntegrationPlatform.HUGGINGFACE,
                    external_id=item["id"],
                    display_name=item["id"].split("/")[-1],
                    description=item.get("description", ""),
                    owner_id=item["id"].split("/")[0] if "/" in item["id"] else "unknown",
                    url=f"{self.hub_base_url}/spaces/{item['id']}",
                    metadata={
                        **base_metadata,
                        "sdk": item.get("sdk"),
                        "app_file": item.get("app_file"),
                        "colorFrom": item.get("colorFrom"),
                        "colorTo": item.get("colorTo"),
                        "runtime": item.get("runtime")
                    }
                )
        
        except Exception as e:
            print(f"⚠️ Error converting Hugging Face item to IntegrationSource: {e}")
            return None
    
    # === Content Metadata Methods ===
    
    async def get_content_metadata(self, external_id: str) -> Dict[str, Any]:
        """
        Get detailed metadata for Hugging Face model, dataset, or space
        
        Args:
            external_id: Hugging Face model/dataset/space ID (e.g., "microsoft/DialoGPT-medium")
            
        Returns:
            Comprehensive metadata dictionary
        """
        try:
            print(f"📋 Fetching Hugging Face metadata for: {external_id}")
            
            # Try to determine content type and get metadata
            # First try as model
            model_metadata = await self._get_model_metadata(external_id)
            if model_metadata and "error" not in model_metadata:
                return model_metadata
            
            # Then try as dataset
            dataset_metadata = await self._get_dataset_metadata(external_id)
            if dataset_metadata and "error" not in dataset_metadata:
                return dataset_metadata
            
            # Finally try as space
            space_metadata = await self._get_space_metadata(external_id)
            if space_metadata and "error" not in space_metadata:
                return space_metadata
            
            return {"error": f"Content not found: {external_id}"}
            
        except Exception as e:
            print(f"❌ Error fetching Hugging Face metadata: {e}")
            return {"error": str(e)}
    
    async def _get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive model metadata"""
        try:
            # Get basic model info
            model_info = await self._make_api_request(f"/models/{model_id}")
            if not model_info:
                return {"error": "Model not found"}
            
            # Get additional metadata in parallel
            tasks = [
                self._get_model_card(model_id),
                self._get_model_files(model_id),
                self._get_model_config(model_id)
            ]
            
            model_card, model_files, model_config = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            
            # Build comprehensive metadata
            metadata = {
                "type": "model",
                "id": model_info["id"],
                "model_id": model_info["id"],
                "creator": model_info["id"].split("/")[0] if "/" in model_info["id"] else "unknown",
                "sha": model_info.get("sha"),
                "pipeline_tag": model_info.get("pipeline_tag"),
                "library_name": model_info.get("library_name"),
                "tags": model_info.get("tags", []),
                "downloads": model_info.get("downloads", 0),
                "likes": model_info.get("likes", 0),
                "created_at": model_info.get("createdAt"),
                "last_modified": model_info.get("lastModified"),
                "private": model_info.get("private", False),
                "gated": model_info.get("gated", False),
                "disabled": model_info.get("disabled", False),
                "transformers_version": model_info.get("transformersVersion"),
                "config": model_info.get("config", {}),
                "model_index": model_info.get("model-index"),
                "cardData": model_info.get("cardData", {}),
                "safetensors": model_info.get("safetensors")
            }
            
            # Add additional metadata if available
            if not isinstance(model_card, Exception) and model_card:
                metadata["model_card"] = model_card
            
            if not isinstance(model_files, Exception) and model_files:
                metadata["files"] = model_files
            
            if not isinstance(model_config, Exception) and model_config:
                metadata["detailed_config"] = model_config
            
            return metadata
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_dataset_metadata(self, dataset_id: str) -> Dict[str, Any]:
        """Get comprehensive dataset metadata"""
        try:
            dataset_info = await self._make_api_request(f"/datasets/{dataset_id}")
            if not dataset_info:
                return {"error": "Dataset not found"}
            
            # Get additional metadata
            tasks = [
                self._get_dataset_card(dataset_id),
                self._get_dataset_files(dataset_id)
            ]
            
            dataset_card, dataset_files = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            
            metadata = {
                "type": "dataset",
                "id": dataset_info["id"],
                "dataset_id": dataset_info["id"],
                "creator": dataset_info["id"].split("/")[0] if "/" in dataset_info["id"] else "unknown",
                "sha": dataset_info.get("sha"),
                "tags": dataset_info.get("tags", []),
                "downloads": dataset_info.get("downloads", 0),
                "likes": dataset_info.get("likes", 0),
                "created_at": dataset_info.get("createdAt"),
                "last_modified": dataset_info.get("lastModified"),
                "private": dataset_info.get("private", False),
                "gated": dataset_info.get("gated", False),
                "disabled": dataset_info.get("disabled", False),
                "cardData": dataset_info.get("cardData", {}),
                "task_categories": dataset_info.get("cardData", {}).get("task_categories", []),
                "language_creators": dataset_info.get("cardData", {}).get("language_creators", []),
                "languages": dataset_info.get("cardData", {}).get("languages", []),
                "multilinguality": dataset_info.get("cardData", {}).get("multilinguality", []),
                "size_categories": dataset_info.get("cardData", {}).get("size_categories", []),
                "source_datasets": dataset_info.get("cardData", {}).get("source_datasets", [])
            }
            
            if not isinstance(dataset_card, Exception) and dataset_card:
                metadata["dataset_card"] = dataset_card
            
            if not isinstance(dataset_files, Exception) and dataset_files:
                metadata["files"] = dataset_files
            
            return metadata
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_space_metadata(self, space_id: str) -> Dict[str, Any]:
        """Get comprehensive space metadata"""
        try:
            space_info = await self._make_api_request(f"/spaces/{space_id}")
            if not space_info:
                return {"error": "Space not found"}
            
            metadata = {
                "type": "space",
                "id": space_info["id"],
                "space_id": space_info["id"],
                "creator": space_info["id"].split("/")[0] if "/" in space_info["id"] else "unknown",
                "sha": space_info.get("sha"),
                "tags": space_info.get("tags", []),
                "likes": space_info.get("likes", 0),
                "created_at": space_info.get("createdAt"),
                "last_modified": space_info.get("lastModified"),
                "private": space_info.get("private", False),
                "gated": space_info.get("gated", False),
                "disabled": space_info.get("disabled", False),
                "runtime": space_info.get("runtime", {}),
                "sdk": space_info.get("sdk"),
                "app_file": space_info.get("app_file"),
                "colorFrom": space_info.get("colorFrom"),
                "colorTo": space_info.get("colorTo"),
                "cardData": space_info.get("cardData", {})
            }
            
            return metadata
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_model_card(self, model_id: str) -> Optional[str]:
        """Get model card content"""
        try:
            # Model cards are typically in README.md
            response = await self._make_raw_request(f"/{model_id}/raw/main/README.md")
            return response.decode('utf-8') if response else None
        except Exception:
            return None
    
    async def _get_dataset_card(self, dataset_id: str) -> Optional[str]:
        """Get dataset card content"""
        try:
            response = await self._make_raw_request(f"/datasets/{dataset_id}/raw/main/README.md")
            return response.decode('utf-8') if response else None
        except Exception:
            return None
    
    async def _get_model_files(self, model_id: str) -> List[Dict[str, Any]]:
        """Get list of model files"""
        try:
            response = await self._make_api_request(f"/models/{model_id}/tree/main")
            if response and isinstance(response, list):
                return [
                    {
                        "path": item.get("path"),
                        "type": item.get("type"),
                        "size": item.get("size"),
                        "blob_id": item.get("oid")
                    }
                    for item in response
                    if item.get("type") == "file"
                ]
            return []
        except Exception:
            return []
    
    async def _get_dataset_files(self, dataset_id: str) -> List[Dict[str, Any]]:
        """Get list of dataset files"""
        try:
            response = await self._make_api_request(f"/datasets/{dataset_id}/tree/main")
            if response and isinstance(response, list):
                return [
                    {
                        "path": item.get("path"),
                        "type": item.get("type"),
                        "size": item.get("size"),
                        "blob_id": item.get("oid")
                    }
                    for item in response
                    if item.get("type") == "file"
                ]
            return []
        except Exception:
            return []
    
    async def _get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model configuration"""
        try:
            response = await self._make_raw_request(f"/{model_id}/raw/main/config.json")
            if response:
                return json.loads(response.decode('utf-8'))
            return None
        except Exception:
            return None
    
    # === Content Download Methods ===
    
    async def download_content(self, external_id: str, target_path: str) -> bool:
        """
        Download Hugging Face model, dataset, or space
        
        Args:
            external_id: Hugging Face content ID
            target_path: Local path to save content
            
        Returns:
            True if download successful
        """
        try:
            print(f"⬇️ Downloading Hugging Face content: {external_id}")
            
            # Get content metadata to determine type
            metadata = await self.get_content_metadata(external_id)
            
            if "error" in metadata:
                print(f"❌ Cannot download: {metadata['error']}")
                return False
            
            content_type = metadata.get("type", "model")
            
            # Download based on content type
            if content_type == "model":
                return await self._download_model(external_id, target_path, metadata)
            elif content_type == "dataset":
                return await self._download_dataset(external_id, target_path, metadata)
            elif content_type == "space":
                return await self._download_space(external_id, target_path, metadata)
            else:
                print(f"❌ Unsupported content type for download: {content_type}")
                return False
                
        except Exception as e:
            print(f"❌ Hugging Face download error: {e}")
            return False
    
    async def _download_model(self, model_id: str, target_path: str, 
                            metadata: Dict[str, Any]) -> bool:
        """Download Hugging Face model"""
        try:
            # Create target directory
            os.makedirs(target_path, exist_ok=True)
            
            # Get list of files to download
            files = metadata.get("files", [])
            
            if not files:
                print("⚠️ No files found in model")
                return False
            
            # Download key model files
            key_files = [
                "config.json", "pytorch_model.bin", "model.safetensors",
                "tokenizer.json", "tokenizer_config.json", "vocab.txt",
                "merges.txt", "special_tokens_map.json", "README.md"
            ]
            
            downloaded_count = 0
            for file_info in files:
                file_path = file_info.get("path", "")
                
                # Download key files or limit total downloads
                if any(key_file in file_path for key_file in key_files) or downloaded_count < 10:
                    success = await self._download_file(model_id, file_path, 
                                                      os.path.join(target_path, file_path))
                    if success:
                        downloaded_count += 1
            
            print(f"✅ Downloaded {downloaded_count} model files to {target_path}")
            return downloaded_count > 0
            
        except Exception as e:
            print(f"❌ Model download error: {e}")
            return False
    
    async def _download_dataset(self, dataset_id: str, target_path: str,
                              metadata: Dict[str, Any]) -> bool:
        """Download Hugging Face dataset"""
        try:
            # Create target directory
            os.makedirs(target_path, exist_ok=True)
            
            # Get list of files
            files = metadata.get("files", [])
            
            if not files:
                print("⚠️ No files found in dataset")
                return False
            
            # Download important dataset files
            important_files = ["README.md", "dataset_infos.json", "dataset_dict.json"]
            data_files = [f for f in files if any(ext in f.get("path", "") 
                         for ext in [".csv", ".json", ".parquet", ".arrow", ".txt"])]
            
            downloaded_count = 0
            
            # Download important metadata files
            for file_info in files:
                file_path = file_info.get("path", "")
                if any(important in file_path for important in important_files):
                    success = await self._download_file(f"datasets/{dataset_id}", file_path,
                                                      os.path.join(target_path, file_path))
                    if success:
                        downloaded_count += 1
            
            # Download sample data files (limit to prevent huge downloads)
            for file_info in data_files[:5]:  # Limit to 5 data files
                file_path = file_info.get("path", "")
                success = await self._download_file(f"datasets/{dataset_id}", file_path,
                                                  os.path.join(target_path, file_path))
                if success:
                    downloaded_count += 1
            
            print(f"✅ Downloaded {downloaded_count} dataset files to {target_path}")
            return downloaded_count > 0
            
        except Exception as e:
            print(f"❌ Dataset download error: {e}")
            return False
    
    async def _download_space(self, space_id: str, target_path: str,
                            metadata: Dict[str, Any]) -> bool:
        """Download Hugging Face Space"""
        try:
            # Create target directory
            os.makedirs(target_path, exist_ok=True)
            
            # Download key space files
            key_files = ["README.md", "app.py", "requirements.txt", "packages.txt"]
            downloaded_count = 0
            
            for filename in key_files:
                success = await self._download_file(f"spaces/{space_id}", filename,
                                                  os.path.join(target_path, filename))
                if success:
                    downloaded_count += 1
            
            print(f"✅ Downloaded {downloaded_count} space files to {target_path}")
            return downloaded_count > 0
            
        except Exception as e:
            print(f"❌ Space download error: {e}")
            return False
    
    async def _download_file(self, repo_id: str, file_path: str, target_path: str) -> bool:
        """Download individual file from Hugging Face"""
        try:
            # Construct download URL
            download_url = f"{self.hub_base_url}/{repo_id}/resolve/main/{file_path}"
            
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
                    
                    return True
                else:
                    print(f"⚠️ Failed to download {file_path}: HTTP {response.status}")
                    return False
                    
        except Exception as e:
            print(f"⚠️ Error downloading {file_path}: {e}")
            return False
    
    # === License Validation Methods ===
    
    async def validate_license(self, external_id: str) -> Dict[str, Any]:
        """
        Validate license compliance for Hugging Face content
        
        Args:
            external_id: Hugging Face content ID
            
        Returns:
            License validation results
        """
        try:
            print(f"📄 Validating Hugging Face license for: {external_id}")
            
            # Get content metadata
            metadata = await self.get_content_metadata(external_id)
            
            if "error" in metadata:
                return {
                    "type": "unknown",
                    "compliant": False,
                    "details": metadata,
                    "issues": ["Failed to fetch content metadata"]
                }
            
            # Extract license information from tags and cardData
            license_info = self._extract_license_from_metadata(metadata)
            
            return license_info
            
        except Exception as e:
            print(f"❌ License validation error: {e}")
            return {
                "type": "unknown",
                "compliant": False,
                "details": {"error": str(e)},
                "issues": [f"License validation error: {str(e)}"]
            }
    
    def _extract_license_from_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract license information from content metadata"""
        issues = []
        
        # Check tags for license information
        tags = metadata.get("tags", [])
        license_tags = [tag for tag in tags if tag.startswith("license:")]
        
        if license_tags:
            license_key = license_tags[0].replace("license:", "").lower()
        else:
            # Check cardData for license
            card_data = metadata.get("cardData", {})
            license_key = card_data.get("license", "unknown").lower()
        
        # Validate license
        permissive_licenses = [
            "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause",
            "unlicense", "cc0-1.0", "isc", "zlib", "cc-by-4.0"
        ]
        
        if license_key == "unknown" or not license_key:
            return {
                "type": "unknown",
                "compliant": False,
                "details": {"license_key": license_key, "tags": tags},
                "issues": ["No license information found"],
                "license_key": license_key
            }
        elif license_key in permissive_licenses:
            return {
                "type": "permissive",
                "compliant": True,
                "details": {"license_key": license_key, "tags": tags},
                "issues": [],
                "license_key": license_key
            }
        elif license_key in ["gpl-2.0", "gpl-3.0", "lgpl-2.1", "lgpl-3.0", "agpl-3.0"]:
            return {
                "type": "copyleft",
                "compliant": False,
                "details": {"license_key": license_key, "tags": tags},
                "issues": [f"Copyleft license not permitted: {license_key}"],
                "license_key": license_key
            }
        else:
            return {
                "type": "unknown",
                "compliant": False,
                "details": {"license_key": license_key, "tags": tags},
                "issues": [f"Unknown or unrecognized license: {license_key}"],
                "license_key": license_key
            }
    
    # === API Helper Methods ===
    
    async def _make_api_request(self, endpoint: str, method: str = "GET",
                              data: Optional[Dict[str, Any]] = None) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """Make request to Hugging Face API"""
        try:
            if not self.session:
                await self._create_session()
            
            url = f"{self.api_base_url}{endpoint}"
            
            if method == "GET":
                async with self.session.get(url) as response:
                    return await self._handle_api_response(response)
            elif method == "POST":
                async with self.session.post(url, json=data) as response:
                    return await self._handle_api_response(response)
            
            return None
            
        except Exception as e:
            print(f"❌ Hugging Face API request error: {e}")
            self.error_count += 1
            return None
    
    async def _make_raw_request(self, path: str) -> Optional[bytes]:
        """Make raw request to Hugging Face for file content"""
        try:
            if not self.session:
                await self._create_session()
            
            url = f"{self.hub_base_url}{path}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                return None
                
        except Exception as e:
            print(f"❌ Hugging Face raw request error: {e}")
            return None
    
    async def _handle_api_response(self, response: aiohttp.ClientResponse) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """Handle Hugging Face API response"""
        try:
            if response.status == 200:
                return await response.json()
            elif response.status == 401:
                print("❌ Hugging Face API authentication failed")
                self.status = ConnectorStatus.AUTH_FAILED
                return None
            elif response.status == 404:
                return None  # Not found is expected for some requests
            elif response.status == 429:
                print("⚠️ Hugging Face API rate limit exceeded")
                self.status = ConnectorStatus.RATE_LIMITED
                return None
            else:
                print(f"⚠️ Hugging Face API error: {response.status}")
                return None
                
        except Exception as e:
            print(f"❌ Error handling Hugging Face API response: {e}")
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