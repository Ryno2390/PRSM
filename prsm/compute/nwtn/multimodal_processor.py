"""
Multimodal Processor
====================

Process multimodal content (text, images, etc.).
"""

from typing import Dict, Any, Optional, List
import logging
import re

logger = logging.getLogger(__name__)


class MultiModalProcessor:
    """Process multimodal content with real text extraction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    async def process(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process multimodal content, extracting text and metadata."""
        result = {
            "processed": True,
            "content": content,
            "extracted_text": None,
            "modalities": [],
            "metadata": {}
        }

        # Extract text if present
        if "text" in content:
            result["extracted_text"] = content["text"]
            result["modalities"].append("text")

        # Check for image references
        if "image" in content or "image_url" in content:
            result["modalities"].append("image")
            result["metadata"]["has_image"] = True

        # Check for audio references
        if "audio" in content or "audio_url" in content:
            result["modalities"].append("audio")
            result["metadata"]["has_audio"] = True

        # Check for structured data
        if "data" in content:
            result["modalities"].append("structured")
            result["metadata"]["data_type"] = type(content["data"]).__name__

        # Extract URLs from text if present
        if result["extracted_text"]:
            url_pattern = r'https?://[^\s]+'
            urls = re.findall(url_pattern, result["extracted_text"])
            if urls:
                result["metadata"]["urls"] = urls

        return result

    async def extract_text(self, content: Dict[str, Any]) -> Optional[str]:
        """Extract text content from multimodal input."""
        result = await self.process(content)
        return result.get("extracted_text")

    async def get_modalities(self, content: Dict[str, Any]) -> List[str]:
        """Get list of modalities present in content."""
        result = await self.process(content)
        return result.get("modalities", [])
