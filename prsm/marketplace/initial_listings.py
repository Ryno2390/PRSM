"""
Initial Model Listings for Marketplace Launch
============================================

Creates initial high-quality model listings to populate the marketplace
at launch with a diverse selection of AI models across categories.
"""

import asyncio
import structlog
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict, Any
from uuid import uuid4

from .models import (
    ModelCategory, PricingTier, ModelProvider, ModelStatus,
    CreateModelListingRequest, ModelMetadata
)
from .real_marketplace_service import RealMarketplaceService

# Initialize service
marketplace_service = RealMarketplaceService()
from ..integrations.security.audit_logger import audit_logger

logger = structlog.get_logger(__name__)


class InitialListingsCreator:
    """Creates initial model listings for marketplace launch"""
    
    def __init__(self):
        self.system_user_id = uuid4()  # System user for initial listings
        
        # Initial model listings data
        self.initial_listings = [
            {
                "name": "GPT-4 Turbo",
                "description": "OpenAI's most capable model, great for complex reasoning, creative writing, and code generation. Supports up to 128k context length with superior performance across diverse tasks.",
                "model_id": "gpt-4-turbo-preview",
                "provider": ModelProvider.OPENAI,
                "category": ModelCategory.LANGUAGE_MODEL,
                "provider_name": "OpenAI",
                "provider_url": "https://openai.com",
                "model_version": "turbo-2024-04-09",
                "pricing_tier": PricingTier.PREMIUM,
                "price_per_token": Decimal('0.00003'),  # $0.03 per 1K tokens
                "context_length": 128000,
                "max_tokens": 4096,
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "languages_supported": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                "tags": ["gpt", "reasoning", "creative", "code", "analysis"],
                "license_type": "Commercial",
                "documentation_url": "https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo",
                "metadata": {
                    "parameter_count": "1.7T",
                    "training_data_size": "Unknown",
                    "model_architecture": "Transformer",
                    "evaluation_metrics": {"MMLU": 86.4, "HellaSwag": 95.3, "HumanEval": 67.0},
                    "use_cases": ["Text generation", "Code completion", "Analysis", "Creative writing"],
                    "limitations": ["May generate plausible but incorrect information", "Knowledge cutoff"],
                    "safety_measures": ["Constitutional AI", "RLHF", "Safety filtering"]
                }
            },
            {
                "name": "Claude-3 Opus",
                "description": "Anthropic's most powerful model for highly complex tasks. Excels at reasoning, analysis, math, coding, and creative tasks with strong safety measures.",
                "model_id": "claude-3-opus-20240229",
                "provider": ModelProvider.ANTHROPIC,
                "category": ModelCategory.LANGUAGE_MODEL,
                "provider_name": "Anthropic",
                "provider_url": "https://anthropic.com",
                "model_version": "opus-20240229",
                "pricing_tier": PricingTier.PREMIUM,
                "price_per_token": Decimal('0.000015'),
                "context_length": 200000,
                "max_tokens": 4096,
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
                "languages_supported": ["en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"],
                "tags": ["claude", "reasoning", "safety", "analysis", "multimodal"],
                "license_type": "Commercial",
                "documentation_url": "https://docs.anthropic.com/claude/docs/models-overview",
                "metadata": {
                    "model_architecture": "Constitutional AI",
                    "evaluation_metrics": {"MMLU": 86.8, "Math": 60.1, "HumanEval": 84.9},
                    "use_cases": ["Complex reasoning", "Data analysis", "Research", "Writing"],
                    "safety_measures": ["Constitutional AI", "Human feedback", "Safety filtering"],
                    "ethical_considerations": "Designed with safety and helpfulness as primary goals"
                }
            },
            {
                "name": "Llama 2 70B Chat",
                "description": "Meta's open-source large language model optimized for dialogue use cases. Great balance of capability and accessibility with commercial-friendly licensing.",
                "model_id": "meta-llama/Llama-2-70b-chat-hf",
                "provider": ModelProvider.META,
                "category": ModelCategory.LANGUAGE_MODEL,
                "provider_name": "Meta",
                "provider_url": "https://ai.meta.com/llama/",
                "model_version": "2-70b-chat",
                "pricing_tier": PricingTier.BASIC,
                "price_per_token": Decimal('0.000008'),
                "context_length": 4096,
                "max_tokens": 2048,
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "languages_supported": ["en", "es", "fr", "de", "it", "pt"],
                "tags": ["llama", "open-source", "chat", "dialogue", "meta"],
                "license_type": "Llama 2 Community License",
                "documentation_url": "https://huggingface.co/meta-llama/Llama-2-70b-chat-hf",
                "metadata": {
                    "parameter_count": "70B",
                    "training_data_size": "2T tokens",
                    "model_architecture": "Transformer",
                    "evaluation_metrics": {"MMLU": 68.9, "HellaSwag": 87.3, "HumanEval": 29.9},
                    "use_cases": ["Conversational AI", "Content generation", "Code assistance"],
                    "limitations": ["May generate harmful content without proper filtering"],
                    "safety_measures": ["RLHF", "Safety fine-tuning", "Red teaming"]
                }
            },
            {
                "name": "DALL-E 3",
                "description": "OpenAI's most advanced text-to-image generation model. Creates highly detailed, creative images from natural language descriptions with improved prompt following.",
                "model_id": "dall-e-3",
                "provider": ModelProvider.OPENAI,
                "category": ModelCategory.IMAGE_GENERATION,
                "provider_name": "OpenAI",
                "provider_url": "https://openai.com",
                "model_version": "3.0",
                "pricing_tier": PricingTier.PREMIUM,
                "price_per_request": Decimal('0.04'),  # $0.04 per image
                "context_length": 4000,
                "max_tokens": 1,  # 1 image per request
                "input_modalities": ["text"],
                "output_modalities": ["image"],
                "languages_supported": ["en"],
                "tags": ["dalle", "image", "art", "creative", "generation"],
                "license_type": "Commercial",
                "documentation_url": "https://platform.openai.com/docs/models/dall-e",
                "metadata": {
                    "model_architecture": "Diffusion Model",
                    "evaluation_metrics": {"FID": "Not disclosed", "Human Preference": "95%+"},
                    "use_cases": ["Digital art", "Marketing materials", "Concept visualization"],
                    "limitations": ["Cannot generate images of real people", "Content policy restrictions"],
                    "safety_measures": ["Content filtering", "Watermarking", "Usage monitoring"]
                }
            },
            {
                "name": "Whisper Large v3",
                "description": "OpenAI's state-of-the-art speech recognition model. Supports 99 languages with high accuracy and robust performance across diverse audio conditions.",
                "model_id": "whisper-large-v3",
                "provider": ModelProvider.OPENAI,
                "category": ModelCategory.SPEECH_TO_TEXT,
                "provider_name": "OpenAI",
                "provider_url": "https://openai.com",
                "model_version": "large-v3",
                "pricing_tier": PricingTier.BASIC,
                "price_per_minute": Decimal('0.006'),  # $0.006 per minute
                "context_length": 30,  # 30 seconds max
                "max_tokens": 1000,
                "input_modalities": ["audio"],
                "output_modalities": ["text"],
                "languages_supported": ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"],
                "tags": ["whisper", "speech", "transcription", "multilingual", "audio"],
                "license_type": "Apache 2.0",
                "documentation_url": "https://platform.openai.com/docs/models/whisper",
                "metadata": {
                    "parameter_count": "1.55B",
                    "training_data_size": "680,000 hours",
                    "model_architecture": "Transformer Encoder-Decoder",
                    "evaluation_metrics": {"WER English": 2.5, "WER Multilingual": 4.7},
                    "use_cases": ["Transcription", "Subtitles", "Voice interfaces", "Accessibility"],
                    "limitations": ["30-second audio limit", "Background noise sensitivity"],
                    "safety_measures": ["Content filtering", "Language detection"]
                }
            },
            {
                "name": "Code Llama 34B Instruct",
                "description": "Meta's specialized code generation model. Excellent for code completion, debugging, explanation, and programming assistance across multiple languages.",
                "model_id": "codellama/CodeLlama-34b-Instruct-hf",
                "provider": ModelProvider.META,
                "category": ModelCategory.CODE_GENERATION,
                "provider_name": "Meta",
                "provider_url": "https://ai.meta.com/blog/code-llama-large-language-model-coding/",
                "model_version": "34b-instruct",
                "pricing_tier": PricingTier.BASIC,
                "price_per_token": Decimal('0.000006'),
                "context_length": 16384,
                "max_tokens": 2048,
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "languages_supported": ["en"],
                "tags": ["code", "programming", "llama", "completion", "debugging"],
                "license_type": "Llama 2 Community License",
                "documentation_url": "https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf",
                "metadata": {
                    "parameter_count": "34B",
                    "training_data_size": "500B code tokens",
                    "model_architecture": "Transformer",
                    "evaluation_metrics": {"HumanEval": 48.8, "MBPP": 55.0},
                    "use_cases": ["Code completion", "Bug fixing", "Code explanation", "Refactoring"],
                    "limitations": ["English only", "May generate insecure code"],
                    "safety_measures": ["Code safety filtering", "Vulnerability detection"]
                }
            },
            {
                "name": "Stable Diffusion XL",
                "description": "Stability AI's flagship image generation model. Creates high-resolution, photorealistic images with excellent prompt adherence and artistic capabilities.",
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "provider": ModelProvider.STABILITY,
                "category": ModelCategory.IMAGE_GENERATION,
                "provider_name": "Stability AI",
                "provider_url": "https://stability.ai",
                "model_version": "xl-base-1.0",
                "pricing_tier": PricingTier.BASIC,
                "price_per_request": Decimal('0.02'),  # $0.02 per image
                "context_length": 77,  # Token limit for prompts
                "max_tokens": 1,
                "input_modalities": ["text"],
                "output_modalities": ["image"],
                "languages_supported": ["en"],
                "tags": ["stable-diffusion", "image", "art", "photorealistic", "open-source"],
                "license_type": "CreativeML Open RAIL++-M",
                "documentation_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
                "metadata": {
                    "model_architecture": "Diffusion Model (U-Net)",
                    "evaluation_metrics": {"FID": 23.4, "Human Preference": "85%"},
                    "use_cases": ["Digital art", "Concept art", "Photo editing", "Creative design"],
                    "limitations": ["May struggle with text generation", "NSFW content filtering"],
                    "safety_measures": ["Content filtering", "Invisible watermarking"]
                }
            },
            {
                "name": "MPT-30B Chat",
                "description": "MosaicML's instruction-following language model. Trained for helpfulness, harmlessness, and honesty with commercial licensing.",
                "model_id": "mosaicml/mpt-30b-chat",
                "provider": ModelProvider.COMMUNITY,
                "category": ModelCategory.LANGUAGE_MODEL,
                "provider_name": "MosaicML",
                "provider_url": "https://www.mosaicml.com",
                "model_version": "30b-chat",
                "pricing_tier": PricingTier.FREE,
                "price_per_token": Decimal('0'),
                "context_length": 8192,
                "max_tokens": 2048,
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "languages_supported": ["en"],
                "tags": ["mpt", "chat", "open-source", "commercial", "instruction"],
                "license_type": "Apache 2.0",
                "documentation_url": "https://huggingface.co/mosaicml/mpt-30b-chat",
                "metadata": {
                    "parameter_count": "30B",
                    "training_data_size": "1T tokens",
                    "model_architecture": "Transformer (MPT)",
                    "evaluation_metrics": {"MMLU": 46.9, "HellaSwag": 79.3},
                    "use_cases": ["Conversational AI", "Q&A", "Instruction following"],
                    "limitations": ["English only", "May generate biased content"],
                    "safety_measures": ["Instruction fine-tuning", "Safety guidelines"]
                }
            }
        ]
    
    async def create_initial_listings(self) -> List[str]:
        """
        Create initial model listings for marketplace launch
        
        Returns:
            List of created listing IDs
        """
        try:
            logger.info("Starting creation of initial marketplace listings")
            
            created_listings = []
            
            for listing_data in self.initial_listings:
                try:
                    # Create metadata object
                    metadata = ModelMetadata(**listing_data.pop("metadata", {}))
                    
                    # Create request object
                    request = CreateModelListingRequest(
                        **listing_data,
                        metadata=metadata.dict()
                    )
                    
                    # Create the listing
                    listing = await marketplace_service.create_model_listing(
                        request=request,
                        owner_user_id=self.system_user_id
                    )
                    
                    # Automatically approve system listings
                    await marketplace_service.update_model_status(
                        listing_id=listing.id,
                        new_status=ModelStatus.ACTIVE,
                        moderator_user_id=self.system_user_id
                    )
                    
                    # Mark some as featured
                    if listing_data.get("name") in ["GPT-4 Turbo", "Claude-3 Opus", "DALL-E 3"]:
                        await self._mark_as_featured(listing.id)
                    
                    created_listings.append(str(listing.id))
                    
                    logger.info("Created initial listing",
                               name=listing_data.get("name"),
                               listing_id=str(listing.id))
                    
                except Exception as e:
                    logger.error("Failed to create initial listing",
                               name=listing_data.get("name"),
                               error=str(e))
                    continue
            
            # Log completion
            await audit_logger.log_security_event(
                event_type="marketplace_initial_listings_created",
                user_id=str(self.system_user_id),
                details={
                    "total_listings": len(self.initial_listings),
                    "created_listings": len(created_listings),
                    "listing_ids": created_listings
                },
                security_level="info"
            )
            
            logger.info("Completed initial marketplace listings creation",
                       total_created=len(created_listings),
                       total_attempted=len(self.initial_listings))
            
            return created_listings
            
        except Exception as e:
            logger.error("Failed to create initial marketplace listings", error=str(e))
            return []
    
    async def _mark_as_featured(self, listing_id: str) -> bool:
        """Mark a listing as featured"""
        try:
            # In a real implementation, this would update the database
            # For now, just log the action
            logger.info("Marked listing as featured", listing_id=listing_id)
            return True
            
        except Exception as e:
            logger.error("Failed to mark listing as featured",
                        listing_id=listing_id,
                        error=str(e))
            return False
    
    async def get_launch_summary(self) -> Dict[str, Any]:
        """Get summary of marketplace launch status"""
        try:
            stats = await marketplace_service.get_marketplace_stats()
            
            categories_with_models = set()
            providers_with_models = set()
            
            # Get category and provider coverage
            for listing_data in self.initial_listings:
                categories_with_models.add(listing_data["category"].value)
                providers_with_models.add(listing_data["provider"].value)
            
            return {
                "total_initial_listings": len(self.initial_listings),
                "categories_covered": len(categories_with_models),
                "providers_included": len(providers_with_models),
                "pricing_tiers": ["free", "basic", "premium"],
                "featured_models": 3,
                "marketplace_stats": stats.dict(),
                "launch_ready": stats.total_models >= 5,
                "launch_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get launch summary", error=str(e))
            return {
                "total_initial_listings": 0,
                "categories_covered": 0,
                "providers_included": 0,
                "launch_ready": False,
                "error": str(e)
            }


# Global instance
initial_listings_creator = InitialListingsCreator()


async def launch_marketplace_with_initial_listings() -> Dict[str, Any]:
    """
    Launch the marketplace with initial model listings
    
    This function:
    1. Creates initial high-quality model listings
    2. Sets up featured models for discovery
    3. Validates marketplace readiness
    4. Returns launch summary
    """
    try:
        logger.info("🚀 Launching PRSM Marketplace with initial listings")
        
        # Create initial listings
        created_listings = await initial_listings_creator.create_initial_listings()
        
        # Get launch summary
        launch_summary = await initial_listings_creator.get_launch_summary()
        launch_summary["created_listing_ids"] = created_listings
        
        if launch_summary.get("launch_ready", False):
            logger.info("✅ Marketplace successfully launched",
                       total_listings=len(created_listings))
        else:
            logger.warning("⚠️ Marketplace launch incomplete",
                          created_listings=len(created_listings))
        
        return {
            "success": True,
            "message": "Marketplace launched with initial listings",
            "launch_summary": launch_summary
        }
        
    except Exception as e:
        logger.error("❌ Failed to launch marketplace", error=str(e))
        return {
            "success": False,
            "message": "Marketplace launch failed",
            "error": str(e)
        }