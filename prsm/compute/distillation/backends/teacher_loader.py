"""
Teacher Model Loader for PRSM Distillation

🎯 PURPOSE:
Provides a unified interface for loading teacher models from various sources:
- HuggingFace Hub (org/model-name format)
- Local checkpoints (.pt/.pth files or HF-format directories)
- API-based teachers (Anthropic Claude, OpenAI GPT, Ollama)

🔧 DESIGN INVARIANT:
load_teacher_model() must NEVER raise. It always returns a TeacherModelWrapper,
even on failure. This ensures training continues with student-only cross-entropy
loss when teacher loading fails.

📚 PHASE 1 PRIORITY 3:
This module replaces the `teacher_model = None` stub that caused a four-layer
failure chain in the distillation system.
"""

import os
import logging
from enum import Enum
from typing import Optional, Any, Dict
from uuid import uuid4

logger = logging.getLogger(__name__)

# Optional dependencies - import with try/except
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - teacher soft labels will be disabled")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available - HuggingFace teacher loading disabled")

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    bnb = None  # type: ignore
    BITSANDBYTES_AVAILABLE = False


class TeacherSource(str, Enum):
    """
    Enumeration of supported teacher model sources.
    
    Each source has different capabilities:
    - HF_HUB / LOCAL_CHECKPOINT: Support soft-label distillation (real logits)
    - API sources: Support hard-label distillation only (text responses)
    """
    HF_HUB = "hf_hub"                    # org/model-name format (HuggingFace)
    LOCAL_CHECKPOINT = "local"          # /path/to/file.pt or /path/to/dir
    ANTHROPIC_API = "anthropic_api"     # claude-* models
    OPENAI_API = "openai_api"           # gpt-*, o1-*, text-*
    OLLAMA = "ollama"                   # ollama:model-name or local Ollama server
    UNKNOWN = "unknown"                 # fallback — try HF Hub, log warning


def classify_teacher_source(model_id: str) -> TeacherSource:
    """
    Classify the source of a teacher model based on its identifier.
    
    Rules are applied in order (first match wins):
    1. If path exists on filesystem → LOCAL_CHECKPOINT
    2. If model_id suggests Anthropic API → ANTHROPIC_API
    3. If model_id suggests OpenAI API → OPENAI_API
    4. If model_id suggests Ollama → OLLAMA
    5. If model_id has org/model format → HF_HUB
    6. Fallback → UNKNOWN
    
    Args:
        model_id: Teacher model identifier (path, HF model name, or API model name)
        
    Returns:
        TeacherSource enum value indicating the detected source
        
    Examples:
        >>> classify_teacher_source("/path/to/model.pt")
        TeacherSource.LOCAL_CHECKPOINT
        >>> classify_teacher_source("meta-llama/Llama-2-7b-hf")
        TeacherSource.HF_HUB
        >>> classify_teacher_source("claude-3-opus")
        TeacherSource.ANTHROPIC_API
        >>> classify_teacher_source("gpt-4")
        TeacherSource.OPENAI_API
    """
    if not model_id:
        logger.warning("Empty model_id provided, defaulting to UNKNOWN")
        return TeacherSource.UNKNOWN
    
    # Rule 1: Check if it's a local path that exists
    if os.path.exists(model_id):
        return TeacherSource.LOCAL_CHECKPOINT
    
    model_id_lower = model_id.lower()
    
    # Rule 2: Anthropic API detection
    if model_id_lower.startswith("claude") or "anthropic" in model_id_lower:
        return TeacherSource.ANTHROPIC_API
    
    # Rule 3: OpenAI API detection
    openai_prefixes = ["gpt-", "o1-", "text-davinci", "davinci"]
    if any(model_id_lower.startswith(prefix) for prefix in openai_prefixes):
        return TeacherSource.OPENAI_API
    
    # Rule 4: Ollama detection
    if model_id_lower.startswith("ollama:") or model_id_lower.startswith("ollama/"):
        return TeacherSource.OLLAMA
    
    # Rule 5: HuggingFace Hub detection (org/model-name format)
    # Must contain "/" but not start with "/" (which would be a local path)
    if "/" in model_id and not model_id.startswith("/"):
        return TeacherSource.HF_HUB
    
    # Rule 6: Fallback to UNKNOWN
    logger.warning(
        f"Could not classify teacher source for '{model_id}', "
        f"defaulting to UNKNOWN (will attempt HF Hub lookup)"
    )
    return TeacherSource.UNKNOWN


class TeacherModelWrapper:
    """
    Uniform interface for both local and API-based teacher models.
    
    This wrapper provides a consistent interface regardless of the teacher source:
    - Local/HF teachers: Support soft-label distillation (real logits)
    - API teachers: Support hard-label distillation (text responses only)
    
    Attributes:
        model_id: The original model identifier
        source: The detected TeacherSource
        model: The loaded local model (torch.nn.Module or HF model), or None for API sources
        tokenizer: The HF tokenizer, or None for API sources
        executor: ModelExecutor instance for API sources
        device: The device the model is loaded on
        load_error: Error message if loading failed, None otherwise
    
    Design Invariant:
        This class never raises during initialization. If loading fails,
        supports_soft_labels returns False and model is None.
    """
    
    def __init__(
        self,
        model_id: str,
        source: TeacherSource,
        local_model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        executor: Optional[Any] = None,
        device: str = "cpu",
        load_error: Optional[str] = None
    ):
        """
        Initialize the TeacherModelWrapper.
        
        Args:
            model_id: Original model identifier
            source: Detected TeacherSource
            local_model: Loaded torch.nn.Module or HF model (None for API sources)
            tokenizer: HF tokenizer (None for API sources)
            executor: ModelExecutor instance for API calls
            device: Device the model is loaded on
            load_error: Error message if loading failed
        """
        self.model_id = model_id
        self.source = source
        self.model = local_model
        self.tokenizer = tokenizer
        self.executor = executor
        self.device = device
        self.load_error = load_error
        self._instance_id = str(uuid4())[:8]
        
        logger.info(
            f"TeacherModelWrapper initialized [{self._instance_id}]: "
            f"model_id='{model_id}', source={source.value}, "
            f"supports_soft_labels={self.supports_soft_labels}"
        )
    
    @property
    def supports_soft_labels(self) -> bool:
        """
        Check if this teacher supports soft-label distillation.
        
        Returns True only when:
        1. A real local model is loaded (self.model is not None)
        2. The source type supports logits (HF_HUB, LOCAL_CHECKPOINT, or UNKNOWN)
        
        API-based teachers (Anthropic, OpenAI, Ollama) cannot return logits
        and therefore only support hard-label distillation.
        
        Returns:
            bool: True if soft-label distillation is supported
        """
        return (
            self.model is not None 
            and self.source in (
                TeacherSource.HF_HUB,
                TeacherSource.LOCAL_CHECKPOINT,
                TeacherSource.UNKNOWN
            )
        )
    
    async def get_soft_labels(
        self,
        input_ids: Any,
        attention_mask: Any
    ) -> Optional[Any]:
        """
        Get teacher logits for soft-label distillation.
        
        This method returns the raw logits from the teacher model, which can
        be used for knowledge distillation with soft labels (KL divergence).
        
        Args:
            input_ids: Tokenized input IDs (torch.Tensor)
            attention_mask: Attention mask (torch.Tensor)
            
        Returns:
            torch.Tensor of logits, or None if soft labels are not supported
            
        Note:
            For API-based teachers, this always returns None. Use get_response()
            for hard-label distillation instead.
        """
        if not self.supports_soft_labels:
            logger.debug(
                f"Teacher [{self._instance_id}] does not support soft labels "
                f"(source={self.source.value}, model_loaded={self.model is not None})"
            )
            return None
        
        if not TORCH_AVAILABLE:
            logger.warning("Cannot get soft labels: PyTorch not available")
            return None
        
        if self.model is None:
            logger.warning(f"Teacher [{self._instance_id}] model is None")
            return None
        
        try:
            # Ensure model is in eval mode
            self.model.eval()
            
            # Move inputs to the correct device
            if hasattr(input_ids, 'to'):
                input_ids = input_ids.to(self.device)
            if hasattr(attention_mask, 'to'):
                attention_mask = attention_mask.to(self.device)
            
            # Run inference without gradients
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                # Return logits
                if hasattr(outputs, 'logits'):
                    return outputs.logits
                elif isinstance(outputs, dict) and 'logits' in outputs:
                    return outputs['logits']
                else:
                    logger.warning(
                        f"Teacher [{self._instance_id}] output has no logits attribute"
                    )
                    return None
                    
        except Exception as e:
            logger.error(
                f"Error getting soft labels from teacher [{self._instance_id}]: {e}"
            )
            return None
    
    async def get_response(self, prompt: str) -> str:
        """
        Get teacher text response for hard-label distillation.
        
        This method generates a text response from the teacher, which can be
        used for hard-label distillation or response-based distillation.
        
        Args:
            prompt: The input prompt to generate a response for
            
        Returns:
            str: The generated response text, or empty string on failure
            
        Note:
            For local models, this uses the model's generate method.
            For API models, this uses the executor to make API calls.
        """
        if not prompt:
            logger.warning(f"Teacher [{self._instance_id}] received empty prompt")
            return ""
        
        # Try local model generation first
        if self.model is not None and self.tokenizer is not None:
            return await self._generate_local(prompt)
        
        # Fall back to API via executor
        if self.executor is not None:
            return await self._generate_api(prompt)
        
        logger.warning(
            f"Teacher [{self._instance_id}] cannot generate response: "
            f"no local model and no executor available"
        )
        return ""
    
    async def _generate_local(self, prompt: str) -> str:
        """Generate response using local model."""
        if not TORCH_AVAILABLE or self.model is None or self.tokenizer is None:
            return ""
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Generate
            self.model.eval()
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            response = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()
            
        except Exception as e:
            logger.error(
                f"Error generating local response [{self._instance_id}]: {e}"
            )
            return ""
    
    async def _generate_api(self, prompt: str) -> str:
        """Generate response using API executor."""
        if self.executor is None:
            return ""
        
        try:
            # The executor should have a method to generate responses
            # This is a generic interface that works with different API backends
            if hasattr(self.executor, 'generate'):
                response = await self.executor.generate(prompt)
                return response if isinstance(response, str) else ""
            elif hasattr(self.executor, 'complete'):
                response = await self.executor.complete(prompt)
                return response if isinstance(response, str) else ""
            elif hasattr(self.executor, 'chat'):
                response = await self.executor.chat(prompt)
                return response if isinstance(response, str) else ""
            else:
                logger.warning(
                    f"Executor for teacher [{self._instance_id}] has no known generation method"
                )
                return ""
                
        except Exception as e:
            logger.error(
                f"Error generating API response [{self._instance_id}]: {e}"
            )
            return ""
    
    def __repr__(self) -> str:
        return (
            f"TeacherModelWrapper(model_id='{self.model_id}', "
            f"source={self.source.value}, "
            f"supports_soft_labels={self.supports_soft_labels}, "
            f"device='{self.device}')"
        )


async def load_teacher_model(
    model_id: str,
    device: str = "auto",
    executor: Optional[Any] = None
) -> TeacherModelWrapper:
    """
    Load a teacher model from various sources.
    
    This is the main entry point for loading teacher models. It handles
    all source types gracefully and never raises exceptions.
    
    CRITICAL DESIGN INVARIANT:
        This function must NEVER raise. It always returns a TeacherModelWrapper,
        even on failure. If loading fails, wrapper.supports_soft_labels will be False
        and wrapper.model will be None. Training continues with student-only
        cross-entropy loss.
    
    Args:
        model_id: Teacher model identifier. Can be:
            - HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
            - Local path to .pt/.pth file or HF-format directory
            - API model name (e.g., "claude-3-opus", "gpt-4", "ollama:llama2")
        device: Target device ("cpu", "cuda", "auto"). Default: "auto"
        executor: Optional ModelExecutor instance for API-based sources
        
    Returns:
        TeacherModelWrapper: A wrapper around the loaded model. Check
            wrapper.supports_soft_labels to determine if soft-label
            distillation is available.
            
    Examples:
        >>> wrapper = await load_teacher_model("meta-llama/Llama-2-7b-hf")
        >>> if wrapper.supports_soft_labels:
        ...     logits = await wrapper.get_soft_labels(input_ids, attention_mask)
        
        >>> wrapper = await load_teacher_model("claude-3-opus", executor=my_executor)
        >>> response = await wrapper.get_response("What is knowledge distillation?")
    """
    # Validate inputs
    if not model_id:
        logger.error("load_teacher_model called with empty model_id")
        return TeacherModelWrapper(
            model_id="",
            source=TeacherSource.UNKNOWN,
            load_error="Empty model_id provided"
        )
    
    # Classify the source
    source = classify_teacher_source(model_id)
    logger.info(f"Loading teacher model '{model_id}' from source: {source.value}")
    
    # Resolve device
    resolved_device = _resolve_device(device)
    
    # Route to appropriate loader based on source
    try:
        if source == TeacherSource.LOCAL_CHECKPOINT:
            return await _load_local_checkpoint(model_id, resolved_device)
        
        elif source == TeacherSource.HF_HUB:
            return await _load_hf_hub(model_id, resolved_device)
        
        elif source == TeacherSource.UNKNOWN:
            # Try HF Hub as fallback
            logger.warning(
                f"Unknown source for '{model_id}', attempting HF Hub lookup"
            )
            wrapper = await _load_hf_hub(model_id, resolved_device)
            if wrapper.load_error:
                wrapper.source = TeacherSource.UNKNOWN
            return wrapper
        
        elif source in (TeacherSource.ANTHROPIC_API, TeacherSource.OPENAI_API, TeacherSource.OLLAMA):
            return _create_api_wrapper(model_id, source, executor, resolved_device)
        
        else:
            # This shouldn't happen, but handle gracefully
            logger.error(f"Unhandled teacher source: {source}")
            return TeacherModelWrapper(
                model_id=model_id,
                source=source,
                load_error=f"Unhandled source type: {source}"
            )
            
    except Exception as e:
        # CRITICAL: Never raise, always return a wrapper
        logger.error(f"Unexpected error loading teacher model '{model_id}': {e}")
        return TeacherModelWrapper(
            model_id=model_id,
            source=source,
            load_error=f"Unexpected error: {str(e)}"
        )


def _resolve_device(device: str) -> str:
    """
    Resolve the target device for model loading.
    
    Args:
        device: Device specification ("cpu", "cuda", "auto")
        
    Returns:
        str: Resolved device string
    """
    if device == "auto":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


async def _load_local_checkpoint(model_id: str, device: str) -> TeacherModelWrapper:
    """
    Load a teacher model from a local checkpoint.
    
    Handles two formats:
    - .pt/.pth files: Loaded with torch.load()
    - Directories: Assumed to be HF format, loaded with from_pretrained()
    
    Args:
        model_id: Path to the local checkpoint
        device: Target device
        
    Returns:
        TeacherModelWrapper with the loaded model or error
    """
    if not TORCH_AVAILABLE:
        return TeacherModelWrapper(
            model_id=model_id,
            source=TeacherSource.LOCAL_CHECKPOINT,
            load_error="PyTorch not available for loading local checkpoint"
        )
    
    try:
        # Check if it's a .pt or .pth file
        if model_id.endswith(".pt") or model_id.endswith(".pth"):
            logger.info(f"Loading PyTorch checkpoint from {model_id}")
            model = torch.load(model_id, map_location=device)
            
            # Try to load tokenizer from same directory
            tokenizer = None
            if TRANSFORMERS_AVAILABLE and AutoTokenizer is not None:
                checkpoint_dir = os.path.dirname(model_id)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        checkpoint_dir,
                        local_files_only=True
                    )
                except Exception as e:
                    logger.warning(f"Could not load tokenizer from {checkpoint_dir}: {e}")
            
            return TeacherModelWrapper(
                model_id=model_id,
                source=TeacherSource.LOCAL_CHECKPOINT,
                local_model=model,
                tokenizer=tokenizer,
                device=device
            )
        
        else:
            # Assume HF format directory
            logger.info(f"Loading HF format model from directory {model_id}")
            
            if not TRANSFORMERS_AVAILABLE or AutoModelForCausalLM is None:
                return TeacherModelWrapper(
                    model_id=model_id,
                    source=TeacherSource.LOCAL_CHECKPOINT,
                    load_error="Transformers not available for loading HF format"
                )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=True,
                device_map=device
            )
            model.eval()
            
            tokenizer = None
            if AutoTokenizer is not None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        local_files_only=True
                    )
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                except Exception as e:
                    logger.warning(f"Could not load tokenizer: {e}")
            
            return TeacherModelWrapper(
                model_id=model_id,
                source=TeacherSource.LOCAL_CHECKPOINT,
                local_model=model,
                tokenizer=tokenizer,
                device=device
            )
            
    except Exception as e:
        logger.error(f"Failed to load local checkpoint '{model_id}': {e}")
        return TeacherModelWrapper(
            model_id=model_id,
            source=TeacherSource.LOCAL_CHECKPOINT,
            load_error=f"Failed to load local checkpoint: {str(e)}"
        )


async def _load_hf_hub(model_id: str, device: str) -> TeacherModelWrapper:
    """
    Load a teacher model from HuggingFace Hub.
    
    Attempts to load with 8-bit quantization if bitsandbytes is available.
    
    Args:
        model_id: HuggingFace model identifier (org/model-name)
        device: Target device
        
    Returns:
        TeacherModelWrapper with the loaded model or error
    """
    if not TRANSFORMERS_AVAILABLE or AutoModelForCausalLM is None:
        return TeacherModelWrapper(
            model_id=model_id,
            source=TeacherSource.HF_HUB,
            load_error="Transformers not available for loading from HF Hub"
        )
    
    if not TORCH_AVAILABLE:
        return TeacherModelWrapper(
            model_id=model_id,
            source=TeacherSource.HF_HUB,
            load_error="PyTorch not available for loading from HF Hub"
        )
    
    try:
        logger.info(f"Loading model from HuggingFace Hub: {model_id}")
        
        # Prepare loading kwargs
        load_kwargs: Dict[str, Any] = {
            "pretrained_model_name_or_path": model_id,
            "device_map": device,
        }
        
        # Try 8-bit quantization if available and using CUDA
        if BITSANDBYTES_AVAILABLE and device == "cuda":
            try:
                load_kwargs["load_in_8bit"] = True
                logger.info("Attempting 8-bit quantization with bitsandbytes")
            except Exception as e:
                logger.warning(f"Could not enable 8-bit quantization: {e}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
        model.eval()
        
        # Load tokenizer
        tokenizer = None
        if AutoTokenizer is not None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    logger.debug("Set pad_token to eos_token")
            except Exception as e:
                logger.warning(f"Could not load tokenizer for {model_id}: {e}")
        
        logger.info(f"Successfully loaded teacher model from HF Hub: {model_id}")
        
        return TeacherModelWrapper(
            model_id=model_id,
            source=TeacherSource.HF_HUB,
            local_model=model,
            tokenizer=tokenizer,
            device=device
        )
        
    except Exception as e:
        logger.error(f"Failed to load model from HF Hub '{model_id}': {e}")
        return TeacherModelWrapper(
            model_id=model_id,
            source=TeacherSource.HF_HUB,
            load_error=f"Failed to load from HF Hub: {str(e)}"
        )


def _create_api_wrapper(
    model_id: str,
    source: TeacherSource,
    executor: Optional[Any],
    device: str
) -> TeacherModelWrapper:
    """
    Create a wrapper for an API-based teacher model.
    
    API teachers don't have local models - they only support hard-label
    distillation via text responses.
    
    Args:
        model_id: API model identifier
        source: TeacherSource (ANTHROPIC_API, OPENAI_API, or OLLAMA)
        executor: Optional ModelExecutor for API calls
        device: Device (not used for API, but stored for reference)
        
    Returns:
        TeacherModelWrapper configured for API access
    """
    logger.info(f"Creating API teacher wrapper for {source.value}: {model_id}")
    
    if executor is None:
        logger.warning(
            f"No executor provided for API teacher '{model_id}'. "
            f"Response generation will be disabled."
        )
    
    return TeacherModelWrapper(
        model_id=model_id,
        source=source,
        local_model=None,  # No local model for API sources
        tokenizer=None,    # No tokenizer for API sources
        executor=executor,
        device=device
    )
