"""
Capability Detection
====================

Detects node capabilities at startup for peer discovery and job routing.

Capabilities include:
- Available backends (anthropic, openai, local)
- GPU availability
- Supported operations (inference, embedding, benchmark)
"""

import logging
import os
import shutil
import subprocess
from typing import List, Tuple

logger = logging.getLogger(__name__)


def detect_gpu_availability() -> bool:
    """Check if GPU is available on this node.

    Returns:
        True if GPU is detected, False otherwise.
    """
    # Check for NVIDIA GPU via nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                count = int(result.stdout.strip())
                if count > 0:
                    logger.info(f"Detected {count} NVIDIA GPU(s)")
                    return True
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass

    # Check for Apple Silicon GPU (M-series)
    if os.path.exists("/System/Library/PrivateFrameworks/MetalPerformance.framework"):
        logger.info("Apple Silicon GPU detected")
        return True

    # Check CUDA environment variables
    if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_DEVICE_ORDER"):
        logger.info("CUDA environment detected")
        return True

    logger.debug("No GPU detected")
    return False


def detect_available_backends() -> Tuple[List[str], bool]:
    """Detect which LLM backends are available on this node.

    Returns:
        Tuple of (list of available backend names, any_real_backend flag).
        Backend names are lowercase: "anthropic", "openai", "local".
    """
    backends = []

    # Check Anthropic API key
    if os.environ.get("ANTHROPIC_API_KEY"):
        backends.append("anthropic")
        logger.info("Anthropic backend available (API key detected)")

    # Check OpenAI API key
    if os.environ.get("OPENAI_API_KEY"):
        backends.append("openai")
        logger.info("OpenAI backend available (API key detected)")

    # Check for local model support
    # This could be Ollama, local transformers, or other local inference
    local_available = False

    # Check for Ollama
    if shutil.which("ollama"):
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                backends.append("local")
                local_available = True
                logger.info("Local backend available (Ollama detected)")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Check for local model URL (transformers/other)
    if not local_available and os.environ.get("PRSM_LOCAL_MODEL_URL"):
        backends.append("local")
        local_available = True
        logger.info("Local backend available (PRSM_LOCAL_MODEL_URL set)")

    # Check for local model path
    if not local_available and os.environ.get("PRSM_LOCAL_MODEL_PATH"):
        model_path = os.environ.get("PRSM_LOCAL_MODEL_PATH")
        if os.path.exists(model_path):
            backends.append("local")
            local_available = True
            logger.info("Local backend available (model path detected)")

    any_real_backend = len(backends) > 0
    if not any_real_backend:
        logger.warning("No real backends detected - node will use mock backend only")

    return backends, any_real_backend


def detect_node_capabilities() -> dict:
    """Detect all node capabilities at startup.

    Returns:
        Dictionary with:
        - capabilities: List of capability strings (e.g., "inference", "embedding")
        - supported_backends: List of available backend names
        - gpu_available: Boolean indicating GPU presence
        - any_real_backend: Boolean indicating if any real backend is available
    """
    backends, any_real_backend = detect_available_backends()
    gpu_available = detect_gpu_availability()

    # Determine capabilities based on available backends
    capabilities = []

    if any_real_backend:
        # All backends support inference
        capabilities.append("inference")

        # OpenAI and local backends typically support embeddings
        if "openai" in backends or "local" in backends:
            capabilities.append("embedding")

        # Benchmark capability requires local execution
        if "local" in backends:
            capabilities.append("benchmark")

        # Training/fine-tuning typically requires local GPU
        if gpu_available and "local" in backends:
            capabilities.append("training")
            capabilities.append("fine_tuning")

    # Distillation capability removed in v1.6.0 (PRSM is no longer an AGI framework)

    # If no real backends, add minimal capability for testing
    if not capabilities:
        capabilities.append("mock")

    logger.info(
        f"Node capabilities detected: capabilities={capabilities}, "
        f"backends={backends}, gpu={gpu_available}"
    )

    return {
        "capabilities": capabilities,
        "supported_backends": backends,
        "gpu_available": gpu_available,
        "any_real_backend": any_real_backend,
    }


def get_capabilities_for_discovery() -> dict:
    """Get capabilities in a format suitable for peer discovery announcement.

    This is a convenience wrapper around detect_node_capabilities.

    Returns:
        Dictionary with capabilities, supported_backends, and gpu_available.
    """
    caps = detect_node_capabilities()
    return {
        "capabilities": caps["capabilities"],
        "supported_backends": caps["supported_backends"],
        "gpu_available": caps["gpu_available"],
    }