"""
BSC Deployment Manager
======================

Handles the choice between local model inference and PRSM network service.

Local mode  — loads a small causal LM via HuggingFace transformers (default) or
              optionally via llama-cpp-python (GGUF) or mlx-lm (Apple Silicon MLX).
              A 0.5B–3B quantized model runs comfortably on any M-series Mac with 16 GB RAM.
              Evaluation-only (no generation) is 3–5× cheaper than a generation call.

Network mode — calls a PRSM BSC service node over HTTP.
               The service node earns FTNS for each evaluation.
               Requires PRSM network connectivity and an FTNS balance.

The interface is identical in both modes; callers never need to know which is active.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class DeploymentMode(str, Enum):
    """Supported BSC deployment backends."""
    LOCAL_TRANSFORMERS = "local_transformers"   # HuggingFace transformers (default, no extras)
    LOCAL_LLAMA_CPP   = "local_llama_cpp"       # llama-cpp-python — requires: pip install llama-cpp-python
    LOCAL_MLX         = "local_mlx"             # mlx-lm — requires: pip install mlx-lm (Apple Silicon only)
    NETWORK_SERVICE   = "network_service"       # PRSM BSC node over HTTP


@dataclass
class BSCDeploymentConfig:
    """
    Full configuration for a BSC deployment.

    Attributes
    ----------
    mode : DeploymentMode
        Which backend to use.  Defaults to LOCAL_TRANSFORMERS.
    model_name : str
        HuggingFace model ID (for LOCAL_TRANSFORMERS / LOCAL_MLX) or path to a
        .gguf file (for LOCAL_LLAMA_CPP).
        Default ``"Qwen/Qwen2.5-0.5B"`` — 353 M parameters, ~700 MB fp16,
        trained on code + natural language, fast on any hardware.
    device : Optional[str]
        Torch device string.  None = auto-detect (MPS → CUDA → CPU).
    max_context_tokens : int
        Maximum tokens of context fed to the predictor.  Longer contexts are
        left-truncated, preserving the most-recent tokens.
    network_endpoint : Optional[str]
        HTTP base URL of the PRSM BSC service node (NETWORK_SERVICE mode only).
        Example: ``"http://bsc-node.prsm-network.com:7890"``
    ftns_budget : float
        Maximum FTNS to spend per session (NETWORK_SERVICE mode only).
        A value of 0.0 means unlimited.
    epsilon : float
        Surprise threshold in [0, 1].  Chunks whose normalised surprise score
        exceeds epsilon are forwarded to semantic de-duplication.
        Tune upward to make the filter more selective; downward to capture more.
    similarity_threshold : float
        Cosine-similarity ceiling for semantic de-duplication.  Candidates
        already represented in the whiteboard at this similarity or above are
        discarded even if they exceed epsilon.
    embedding_model : str
        Sentence-transformer model used for semantic de-duplication embeddings.
        Must be compatible with ``sentence-transformers`` or HuggingFace
        ``AutoModel``.
    """

    mode: DeploymentMode = DeploymentMode.LOCAL_TRANSFORMERS

    # Predictor model
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: Optional[str] = None
    max_context_tokens: int = 2048

    # Network mode only
    network_endpoint: Optional[str] = None
    ftns_budget: float = 0.0

    # Thresholds
    epsilon: float = 0.55
    similarity_threshold: float = 0.85

    # Embedding model for semantic de-duplication
    embedding_model: str = "all-MiniLM-L6-v2"

    # Extra flags stored as a dict for future extensibility
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError if the config is internally inconsistent."""
        if self.mode == DeploymentMode.NETWORK_SERVICE and not self.network_endpoint:
            raise ValueError(
                "network_endpoint must be set when mode=NETWORK_SERVICE"
            )
        if self.mode == DeploymentMode.LOCAL_MLX:
            try:
                import mlx  # noqa: F401
            except ImportError:
                raise ValueError(
                    "mlx-lm is not installed.  Run: pip install mlx-lm"
                )
        if self.mode == DeploymentMode.LOCAL_LLAMA_CPP:
            try:
                import llama_cpp  # noqa: F401
            except ImportError:
                raise ValueError(
                    "llama-cpp-python is not installed.  "
                    "Run: pip install llama-cpp-python"
                )
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError("epsilon must be strictly between 0 and 1")
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")

    @classmethod
    def auto(cls) -> "BSCDeploymentConfig":
        """
        Return a sensible default config for the current hardware.

        Priority:
          1. Apple Silicon (MPS available)  → LOCAL_TRANSFORMERS on MPS
          2. CUDA GPU available             → LOCAL_TRANSFORMERS on CUDA
          3. CPU fallback                   → LOCAL_TRANSFORMERS on CPU
        """
        import torch

        device: Optional[str]
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("BSC auto-config: Apple Silicon MPS detected")
        elif torch.cuda.is_available():
            device = "cuda"
            logger.info("BSC auto-config: CUDA GPU detected")
        else:
            device = "cpu"
            logger.info("BSC auto-config: CPU-only mode")

        return cls(
            mode=DeploymentMode.LOCAL_TRANSFORMERS,
            device=device,
        )
