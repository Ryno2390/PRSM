"""
BSC Predictor — Perplexity Evaluator
=====================================

Wraps a small (0.5B–3B parameter) causal language model to evaluate the
cross-entropy loss (perplexity) of incoming text chunks relative to the
current compressed context.

    High perplexity  → high surprise  → candidate for whiteboard promotion
    Low perplexity   → expected info  → discard

Mathematical basis
------------------
For a sequence of tokens t_1 … t_n the cross-entropy loss is:

    H = -1/n  Σ  log P(t_i | t_1 … t_{i-1})

Perplexity is PP = exp(H).  This equals the KL divergence from the one-hot
"actual" distribution to the model's predicted distribution at each token
position, averaged over the chunk — i.e. it IS the KL divergence under the
standard simplification where q is one-hot.

Adaptive baseline
-----------------
A fixed epsilon works poorly when domain shifts (e.g. natural language → code).
The predictor maintains an exponential moving average of recent perplexity
values.  The final normalised score is sigmoid-shaped relative to this
adaptive baseline so that the threshold remains meaningful as context evolves.

Supported backends
------------------
  LOCAL_TRANSFORMERS  — HuggingFace AutoModelForCausalLM (default; no extras)
  LOCAL_LLAMA_CPP     — llama-cpp-python, GGUF models (optional extra)
  LOCAL_MLX           — mlx-lm, Apple Silicon MLX (optional extra)
  NETWORK_SERVICE     — HTTP call to a PRSM BSC service node
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .deployment import BSCDeploymentConfig, DeploymentMode

logger = logging.getLogger(__name__)


@dataclass
class SurpriseScore:
    """Output of a single BSC predictor evaluation."""

    score: float
    """Normalised surprise in [0, 1].  Values above BSCDeploymentConfig.epsilon
    are forwarded to the semantic de-duplication step."""

    raw_perplexity: float
    """Raw perplexity of the chunk given the context (exp of mean cross-entropy)."""

    token_count: int
    """Number of tokens in the evaluated chunk."""

    context_tokens: int
    """Number of context tokens that were actually fed to the model (may be
    less than the full context after left-truncation)."""

    adaptive_baseline: float
    """The baseline perplexity value at the time of this evaluation, for
    debugging and logging."""


class BSCPredictor:
    """
    Evaluates the informational surprise of a new text chunk.

    Usage
    -----
    .. code-block:: python

        config = BSCDeploymentConfig.auto()
        predictor = BSCPredictor(config)

        score = await predictor.score_surprise(
            context="We are building a distributed AI protocol.",
            chunk="The codebase now requires a PostgreSQL migration."
        )
        print(score.score)        # e.g. 0.83 — high surprise
    """

    def __init__(self, config: Optional[BSCDeploymentConfig] = None) -> None:
        self._config = config or BSCDeploymentConfig.auto()
        self._model = None
        self._tokenizer = None
        self._loaded = False

        # Adaptive baseline: EMA of recent perplexity values.
        # Seed with a reasonable estimate for mixed code + English text.
        self._baseline: float = 45.0
        self._baseline_alpha: float = 0.05  # EMA decay rate

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def score_surprise(self, context: str, chunk: str) -> SurpriseScore:
        """
        Score how surprising *chunk* is relative to *context*.

        Parameters
        ----------
        context:
            The current compressed whiteboard state or recent conversation
            summary — what the model "knows" so far.
        chunk:
            The new agent output to evaluate.

        Returns
        -------
        SurpriseScore
            Contains the normalised score, raw perplexity, and token counts.
        """
        if not chunk.strip():
            return SurpriseScore(
                score=0.0,
                raw_perplexity=0.0,
                token_count=0,
                context_tokens=0,
                adaptive_baseline=self._baseline,
            )

        mode = self._config.mode
        if mode == DeploymentMode.NETWORK_SERVICE:
            return await self._score_network(context, chunk)
        else:
            await self._ensure_loaded()
            loop = asyncio.get_event_loop()
            perplexity, n_chunk, n_ctx = await loop.run_in_executor(
                None, self._compute_perplexity, context, chunk
            )
            return self._build_score(perplexity, n_chunk, n_ctx)

    async def warmup(self) -> None:
        """Pre-load the model so the first real call is not slow."""
        await self._ensure_loaded()
        logger.info("BSC predictor warmed up")

    # ------------------------------------------------------------------
    # Internal: model loading
    # ------------------------------------------------------------------

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

    def _load_model(self) -> None:
        mode = self._config.mode
        if mode == DeploymentMode.LOCAL_TRANSFORMERS:
            self._load_transformers()
        elif mode == DeploymentMode.LOCAL_LLAMA_CPP:
            self._load_llama_cpp()
        elif mode == DeploymentMode.LOCAL_MLX:
            self._load_mlx()
        self._loaded = True

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self._config.model_name
        device = self._resolve_device()
        dtype = torch.float16 if device != "cpu" else torch.float32

        logger.info(f"Loading BSC predictor: {model_name} on {device} ({dtype})")
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        self._model.eval()
        n_params = sum(p.numel() for p in self._model.parameters())
        logger.info(f"BSC predictor loaded: {n_params / 1e6:.0f}M params")

    def _load_llama_cpp(self) -> None:
        from llama_cpp import Llama  # type: ignore[import]

        logger.info(f"Loading GGUF model: {self._config.model_name}")
        self._model = Llama(
            model_path=self._config.model_name,
            n_ctx=self._config.max_context_tokens,
            logits_all=True,
            verbose=False,
        )
        logger.info("GGUF model loaded")

    def _load_mlx(self) -> None:
        import mlx.core as mx  # type: ignore[import]
        from mlx_lm import load  # type: ignore[import]

        logger.info(f"Loading MLX model: {self._config.model_name}")
        self._model, self._tokenizer = load(self._config.model_name)
        logger.info("MLX model loaded")

    # ------------------------------------------------------------------
    # Internal: perplexity computation per backend
    # ------------------------------------------------------------------

    def _compute_perplexity(
        self, context: str, chunk: str
    ) -> tuple[float, int, int]:
        """
        Returns (perplexity, n_chunk_tokens, n_context_tokens_used).
        Dispatches to the correct backend implementation.
        """
        mode = self._config.mode
        if mode == DeploymentMode.LOCAL_TRANSFORMERS:
            return self._perplexity_transformers(context, chunk)
        elif mode == DeploymentMode.LOCAL_LLAMA_CPP:
            return self._perplexity_llama_cpp(context, chunk)
        elif mode == DeploymentMode.LOCAL_MLX:
            return self._perplexity_mlx(context, chunk)
        raise RuntimeError(f"Unknown mode: {mode}")

    def _perplexity_transformers(
        self, context: str, chunk: str
    ) -> tuple[float, int, int]:
        import torch
        import torch.nn.functional as F

        device = self._resolve_device()

        context_ids: list[int] = self._tokenizer.encode(
            context, add_special_tokens=True
        )
        chunk_ids: list[int] = self._tokenizer.encode(
            chunk, add_special_tokens=False
        )

        if not chunk_ids:
            return 0.0, 0, len(context_ids)

        # Left-truncate context to leave room for the chunk
        max_ctx = self._config.max_context_tokens - len(chunk_ids) - 4
        if len(context_ids) > max_ctx:
            context_ids = context_ids[-max_ctx:]

        full_ids = context_ids + chunk_ids
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = self._model(input_ids)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # Position chunk_start-1 in logits predicts chunk_ids[0], etc.
        chunk_start = len(context_ids)
        n_chunk = len(chunk_ids)

        chunk_logits = logits[0, chunk_start - 1 : chunk_start - 1 + n_chunk, :]
        chunk_labels = torch.tensor(chunk_ids, dtype=torch.long, device=device)

        loss = F.cross_entropy(chunk_logits, chunk_labels)
        perplexity = math.exp(min(loss.item(), 20.0))  # cap at exp(20) to avoid inf

        return perplexity, n_chunk, len(context_ids)

    def _perplexity_llama_cpp(
        self, context: str, chunk: str
    ) -> tuple[float, int, int]:
        import numpy as _np

        prompt = context + chunk
        output = self._model(
            prompt,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )
        token_logprobs: list[Optional[float]] = (
            output["choices"][0]["logprobs"]["token_logprobs"]
        )
        # Only take log-probs for the chunk tokens (last len(chunk) encoded tokens)
        context_tokens = self._model.tokenize(context.encode())
        chunk_tokens = self._model.tokenize(chunk.encode(), add_bos=False)
        n_ctx = len(context_tokens)
        n_chunk = len(chunk_tokens)

        chunk_logprobs = [
            lp
            for lp in token_logprobs[n_ctx:]
            if lp is not None
        ]
        if not chunk_logprobs:
            return 0.0, 0, n_ctx

        avg_nll = -_np.mean(chunk_logprobs)
        perplexity = math.exp(min(avg_nll, 20.0))
        return perplexity, n_chunk, n_ctx

    def _perplexity_mlx(
        self, context: str, chunk: str
    ) -> tuple[float, int, int]:
        import mlx.core as mx
        import mlx.nn as nn

        context_ids = self._tokenizer.encode(context)
        chunk_ids = self._tokenizer.encode(chunk)

        if not chunk_ids:
            return 0.0, 0, len(context_ids)

        max_ctx = self._config.max_context_tokens - len(chunk_ids) - 4
        if len(context_ids) > max_ctx:
            context_ids = context_ids[-max_ctx:]

        full_ids = mx.array([context_ids + chunk_ids])
        logits = self._model(full_ids)

        chunk_start = len(context_ids)
        n_chunk = len(chunk_ids)
        chunk_logits = logits[0, chunk_start - 1 : chunk_start - 1 + n_chunk, :]
        chunk_labels = mx.array(chunk_ids)

        loss = nn.losses.cross_entropy(chunk_logits, chunk_labels)
        perplexity = math.exp(min(float(loss.mean()), 20.0))
        return perplexity, n_chunk, len(context_ids)

    # ------------------------------------------------------------------
    # Internal: network mode
    # ------------------------------------------------------------------

    async def _score_network(self, context: str, chunk: str) -> SurpriseScore:
        import aiohttp

        payload = {
            "context": context,
            "chunk": chunk,
            "max_context_tokens": self._config.max_context_tokens,
        }
        endpoint = self._config.network_endpoint.rstrip("/") + "/v1/bsc/score"
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                resp.raise_for_status()
                data = await resp.json()

        perplexity = float(data["raw_perplexity"])
        n_chunk = int(data["token_count"])
        n_ctx = int(data["context_tokens"])
        return self._build_score(perplexity, n_chunk, n_ctx)

    # ------------------------------------------------------------------
    # Internal: score normalisation and baseline adaptation
    # ------------------------------------------------------------------

    def _build_score(
        self, perplexity: float, n_chunk: int, n_ctx: int
    ) -> SurpriseScore:
        """
        Normalise raw perplexity to [0, 1] using an adaptive sigmoid relative
        to the running baseline, then update the baseline.
        """
        baseline = self._baseline

        # Sigmoid centred at baseline, scaled by baseline so the shape is
        # self-similar as baseline drifts (e.g. from English to code).
        if baseline > 0:
            z = (perplexity - baseline) / (baseline * 0.5)
        else:
            z = 0.0
        score = 1.0 / (1.0 + math.exp(-z))

        # Update adaptive baseline (EMA, biased toward lower values to avoid
        # drift on a single anomalous high-surprise burst).
        alpha = self._baseline_alpha
        self._baseline = (1 - alpha) * baseline + alpha * min(perplexity, baseline * 2)

        return SurpriseScore(
            score=float(np.clip(score, 0.0, 1.0)),
            raw_perplexity=perplexity,
            token_count=n_chunk,
            context_tokens=n_ctx,
            adaptive_baseline=baseline,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        if self._config.device:
            return self._config.device
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    @property
    def baseline_perplexity(self) -> float:
        """Current adaptive baseline perplexity (read-only)."""
        return self._baseline
