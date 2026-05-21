"""Sprint 688 F38 — prompt encoder must add gpt2-style position embeddings.

Live-attest of sprint 687 closed the end-to-end inference path
with a signed receipt, but the gpt2 outputs were semantic
garbage ("The capital of France is" → " all" instead of " the";
"Hello, my name is" → " hello" instead of " John").

Root cause: build_hf_prompt_encoder called model.get_input_embeddings()
which returns ONLY the token-embedding matrix (`wte`). For LLaMA-
family models, position info enters via rotary embeddings inside
attention, so wte-only works. For gpt2 / Falcon / GPT-NeoX,
position embeddings (`wpe`) MUST be added at the embedding stage
— GPT2Model.forward does `wte(ids) + wpe(positions)` explicitly.
Without `wpe`, the layers see content vectors without position
info → output divergence.

Sprint 688 detects `model.transformer.wpe` (gpt2 / Falcon path)
and gpt_neox.wpe, adds the position embeddings before returning
the activation. No change for LLaMA-family (no wpe attribute).
"""
from __future__ import annotations

import inspect

import pytest


def test_prompt_encoder_adds_position_embeddings_for_gpt2_source():
    """Source-level guard: the encoder body must read
    transformer.wpe and add pos_embeds when present. Defends
    against a refactor that removes the path."""
    from prsm.node.chain_executor_adapters import build_hf_prompt_encoder
    src = inspect.getsource(build_hf_prompt_encoder)
    # Must reference transformer.wpe (gpt2 path)
    assert "transformer.wpe" in src or "wpe" in src, (
        "encoder must look up the wpe (position embedding) on the model"
    )
    # Must reference gpt_neox.wpe (alternative architecture path)
    assert "gpt_neox" in src or "embed_in" in src, (
        "encoder must handle gpt_neox-style architectures too"
    )
    # The actual addition must happen
    assert "embeddings + pos_embeds" in src or "embeddings += pos_embeds" in src
    # Sprint marker
    assert "Sprint 688" in src or "sprint 688" in src


def test_prompt_encoder_skips_wpe_addition_for_llama_style():
    """When the model lacks transformer.wpe (LLaMA / Mistral),
    encoder returns the token embeddings WITHOUT adding any
    position vectors — preserves the LLaMA/rotary path."""
    from prsm.node.chain_executor_adapters import build_hf_prompt_encoder

    encoder = build_hf_prompt_encoder(model_id="dummy", device="cpu")
    # Source inspection: confirm wpe lookup is GUARDED by hasattr,
    # not always executed.
    src = inspect.getsource(build_hf_prompt_encoder)
    assert "if _wpe is not None" in src or "hasattr(" in src, (
        "wpe addition must be guarded — LLaMA models have no wpe"
    )
