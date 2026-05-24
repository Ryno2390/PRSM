"""
PRSM Node Client
=================

Async Python client for PRSM node Ring 1-10 APIs.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PRSMClient:
    """Client for interacting with a PRSM node's API.

    Usage:
        client = PRSMClient("http://localhost:8000")

        # Full forge pipeline
        result = await client.query("EV adoption trends in NC", budget=10.0)

        # Get cost quote first
        quote = await client.quote("EV trends", shards=5, tier="t2")

        # Check node hardware
        status = await client.status()
    """

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session = None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _ensure_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def _get(self, path: str) -> Dict[str, Any]:
        await self._ensure_session()
        async with self._session.get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            timeout=__import__("aiohttp").ClientTimeout(total=30),
        ) as resp:
            return await resp.json()

    async def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_session()
        async with self._session.post(
            f"{self.base_url}{path}",
            json=data,
            headers=self._headers(),
            timeout=__import__("aiohttp").ClientTimeout(total=120),
        ) as resp:
            return await resp.json()

    # ── Core Endpoints ────────────────────────────────────────────

    async def status(self) -> Dict[str, Any]:
        """Get node status."""
        return await self._get("/status")

    async def peers(self) -> Dict[str, Any]:
        """Get connected peers."""
        return await self._get("/peers")

    # ── Ring 5: Forge Pipeline ────────────────────────────────────

    async def query(
        self,
        query: str,
        budget: float = 10.0,
        privacy: str = "standard",
        shard_cids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Submit a query through the full Ring 1-10 forge pipeline.

        Args:
            query: Natural language query.
            budget: Maximum FTNS to spend.
            privacy: Privacy level (none, standard, high, maximum).
            shard_cids: Optional specific shard CIDs to target.

        Returns:
            Dict with route, response, result, traces_collected.
        """
        payload = {
            "query": query,
            "budget_ftns": budget,
            "privacy_level": privacy,
        }
        if shard_cids:
            payload["shard_cids"] = shard_cids
        return await self._post("/compute/forge", payload)

    async def prompt(self, prompt: str, budget: float = 0.0) -> Dict[str, Any]:
        """Submit a prompt via legacy NWTN path."""
        return await self._post("/compute/query", {"prompt": prompt, "budget": budget})

    # ── Sprint 819 — Verifiable Inference ─────────────────────────

    async def infer(
        self,
        prompt: str,
        *,
        model_id: str = "gpt2",
        max_tokens: int = 8,
        budget_ftns: float = 1.0,
        privacy_tier: str = "none",
        content_tier: str = "A",
        verify_pubkey_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sprint 819 — POST /compute/inference for verifiable
        inference + signed receipt.

        Returns the parsed server payload:
          {success, output, ftns_charged, receipt, ...}

        When ``verify_pubkey_b64`` is set, the returned dict
        gains a ``receipt_verified`` boolean computed via
        sprint-706 verify_receipt against the supplied pubkey.
        Useful when a caller pins an operator's published pubkey
        and wants the verify check inline with the inference
        request (avoids a separate verify-receipt round-trip).

        Defaults match the `prsm compute infer` CLI (sprint 802)
        so users moving between CLI + SDK see consistent
        behavior.
        """
        body = {
            "prompt": prompt,
            "model_id": model_id,
            "budget_ftns": budget_ftns,
            "privacy_tier": privacy_tier,
            "content_tier": content_tier,
            "max_tokens": max_tokens,
        }
        result = await self._post("/compute/inference", body)
        if verify_pubkey_b64:
            try:
                from prsm.compute.inference.models import (
                    InferenceReceipt,
                )
                from prsm.compute.inference.receipt import (
                    verify_receipt,
                )
                receipt = InferenceReceipt.from_dict(
                    result.get("receipt") or {},
                )
                result["receipt_verified"] = bool(
                    verify_receipt(
                        receipt, public_key_b64=verify_pubkey_b64,
                    ),
                )
            except Exception:
                result["receipt_verified"] = False
        return result

    # ── Sprint 821 — Content publish + fetch ────────────────────

    async def publish_content(
        self,
        text: str,
        *,
        filename: str = "document.txt",
        replicas: int = 3,
        royalty_rate: Optional[float] = None,
        parent_cids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Sprint 821 — POST /content/upload. Returns server
        payload with the assigned CID + filename + size.

        Mirror of `prsm content publish` CLI (sprint 806).
        Defaults match the CLI + server-side defaults.

        For binary content or text > 100MB, use the shard
        endpoint instead (no SDK wrapper yet; see sprint 817's
        CLI for the canonical body shape).
        """
        body: Dict[str, Any] = {
            "text": text,
            "filename": filename,
            "replicas": replicas,
            "parent_cids": list(parent_cids) if parent_cids else [],
        }
        if royalty_rate is not None:
            body["royalty_rate"] = royalty_rate
        return await self._post("/content/upload", body)

    async def search_content(
        self,
        query: str,
        *,
        limit: int = 20,
        min_tier: Optional[str] = None,
        exclude_new: bool = False,
    ) -> Dict[str, Any]:
        """Sprint 822 — GET /content/search; return server payload.

        Mirror of `prsm content search` CLI (sprint 808).
        Defaults match the CLI.

        Returns ``{results, count}``. Each result row carries
        ``cid``, ``filename``, ``creator_tier`` (server-side
        sprint 289).
        """
        await self._ensure_session()
        import aiohttp as _aiohttp

        params: Dict[str, Any] = {"q": query, "limit": limit}
        if min_tier:
            params["min_tier"] = min_tier
        if exclude_new:
            params["exclude_new"] = "true"

        async with self._session.get(
            f"{self.base_url}/content/search",
            headers=self._headers(),
            timeout=_aiohttp.ClientTimeout(total=15.0),
            params=params,
        ) as resp:
            return await resp.json()

    async def fetch_content(
        self,
        cid: str,
        *,
        timeout: float = 30.0,
        verify_hash: bool = True,
    ) -> Dict[str, Any]:
        """Sprint 821 — GET /content/retrieve/{cid}. Returns
        the server payload with `data` as base64-encoded bytes
        (caller decodes for binary content).

        Mirror of `prsm content fetch` CLI (sprint 805).

        Threads ``timeout`` + ``verify_hash`` as query params;
        the server-side timeout cap (PRSM_MAX_RETRIEVE_TIMEOUT_SEC,
        default 300) applies.
        """
        await self._ensure_session()
        import aiohttp as _aiohttp

        params: Dict[str, Any] = {"timeout": timeout}
        # The fastapi server reads bool from "false" / "true"
        # lowercased strings — serialize accordingly.
        params["verify_hash"] = "true" if verify_hash else "false"

        async with self._session.get(
            f"{self.base_url}/content/retrieve/{cid}",
            headers=self._headers(),
            timeout=_aiohttp.ClientTimeout(total=timeout + 5.0),
            params=params,
        ) as resp:
            return await resp.json()

    # ── Sprint 820 — Streaming verifiable inference ─────────────

    async def infer_stream(
        self,
        prompt: str,
        *,
        model_id: str = "gpt2",
        max_tokens: int = 8,
        budget_ftns: float = 1.0,
        privacy_tier: str = "none",
        content_tier: str = "A",
    ):
        """Sprint 820 — async generator consuming SSE from
        /compute/inference/stream.

        Yields events as dicts with a `type` discriminator:
          {"type": "token", "sequence_index": N, "text_delta",
           "token_id", "finish_reason"}
          {"type": "result", "success", "output",
           "ftns_charged", "receipt", ...}
          {"type": "error", "detail"}

        Iteration terminates after the first `result` or
        `error` event — extra frames after a terminal event
        are ignored (defense against misbehaving servers).

        Defaults match sprint 819 .infer() so users can swap
        between unary and streaming with one keyword.

        Usage:
            async for ev in client.infer_stream(prompt="..."):
                if ev["type"] == "token":
                    print(ev["text_delta"], end="", flush=True)
                elif ev["type"] == "result":
                    print()
                    print("Receipt:", ev["receipt"])
        """
        await self._ensure_session()
        body = {
            "prompt": prompt,
            "model_id": model_id,
            "budget_ftns": budget_ftns,
            "privacy_tier": privacy_tier,
            "content_tier": content_tier,
            "max_tokens": max_tokens,
        }
        import aiohttp as _aiohttp

        async with self._session.post(
            f"{self.base_url}/compute/inference/stream",
            json=body,
            headers=self._headers(),
            timeout=_aiohttp.ClientTimeout(total=300),
        ) as resp:
            # Parse SSE event/data frames from line-streamed bytes.
            current_event = None
            buffer = b""
            terminal_seen = False
            async for chunk in resp.content:
                buffer += chunk
                # Split on \n to extract complete lines; keep
                # the (possibly partial) trailing fragment.
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_text = line.decode("utf-8", "replace").rstrip("\r")
                    if not line_text:
                        # blank line = end of an event
                        current_event = None
                        continue
                    if line_text.startswith("event: "):
                        current_event = line_text[len("event: "):].strip()
                    elif line_text.startswith("data: "):
                        if current_event is None:
                            continue
                        try:
                            import json as _json
                            payload = _json.loads(
                                line_text[len("data: "):],
                            )
                        except Exception:
                            continue
                        ev = {"type": current_event, **payload}
                        yield ev
                        if current_event in ("result", "error"):
                            terminal_seen = True
                            return

    # ── Ring 4: Pricing ───────────────────────────────────────────

    async def quote(
        self,
        query: str,
        shards: int = 3,
        tier: str = "t2",
    ) -> Dict[str, Any]:
        """Get a cost estimate for a query.

        Note: This is a client-side estimate using the pricing engine.
        For server-side quotes, use query() with a small budget.
        """
        # Client-side pricing estimate
        from prsm.economy.pricing import PricingEngine
        engine = PricingEngine()
        q = engine.quote_swarm_job(
            shard_count=shards,
            hardware_tier=tier,
            estimated_pcu_per_shard=50.0,
        )
        return q.to_dict()

    # ── Ring 3: Data Upload ───────────────────────────────────────

    async def upload_dataset(
        self,
        dataset_id: str,
        title: str,
        content: bytes,
        shard_count: int = 4,
        base_access_fee: float = 0.0,
        per_shard_fee: float = 0.0,
    ) -> Dict[str, Any]:
        """Upload a dataset with semantic sharding."""
        import base64
        return await self._post("/content/upload/shard", {
            "dataset_id": dataset_id,
            "title": title,
            "content_b64": base64.b64encode(content).decode(),
            "shard_count": shard_count,
            "base_access_fee": base_access_fee,
            "per_shard_fee": per_shard_fee,
        })

    # ── Settlement ────────────────────────────────────────────────

    async def settlement_stats(self) -> Dict[str, Any]:
        """Get FTNS settlement queue stats."""
        return await self._get("/settlement/stats")
