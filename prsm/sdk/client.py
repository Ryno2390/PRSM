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
