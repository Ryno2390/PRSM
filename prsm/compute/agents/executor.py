"""
AgentExecutor — Provider-Side WASM Agent Execution
===================================================

Receives mobile WASM agents, validates them against this node's
capabilities, executes them in a Ring 1 sandbox (WasmtimeRuntime),
and signs the results for verifiability.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from typing import Any, Dict, Tuple

from prsm.compute.agents.models import AgentManifest, MobileAgent
from prsm.compute.wasm.models import ResourceLimits
from prsm.compute.wasm.runtime import WasmtimeRuntime

logger = logging.getLogger(__name__)

# Hardware tier ordering — higher number = more capable
TIER_ORDER: Dict[str, int] = {"t1": 1, "t2": 2, "t3": 3, "t4": 4}


class AgentExecutor:
    """Executes WASM mobile agents in a sandboxed runtime and signs results.

    Parameters
    ----------
    identity:
        NodeIdentity-like object with ``node_id``, ``sign()``, ``public_key_b64``.
    gossip:
        GossipProtocol instance for publishing results.
    hardware_tier:
        This node's hardware tier (t1–t4).
    """

    def __init__(self, identity: Any, gossip: Any, hardware_tier: str = "t1") -> None:
        self.identity = identity
        self.gossip = gossip
        self.hardware_tier = hardware_tier
        self._runtime = WasmtimeRuntime()

    # ── Validation ───────────────────────────────────────────────────

    def validate_manifest(self, manifest: AgentManifest) -> Tuple[bool, str]:
        """Check whether this node meets the manifest's hardware requirements.

        Returns ``(True, "ok")`` on success, or ``(False, reason)`` on failure.
        """
        required = TIER_ORDER.get(manifest.min_hardware_tier, 0)
        available = TIER_ORDER.get(self.hardware_tier, 0)

        if available < required:
            return (
                False,
                f"Insufficient tier: have {self.hardware_tier} "
                f"(level {available}), need {manifest.min_hardware_tier} "
                f"(level {required})",
            )
        return (True, "ok")

    # ── Execution ────────────────────────────────────────────────────

    async def execute_agent(
        self,
        agent: MobileAgent,
        input_data: bytes,
        publish_result: bool = False,
    ) -> Dict[str, Any]:
        """Execute a mobile agent's WASM binary in the sandbox.

        Returns a result dict with status, output, metrics, and signature.
        Never raises — all exceptions are caught and returned as error results.
        """
        try:
            return await self._execute_inner(agent, input_data, publish_result)
        except Exception as exc:
            logger.exception("Unexpected error executing agent %s", agent.agent_id)
            return self._error_result(agent.agent_id, str(exc))

    async def _execute_inner(
        self,
        agent: MobileAgent,
        input_data: bytes,
        publish_result: bool,
    ) -> Dict[str, Any]:
        # 1. Check expiry
        if agent.is_expired():
            return self._error_result(
                agent.agent_id,
                f"Agent {agent.agent_id} has expired (TTL elapsed)",
            )

        # 2. Validate hardware tier
        ok, reason = self.validate_manifest(agent.manifest)
        if not ok:
            return self._error_result(agent.agent_id, reason)

        # 3. Load and execute in sandbox
        resource_limits = ResourceLimits(
            max_memory_bytes=agent.manifest.max_memory_bytes,
            max_execution_seconds=agent.manifest.max_execution_seconds,
            max_output_bytes=agent.manifest.max_output_bytes,
        )

        module = self._runtime.load(agent.wasm_binary)
        exec_result = self._runtime.execute(module, input_data, resource_limits)

        # 4. Build result dict
        output_b64 = base64.b64encode(exec_result.output).decode()

        result: Dict[str, Any] = {
            "agent_id": agent.agent_id,
            "provider_id": self.identity.node_id,
            "status": exec_result.status.value,
            "output_b64": output_b64,
            "execution_time_seconds": exec_result.execution_time_seconds,
            "memory_used_bytes": exec_result.memory_used_bytes,
            "pcu": exec_result.pcu(),
            "error": exec_result.error,
            "provider_signature": "",
            "provider_public_key": self.identity.public_key_b64,
        }

        # 5. Sign result hash
        result_hash = hashlib.sha256(
            json.dumps(
                {k: v for k, v in result.items() if k != "provider_signature"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        result["provider_signature"] = self.identity.sign(result_hash.encode() if isinstance(result_hash, str) else result_hash)

        # 6. Optionally publish via gossip
        if publish_result:
            await self.gossip.publish("agent_result", result)

        return result

    # ── Helpers ──────────────────────────────────────────────────────

    def _error_result(self, agent_id: str, error: str) -> Dict[str, Any]:
        """Build an error result dict."""
        result: Dict[str, Any] = {
            "agent_id": agent_id,
            "provider_id": self.identity.node_id,
            "status": "error",
            "output_b64": "",
            "execution_time_seconds": 0.0,
            "memory_used_bytes": 0,
            "pcu": 0.0,
            "error": error,
            "provider_signature": "",
            "provider_public_key": self.identity.public_key_b64,
        }

        result_hash = hashlib.sha256(
            json.dumps(
                {k: v for k, v in result.items() if k != "provider_signature"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        result["provider_signature"] = self.identity.sign(result_hash.encode() if isinstance(result_hash, str) else result_hash)

        return result
