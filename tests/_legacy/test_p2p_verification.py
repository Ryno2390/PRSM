import pytest
import asyncio
import hashlib
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
from prsm.compute.network.distributed_rlt_network import DistributedRLTNetwork, NetworkMessage, MessageType


def _make_mock_orchestrator(seed=42):
    """Create a mock orchestrator that returns deterministic verifiable results.

    Uses the real NeuroSymbolicOrchestrator for verification but mocks solve_task
    to avoid external dependencies (API calls, model loading, etc.).
    """
    from prsm.compute.nwtn.reasoning.s1_neuro_symbolic import NeuroSymbolicOrchestrator
    from prsm.core.utils.deterministic import get_local_generator, generate_verification_hash

    orchestrator = NeuroSymbolicOrchestrator(node_id="mock_node", seed=seed)

    # Override solve_task to avoid real model inference
    async def mock_solve_task(query, context=""):
        input_hash = hashlib.sha256(query.encode()).hexdigest()
        output = f"Mock response for: {query}"

        # Use the same deterministic mode selection as the real orchestrator
        task_salt = int(hashlib.sha256(query.encode()).hexdigest(), 16) % 10**8
        decision_rng = get_local_generator(seed + task_salt)
        mode = "light"
        roll = decision_rng.next_float()
        if roll > 0.7:
            mode = "deep"

        v_hash = generate_verification_hash(output, "nwtn_v1", input_hash)
        return {
            "output": output,
            "input_hash": input_hash,
            "verification_hash": v_hash,
            "trace": [{"step": "mock", "result": output}],
            "reward": 1.0,
            "mode": mode,
            "metadata": {"query": query, "mode": mode, "seed": seed},
        }

    orchestrator.solve_task = mock_solve_task
    return orchestrator


@pytest.mark.asyncio
async def test_p2p_verifiable_request():
    """Verify that a P2P request-response cycle includes successful verification"""

    # Initialize two nodes
    node_a = DistributedRLTNetwork(node_id="node_a")
    node_b = DistributedRLTNetwork(node_id="node_b")

    # Replace orchestrators with mocks to avoid external dependencies
    node_a.orchestrator = _make_mock_orchestrator()
    node_b.orchestrator = _make_mock_orchestrator()

    await node_a.initialize_network()
    await node_b.initialize_network()

    # Register each other (simplified for test)
    node_a.nodes["node_b"] = node_b.local_node
    node_b.nodes["node_a"] = node_a.local_node

    # Node B registers a teacher
    await node_b.register_teacher("physics_expert")
    node_a.teacher_registry["physics_expert"] = ["node_b"]

    # Node A requests from Node B
    task_context = {"query": "What is refraction?", "context": "Optics"}

    # Monkey-patch _send_message to direct messages between local instances
    async def mock_send(self, message):
        if message.recipient_id == "node_b":
            await node_b._process_message(message)
        elif message.recipient_id == "node_a":
            await node_a._process_message(message)

    node_a._send_message = mock_send.__get__(node_a, DistributedRLTNetwork)
    node_b._send_message = mock_send.__get__(node_b, DistributedRLTNetwork)

    response = await asyncio.wait_for(
        node_a.request_teacher("physics_expert", task_context),
        timeout=10.0
    )

    assert response is not None
    assert response["verified"] is True
    assert "verification_hash" in response["response"]
    assert "trace" in response["response"]


@pytest.mark.asyncio
async def test_p2p_malicious_detection():
    """Verify that a P2P request fails verification if the response is tampered with"""

    node_a = DistributedRLTNetwork(node_id="node_a")
    node_b = DistributedRLTNetwork(node_id="node_b")

    # Replace orchestrators with mocks
    node_a.orchestrator = _make_mock_orchestrator()
    node_b.orchestrator = _make_mock_orchestrator()

    await node_a.initialize_network()
    await node_b.initialize_network()

    node_a.nodes["node_b"] = node_b.local_node
    node_b.nodes["node_a"] = node_a.local_node

    # Node B registers a teacher
    await node_b.register_teacher("malicious_expert")
    node_a.teacher_registry["malicious_expert"] = ["node_b"]

    # Node B is malicious and returns garbage
    async def malicious_handle_request(self, message):
        malicious_output = "I am a malicious node"
        mal_query = "Cheat"
        mal_input_hash = hashlib.sha256(mal_query.encode()).hexdigest()
        from prsm.core.utils.deterministic import generate_verification_hash
        v_hash = generate_verification_hash(malicious_output, "nwtn_v1", mal_input_hash)

        response_message = NetworkMessage(
            message_id=str(uuid4()),
            message_type=MessageType.TEACHER_RESPONSE,
            sender_id=self.node_id,
            recipient_id=message.sender_id,
            payload={
                "request_id": message.payload.get("request_id"),
                "teacher_type": "malicious_expert",
                "response": {
                    "output": malicious_output,
                    "input_hash": mal_input_hash,
                    "verification_hash": v_hash,
                    "trace": [],
                    "reward": 1.0,
                    "metadata": {
                        "query": mal_query,
                        "mode": "deep",  # LIE: validator will expect 'light'
                        "seed": 42
                    }
                }
            }
        )
        await self._send_message(response_message)

    # Must update the message_handlers dict directly, not the instance method,
    # because _setup_message_handlers stores bound method references at init time.
    node_b.message_handlers[MessageType.TEACHER_REQUEST] = malicious_handle_request.__get__(node_b, DistributedRLTNetwork)

    async def mock_send(self, message):
        if message.recipient_id == "node_b":
            await node_b._process_message(message)
        elif message.recipient_id == "node_a":
            await node_a._process_message(message)

    node_a._send_message = mock_send.__get__(node_a, DistributedRLTNetwork)
    node_b._send_message = mock_send.__get__(node_b, DistributedRLTNetwork)

    response = await asyncio.wait_for(
        node_a.request_teacher("malicious_expert", {"query": "Cheat"}),
        timeout=10.0
    )

    assert response is not None
    assert response["verified"] is False