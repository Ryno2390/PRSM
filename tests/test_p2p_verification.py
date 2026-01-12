import pytest
import asyncio
import hashlib
from uuid import uuid4
from prsm.compute.network.distributed_rlt_network import DistributedRLTNetwork, NetworkMessage, MessageType

@pytest.mark.asyncio
async def test_p2p_verifiable_request():
    """Verify that a P2P request-response cycle includes successful verification"""
    
    # Initialize two nodes
    node_a = DistributedRLTNetwork(node_id="node_a")
    node_b = DistributedRLTNetwork(node_id="node_b")
    
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
    
    # We need to monkey-patch _send_message to direct messages between local instances
    async def mock_send(self, message):
        if message.recipient_id == "node_b":
            await node_b._process_message(message)
        elif message.recipient_id == "node_a":
            await node_a._process_message(message)
            
    node_a._send_message = mock_send.__get__(node_a, DistributedRLTNetwork)
    node_b._send_message = mock_send.__get__(node_b, DistributedRLTNetwork)
    
    response = await node_a.request_teacher("physics_expert", task_context)
    
    assert response is not None
    assert response["verified"] is True
    assert "verification_hash" in response["response"]
    assert "trace" in response["response"]

@pytest.mark.asyncio
async def test_p2p_malicious_detection():
    """Verify that a P2P request fails verification if the response is tampered with"""
    
    node_a = DistributedRLTNetwork(node_id="node_a")
    node_b = DistributedRLTNetwork(node_id="node_b")
    
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
        mal_query = "Cheat" # Salted roll for "Cheat" + seed 42 is 0.3584 -> 'light'
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
                            "mode": "deep", # LIE: validator will expect 'light'
                            "seed": 42
                        }
                }
            }
        )
        await self._send_message(response_message)
        
    node_b._handle_teacher_request = malicious_handle_request.__get__(node_b, DistributedRLTNetwork)

    async def mock_send(self, message):
        if message.recipient_id == "node_b":
            await node_b._process_message(message)
        elif message.recipient_id == "node_a":
            await node_a._process_message(message)
            
    node_a._send_message = mock_send.__get__(node_a, DistributedRLTNetwork)
    node_b._send_message = mock_send.__get__(node_b, DistributedRLTNetwork)
    
    response = await node_a.request_teacher("malicious_expert", {"query": "Cheat"})
    
    assert response is not None
    assert response["verified"] is False