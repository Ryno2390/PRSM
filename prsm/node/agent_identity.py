"""
Agent Identity
==============

Extends the node identity system with delegation certificates for AI agents.

A human (node principal) can issue delegation certificates to agent keypairs,
allowing agents to act on the principal's behalf within defined bounds
(capabilities, spending caps, etc.).  Any node can verify a delegation
certificate without contacting the principal.

The delegation certificate is a JSON object signed by the principal's
private key, binding the agent's public key to a set of permissions.
"""

import base64
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from prsm.node.identity import NodeIdentity, verify_signature

logger = logging.getLogger(__name__)


@dataclass
class AgentIdentity:
    """A delegated AI agent identity on the PRSM network."""
    agent_id: str               # hex(sha256(agent_public_key))[:32]
    agent_name: str             # Human-readable name (e.g. "prsm-coder")
    agent_type: str             # "coding", "research", "devops", etc.
    principal_id: str           # Node ID of the human who controls this agent
    principal_public_key: str   # Base64 public key of the principal (for verification)
    public_key_b64: str         # Agent's own Ed25519 public key (base64)
    private_key_b64: str        # Agent's own Ed25519 private key (base64, kept local)
    delegation_cert: str        # Signature from principal proving delegation
    capabilities: List[str] = field(default_factory=list)
    max_spend_ftns: float = 10.0  # Spending cap per epoch
    created_at: float = field(default_factory=time.time)

    def sign(self, data: bytes) -> str:
        """Sign data with this agent's private key."""
        priv_bytes = base64.b64decode(self.private_key_b64)
        priv_key = Ed25519PrivateKey.from_private_bytes(priv_bytes)
        signature = priv_key.sign(data)
        return base64.b64encode(signature).decode()

    def to_public_dict(self) -> Dict[str, Any]:
        """Serialize for network sharing (no private key)."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "principal_id": self.principal_id,
            "principal_public_key": self.principal_public_key,
            "public_key_b64": self.public_key_b64,
            "delegation_cert": self.delegation_cert,
            "capabilities": self.capabilities,
            "max_spend_ftns": self.max_spend_ftns,
            "created_at": self.created_at,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Full serialization including private key (for local persistence only)."""
        d = self.to_public_dict()
        d["private_key_b64"] = self.private_key_b64
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentIdentity":
        """Deserialize from a dict."""
        return cls(
            agent_id=data["agent_id"],
            agent_name=data["agent_name"],
            agent_type=data.get("agent_type", "general"),
            principal_id=data["principal_id"],
            principal_public_key=data["principal_public_key"],
            public_key_b64=data["public_key_b64"],
            private_key_b64=data.get("private_key_b64", ""),
            delegation_cert=data["delegation_cert"],
            capabilities=data.get("capabilities", []),
            max_spend_ftns=data.get("max_spend_ftns", 10.0),
            created_at=data.get("created_at", time.time()),
        )


def create_agent_identity(
    principal: NodeIdentity,
    agent_name: str,
    agent_type: str = "general",
    capabilities: Optional[List[str]] = None,
    max_spend_ftns: float = 10.0,
) -> AgentIdentity:
    """Create a new agent identity delegated by a node principal.

    Generates a fresh Ed25519 keypair for the agent and signs a delegation
    certificate with the principal's private key.

    Args:
        principal: The human node's identity (must have private key)
        agent_name: Human-readable name for the agent
        agent_type: Category (coding, research, devops, etc.)
        capabilities: List of declared capabilities
        max_spend_ftns: Maximum FTNS the agent can spend per epoch

    Returns:
        A fully initialized AgentIdentity with delegation certificate
    """
    # Generate agent keypair
    agent_priv_key = Ed25519PrivateKey.generate()
    agent_priv_bytes = agent_priv_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    agent_pub_bytes = agent_priv_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    agent_id = hashlib.sha256(agent_pub_bytes).hexdigest()[:32]
    agent_pub_b64 = base64.b64encode(agent_pub_bytes).decode()
    agent_priv_b64 = base64.b64encode(agent_priv_bytes).decode()

    caps = capabilities or []
    created_at = time.time()

    # Build delegation certificate payload
    cert_payload = {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "agent_type": agent_type,
        "agent_public_key": agent_pub_b64,
        "principal_id": principal.node_id,
        "capabilities": caps,
        "max_spend_ftns": max_spend_ftns,
        "issued_at": created_at,
    }
    cert_bytes = json.dumps(cert_payload, sort_keys=True).encode()
    delegation_cert = principal.sign(cert_bytes)

    agent = AgentIdentity(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_type=agent_type,
        principal_id=principal.node_id,
        principal_public_key=principal.public_key_b64,
        public_key_b64=agent_pub_b64,
        private_key_b64=agent_priv_b64,
        delegation_cert=delegation_cert,
        capabilities=caps,
        max_spend_ftns=max_spend_ftns,
        created_at=created_at,
    )

    logger.info(f"Created agent {agent_name} ({agent_id[:12]}...) delegated by {principal.node_id[:12]}...")
    return agent


def verify_delegation(agent_data: Dict[str, Any]) -> bool:
    """Verify that an agent's delegation certificate is valid.

    Checks that the delegation_cert was signed by the claimed principal's
    public key, proving the agent was authorized by that principal.

    Args:
        agent_data: Public agent identity dict (from to_public_dict or gossip)

    Returns:
        True if the delegation certificate is valid
    """
    principal_pub_key = agent_data.get("principal_public_key", "")
    delegation_cert = agent_data.get("delegation_cert", "")

    if not principal_pub_key or not delegation_cert:
        return False

    # Reconstruct the canonical cert payload
    cert_payload = {
        "agent_id": agent_data.get("agent_id", ""),
        "agent_name": agent_data.get("agent_name", ""),
        "agent_type": agent_data.get("agent_type", ""),
        "agent_public_key": agent_data.get("public_key_b64", ""),
        "principal_id": agent_data.get("principal_id", ""),
        "capabilities": agent_data.get("capabilities", []),
        "max_spend_ftns": agent_data.get("max_spend_ftns", 10.0),
        "issued_at": agent_data.get("created_at", 0),
    }
    cert_bytes = json.dumps(cert_payload, sort_keys=True).encode()

    return verify_signature(principal_pub_key, cert_bytes, delegation_cert)
