# Hacker News — Show HN Post

## Title
Show HN: PRSM – P2P decentralized compute network for AI/scientific research (alpha)

## Body

We built PRSM (Protocol for Recursive Scientific Modeling) because we wanted a way to run collaborative AI inference and embedding jobs across untrusted peer nodes without routing everything through a centralized API. Today we're opening it up.

The core is a P2P node network where contributors share compute and storage, get paid in FTNS tokens, and spend those tokens to submit jobs. Orchestration runs through NWTN (Neural Web for Transformation Networking), a 5-layer agent pipeline that handles task decomposition, routing, execution, and result validation. Storage is IPFS-based with semantic provenance tracking so you can trace outputs back to their source models and datasets. The safety layer includes circuit breakers, Ed25519 signatures, and post-quantum crypto for message integrity.

Getting started takes about 30 seconds:

```bash
pip install prsm-network
prsm node start
# Auto-connects to wss://bootstrap1.prsm-network.com:8765
# You receive a 100 FTNS welcome grant
# REST API live at localhost:8000
```

Real talk on state: this is v0.2.1 alpha. Single-node and multi-node compute both work. Real AI inference runs via Anthropic and OpenAI backends. We have 1,391+ passing tests. The FTNS token is deployed on Ethereum Sepolia testnet (`0xd979c096BE297F4C3a85175774Bc38C22b95E6a4`). This is not production software — don't run it on critical workloads, and the token economics are still being tuned. What we want right now is node operators, early integrators, and people who will break things and tell us about it.

GitHub: https://github.com/Ryno2390/PRSM

Happy to answer questions about the NWTN pipeline architecture, the token model, or the neuro-symbolic approach we're taking to research task decomposition.
