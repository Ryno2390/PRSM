# Blog Post — Launch Announcement

## We Built a Peer-to-Peer AI Compute Network for Scientific Research. Here's Where We Are.

The way most researchers access AI compute today is structurally broken for science. You pay a centralized provider, your data and outputs pass through infrastructure you don't control, provenance is opaque, and costs scale in ways that make large collaborative projects economically unworkable. When the outputs of an AI-assisted research pipeline can't be reliably traced back to the models and data that produced them, reproducibility — the bedrock of science — is at risk.

We built PRSM to take a different path.

---

### What PRSM Is

PRSM stands for Protocol for Recursive Scientific Modeling. It's a peer-to-peer framework where nodes contribute compute and storage, earn tokens for doing so, and spend those tokens to submit AI inference and embedding jobs. The goal is a self-sustaining network where the people doing the work are also the people running the infrastructure.

The orchestration layer is called NWTN — Neural Web for Transformation Networking. It's a 5-layer agent pipeline that handles everything from breaking down a complex research task into subtasks, routing those subtasks to appropriate nodes, executing inference against real AI backends (Anthropic and OpenAI today, with the architecture designed to be backend-agnostic), validating results, and returning composed outputs. The approach is neuro-symbolic: NWTN reasons over structured representations of tasks and model capabilities, not just raw prompts.

Storage is handled through IPFS with semantic provenance tracking built in. Every stored artifact carries metadata about what model produced it, what data it was derived from, and who contributed compute. This isn't cosmetic — it feeds directly into the royalty distribution logic, so contributors get credited and compensated when their stored artifacts are referenced downstream.

---

### What Actually Works Today

We shipped v0.2.1 alpha this week. Here's what's real:

`pip install prsm-network` works. `prsm node start` connects your node to the bootstrap server at `wss://bootstrap1.prsm-network.com:8765`, and you receive a 100 FTNS token welcome grant. The REST API comes up at localhost:8000. You can submit compute jobs against live AI backends and get real results back. Single-node and multi-node compute are both functional. We have 1,391+ passing tests.

The FTNS token is deployed on Ethereum Sepolia testnet at `0xd979c096BE297F4C3a85175774Bc38C22b95E6a4`. The on-chain token economy — earning for contributing, spending to submit jobs — is implemented and testable today. The safety infrastructure includes circuit breakers, emergency halt mechanisms, rule-based monitoring, Ed25519 signatures, and post-quantum cryptography for message integrity.

This is alpha software. We're being direct about that. The token economics are still being calibrated, the network has a small number of nodes, and there are rough edges. We're not asking you to trust PRSM with production workloads. We're asking you to run a node, poke at it, and tell us what breaks.

---

### What's Coming

The immediate roadmap focuses on three things: expanding the node network so there's meaningful redundancy and load distribution, refining the FTNS economic model based on real usage data from the testnet, and deepening the neuro-symbolic reasoning layer in NWTN so it can handle more complex multi-step research tasks with less hand-holding.

Longer term, we're working toward a mainnet FTNS deployment, a model distillation pipeline that lets nodes contribute training as well as inference, and deeper integrations with scientific data repositories so that provenance tracking spans the full research lifecycle from raw data to published output.

---

### Get Involved

If you're a researcher who's frustrated with the cost and opacity of current AI compute access, a developer who wants to work on distributed systems problems that actually matter, or a node operator looking to contribute to something early — we want to hear from you.

```bash
pip install prsm-network
prsm node start
```

GitHub: https://github.com/Ryno2390/PRSM

The network gets more useful with every node that joins. Come help us build it.
