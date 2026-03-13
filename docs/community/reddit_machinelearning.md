# Reddit — r/MachineLearning Post

## Title
We built a decentralized P2P inference network with a 5-layer neuro-symbolic orchestration pipeline — v0.2.1 alpha, looking for early testers [project]

## Body

Hi r/MachineLearning — posting to share PRSM (Protocol for Recursive Scientific Modeling), a P2P compute network we've been building for collaborative AI/scientific research workloads. Flagging upfront that this is alpha software — functional, tested, but not production.

**The architecture I think is most interesting here:**

The orchestration layer is NWTN (Neural Web for Transformation Networking), a 5-layer agent pipeline. From the outside in: a task decomposition layer that breaks research queries into structured subtasks, a routing layer that matches subtasks to nodes based on model capabilities and availability, an execution layer that runs actual inference against real backends (Anthropic and OpenAI currently), a validation layer that checks result coherence before composition, and a synthesis layer that assembles final outputs with provenance metadata attached.

The approach is neuro-symbolic in the sense that NWTN operates over structured representations of tasks, model capabilities, and data relationships — not just prompt chaining. This matters for scientific tasks specifically because you often need to compose heterogeneous operations (embedding, retrieval, generation, structured reasoning) in ways that pure LLM pipelines handle badly. The pipeline is designed to make those composition steps explicit and auditable rather than implicit.

The storage layer uses IPFS with semantic provenance tracking. Every artifact carries structured metadata about its production lineage — which model, which node, which source data. This feeds a royalty distribution system so nodes that contribute compute and stored artifacts get credited downstream when those artifacts are referenced. We're treating provenance as a first-class concern rather than an afterthought, because reproducibility matters for science in ways it doesn't for most other LLM use cases.

On the roadmap: a model distillation pipeline where teacher models running on higher-resource nodes can distill into smaller models that run more efficiently across the broader network. The idea is that the network itself becomes a vehicle for propagating capability, not just sharing raw compute.

**Current state:**

- v0.2.1 alpha, 1,391+ passing tests
- Single-node and multi-node compute functional
- Real inference via Anthropic/OpenAI backends
- FTNS token on Ethereum Sepolia testnet for the economic layer

```bash
pip install prsm-network
prsm node start
# Connects to bootstrap, 100 FTNS welcome grant, REST API at localhost:8000
```

GitHub: https://github.com/Ryno2390/PRSM

What we want right now is people who will stress-test the NWTN pipeline, experiment with multi-node job routing, and give us technically specific feedback on where the architecture is weak. Happy to go deep on any of the design decisions in the comments.
