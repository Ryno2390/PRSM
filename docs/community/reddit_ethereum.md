# Reddit — r/ethereum Post

## Title
FTNS token live on Sepolia — building a P2P AI compute network where nodes earn for contributing inference/storage [alpha]

## Body

We just deployed the FTNS (Fungible Token for Node Services) contract to Ethereum Sepolia testnet as part of PRSM — a decentralized P2P network for AI compute and scientific research. Sharing here because the token economic model is genuinely novel and I'd value feedback from people who think carefully about this stuff.

**Contract address (Sepolia):** `0xd979c096BE297F4C3a85175774Bc38C22b95E6a4`

**How the token economy works:**

FTNS is the native unit of exchange in the PRSM compute marketplace. Nodes earn FTNS by contributing compute (running inference jobs routed by the NWTN orchestration layer) and storage (hosting artifacts on the IPFS-based provenance layer). Nodes spend FTNS to submit jobs to the network. The incentive is symmetric: the more you contribute, the more you can consume.

New nodes receive a 100 FTNS welcome grant on first connection to the bootstrap server — this lets you start submitting jobs immediately without needing to front compute first. The intent is to lower the barrier to entry while the network bootstraps toward sufficient node density.

The royalty distribution mechanism is one of the more interesting parts: when stored artifacts (model outputs, embeddings, intermediate research results) are referenced in downstream jobs, the nodes that produced and hosted those artifacts receive a share of the job's FTNS cost. This is tracked via semantic provenance metadata attached to every stored artifact. The goal is to make contribution economically meaningful across time, not just for the node that runs the immediate inference job.

**Current state and honest caveats:**

This is Sepolia — testnet only, no mainnet deployment yet. The token economics are implemented and functional but are still being calibrated based on real usage data. We're deliberately not rushing to mainnet because we want the incentive model to actually work before we make it real-money. The staking mechanics and bridge architecture for mainnet are on the roadmap but not yet implemented.

The broader network is v0.2.1 alpha with 1,391+ passing tests. The safety layer includes circuit breakers, Ed25519 signatures, and post-quantum crypto for message integrity — relevant because you don't want nodes manipulating job results to inflate their earnings.

```bash
pip install prsm-network
prsm node start
# Auto-connects to bootstrap at wss://bootstrap1.prsm-network.com:8765
# 100 FTNS welcome grant on first connect
```

GitHub: https://github.com/Ryno2390/PRSM

If you've thought about token-incentivized compute markets, stake-weighted job routing, or Sybil resistance in permissionless node networks — I'd genuinely like to hear your perspective on where the economic model has holes. That's the kind of feedback that's hard to get without posting in a community that actually understands this stuff.
