# PRSM: Protocol for Recursive Scientific Modeling

**A decentralized AGI framework for verified scientific discovery.**

---

## 🏗️ Revolutionary Project Architecture

PRSM leverages three converged technologies to decentralize artificial intelligence:

### 1. 🧠 Linear-Scaling Brain (SSM Core)
*   **Tech:** State Space Models (SSM) inspired by Mamba/S4.
*   **Advantage:** Replaces memory-heavy Transformers with $O(N)$ complexity. This enables complex scientific reasoning on edge nodes (laptops/mobiles) with a constant memory footprint.
*   **Module:** `prsm/compute/nwtn/architectures/ssm_core.py`

### 2. 🔗 Ledger of Truth (Deterministic Consensus)
*   **Tech:** Deterministic PRNG + Quantized SHA-256 Hashing + Byzantine Fault Tolerance (BFT).
*   **Advantage:** Ensures that every node reaches the *exact same* conclusion for a given seed. This creates a **Proof-of-Inference**, allowing the blockchain to reach consensus on task rewards without trust.
*   **Module:** `prsm/core/utils/deterministic.py`

### 3. 🔍 Exploratory Discovery (Search-Augmented Reasoning)
*   **Tech:** Monte Carlo Tree Search (MCTS) + Value Functions.
*   **Advantage:** Moves AI from "Predictive" (probabilistic guessing) to "Exploratory" (searching for proofs). NWTN explores a tree of hypotheses and prunes weak logical paths.
*   **Module:** `prsm/compute/nwtn/engines/search_reasoning_engine.py`

---

## ✅ Verified Features & Tests

The following core components have been tested and verified:

### 🚀 P2P Networking
*   **Status:** ✅ Verified. Nodes discover peers and share specialized "Teacher" capabilities.
*   **Test:** `tests/test_p2p_network.py`

### 💎 Blockchain Oracle
*   **Status:** ✅ Verified. Aggregates economic data and syncs with the FTNS ledger.
*   **Test:** `tests/test_blockchain_oracle.py`

### 📦 Decentralized Storage
*   **Status:** ✅ Verified. IPFS-based sharding and retrieval verified with mocked node.
*   **Test:** `tests/test_ipfs_integration.py`

### 🔢 Deterministic SSM
*   **Status:** ✅ Verified. Confirmed that local generators and quantization enable identical inference hashes across instances.
*   **Test:** `tests/test_deterministic_ssm.py`

### 🌲 MCTS Reasoning
*   **Status:** ✅ Verified. The engine successfully searches through a hypothesis space to find high-value scientific insights.
*   **Test:** `tests/test_search_reasoning.py`

---

## 🚀 Usage

1.  **Clone & Setup:**
    ```bash
    git clone https://github.com/Ryno2390/PRSM.git
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    pip install torch pyyaml structlog pydantic aiohttp aiofiles web3 eth-account hexbytes
    ```

2.  **Run Core Tests:**
    ```bash
    PYTHONPATH=. python3 tests/test_ssm_system.py
    PYTHONPATH=. python3 tests/verify_consensus_logic.py
    ```

---

**© 2026 PRSM Project - Decentralizing the Future of Science.**
