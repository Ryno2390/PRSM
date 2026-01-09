# PRSM: Protocol for Recursive Scientific Modeling

**A decentralized AI framework for scientific discovery, leveraging edge computing, blockchain economics, and distributed storage.**

---

## 🏗️ Project Architecture

PRSM has been refactored into a clean, domain-driven architecture to ensure scalability and maintainability.

### 1. `prsm/compute/` (The "Brain")
Contains all logic related to AI execution and distributed task processing.
*   **`nwtn/`**: The core "Newton" reasoning engine and neuro-symbolic AI logic.
*   **`network/`**: The **Distributed RLT Network** implementation. Handles P2P peer discovery, teacher coordination, and load balancing.
*   **`agents/`**: Autonomous agents for specific scientific tasks.

### 2. `prsm/economy/` (The "Ledger")
Manages the economic incentives and value transfer within the network.
*   **`blockchain/`**: The **FTNS Oracle** and smart contract integrations. Syncs off-chain state with Ethereum/Polygon/BSC.
*   **`tokenomics/`**: Logic for token supply, inflation control, and staking rewards.
*   **`marketplace/`**: Decentralized exchange for trading AI models and datasets.

### 3. `prsm/data/` (The "Memory")
Handles decentralized storage and data ingestion.
*   **`ipfs/`**: The **IPFS Client** for model sharding and content-addressed storage.
*   **`ingestion/`**: Pipelines for processing scientific papers and datasets.
*   **`storage/`**: Local and distributed persistence layers.

### 4. `prsm/interface/` (The "Face")
The interaction layer for users and external applications.
*   **`api/`**: RESTful API endpoints for the PRSM platform.
*   **`dashboard/`**: Real-time visualization of network health and compute tasks.
*   **`cli.py`**: Command-line tools for node management.

### 5. `prsm/core/` (The "Foundation")
Shared utilities used across all domains.
*   **`config/`**: Centralized configuration management.
*   **`security/`**: Authentication, authorization, and cryptography primitives.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Node.js & NPM (for smart contract tests)
*   Docker (optional, for local IPFS/Blockchain nodes)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ryno2390/PRSM.git
    cd PRSM
    ```

2.  **Set up a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    # Install additional core dependencies
    pip install pyyaml structlog pydantic aiohttp aiofiles web3 eth-account hexbytes
    ```

---

## ✅ Verified Features

The following core components have been tested and verified:

### 1. Distributed Edge Computing (P2P)
The **Distributed RLT Network** allows nodes to discover each other and share "teacher" capabilities.
*   **Test:** `tests/test_p2p_network.py`
*   **Run:** `python tests/test_p2p_network.py`
*   **Status:** ✅ Verified (Teacher registration & discovery works).

### 2. Blockchain Economy
The **FTNS Oracle** aggregates prices from multiple sources and syncs with the ledger.
*   **Test:** `tests/test_blockchain_oracle.py`
*   **Run:** `python tests/test_blockchain_oracle.py`
*   **Status:** ✅ Verified (Price aggregation & DB persistence works).

### 3. Decentralized Storage (IPFS)
The **IPFS Integration** enables adding and retrieving content via content addressing (CID).
*   **Test:** `tests/test_ipfs_integration.py`
*   **Run:** `python tests/test_ipfs_integration.py`
*   **Status:** ✅ Verified (Content add/get with mocking works).

---

## 📂 Legacy Content
All original demos, archived scripts, and examples have been moved to the `examples_and_demos/` directory to keep the root clean.

---

## 🤝 Contributing
1.  Fork the repo.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add some amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

---

**© 2026 PRSM Project**