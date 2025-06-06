# PRSM Architecture

The Protocol for Recursive Scientific Modeling (PRSM) is a decentralized, multi-agent AI framework designed to coordinate, execute, and refine complex scientific tasks through a recursive, parallelized system of lightweight models. This document provides an overview of PRSM’s architectural layers, agents, data routing, and integration with decentralized technologies.

---

## 🔧 Core Architectural Concepts

- **Recursive Decomposition ("Refraction")**: Complex user prompts are broken into successively smaller subtasks by a hierarchy of Architect AIs.
- **Parallel Execution**: Elemental tasks are processed simultaneously using a distributed network of lightweight, purpose-built models.
- **Decentralized Coordination**: All compute, data routing, and governance occur through decentralized infrastructure and peer-contributed resources.
- **Token-Gated Access**: PRSM’s core AGI, NWTN, consumes FTNS tokens to grant contextual reasoning and compute allocation.

---

## 🧠 NWTN: The Orchestrator AGI

**NWTN (Neural Web for Transformation Networking)** is the central orchestrator that mediates user queries and governs the distributed AI network. Its responsibilities include:

- Parsing and refining ambiguous user prompts
- Delegating prompt segments to the Architect AI hierarchy
- Routing tasks through Prompter, Router, and Compiler sub-AIs
- Reconstructing final outputs and reasoning chains
- Charging FTNS per unit of context requested

---

## 🏗️ Agent Layers

### 1. **Architect AIs**
- Recursive prompt decomposers
- Multiple levels of abstraction
- Transform user intent into subtasks via “refraction”

### 2. **Prompter AIs**
- Specialized in prompt engineering
- Translates subtasks into optimized queries
- Enhances alignment, context, and clarity

### 3. **Router AIs**
- Match tasks to available distilled sub-models
- Access the open marketplace of LLMs
- Fall back to commercial or open-source general LLMs if needed

### 4. **Sub-LLMs (Distilled AIs)**
- Lightweight, purpose-built models
- Hosted by participants across the network
- Continuously refined and governed via the PRSM teacher ecosystem

### 5. **Compiler AIs**
- Aggregate outputs from multiple sub-models
- Reconstruct answers and metadata for traceability
- Work in hierarchical chains (elemental → mid → final compiler)

---

## 🔁 Process Flow

1. **User Prompt → NWTN**
2. **NWTN Clarifies** → Passes to Level-1 Architect
3. **Refraction Process** → Subtasks → Lower-Level Architects
4. **Prompter AIs** optimize prompt construction
5. **Router AIs** assign tasks to best-fit sub-LLMs
6. **Parallel Execution** of tasks
7. **Compilers** merge output → Results percolate up architecture
8. **NWTN returns** final answer + transparency trace

---

## 🧩 Modular AI Components

- Each agent (Architect, Prompter, Router, etc.) can be:
  - Distilled and improved by the community
  - Hosted on local compute or edge networks
  - Replaced/upgraded based on performance

- Incentive for hosting or improving components:
  - Earn FTNS (Fungible Tokens for Node Support)
  - Gain governance voting weight
  - Access more NWTN context bandwidth

## 🤖 Automated Distillation System

PRSM's revolutionary Automated Distillation System enables any user to create specialized AI models without machine learning expertise. This system directly contributes to PRSM's agent network by generating purpose-built models.

### **Integration with PRSM Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                 AUTOMATED DISTILLATION SYSTEM                   │
├─────────────────────────────────────────────────────────────────┤
│  🎯 Distillation Orchestrator                                  │
│    ↓ ← Receives requests from NWTN for specialized agents      │
│  📊 Knowledge Extraction Engine                                 │
│    ↓ ← Analyzes teacher models for capability mapping          │
│  🏗️ Model Architecture Generator                               │
│    ↓ ← Designs optimal student architectures                   │
│  🎓 Automated Training Pipeline                                 │
│    ↓ ← Executes multi-strategy distillation training          │
│  📈 Performance Evaluator                                      │
│    ↓ ← Validates quality and performance metrics              │
│  ✅ Safety & Validation                                        │
│    ↓ ← Ensures circuit breaker compliance                     │
│  🏪 Marketplace Integration                                     │
│    ↓ ← Deploys to P2P federation and marketplace              │
│  🔄 Agent Network Integration                                   │
│    ↓ ← Becomes Prompter, Router, Compiler agents              │
└─────────────────────────────────────────────────────────────────┘
```

### **System Components**

**🎯 Distillation Orchestrator**
- Central coordination for distillation lifecycle
- Integration with FTNS tokenomics for cost management
- Queue management for fair resource allocation
- Real-time progress tracking and user communication

**📊 Knowledge Extraction Engine**
- Analyzes teacher models to identify distillable capabilities
- Maps domain expertise and reasoning patterns
- Determines optimal compression strategies
- Assesses distillation feasibility and difficulty

**🏗️ Model Architecture Generator**
- Generates optimal student architectures based on requirements
- Balances performance vs. efficiency trade-offs
- Considers deployment constraints and hardware targets
- Integrates with PRSM's agent architecture patterns

**🎓 Automated Training Pipeline**
- Six training strategies for different use cases
- Multi-stage training with progress tracking
- Resource management and optimization
- Error handling and recovery mechanisms

**📈 Performance Evaluator**
- Comprehensive quality assessment
- Benchmarking against teacher models
- Efficiency and resource utilization analysis
- Deployment readiness validation

**✅ Safety & Validation**
- Circuit breaker compliance testing
- Content safety and bias detection
- Governance standard alignment
- Emergency halt capability integration

### **Training Strategies for PRSM Agents**

1. **BASIC**: Simple teacher-student transfer for general-purpose agents
2. **PROGRESSIVE**: Multi-stage training for complex reasoning agents
3. **ENSEMBLE**: Multi-teacher training for robust agent capabilities
4. **ADVERSARIAL**: Robust training for safety-critical agents
5. **CURRICULUM**: Structured learning for educational and tutorial agents
6. **SELF_SUPERVISED**: Learning from unlabeled data for specialized domains

### **Agent Network Integration**

Distilled models automatically integrate into PRSM's agent ecosystem:

- **Prompter AIs**: Specialized prompt optimization for specific domains
- **Router AIs**: Expert task-model matching within domains
- **Compiler AIs**: Domain-specific result synthesis and validation
- **Architect AIs**: Specialized decomposition for complex domain problems
- **Executor AIs**: Optimized execution engines for specific tasks

### **Economic Integration**

- **FTNS Cost Management**: Transparent pricing and budget control
- **Revenue Sharing**: Automatic distribution to teacher model owners
- **Marketplace Listing**: Instant deployment to PRSM marketplace
- **Performance Incentives**: Rewards for high-quality, efficient models
- **Community Contributions**: Earning mechanisms for model creators

### **Democratization of AI Development**

The Automated Distillation System transforms PRSM from requiring pre-existing specialized models to empowering users to create their own:

- **No ML Expertise Required**: Simple API for non-technical users
- **Cost-Effective**: 90%+ reduction in development costs
- **Rapid Deployment**: Models ready in hours, not months
- **Quality Assurance**: Automated validation and safety compliance
- **Community Driven**: Shared knowledge and collaborative improvement

---

## 🛰️ Decentralized Infrastructure

- **Storage**: IPFS is used to store and version research artifacts, models, and training data.
- **Networking**: DAG-based protocol (e.g., IOTA) for scalable, feeless microtransactions and tracking.
- **Execution**: Distributed compute across participant hardware (desktop, mobile, edge devices).
- **Security**: Sharded model execution with circuit breaker fail-safes and zero-trust sandboxing.

---

## ⚙️ FTNS Utility and Cost Model

- NWTN access requires FTNS, charged per unit of compute/context.
- Initial users receive FTNS grants but must contribute (storage, compute, novel data) to continue use.
- FTNS are earned via:
  - Hosting frequently accessed files (IPFS provenance)
  - Contributing useful research (judged by access rate, not subjective novelty)
  - Running pre/post-processing on edge devices
  - Distilling or improving sub-models

---

## 🔒 Safety and Governance

- Recursive decomposition inherently limits AGI generalization risks.
- Specialized sub-models lack broad capabilities.
- Any participant can trip a "kill switch" if unsafe models arise (Toyota-style assembly line halt).
- Open voting governs architectural upgrades and economic policy.
- Token-weighted votes with decay-based term limits.

---

## 📈 Future Expansion

- Forks of NWTN for enterprise/private use (e.g., internal Apple R&D instance)
- Real-time token micro-incentives based on usage metrics
- Modular plugin support for hardware interfacing, simulation, and robotics
- IPFS-wide data mirroring with automated semantic indexing

---

## 📌 Summary

PRSM replaces centralized monolithic AI with a distributed, recursive, modular, and incentivized system. Its architecture is designed for scalability, resilience, scientific transparency, and continuous improvement. By fragmenting large models into decentralized, purpose-built intelligences and connecting them with robust orchestration and incentive systems, PRSM offers a new paradigm for open AI.
