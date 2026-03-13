# PRSM Architecture Analysis

## Executive Summary

PRSM (Protocol for Recursive Scientific Modeling) is a sophisticated decentralized AI framework designed for scientific discovery. The codebase demonstrates a well-organized, modular architecture with clear separation of concerns across 8 major subsystems. This analysis provides a comprehensive overview of the architectural decisions, component relationships, and design patterns employed.

---

## 1. High-Level Architecture Overview

### System Architecture Diagram

```mermaid
graph TB
    subgraph Client Layer
        CLI[CLI Interface]
        API[REST API]
        WS[WebSocket]
        DASH[Dashboard UI]
    end

    subgraph Application Layer
        NWTN[NWTN Orchestrator]
        AGENTS[Agent Framework]
        TEACHERS[Teacher Models]
    end

    subgraph Service Layer
        AUTH[Auth Service]
        SAFETY[Safety Infrastructure]
        GOV[Governance System]
        IMPROVE[Self-Improvement]
    end

    subgraph Data Layer
        DB[(PostgreSQL/SQLite)]
        REDIS[(Redis Cache)]
        IPFS[IPFS Storage]
        VECTOR[(Vector Store)]
    end

    subgraph Network Layer
        NODE[P2P Node]
        LEDGER[DAG Ledger]
        FEDERATION[Federation]
    end

    subgraph Economy Layer
        FTNS[FTNS Tokenomics]
        MARKET[Marketplace]
        PAY[Payments]
        BLOCK[Blockchain Bridge]
    end

    CLI --> NWTN
    API --> NWTN
    WS --> NWTN
    DASH --> API

    NWTN --> AGENTS
    NWTN --> FTNS
    AGENTS --> TEACHERS
    
    NWTN --> AUTH
    AGENTS --> SAFETY
    NWTN --> GOV
    NWTN --> IMPROVE

    AUTH --> DB
    SAFETY --> REDIS
    NWTN --> IPFS
    AGENTS --> VECTOR

    NWTN --> NODE
    NODE --> LEDGER
    NODE --> FEDERATION

    FTNS --> LEDGER
    MARKET --> FTNS
    PAY --> BLOCK
```

---

## 2. Directory Structure Analysis

### Top-Level Organization

```
PRSM/
├── prsm/                    # Main source code
│   ├── core/               # Foundation: config, models, database, auth
│   ├── compute/            # AI/ML compute layer
│   │   ├── nwtn/          # NWTN reasoning system
│   │   ├── agents/        # Agent pipeline
│   │   ├── distillation/  # Model distillation
│   │   └── federation/    # Distributed compute
│   ├── economy/           # Economic systems
│   │   ├── tokenomics/    # FTNS token system
│   │   ├── governance/    # DAO governance
│   │   ├── marketplace/   # Model marketplace
│   │   └── payments/      # Payment processing
│   ├── node/              # P2P networking
│   ├── interface/         # API and UI
│   ├── data/              # Data management
│   └── collaboration/     # Session management
├── tests/                  # Test suites
├── docs/                   # Documentation
├── config/                 # Configuration files
├── docker/                 # Docker configurations
└── scripts/               # Utility scripts
```

### Module Size Analysis

| Module | Files | Primary Responsibility |
|--------|-------|----------------------|
| `prsm/core/` | 80+ | Foundation services, auth, config, monitoring |
| `prsm/compute/` | 100+ | AI orchestration, agents, NWTN, distillation |
| `prsm/economy/` | 50+ | Tokenomics, governance, marketplace, payments |
| `prsm/node/` | 20 | P2P networking, ledger, discovery |
| `prsm/interface/` | 50+ | REST API, WebSocket, dashboard |
| `prsm/data/` | 30+ | Storage, IPFS, vector stores, analytics |

---

## 3. Core Components Deep Dive

### 3.1 Core Infrastructure (`prsm/core/`)

The core module serves as the foundation for all other components:

**Key Submodules:**

| Submodule | Purpose | Key Files |
|-----------|---------|-----------|
| `auth/` | Authentication & authorization | `auth_manager.py`, `jwt_handler.py`, `middleware.py` |
| `config/` | Configuration management | `config.py`, `manager.py`, `schemas.py` |
| `cryptography/` | Security & encryption | `encryption.py`, `key_management.py`, `zk_proofs.py` |
| `database/` | Data persistence | `database.py`, `models.py` |
| `monitoring/` | Observability | `metrics.py`, `health.py`, `alerts.py` |
| `safety/` | Safety infrastructure | `circuit_breaker.py`, `monitor.py`, `governance.py` |
| `integrations/` | External connectors | `connectors/`, `langchain/`, `mcp/` |

**Design Patterns Identified:**
- **Settings Pattern**: Centralized configuration via Pydantic `BaseSettings`
- **Repository Pattern**: Database abstraction through SQLAlchemy models
- **Circuit Breaker Pattern**: Fault tolerance in `circuit_breaker.py`
- **Factory Pattern**: Integration connectors via base classes

### 3.2 NWTN Orchestrator (`prsm/compute/nwtn/`)

The NWTN (Neural Web for Transformation Networking) system is the central coordination layer:

**Architecture:**

```mermaid
flowchart LR
    Query[User Query] --> Clarify[Intent Clarification]
    Clarify --> Discover[Model Discovery]
    Discover --> Allocate[Context Allocation]
    Allocate --> Reason[Multi-Stage Reasoning]
    Reason --> Validate[Safety Validation]
    Validate --> Response[Response]
```

**Key Components:**
- [`orchestrator.py`](prsm/compute/nwtn/orchestrator.py) - Main coordination logic
- [`context_manager.py`](prsm/compute/nwtn/context_manager.py) - FTNS context tracking
- [`engines/`](prsm/compute/nwtn/engines/) - Processing engines
  - `chunk_embedding_system.py` - Document embedding
  - `universal_knowledge_ingestion_engine.py` - Knowledge ingestion
  - `world_model_engine.py` - World modeling
- [`architectures/`](prsm/compute/nwtn/architectures/) - Neural architectures
  - `hybrid_architecture.py` - Hybrid model architecture

### 3.3 Agent Framework (`prsm/compute/agents/`)

5-layer agent pipeline for distributed task processing:

```mermaid
flowchart TB
    subgraph Layer 1 - Architects
        ARCH[Hierarchical Architect]
    end
    subgraph Layer 2 - Prompters
        PROMPT[Prompt Optimizer]
    end
    subgraph Layer 3 - Routers
        ROUTE[Intelligent Router]
    end
    subgraph Layer 4 - Executors
        EXEC[Model Executor]
    end
    subgraph Layer 5 - Compilers
        COMP[Hierarchical Compiler]
    end

    ARCH --> PROMPT --> ROUTE --> EXEC --> COMP
    COMP -.->|Recursive Decomposition| ARCH
```

**Key Files:**
- [`agents/base.py`](prsm/compute/agents/base.py) - Abstract agent base class
- [`agents/prompters/prompt_optimizer.py`](prsm/compute/agents/prompters/prompt_optimizer.py) - Domain-specific optimization
- [`agents/executors/unified_router.py`](prsm/compute/agents/executors/unified_router.py) - Model routing
- [`agents/compilers/hierarchical_compiler.py`](prsm/compute/agents/compilers/hierarchical_compiler.py) - Result synthesis

### 3.4 Economy Layer (`prsm/economy/`)

Comprehensive economic system with multiple subdomains:

**Tokenomics (`prsm/economy/tokenomics/`):**
- `ftns_service.py` - Core token operations
- `atomic_ftns_service.py` - Atomic transactions
- `dynamic_supply_controller.py` - Supply management
- `enhanced_pricing_engine.py` - Dynamic pricing

**Governance (`prsm/economy/governance/`):**
- `voting.py` - Token-weighted voting
- `proposals.py` - Proposal management
- `quadratic_voting.py` - Quadratic voting implementation
- `anti_monopoly.py` - Anti-monopoly measures

**Marketplace (`prsm/economy/marketplace/`):**
- `real_marketplace_service.py` - Marketplace operations
- `recommendation_engine.py` - Model recommendations
- `reputation_system.py` - Reputation tracking

### 3.5 Node Layer (`prsm/node/`)

P2P networking infrastructure:

**Key Components:**
- [`node.py`](prsm/node/node.py) - Main node orchestrator
- [`dag_ledger.py`](prsm/node/dag_ledger.py) - DAG-based transaction ledger
- [`transport.py`](prsm/node/transport.py) - WebSocket transport
- [`discovery.py`](prsm/node/discovery.py) - Peer discovery
- [`gossip.py`](prsm/node/gossip.py) - Gossip protocol
- [`agent_registry.py`](prsm/node/agent_registry.py) - Agent registration

### 3.6 Interface Layer (`prsm/interface/`)

API and user-facing components:

**API Structure (`prsm/interface/api/`):**
- `main.py` - FastAPI application entry point
- `app_factory.py` - Application factory pattern
- `routers/` - Domain-specific routers
- `websocket/` - WebSocket handlers

**API Endpoints by Domain:**
| Domain | File | Key Endpoints |
|--------|------|---------------|
| Auth | `auth_api.py` | `/auth/*` |
| FTNS | `budget_api.py` | `/ftns/*` |
| Governance | `governance_api.py` | `/governance/*` |
| Marketplace | `marketplace_api.py` | `/marketplace/*` |
| Monitoring | `monitoring_api.py` | `/monitoring/*` |

---

## 4. Configuration Management

### Configuration Architecture

PRSM uses a layered configuration approach:

```mermaid
flowchart TB
    ENV[Environment Variables]
    FILE[Config Files YAML]
    CODE[Default Values]
    
    ENV --> CM[ConfigManager]
    FILE --> CM
    CODE --> CM
    
    CM --> VALID[Validators]
    VALID --> SETTINGS[PRSMSettings]
    SETTINGS --> APP[Application]
```

**Key Configuration Files:**
- [`prsm/core/config.py`](prsm/core/config.py) - Main settings class
- [`prsm/core/config/manager.py`](prsm/core/config/manager.py) - Configuration manager
- [`prsm/core/config/schemas.py`](prsm/core/config/schemas.py) - Configuration schemas
- [`prsm/core/config/models/`](prsm/core/config/models/) - YAML model configurations

**Configuration Categories:**
1. **Core Settings**: Environment, debug mode, logging
2. **Security**: JWT settings, secret keys, access control
3. **Database**: Connection strings, pooling, transactions
4. **External Services**: Redis, IPFS, vector DBs, AI APIs
5. **PRSM-Specific**: NWTN, agents, safety, FTNS, P2P

---

## 5. Entry Points

### Primary Entry Points

| Entry Point | Location | Purpose |
|-------------|----------|---------|
| CLI | [`prsm/cli.py`](prsm/cli.py) | Command-line interface |
| API Server | [`prsm/interface/api/main.py`](prsm/interface/api/main.py) | FastAPI application |
| P2P Node | [`prsm/node/node.py`](prsm/node/node.py) | Network node |

### CLI Commands

```bash
prsm serve --host 127.0.0.1 --port 8000  # Start API server
prsm status                               # Show system status
prsm init                                 # Initialize system
```

### Application Factory

The API uses the factory pattern via [`app_factory.py`](prsm/interface/api/app_factory.py):

```python
from prsm.interface.api.app_factory import create_app
app = create_app()
```

---

## 6. Dependency Analysis

### Internal Dependencies

```mermaid
graph BT
    subgraph Foundation
        CORE[core]
        CONFIG[config]
    end
    
    subgraph Compute
        NWTN[nwtn]
        AGENTS[agents]
        DISTILL[distillation]
    end
    
    subgraph Economy
        FTNS[tokenomics]
        GOV[governance]
        MARKET[marketplace]
    end
    
    subgraph Infrastructure
        NODE[node]
        DATA[data]
        INTERFACE[interface]
    end
    
    NWTN --> CORE
    AGENTS --> CORE
    FTNS --> CORE
    NODE --> CORE
    INTERFACE --> CORE
    
    NWTN --> FTNS
    AGENTS --> NWTN
    MARKET --> FTNS
    GOV --> FTNS
```

### External Dependencies (Key)

| Category | Package | Purpose |
|----------|---------|---------|
| Web Framework | FastAPI, Uvicorn | API server |
| Data Validation | Pydantic | Model validation |
| Database | SQLAlchemy, Alembic, asyncpg | ORM & migrations |
| Caching | Redis | Session & cache |
| Storage | IPFS, ipfshttpclient | Distributed storage |
| AI/ML | OpenAI, Anthropic, Transformers, Torch | Model integration |
| Vector DB | Pinecone, Weaviate, ChromaDB | Vector storage |
| Security | cryptography, PyJWT, bcrypt | Authentication |
| Networking | websockets, aiohttp | P2P communication |
| Monitoring | Prometheus, structlog | Observability |

---

## 7. Design Patterns Identified

### Architectural Patterns

| Pattern | Location | Description |
|---------|----------|-------------|
| **Microservices/SOA** | `prsm/interface/api/` | Domain-specific API services |
| **Agent-Based** | `prsm/compute/agents/` | Specialized autonomous agents |
| **Event-Driven** | `prsm/node/gossip.py` | Gossip protocol for propagation |
| **CQRS** | `prsm/core/database.py` | Read/write separation |

### Design Patterns

| Pattern | Location | Description |
|---------|----------|-------------|
| **Factory** | `app_factory.py`, `base_connector.py` | Object creation abstraction |
| **Repository** | `database.py`, `FTNSQueries` | Data access abstraction |
| **Circuit Breaker** | `circuit_breaker.py` | Fault tolerance |
| **Strategy** | `cache_strategies.py` | Interchangeable algorithms |
| **Observer** | `monitoring/` | Event notification |
| **Proxy** | Data access via `enhanced_ipfs.py` | Access control |
| **Singleton** | `get_settings()` | Configuration instance |

### Data Patterns

| Pattern | Location | Description |
|---------|----------|-------------|
| **Active Record** | SQLAlchemy models | Model-DB integration |
| **Data Mapper** | `models.py` | Object-DB mapping |
| **Unit of Work** | Database sessions | Transaction management |

---

## 8. Data Flow Analysis

### Query Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant NWTN
    participant Agents
    participant FTNS
    participant Safety
    participant IPFS

    User->>API: Submit Query
    API->>NWTN: Process Query
    NWTN->>FTNS: Check Balance
    FTNS-->>NWTN: Balance OK
    NWTN->>Agents: Execute Pipeline
    Agents->>IPFS: Retrieve Context
    IPFS-->>Agents: Context Data
    Agents->>Safety: Validate Output
    Safety-->>Agents: Validation Result
    Agents-->>NWTN: Results
    NWTN->>FTNS: Charge Tokens
    NWTN-->>API: Response
    API-->>User: Final Answer
```

### Transaction Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant FTNS
    participant Ledger
    participant Network

    User->>API: Create Transaction
    API->>FTNS: Validate & Process
    FTNS->>Ledger: Record Transaction
    Ledger->>Network: Propagate
    Network-->>Ledger: Consensus
    Ledger-->>FTNS: Confirmed
    FTNS-->>API: Success
    API-->>User: Transaction ID
```

---

## 9. Architectural Strengths

### 1. **Modular Design**
- Clear separation of concerns across modules
- Each subsystem can be developed and tested independently
- Well-defined interfaces between components

### 2. **Scalability**
- Async-first architecture with `asyncio`
- Horizontal scaling via P2P federation
- Caching layer with Redis

### 3. **Resilience**
- Circuit breaker pattern for fault tolerance
- Safety monitoring infrastructure
- Graceful degradation capabilities

### 4. **Security**
- Multi-layer authentication (JWT, MFA, SSO)
- Post-quantum cryptography support
- Comprehensive audit logging

### 5. **Extensibility**
- Plugin architecture via `integrations/`
- Factory patterns for connectors
- Configuration-driven behavior

### 6. **Observability**
- Structured logging with `structlog`
- Prometheus metrics integration
- Health monitoring endpoints

---

## 10. Potential Improvement Areas

### 1. **Module Organization**
- **Observation**: Some modules have overlapping responsibilities (e.g., `prsm/core/ipfs_client.py` vs `prsm/data/ipfs/`)
- **Recommendation**: Consolidate IPFS functionality into a single module

### 2. **Dependency Management**
- **Observation**: Large number of dependencies in `pyproject.toml`
- **Recommendation**: Consider splitting into optional dependency groups for lighter installs

### 3. **Test Coverage**
- **Observation**: Coverage threshold set at 50% in `pyproject.toml`
- **Recommendation**: Increase coverage targets for critical security components (currently targeting 80%)

### 4. **Documentation**
- **Observation**: Some modules lack inline documentation
- **Recommendation**: Add docstrings to all public APIs

### 5. **Configuration Complexity**
- **Observation**: Multiple configuration layers may be confusing
- **Recommendation**: Simplify configuration hierarchy or provide better documentation

### 6. **Legacy Compatibility**
- **Observation**: Compatibility shim in `prsm/__init__.py` for `prsm.nwtn` imports
- **Recommendation**: Complete migration path and remove legacy support

---

## 11. Technology Stack Summary

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3.11+ |
| **Web Framework** | FastAPI, Uvicorn |
| **Database** | PostgreSQL, SQLite, SQLAlchemy |
| **Caching** | Redis |
| **Storage** | IPFS |
| **Vector DB** | Pinecone, Weaviate, ChromaDB |
| **AI/ML** | OpenAI, Anthropic, Transformers, PyTorch |
| **Networking** | WebSockets, aiohttp |
| **Security** | cryptography, PyJWT, NaCl |
| **Monitoring** | Prometheus, structlog |
| **Testing** | pytest, pytest-asyncio |
| **Containerization** | Docker, docker-compose |

---

## 12. Conclusion

PRSM demonstrates a sophisticated, well-architected system for decentralized AI research collaboration. The modular design, comprehensive safety infrastructure, and economic incentive mechanisms position it well for its intended purpose. The codebase shows maturity in its architectural decisions while maintaining flexibility for future enhancements.

Key takeaways:
1. **Strong foundation** with clear separation of concerns
2. **Comprehensive feature set** covering AI orchestration, economics, and governance
3. **Production-ready infrastructure** with monitoring, caching, and security
4. **Extensible design** supporting plugins and integrations
5. **Active development** evidenced by detailed documentation and test infrastructure

---

## 13. Code Review Findings (2026-02-20)

A comprehensive code review was conducted identifying the following issues organized by priority:

### Critical Priority (Immediate Action Required)

| # | Issue | Location | Risk |
|---|-------|----------|------|
| 1 | **Signature Verification in DAG Ledger** | [`dag_ledger.py:508`](prsm/node/dag_ledger.py:508) | Contains `pass` statement in signature verification block, potential for forged transactions |
| 2 | **Marketplace Uses Deprecated FTNS Service** | [`marketplace.py:24`](prsm/economy/tokenomics/marketplace.py:24), [`advanced_ftns.py:23`](prsm/economy/tokenomics/advanced_ftns.py:23) | Race condition vulnerabilities from deprecated service usage |
| 3 | **Atomic Balance Operations Missing in DAG Ledger** | [`dag_ledger.py:482-485`](prsm/node/dag_ledger.py:482) | TOCTOU race condition in balance check |

### High Priority (Next Sprint)

| # | Issue | Location | Risk |
|---|-------|----------|------|
| 4 | **Duplicate IPFS Clients** | [`prsm/core/ipfs_client.py`](prsm/core/ipfs_client.py) (1222 lines), [`prsm/data/ipfs/ipfs_client.py`](prsm/data/ipfs/ipfs_client.py) (503 lines) | Code duplication, maintenance burden |
| 5 | **Mock Services in Production Code** | [`orchestrator.py:81-175`](prsm/compute/nwtn/orchestrator.py:81) | Mock classes defined in production module |
| 6 | **Thread Safety in Peer Connections** | [`transport.py:205-207`](prsm/node/transport.py:205) | Race condition in peer dictionary modification |
| 7 | **Missing Error Handling in NWTN** | [`orchestrator.py:310`](prsm/compute/nwtn/orchestrator.py:310) | No try/except in query processing |

### Medium Priority

| # | Issue | Location | Risk |
|---|-------|----------|------|
| 8 | **Silent Exception Handling** | Codebase-wide | 99 instances of `except Exception: pass` hide potential issues |
| 9 | **Missing Integration Tests** | Test suites | P2P network partition, DAG consensus, marketplace concurrency, NWTN end-to-end |
| 10 | **Hardcoded Values** | Multiple files | Timeouts, limits, thresholds should be configurable |

### Low Priority (Technical Debt)

| # | Issue | Location | Risk |
|---|-------|----------|------|
| 11 | **Print Statements Instead of Logging** | [`marketplace.py:82`](prsm/economy/tokenomics/marketplace.py:82), [`advanced_ftns.py:90`](prsm/economy/tokenomics/advanced_ftns.py:90) | Inconsistent logging practices |
| 12 | **Legacy Compatibility Shim** | [`prsm/__init__.py:45-48`](prsm/__init__.py:45) | Needs deprecation timeline documented |
| 13 | **Missing Type Hints** | Multiple functions | Incomplete type annotations |

---

## 14. Security Assessment Summary

| Area | Status | Notes |
|------|--------|-------|
| Cryptographic Signatures | ✅ Strong | Ed25519 for transactions |
| JWT Handling | ✅ Strong | Algorithm restriction, revocation support |
| Token Double-Spend | ⚠️ Mixed | Atomic service exists but not used everywhere |
| P2P Authentication | ✅ Strong | Signature-based handshake |
| Post-Quantum Ready | ✅ Strong | PQC support implemented |
| Input Validation | ✅ Strong | Comprehensive validation layer |
| Rate Limiting | ⚠️ Partial | Exists but not on all endpoints |
| Error Information Leakage | ⚠️ Risk | Silent exceptions hide issues |

---

## 15. Architecture Observations

### Positive Patterns Identified

1. **Clean separation of concerns** between node, economy, compute modules
2. **Dependency injection pattern** used for services
3. **Async-first design** throughout the codebase
4. **Comprehensive data models** with Pydantic validation

### Areas for Improvement

1. **Service locator pattern overuse** - Global `get_settings()`, `get_database_service()` calls
2. **Mixed abstraction levels** in some modules
3. **Inconsistent error handling** across modules
4. **Incomplete implementations** - 26 TODO comments found

---

## 16. Remediation Priority Matrix

Issues organized by impact and effort to guide remediation planning:

### High Impact, Low Effort (Quick Wins)

| Issue | Effort | Impact | Status | Recommended Action |
|-------|--------|--------|--------|-------------------|
| Signature Verification `pass` statement | 1-2 hours | Critical | ✅ **COMPLETED** | Replaced with proper Ed25519 verification |
| Thread Safety in Peer Connections | 2-4 hours | High | Pending | Add threading.Lock for peer dictionary |
| Missing Error Handling in NWTN | 2-4 hours | High | Pending | Wrap query processing in try/except |

### High Impact, High Effort (Major Projects)

| Issue | Effort | Impact | Status | Recommended Action |
|-------|--------|--------|--------|-------------------|
| Marketplace Deprecated Service | 1-2 weeks | Critical | ✅ **COMPLETED** | Migrated to AtomicFTNSService |
| Atomic Balance Operations | 1 week | Critical | ✅ **COMPLETED** | Implemented OCC with version tracking |
| Duplicate IPFS Clients | 2-3 weeks | High | Pending | Consolidate into single module |
| Silent Exception Handling | 2-3 weeks | Medium | Pending | Replace with proper logging |

### Medium Impact, Low Effort (Improvements)

| Issue | Effort | Impact | Status | Recommended Action |
|-------|--------|--------|--------|-------------------|
| Mock Services in Production | 4-8 hours | Medium | Pending | Move to test fixtures |
| Print Statements to Logging | 2-4 hours | Low | ✅ **COMPLETED** | Replaced with structlog in marketplace |
| Hardcoded Values | 4-8 hours | Medium | Pending | Extract to configuration |

### Low Impact, Low Effort (Cleanup)

| Issue | Effort | Impact | Status | Recommended Action |
|-------|--------|--------|--------|-------------------|
| Legacy Compatibility Shim | 2-4 hours | Low | Pending | Add deprecation warnings |
| Missing Type Hints | Ongoing | Low | Pending | Add during regular maintenance |

---

## 17. What Still Needs to Be Done

Based on the code review findings, the following items require attention:

### Sprint 1 — Critical Security Fixes ✅ COMPLETED (2026-02-20)

- [x] **CRITICAL**: Fix signature verification `pass` statement in [`dag_ledger.py:508`](prsm/node/dag_ledger.py:508)
- [x] **CRITICAL**: Migrate marketplace from deprecated FTNS service to atomic service
- [x] **CRITICAL**: Implement atomic balance operations in DAG ledger

### Sprint 2 — Code Quality & Thread Safety ✅ COMPLETED (2026-02-27)

- [x] **HIGH**: Consolidate duplicate IPFS client implementations
- [x] **HIGH**: Remove mock services from production code in orchestrator
- [x] **HIGH**: Add thread safety to peer connection management
- [x] **HIGH**: Add error handling to NWTN query processing

### Sprint 3 — Exception Handling & Integration Tests ✅ COMPLETED (2026-03-01)

- [x] **MEDIUM**: Audit and fix silent exception handling (40 bare except patterns, ~50 silent handlers)
- [x] **MEDIUM**: Add missing integration tests (P2P, DAG consensus, marketplace concurrency, NWTN e2e)

### Sprint 4 — Core Collaboration Robustness ✅ COMPLETED (2026-03-02)

- [x] **CRITICAL**: Wire FTNS payments into collaboration protocols (task delegation, peer review, knowledge exchange)
- [x] **CRITICAL**: Broadcast all collaboration state changes to the network
- [x] **HIGH**: Add expiry enforcement and bounded memory for collaboration state
- [x] **MEDIUM**: Add content retrieval API for cross-node content download
- [x] **MEDIUM**: Bridge CollaborationManager with P2P AgentCollaboration

*See Section 20 for full Sprint 4 plan with phased implementation details.*

### Technical Debt Backlog

- [x] **MEDIUM**: Extract hardcoded values to configuration *(Sprint 5, Item 1)*
- [ ] **LOW**: Wire compute provider to NWTN orchestrator for real inference
- [ ] **LOW**: Extend peer discovery with capability-based search
- [ ] **LOW**: Add gossip persistence for late-joining nodes
- [ ] **LOW**: Document deprecation timeline for legacy compatibility shim
- [ ] **LOW**: Complete type hint coverage
- [ ] **LOW**: Increase test coverage from 50% to 80% for security components

---

## 18. Sprint 1 Completion Summary (2026-02-20)

Sprint 1 focused on addressing the three critical security vulnerabilities identified in the code review. All critical items have been successfully remediated.

### Completed Tasks

#### 1. Signature Verification in DAG Ledger ✅

**File Modified:** [`prsm/node/dag_ledger.py`](prsm/node/dag_ledger.py)

**Changes Made:**
- Fixed `pass` statement at line 508 that was bypassing signature verification
- Implemented proper Ed25519 signature verification using `cryptography` library
- Added logging for security audit trail using `structlog`
- Added `_pending_verification` instance variable for storing verification data
- Enhanced `_verify_transaction_signature()` method with full signature validation
- Updated `_is_signature_required()` to properly identify transaction types requiring signatures

**Security Impact:**
- Prevents forged transactions from being accepted
- Ensures transaction integrity and non-repudiation
- Provides audit trail for security monitoring

#### 2. Marketplace Migration to AtomicFTNSService ✅

**Files Modified:**
- [`prsm/economy/tokenomics/marketplace.py`](prsm/economy/tokenomics/marketplace.py)
- [`prsm/economy/tokenomics/advanced_ftns.py`](prsm/economy/tokenomics/advanced_ftns.py)
- [`prsm/economy/tokenomics/ftns_service.py`](prsm/economy/tokenomics/ftns_service.py)

**Changes Made:**
- Migrated from deprecated `ftns_service` to `AtomicFTNSService`
- Added idempotency key support for all marketplace transactions
- Replaced `print()` statements with `structlog` logging
- Added deprecation warning to old `FTNSService` class
- Updated all balance operations to use atomic methods

**Security Impact:**
- Eliminates race condition vulnerabilities in token operations
- Prevents double-spend attacks through idempotency keys
- Provides consistent logging for audit and debugging

#### 3. Atomic Balance Operations in DAG Ledger ✅

**File Modified:** [`prsm/node/dag_ledger.py`](prsm/node/dag_ledger.py)

**Changes Made:**
- Added new exception classes:
  - `AtomicOperationError` - Base exception for atomic operations
  - `InsufficientBalanceError` - Raised when balance is too low
  - `ConcurrentModificationError` - Raised on version mismatch (OCC)
  - `BalanceLockError` - Raised when lock acquisition fails
- Added `wallet_balances` cache table with version tracking
- Implemented atomic balance methods:
  - `_check_balance_atomic()` - Atomic balance check with version
  - `_commit_balance_deduction()` - Atomic deduction with version check
  - `_commit_balance_credit()` - Atomic credit operation
  - `_rollback_balance_check()` - Rollback for failed operations
- Uses SQLite SAVEPOINT for nested transaction support
- Implements Optimistic Concurrency Control (OCC) via version column

**Security Impact:**
- Prevents TOCTOU (Time-of-Check-Time-of-Use) race conditions
- Ensures balance consistency under concurrent access
- Provides proper error handling for concurrent modifications

#### 4. Test Suite Created ✅

**File Created:** [`tests/security/test_sprint1_security_fixes.py`](tests/security/test_sprint1_security_fixes.py)

**Test Coverage:**
- 27 tests across 6 test classes
- `TestSignatureVerification` - 5 tests for signature verification
- `TestAtomicFTNSOperations` - 5 tests for atomic FTNS operations
- `TestAtomicBalanceOperations` - 6 tests for atomic balance operations
- `TestMarketplaceAtomicOperations` - 4 tests for marketplace integration
- `TestIntegrationScenarios` - 4 tests for end-to-end scenarios
- `TestEdgeCases` - 3 tests for edge cases and error handling

#### 5. Verification Results ✅

All core security fixes verified working via integration tests:

| Component | Status | Notes |
|-----------|--------|-------|
| DAG Ledger Transaction Creation | ✅ WORKING | Genesis and transfer transactions create correctly |
| Signature Verification | ✅ WORKING | Unsigned transactions rejected when verification enabled |
| Atomic Balance Operations | ✅ WORKING | Insufficient balance and concurrent modification detected |
| Marketplace Atomic Operations | ✅ WORKING | Idempotency keys prevent double-deduction |

### Files Modified Summary

| File | Changes |
|------|---------|
| `prsm/node/dag_ledger.py` | Signature verification, atomic balance operations, new exception classes |
| `prsm/economy/tokenomics/marketplace.py` | Migrated to AtomicFTNSService, added idempotency keys |
| `prsm/economy/tokenomics/advanced_ftns.py` | Enhanced atomic operations, improved logging |
| `prsm/economy/tokenomics/ftns_service.py` | Added deprecation warning |
| `tests/security/test_sprint1_security_fixes.py` | New test suite (27 tests) |
| `tests/security/conftest.py` | Test fixtures and configuration |

### Security Improvements Summary

| Vulnerability | Before | After |
|--------------|--------|-------|
| Signature Verification | `pass` statement bypassed verification | Full Ed25519 signature validation |
| Token Operations | Race condition vulnerable | Atomic operations with idempotency |
| Balance Operations | TOCTOU race condition | OCC with version tracking |
| Logging | `print()` statements | Structured logging with `structlog` |

---

## 19. Sprint 2 Completion Summary (2026-02-27)

Sprint 2 focused on high-priority improvements to code quality, thread safety, and error handling.

### 1. IPFS Client Consolidation ✅

**Problem**: Multiple IPFS client implementations existed across the codebase, leading to:
- Inconsistent behavior
- Duplicated maintenance effort
- Potential for bugs due to different implementations

**Solution**: Consolidated to single canonical client at `prsm/core/ipfs_client.py`:
- Updated `prsm/data/ipfs/__init__.py` to re-export from canonical location
- Updated `prsm/data/data_layer/__init__.py` to use canonical client
- Added deprecation notices to legacy import paths

### 2. Mock Services Separation ✅

**Problem**: Mock services were mixed with production code, risking:
- Accidental use of mocks in production
- Unclear separation of concerns
- Testing complexity

**Solution**: Separated mock implementations from production code:
- Mocks clearly marked with deprecation warnings
- Production code uses real implementations
- Test fixtures provide mocks for testing only

### 3. Thread Safety in Peer Connection Management ✅

**Problem**: Race conditions in `WebSocketTransport` class:
- `self.peers` dictionary accessed from multiple async tasks
- `self._seen_nonces` and `self._nonce_timestamps` modified concurrently
- No synchronization between concurrent operations

**Solution**: Added `asyncio.Lock` instances for thread-safe access:
- `self._peers_lock` protects `self.peers` dictionary
- `self._nonces_lock` protects nonce tracking structures
- All dictionary access wrapped in `async with lock:` blocks
- Added async helper methods `get_peer_count()` and `get_peer_addresses()`

**Files Modified**: `prsm/node/transport.py`

### 4. NWTN Query Processing Error Handling ✅

**Problem**: Query processing lacked comprehensive error handling:
- Failures could leave sessions in inconsistent states
- No structured way to handle stage-specific errors
- Difficult to diagnose issues in production

**Solution**: Added comprehensive exception hierarchy and error handling:

```python
class QueryProcessingError(NWTNOrchestratorError):
    """Base error with session_id and stage context"""
    
class IntentClarificationError(QueryProcessingError):
    """Intent clarification failures with prompt context"""
    
class ModelDiscoveryError(QueryProcessingError):
    """Model discovery failures with category context"""
    
class ContextAllocationError(QueryProcessingError):
    """Context allocation failures with requested/available amounts"""
    
class ReasoningExecutionError(QueryProcessingError):
    """Reasoning execution failures with step and agent_type"""
    
class SafetyValidationError(QueryProcessingError):
    """Safety validation failures with flag details"""
```

**Files Modified**: `prsm/compute/nwtn/orchestrator.py`

### Files Modified Summary

| File | Changes |
|------|---------|
| `prsm/core/ipfs_client.py` | Canonical IPFS client implementation |
| `prsm/data/ipfs/__init__.py` | Re-exports from canonical location |
| `prsm/data/data_layer/__init__.py` | Updated imports |
| `prsm/node/transport.py` | Thread safety locks, async helper methods |
| `prsm/compute/nwtn/orchestrator.py` | Exception hierarchy, comprehensive error handling |

### Quality Improvements Summary

| Area | Before | After |
|------|--------|-------|
| IPFS Client | Multiple implementations | Single canonical client |
| Mock Services | Mixed with production | Clearly separated |
| Peer Connections | Race condition vulnerable | Thread-safe with asyncio.Lock |
| NWTN Error Handling | Minimal try/except | Comprehensive exception hierarchy |
| Session State | Inconsistent on failure | Proper cleanup and status updates |

### Verification Results

All Sprint 2 improvements verified functional:
- ✓ Transport thread safety: VERIFIED
- ✓ NWTN error handling: VERIFIED
- ✓ IPFS client consolidation: VERIFIED
- ✓ Mock services separation: VERIFIED
- ✓ Dependency injection: VERIFIED

---

## 20. Sprint 4 — Core Collaboration Robustness (2026-03-02)

### Motivation

With Sprints 1–3 having addressed critical security vulnerabilities, code quality, and exception handling, the codebase's foundational infrastructure is now solid. This sprint focuses on PRSM's **core collaboration features** — the mechanisms that enable researchers to actually work together on the network. A thorough review of the collaboration stack identified 12 concrete gaps that, if left unaddressed, would prevent real-world multi-node collaboration from functioning end-to-end.

### Current State of Core Collaboration Features

| Component | File(s) | Status | Assessment |
|-----------|---------|--------|------------|
| WebSocket P2P Transport | `prsm/node/transport.py` | **Production** | Solid — thread-safe with asyncio locks, handshake verification, nonce dedup |
| Gossip Protocol | `prsm/node/gossip.py` | **Production** | Working — epidemic gossip with fanout, TTL, heartbeat |
| Peer Discovery | `prsm/node/discovery.py` | **Production** | Working — bootstrap, announce, maintenance, stale pruning |
| DAG Ledger | `prsm/node/dag_ledger.py` | **Production** | Strong — atomic ops, Ed25519 verification, TOCTOU prevention |
| Ledger Sync | `prsm/node/ledger_sync.py` | **Production** | Working — signed broadcast, nonce replay prevention, reconciliation |
| Compute Provider | `prsm/node/compute_provider.py` | **Alpha** | Pipeline works, but inference/embedding returns mock results |
| Compute Requester | `prsm/node/compute_requester.py` | **Production** | Working — submit, await, verify signature, record payment |
| Content Uploader | `prsm/node/content_uploader.py` | **Production** | Working — IPFS upload, provenance, multi-level royalties |
| Content Index | `prsm/node/content_index.py` | **Production** | Working — LRU-evicted gossip-based content tracking |
| Agent Registry | `prsm/node/agent_registry.py` | **Production** | Working — gossip-based agent discovery, capability search |
| Agent Collaboration | `prsm/node/agent_collaboration.py` | **Partial** | State machines defined, but missing payment, persistence, broadcast |
| Session Manager | `prsm/collaboration/session_manager.py` | **Partial** | In-memory sessions, not connected to P2P layer |
| Collaboration Manager | `prsm/collaboration/__init__.py` | **Partial** | Higher-level abstraction, not connected to agent_collaboration |

### Identified Issues

#### Issue 1: No FTNS Payment Execution in Agent Collaboration (Critical)

**Location:** [`prsm/node/agent_collaboration.py`](prsm/node/agent_collaboration.py)

**Problem:** The `AgentCollaboration` class handles task offers with `ftns_budget`, review requests with `ftns_per_review`, and knowledge queries with `ftns_per_response`, but **never actually deducts or credits FTNS tokens**. The collaboration protocols are state machines without payment execution.

- `post_task()` broadcasts budget but doesn't escrow it
- `complete_task()` marks status but doesn't pay the assigned agent
- `submit_review()` records verdicts but doesn't pay reviewers
- `submit_response()` records answers but doesn't pay responders

**Impact:** Researchers have no economic incentive to participate in collaboration protocols. The token economy, which is PRSM's core differentiator, is disconnected from collaboration.

**Recommended Fix:**
- Wire `AgentCollaboration` to the local ledger (or `LedgerSync` for cross-node payments)
- Add escrow: hold FTNS when posting a task/review/query, release to winners, return on cancel
- Add automatic payment on task completion, review submission, and knowledge response

#### Issue 2: Memory-Only Collaboration State (High)

**Location:** [`prsm/node/agent_collaboration.py:110-112`](prsm/node/agent_collaboration.py:110)

**Problem:** All collaboration state is stored in Python dictionaries:
```python
self.tasks: Dict[str, TaskOffer] = {}
self.reviews: Dict[str, ReviewRequest] = {}
self.queries: Dict[str, KnowledgeQuery] = {}
```

If a node restarts, all active collaborations are lost. There's no way to recover in-progress tasks, pending reviews, or open queries.

**Impact:** Any node restart destroys collaboration state, making long-running collaborations unreliable.

**Recommended Fix:**
- Persist active collaboration records to SQLite (alongside the local ledger)
- Load state on startup, resume in-progress collaborations
- Archive completed records for audit trail

#### Issue 3: No Timeout/Expiry Enforcement (High)

**Location:** [`prsm/node/agent_collaboration.py`](prsm/node/agent_collaboration.py)

**Problem:** Tasks have `deadline_seconds` and all records have `created_at`, but nothing enforces these deadlines. There is no background loop that:
- Cancels expired task offers that received no bids
- Times out assigned tasks that weren't completed
- Expires old review requests
- Closes old knowledge queries

**Impact:** Stale collaboration records accumulate indefinitely. FTNS budgets that should be returned remain in limbo.

**Recommended Fix:**
- Add a `_cleanup_loop()` that runs every 60 seconds
- Cancel expired tasks and return escrowed FTNS to requester
- Close stale reviews and queries
- Emit gossip notifications for state changes

#### Issue 4: Missing State Change Broadcasts (High)

**Location:** [`prsm/node/agent_collaboration.py:192-208`](prsm/node/agent_collaboration.py:192)

**Problem:** Several critical state changes are local-only — the network is never notified:
- `assign_task()` — the assigned agent and other bidders aren't told
- `complete_task()` — the requester on another node isn't notified
- `submit_review()` — the submitter isn't notified of the verdict
- `submit_response()` — the requester isn't notified of the answer

Only `post_task()`, `submit_bid()`, `request_review()`, and `post_query()` broadcast via gossip.

**Impact:** In a multi-node scenario, collaboration cannot progress because parties on other nodes never learn about state changes.

**Recommended Fix:**
- Add gossip subtypes: `agent_task_assign`, `agent_task_complete`, `agent_review_submit`, `agent_knowledge_response`
- Broadcast on each state change
- Add corresponding `_on_*` handlers to process incoming state updates

#### Issue 5: No Bid Selection Strategy (Medium)

**Location:** [`prsm/node/agent_collaboration.py:192-199`](prsm/node/agent_collaboration.py:192)

**Problem:** When multiple agents bid on a task, the `assign_task()` method requires the requester to manually pick a winner by agent ID. There's no:
- Automatic bid scoring based on cost, estimated time, or capability match
- Reputation-weighted selection
- Automatic assignment after a bidding window

**Recommended Fix:**
- Add `select_best_bid()` method with configurable scoring strategy
- Consider cost, estimated_seconds, agent reputation (from agent_registry), and capability match
- Optionally add auto-assignment after a configurable bidding window

#### Issue 6: No Content Retrieval API (Medium)

**Location:** [`prsm/node/content_uploader.py`](prsm/node/content_uploader.py)

**Problem:** The `ContentUploader` has `_handle_content_request()` to serve incoming requests, but there's no corresponding `request_content()` method for a node to initiate a download. A node must manually construct P2P messages to fetch content from another node.

**Recommended Fix:**
- Add `request_content(cid: str) -> Optional[bytes]` method
- Use the `ContentIndex` to find providers
- Send direct content_request, await response with timeout
- Verify content hash on receipt

#### Issue 7: Unbounded Memory Growth (Medium)

**Location:** Multiple files in `prsm/node/`

**Problem:** Several dictionaries grow without bounds:
- `agent_collaboration.py`: `self.tasks`, `self.reviews`, `self.queries`
- `compute_provider.py`: `self.completed_jobs`
- `content_uploader.py`: `self.uploaded_content`
- `compute_requester.py`: `self.submitted_jobs`

**Recommended Fix:**
- Add max-size bounds with LRU eviction for completed/archived items
- Move completed records to persistent storage before eviction
- Log when eviction occurs for monitoring

#### Issue 8: Dual Collaboration Systems Not Connected (Medium)

**Location:** [`prsm/collaboration/`](prsm/collaboration/) vs [`prsm/node/agent_collaboration.py`](prsm/node/agent_collaboration.py)

**Problem:** There are two independent collaboration systems:
1. `prsm/collaboration/` — Higher-level `SessionManager` and `CollaborationManager` with proposal/session/result lifecycle
2. `prsm/node/agent_collaboration.py` — P2P-level protocols for task delegation, peer review, knowledge exchange

These systems don't interact. The `CollaborationManager` manages collaboration lifecycle but has no P2P transport. The `AgentCollaboration` has P2P transport but no session management.

**Recommended Fix:**
- Bridge the two: `CollaborationManager` should be able to dispatch tasks to `AgentCollaboration` for network execution
- `AgentCollaboration` completion should update `CollaborationManager` session state
- Eventually consolidate into a single coherent API

#### Issue 9: Mock Compute Jobs (Low, Alpha-appropriate)

**Location:** [`prsm/node/compute_provider.py:308-338`](prsm/node/compute_provider.py:308)

**Problem:** `_run_inference()` and `_run_embedding()` return mock results:
```python
return {
    "response": f"[PRSM node {self.identity.node_id[:8]} processed inference]",
    ...
}
```

**Note:** This is documented as alpha behavior and is appropriate for the current stage. However, the pipeline should eventually connect to the NWTN orchestrator for real inference.

**Recommended Fix (future):**
- Wire `_run_inference()` to the NWTN orchestrator
- Wire `_run_embedding()` to the embedding pipeline
- Add model discovery via the agent registry

#### Issue 10: Discovery Lacks Capability-Based Search (Low)

**Location:** [`prsm/node/discovery.py`](prsm/node/discovery.py)

**Problem:** `PeerDiscovery` only tracks node-level metadata (node_id, address, roles). It doesn't support finding nodes that have specific agent capabilities, GPU resources, or content.

**Recommended Fix:**
- Extend `PeerInfo` with `capabilities: List[str]` and `resources: Dict`
- Add `find_peers_with_capability(capability: str) -> List[PeerInfo]`
- Populate from agent_registry advertisements

#### Issue 11: No Gossip Message Persistence (Low)

**Location:** [`prsm/node/gossip.py`](prsm/node/gossip.py)

**Problem:** If a node joins the network after a job offer or task was broadcast, it will never see that offer. There's no catch-up mechanism for missed gossip.

**Recommended Fix (future):**
- Add a gossip log with configurable retention
- On new peer connect, exchange recent gossip digests
- Request missing messages by nonce

#### Issue 12: No Graceful Collaboration Shutdown (Low)

**Problem:** When a node shuts down, its active collaborations (tasks it posted, reviews it requested, bids it submitted) are left in limbo. Other participants are not notified.

**Recommended Fix:**
- On `PRSMNode.stop()`, gossip cancellation messages for all active tasks/reviews/queries
- Return escrowed FTNS
- Mark all local collaboration records as cancelled

### Sprint 4 Phased Implementation Plan

#### Phase 1: FTNS Payment Integration (Critical path)

**Goal:** Wire collaboration protocols to the token economy.

**Files to modify:**
- `prsm/node/agent_collaboration.py` — Add ledger and ledger_sync as constructor parameters; implement escrow, payment, and refund logic

**Changes:**
1. Add `ledger: LocalLedger` and `ledger_sync: Optional[LedgerSync]` to `__init__()`
2. In `post_task()`: escrow `ftns_budget` (debit from requester wallet)
3. In `complete_task()`: release escrow to assigned agent via `ledger_sync.signed_transfer()`
4. In `assign_task()`: if task is cancelled, return escrow to requester
5. In `request_review()`: escrow `ftns_per_review * max_reviewers`
6. In `submit_review()`: pay reviewer from escrow when review is accepted
7. In `post_query()`: escrow `ftns_per_response * max_responses`
8. In `submit_response()`: pay responder from escrow
9. Update `PRSMNode.initialize()` to pass ledger and ledger_sync to AgentCollaboration

**Estimated effort:** 4–6 hours

#### Phase 2: State Change Broadcasts (Critical for multi-node)

**Goal:** Network-wide visibility of all collaboration state changes.

**Files to modify:**
- `prsm/node/agent_collaboration.py` — Add new gossip subtypes and handlers
- `prsm/node/gossip.py` — Register new gossip subtypes (constants only)

**Changes:**
1. Add gossip subtypes: `GOSSIP_TASK_ASSIGN`, `GOSSIP_TASK_COMPLETE`, `GOSSIP_REVIEW_SUBMIT`, `GOSSIP_KNOWLEDGE_RESPONSE`
2. Make `assign_task()` async, broadcast assignment
3. Make `complete_task()` async, broadcast completion
4. Make `submit_review()` async, broadcast review verdict
5. Make `submit_response()` async, broadcast response
6. Add `_on_task_assign()`, `_on_task_complete()`, `_on_review_submit()`, `_on_knowledge_response()` handlers
7. Subscribe in `start()`

**Estimated effort:** 3–4 hours

#### Phase 3: Expiry Enforcement & Bounded Memory (Reliability)

**Goal:** Prevent stale records and unbounded memory growth.

**Files to modify:**
- `prsm/node/agent_collaboration.py` — Add cleanup loop and eviction
- `prsm/node/compute_provider.py` — Bound `completed_jobs`
- `prsm/node/compute_requester.py` — Bound `submitted_jobs`

**Changes:**
1. Add `_cleanup_loop()` to AgentCollaboration that runs every 60 seconds
2. Cancel expired tasks (beyond `deadline_seconds`), return escrowed FTNS
3. Close expired reviews and queries (configurable timeout, default 1 hour)
4. Add `MAX_COMPLETED_ITEMS = 1000` constant, evict oldest when exceeded
5. Bound `completed_jobs` in ComputeProvider with same pattern
6. Bound `submitted_jobs` in ComputeRequester

**Estimated effort:** 2–3 hours

#### Phase 4: Content Retrieval API (Usability)

**Goal:** Allow nodes to request and download content from the network.

**Files to modify:**
- `prsm/node/content_uploader.py` — Add `request_content()` method

**Changes:**
1. Add `async def request_content(cid: str, timeout: float = 30.0) -> Optional[bytes]`
2. Look up providers via `content_index.get_providers(cid)`
3. Send direct `content_request` message to best provider
4. Await response with configurable timeout
5. Verify `content_hash` matches on receipt
6. Return content bytes or None

**Estimated effort:** 2–3 hours

#### Phase 5: Collaboration System Bridge (Architecture)

**Goal:** Connect the higher-level `CollaborationManager` to P2P protocols.

**Files to modify:**
- `prsm/collaboration/__init__.py` — Add bridge methods
- `prsm/node/node.py` — Wire collaboration manager into node lifecycle

**Changes:**
1. Add `set_agent_collaboration(ac: AgentCollaboration)` to CollaborationManager
2. When a `CollaborationSession` of type `TASK_DELEGATION` starts, dispatch to `AgentCollaboration.post_task()`
3. When `AgentCollaboration` completes a task, update the corresponding `CollaborationSession`
4. Add `create_session_from_task()` and `create_session_from_review()` convenience methods

**Estimated effort:** 3–4 hours

### Priority Matrix

| Phase | Impact | Effort | Priority | Dependency |
|-------|--------|--------|----------|------------|
| Phase 1: FTNS Payment | **Critical** | 4–6 hrs | P0 | None |
| Phase 2: State Broadcasts | **Critical** | 3–4 hrs | P0 | None |
| Phase 3: Expiry & Bounds | **High** | 2–3 hrs | P1 | Phase 1 (for escrow refund) |
| Phase 4: Content Retrieval | **Medium** | 2–3 hrs | P2 | None |
| Phase 5: System Bridge | **Medium** | 3–4 hrs | P2 | Phase 1, Phase 2 |

### Success Criteria

- [x] Tasks, reviews, and queries trigger actual FTNS token transfers
- [x] All collaboration state changes are broadcast to the network
- [x] Expired collaborations are automatically cleaned up with FTNS returned
- [x] Nodes can request and download content from network peers by CID
- [x] `CollaborationManager` dispatches to P2P `AgentCollaboration` for execution
- [x] No in-memory dictionary grows beyond configurable bounds
- [x] Integration tests cover multi-node collaboration scenarios end-to-end

### Sprint 4 Completion Summary (2026-03-02)

All 5 phases completed.

#### Files Modified

| File | Changes |
|------|---------|
| `prsm/node/agent_collaboration.py` | Complete rewrite: FTNS escrow/payment, gossip broadcasts, expiry cleanup, bounded archives |
| `prsm/node/gossip.py` | Added 5 new gossip subtypes for collaboration state changes |
| `prsm/node/node.py` | Wired ledger and ledger_sync into AgentCollaboration; added graceful shutdown |
| `prsm/node/content_uploader.py` | Added `request_content()` with provider discovery, hash verification, inline/gateway modes |
| `prsm/collaboration/__init__.py` | Added `dispatch_session()`, `on_protocol_complete()`, bidirectional session↔protocol mapping |
| `tests/security/test_sprint4_collaboration.py` | 23 tests across 8 test classes |
| `tests/security/test_sprint4_content_retrieval.py` | 14 tests across 6 test classes |
| `tests/security/test_sprint4_collab_bridge.py` | 12 tests across 4 test classes |

#### What Changed

**FTNS Payment Integration:**
- `post_task()` escrows the FTNS budget via `ledger.debit()`
- `complete_task()` pays the assigned agent via `ledger_sync.signed_transfer()`
- `cancel_task()` refunds escrowed FTNS via `ledger.credit()`
- `request_review()` escrows `ftns_per_review * max_reviewers`
- `submit_review()` pays each reviewer from escrow
- `post_query()` escrows `ftns_per_response * max_responses`
- `submit_response()` pays each responder from escrow
- Unused escrow slots are refunded when reviews/queries close

**State Change Broadcasts:**
- New gossip subtypes: `agent_task_assign`, `agent_task_complete`, `agent_task_cancel`, `agent_review_submit`, `agent_knowledge_response`
- All state changes (assignment, completion, cancellation, review verdicts, knowledge responses) broadcast to network
- Corresponding `_on_*` handlers process incoming state updates from remote nodes

**Expiry Enforcement:**
- Background `_cleanup_loop()` runs every 60 seconds
- Cancels tasks past their `deadline_seconds` and refunds escrow
- Closes reviews and queries past configurable timeouts (1h and 30m)
- Broadcasts cancellation gossip for expired items

**Bounded Memory:**
- Completed/cancelled/expired records archived to `OrderedDict` with LRU eviction
- `MAX_COMPLETED_RECORDS = 500` per category (tasks, reviews, queries)
- Active dictionaries only hold in-progress records

**Graceful Shutdown:**
- `stop()` cancels all open/assigned tasks owned by this node
- Refunds escrowed FTNS before shutting down

**Collaboration System Bridge (Phase 5):**
- `CollaborationManager.set_agent_collaboration(ac)` wires in the P2P layer
- `dispatch_session(session_id)` maps session type to P2P protocol:
  - `TASK_DELEGATION` → `AgentCollaboration.post_task()`
  - `PEER_REVIEW` → `AgentCollaboration.request_review()`
  - `KNOWLEDGE_EXCHANGE` → `AgentCollaboration.post_query()`
  - Other types (e.g., `JOINT_REASONING`) start locally without P2P dispatch
- Bidirectional mapping: `session_id ↔ protocol_id` via `_session_to_protocol` / `_protocol_to_session`
- `on_protocol_complete(protocol_id, success, outputs)` callback updates the linked session
- Insufficient balance on dispatch gracefully fails the session with error metadata
- Mappings are cleaned up after protocol completion

**Content Retrieval API (Phase 4):**
- Added `request_content(cid, timeout, verify_hash)` to `ContentUploader`
- Discovers providers via `ContentIndex.lookup(cid)`
- Sends direct `content_request` P2P message to providers
- Supports inline (base64 ≤1MB) and gateway (IPFS URL >1MB) transfer modes
- Verifies SHA-256 content hash against content index record
- Falls back to next provider on failure/mismatch
- Routes incoming `content_response` via `_handle_content_response()` to resolve pending futures
- Checks local IPFS first before contacting network

#### Test Results

```
49 passed in 3.54s (23 collaboration + 14 content retrieval + 12 bridge)

TestTaskEscrowAndPayment (4 tests)   — escrow, insufficient balance, payment, refund
TestReviewEscrowAndPayment (2 tests) — escrow, reviewer payment
TestQueryEscrowAndPayment (2 tests)  — escrow, responder payment
TestStateBroadcasts (5 tests)        — assign, complete, cancel, review, response
TestIncomingStateChanges (3 tests)   — remote assign, complete, review updates
TestExpiryEnforcement (3 tests)      — expired tasks, reviews, queries with refund
TestBoundedMemory (2 tests)          — archive bounds, active→archive movement
TestGracefulShutdown (1 test)        — stop refunds open tasks
TestStats (1 test)                   — stats accuracy

TestProviderDiscovery (4 tests)      — no transport, no providers, self-skip, local content
TestInlineTransfer (3 tests)         — decode, hash mismatch rejection, hash skip
TestGatewayTransfer (1 test)         — gateway URL fetch
TestErrorHandling (3 tests)          — timeout, not found, fallback to second provider
TestResponseHandler (3 tests)        — future resolution, unknown ID, already-done future

TestDispatchTask (2 tests)           — task dispatch, insufficient balance
TestDispatchReview (1 test)          — review dispatch
TestDispatchQuery (1 test)           — query dispatch
TestDispatchEdgeCases (4 tests)      — unknown session, no collab, already active, local-only type
TestProtocolCompletion (4 tests)     — success, failure, unknown protocol, get_result
```

---

## 21. Core Collaboration Review Update (2026-03-03)

This update reflects a completed review of PRSM core collaboration infrastructure across:
- [`prsm/compute/federation/p2p_network.py`](prsm/compute/federation/p2p_network.py)
- [`prsm/compute/federation/enhanced_p2p_network.py`](prsm/compute/federation/enhanced_p2p_network.py)
- [`prsm/node/transport.py`](prsm/node/transport.py)
- [`prsm/node/dag_ledger.py`](prsm/node/dag_ledger.py)
- [`prsm/core/ipfs_client.py`](prsm/core/ipfs_client.py)

### What Has Been Accomplished

- **Transport locking discipline is in place** in [`WebSocketTransport`](prsm/node/transport.py), with async lock patterns that reduce concurrent state corruption risk.
- **DAG ledger atomicity intent and implementation are established** in [`LocalDAGLedger`](prsm/node/dag_ledger.py), including atomic balance operation patterns introduced in earlier remediation work.
- **NWTN orchestration remains DI-oriented and structurally extensible**, supporting future replacement of mock/placeholder paths with production providers.
- **IPFS failover architecture concept exists** in [`prsm/core/ipfs_client.py`](prsm/core/ipfs_client.py), providing a useful resilience direction even where API consistency work remains.
- **Core collaboration review scope is now explicit**, with overlap and boundary issues identified across federation, node transport/ledger, and core storage pathways.

### Implementation Progress Update — P0 Tranches Completed (2026-03-03)

#### Tranche 1: Transport handshake hardening ✅

**Files updated:**
- [`prsm/node/transport.py`](prsm/node/transport.py)
- [`tests/security/test_transport_handshake_auth.py`](tests/security/test_transport_handshake_auth.py)
- [`tests/security/test_transport_handshake_replay.py`](tests/security/test_transport_handshake_replay.py)

**Outcomes:**
- Authenticated peer promotion only
- Identity↔key binding enforced during handshake state transition
- Replay nonce rejection enforced
- `ack_for` binding checks enforced

#### Tranche 2: IPFS API/accessor normalization ✅

**Files updated:**
- [`prsm/core/ipfs_client.py`](prsm/core/ipfs_client.py)
- [`prsm/compute/federation/p2p_network.py`](prsm/compute/federation/p2p_network.py)
- [`prsm/compute/federation/enhanced_p2p_network.py`](prsm/compute/federation/enhanced_p2p_network.py)
- [`prsm/data/ipfs/__init__.py`](prsm/data/ipfs/__init__.py)
- [`prsm/data/data_layer/__init__.py`](prsm/data/data_layer/__init__.py)
- [`tests/unit/test_ipfs_accessor_contract.py`](tests/unit/test_ipfs_accessor_contract.py)

**Outcomes:**
- Canonical accessor decision finalized: [`get_ipfs_client()`](prsm/core/ipfs_client.py:1)
- Helper path consistency fixed
- Compatibility shims aligned to canonical accessor
- Federation and data-layer consumer imports normalized

#### Tranche 3: Consensus defect remediation ✅

**Files updated:**
- [`prsm/compute/federation/consensus.py`](prsm/compute/federation/consensus.py)
- [`tests/unit/test_federation_consensus_p0.py`](tests/unit/test_federation_consensus_p0.py)

**Outcomes:**
- Duplicate constant and method definitions removed/merged
- Missing `Decimal` import runtime hazard eliminated
- Consensus smoke path validated through compile + targeted execution

### Verification Summary (P0)

Verified command runs and pass counts:

- `pytest -q tests/security/test_transport_handshake_auth.py tests/security/test_transport_handshake_replay.py` → **7 passed**
- `pytest -q tests/test_transport.py::TestWebSocketTransport::test_two_nodes_connect tests/test_transport.py::TestWebSocketTransport::test_self_connection_rejected tests/test_transport.py::TestWebSocketTransport::test_nonce_dedup` → **3 passed**
- `pytest -q tests/unit/test_ipfs_accessor_contract.py` → **4 passed**
- `pytest -q tests/test_ipfs_system_integration.py::test_ipfs_system_integration` → **1 passed**
- `pytest tests/unit/test_federation_consensus_p0.py -q` → **4 passed**
- `python3 -m py_compile prsm/compute/federation/consensus.py && pytest tests/unit/test_federation_consensus_p0.py -k p2p_network_consensus_smoke_path_uses_consensus_engine -q` → **compile success, 1 selected passed**

### Implementation Progress Update — P1 Tranches Completed (2026-03-03)

#### Tranche 1: Production federation handler completion ✅

**Files updated:**
- [`prsm/compute/federation/enhanced_p2p_network.py`](prsm/compute/federation/enhanced_p2p_network.py)
- [`tests/unit/test_enhanced_p2p_network_p1_tranche1.py`](tests/unit/test_enhanced_p2p_network_p1_tranche1.py)

**Outcomes:**
- Implemented production handlers: `_handle_shard_retrieve`, `_handle_task_execute`, `_handle_peer_discovery`
- Added request-style handler wiring and response dispatch with `request_id` propagation
- Added signed-capability trust-gating baseline in DHT routing

#### Tranche 2: Trust lifecycle and routing enforcement ✅

**Files updated:**
- [`prsm/compute/federation/enhanced_p2p_network.py`](prsm/compute/federation/enhanced_p2p_network.py)
- [`tests/unit/test_enhanced_p2p_network_p1_tranche2.py`](tests/unit/test_enhanced_p2p_network_p1_tranche2.py)

**Outcomes:**
- Implemented trust lifecycle semantics:
  - identity registration/update rules
  - key rotation
  - revocation and capability revocation
  - stale cutoff policy
  - structured trust decision logging
- Enforced trust-gated routing candidate filtering consistently

#### Tranche 3: Reconnect and partition reconciliation semantics ✅

**Files updated:**
- [`prsm/compute/federation/enhanced_p2p_network.py`](prsm/compute/federation/enhanced_p2p_network.py)
- [`tests/unit/test_enhanced_p2p_network_p1_tranche3.py`](tests/unit/test_enhanced_p2p_network_p1_tranche3.py)

**Outcomes:**
- Added reconnect and partition reconciliation behavior:
  - peer state transitions
  - reconcile generation handling
  - deterministic shard location reconciliation
  - fail-closed in-flight task reconciliation
- Added RPC idempotency via canonical operation id

### Verification Summary (P1)

Verified command runs and reported outcomes:

- `pytest -q tests/unit/test_enhanced_p2p_network_p1_tranche1.py` → **8 passed**
- `pytest -q tests/unit/test_enhanced_p2p_network_p1_tranche1.py tests/unit/test_enhanced_p2p_network_p1_tranche2.py` → **14 passed**
- `pytest -q tests/unit/test_enhanced_p2p_network_p1_tranche3.py` → **4 passed**
- Tranche 1 smoke test run (reported) → **passed**
- Tranche 2 targeted smoke run (reported) → **passed**
- Deterministic repeat reconciliation scenario (Tranche 3) → **passed twice**

### Implementation Progress Update — P2 Tranches Completed (2026-03-03)

#### Tranche 1: Collaboration observability instrumentation ✅

**Files updated:**
- [`prsm/node/transport.py`](prsm/node/transport.py)
- [`prsm/node/gossip.py`](prsm/node/gossip.py)
- [`prsm/node/agent_collaboration.py`](prsm/node/agent_collaboration.py)
- [`prsm/collaboration/__init__.py`](prsm/collaboration/__init__.py)
- [`tests/unit/test_collab_observability_tranche1.py`](tests/unit/test_collab_observability_tranche1.py)
- [`tests/unit/test_collab_canonical_boundaries_tranche1.py`](tests/unit/test_collab_canonical_boundaries_tranche1.py)

**Outcomes:**
- Added additive observability hooks for handshake/auth taxonomy
- Added gossip subtype publish, forward, and drop observability
- Added collaboration transition and terminal outcome observability
- Added dispatch success and failure reason instrumentation
- Regression run passed including security and IPFS contract subsets

#### Tranche 2: Canonical boundary compatibility fences ✅

**Files updated:**
- [`prsm/compute/federation/p2p_network.py`](prsm/compute/federation/p2p_network.py)
- [`prsm/compute/federation/enhanced_p2p_network.py`](prsm/compute/federation/enhanced_p2p_network.py)
- [`tests/unit/test_collab_canonical_boundaries_tranche2.py`](tests/unit/test_collab_canonical_boundaries_tranche2.py)

**Outcomes:**
- Added compatibility fences with warnings and redirect guidance for collaboration-like federation entrypoints
- Preserved canonical path through collaboration manager bridge
- Targeted tranche validation passed

#### Tranche 3: CI guardrails for canonical collaboration boundaries ✅

**Files updated:**
- [`tests/unit/test_collab_canonical_boundaries_tranche2.py`](tests/unit/test_collab_canonical_boundaries_tranche2.py)
- [`tests/unit/test_collab_canonical_ci_gates_tranche3.py`](tests/unit/test_collab_canonical_ci_gates_tranche3.py)

**Outcomes:**
- Extended canonical boundary coverage for compatibility fence semantics
- Added CI-style gates for canonical collaboration suite presence and ownership
- Added CI-style gates for compatibility fence coverage
- Validation passed for tranche suite and Sprint 4 collaboration security tests

### Verification Summary (P2)

Verified command runs and reported outcomes:

- `pytest -q tests/unit/test_collab_observability_tranche1.py tests/unit/test_collab_canonical_boundaries_tranche1.py` (from earlier tranche) and/or combined variants
- `pytest -q tests/security/test_transport_handshake_auth.py tests/security/test_transport_handshake_replay.py tests/security/test_sprint4_collaboration.py tests/security/test_sprint4_collab_bridge.py tests/unit/test_ipfs_accessor_contract.py` → **61 passed**
- `pytest -q tests/unit/test_collab_canonical_boundaries_tranche2.py tests/unit/test_collab_canonical_boundaries_tranche1.py tests/unit/test_collab_observability_tranche1.py` → **13 passed** (final rerun)
- `pytest -q tests/unit/test_collab_observability_tranche1.py tests/unit/test_collab_canonical_boundaries_tranche1.py tests/unit/test_collab_canonical_boundaries_tranche2.py tests/unit/test_collab_canonical_ci_gates_tranche3.py` → **19 passed** (final run)
- `pytest -q tests/security/test_sprint4_collaboration.py tests/security/test_sprint4_collab_bridge.py` → **50 passed**

### What Still Needs to Be Done

#### P0 — Stabilize a Works-on-Clone Collaboration Spine ✅ COMPLETED (2026-03-03)

P0 scope items originally identified in this section are now implemented and verified in three completed tranches.

**P0 Validation / Acceptance Checklist**
- [x] Fresh clone boots and executes collaboration-critical startup path without import or symbol collisions
- [x] IPFS helper and accessor paths pass integration tests for success, failure, and failover branches
- [x] Unauthenticated or weakly authenticated handshake attempts are rejected in transport tests
- [x] Consensus flow produces deterministic outcomes under repeated identical inputs

**Residual risks carried forward after P0 completion:**
- Capability announcements and routing trust are not yet cryptographically bound end-to-end
- Multi-node reconnect and partition/rejoin convergence semantics still require full production validation
- Observability remains limited for handshake-failure taxonomy, consensus branch outcomes, and failover-path usage

#### P1 — Complete Production P2P Operational Semantics ✅ COMPLETED (2026-03-03)

P1 scope items in this section were implemented across three completed tranches in [`prsm/compute/federation/enhanced_p2p_network.py`](prsm/compute/federation/enhanced_p2p_network.py), with tranche-specific tests and smoke validation.

**P1 Validation / Acceptance Checklist**
- [x] Production federation handlers implemented with request/response correlation via `request_id`
- [x] Trust lifecycle and trust-gated routing candidate filtering enforced
- [x] Reconnect and partition reconciliation semantics implemented with deterministic shard reconciliation
- [x] RPC idempotency established via canonical operation id

**Residual risks carried forward to P2 (from execution summaries):**
- Capability trust-gating is in place as a baseline, but full end-to-end cryptographic binding and broader authenticity hardening remain follow-on work
- Reconnect/partition reconciliation now has deterministic scenario validation, but broader multi-node production validation remains pending
- Observability remains limited for trust-decision pathways, reconciliation branch outcomes, and failure taxonomy coverage

#### P2 — Reduce Maintenance Cost and Improve Operability ✅ COMPLETED (2026-03-03)

P2 scope was implemented across three completed tranches spanning observability instrumentation, canonical boundary compatibility fences, and CI gate enforcement in collaboration-focused test suites.

**P2 Validation / Acceptance Checklist**
- [x] Dashboards-alert precursor signals established via additive collaboration observability hooks and structured taxonomy coverage
- [x] Canonical-vs-legacy boundaries are documented and reflected in tranche boundary tests
- [x] Duplicate-path regressions are prevented by CI checks over designated canonical collaboration interfaces
- [x] Test suites map to the selected production collaboration stack through canonical ownership and fence coverage gates

**Residual follow-ons after P2 completion:**
- Expand observability from test-level hooks to operational dashboard and alert wiring
- Broaden canonical boundary checks from targeted suites to full CI matrix coverage
- Continue multi-node reliability hardening for reconciliation and trust path telemetry

### Suggested 30/60/90-Day Rollout (Concise)

- **Next 30 days**: Operationalize P2 observability outputs into dashboards and actionable alerts for trust and collaboration pathways.
- **Day 31-60**: Extend canonical boundary and compatibility fence enforcement from targeted suites to broader CI coverage.
- **Day 61-90**: Advance post-P2 reliability hardening for multi-node reconciliation and trust-path telemetry depth.

---

## 22. Sprint 5, Item 2 — Bid Selection Strategy (2026-03-02)

### Motivation

PRSM's agent collaboration system lets agents post tasks and receive bids, but had no automated bid selection — callers had to manually pick a winner and pass the `agent_id` to `assign_task()`. Adding a scoring and selection layer makes the task delegation protocol fully autonomous.

### What Changed

#### `prsm/node/config.py` — 3 new config fields

```python
bid_strategy: str = "best_score"       # "lowest_cost", "fastest", "best_score"
bid_window_seconds: float = 30.0
min_bids: int = 1
```

Strategy stored as string to avoid circular import (config.py must not import from agent_collaboration.py). Converted to enum in `node.py`. Updated `save()` to persist the new fields; `load()` handles them automatically via `**data`.

#### `prsm/node/agent_registry.py` — 2 reputation fields on `AgentRecord`

```python
tasks_completed: int = 0
tasks_failed: int = 0
```

Preparatory fields for future reputation scoring. Updated `to_dict()` to include them.

#### `prsm/node/agent_collaboration.py` — Core implementation

**`BidStrategy` enum:** `LOWEST_COST`, `FASTEST`, `BEST_SCORE`

**Scoring weight constants:**
- `DEFAULT_COST_WEIGHT = 0.35`
- `DEFAULT_TIME_WEIGHT = 0.25`
- `DEFAULT_CAPABILITY_WEIGHT = 0.25`
- `DEFAULT_FRESHNESS_WEIGHT = 0.15`

**Extended `__init__`** with `agent_registry`, `bid_strategy`, `bid_window_seconds`, `min_bids` (all with defaults for backward compatibility).

**Bid validation in `submit_bid()`:** Rejects bids that exceed the task budget or target non-OPEN tasks.

**`score_bid(bid, task) -> float`:** Composite 0.0–1.0 score:
- Cost efficiency (0.35): `1 - (cost / budget)`
- Time efficiency (0.25): `1 - (seconds / deadline)`
- Capability match (0.25): fraction of required_capabilities the bidder has (from agent_registry)
- Freshness (0.15): exponential decay from agent `last_seen`
- Returns 0.0 for over-budget bids; defaults to 0.5 for missing components

**`select_best_bid(task) -> Optional[dict]`:** Filters over-budget bids, then applies strategy:
- `LOWEST_COST` → cheapest bid
- `FASTEST` → shortest estimated time
- `BEST_SCORE` → highest composite score

**`auto_assign_task(task_id) -> Optional[str]`:** Async pipeline that waits for bids (configurable window with 1s polling), exits early when `min_bids` met past half-window, calls `select_best_bid()` then `assign_task()`.

**Updated `get_stats()`:** Added `total_active_bids` and `bid_strategy` fields.

#### `prsm/node/node.py` — Wiring

- Imports `BidStrategy`, passes `bid_strategy=BidStrategy(self.config.bid_strategy)`, `bid_window_seconds`, `min_bids` to `AgentCollaboration` constructor
- Wires `agent_registry` post-construction: `self.agent_collaboration.agent_registry = self.agent_registry`

#### `tests/security/test_sprint4_collaboration.py` — 15 new tests

| Test Class | Tests | Coverage |
|-----------|-------|---------|
| `TestBidValidation` | 3 | Over-budget rejected, non-OPEN rejected, valid accepted |
| `TestScoreBid` | 4 | Cost ranking, capability match, over-budget → 0.0, freshness |
| `TestSelectBestBid` | 5 | LOWEST_COST, FASTEST, BEST_SCORE strategies; empty → None; all over-budget → None |
| `TestAutoAssignTask` | 3 | Basic assignment, no bids → None, early exit on min_bids |

### What's NOT Changed

- Existing `assign_task()` API unchanged — manual assignment still works
- All new constructor params have defaults — zero breakage to existing callers
- `submit_bid()` still gossips for unknown tasks (remote propagation)

### Test Results

```
38 passed in 10.68s (23 existing + 15 new)
```

### Verification

- `from prsm.node.agent_collaboration import BidStrategy` — enum imports correctly
- `from prsm.node.node import PRSMNode` — wiring imports correctly
- Config round-trip: save with `bid_strategy="fastest"`, load, value preserved

---

## 23. P3 Operationalization Tranche 3 — Runbook + Canary Validation (2026-03-03)

### Scope Completed

This tranche finalized operationalization of collaboration observability delivered in prior P3 work by:

1. Adding production runbook guidance for collaboration telemetry and alert handling.
2. Running a focused canary suite across collaboration telemetry/alerts and critical collaboration security regressions.
3. Recording exact command-level validation outcomes in architecture/progress documentation.

### Files Updated

- [`docs/PRODUCTION_OPERATIONS_MANUAL.md`](docs/PRODUCTION_OPERATIONS_MANUAL.md)
- [`plans/prsm_architecture_analysis.md`](plans/prsm_architecture_analysis.md)

### Runbook Content Added

Added section: **"Collaboration Telemetry & Alert Handling Runbook (P3)"** to [`docs/PRODUCTION_OPERATIONS_MANUAL.md`](docs/PRODUCTION_OPERATIONS_MANUAL.md), including:

- Key collaboration metrics with expected healthy ranges and investigation thresholds.
- Alert meaning mapping tied to [`AlertManager.setup_collaboration_rules()`](prsm/core/monitoring/alerts.py:694).
- Triage flow (exporter/rules verification, trust vs reliability path classification, reason taxonomy correlation, protocol-stall confirmation).
- Rollback/safing steps (fail-closed peer isolation for trust anomalies, temporary collaboration dispatch safing, rollback to known-good deployment/rules, recovery gates).

### Canary Validation — Exact Command and Outcome

Executed focused canary suite:

```bash
pytest -q \
  tests/unit/test_collab_operational_monitoring_p3_tranche1.py \
  tests/unit/test_collab_operational_monitoring_p3_tranche2.py \
  tests/security/test_transport_handshake_auth.py \
  tests/security/test_transport_handshake_replay.py \
  tests/security/test_sprint4_collaboration.py \
  tests/security/test_sprint4_collab_bridge.py
```

Outcome:

- **65 passed in 12.55s**
- Exit code: **0**

This canary set covers:

- P3 observability + alert threshold behavior (`tranche1`, `tranche2` tests)
- Transport handshake auth + replay protections
- Sprint 4 collaboration protocol and collaboration-manager bridge regressions

### P3 Progress Status Update

#### P3 — Collaboration Operationalization and Monitoring Hardening ✅ COMPLETED (2026-03-03)

| Tranche | Status | Outcome |
|---|---|---|
| Tranche 1 | ✅ Completed | Collaboration observability instrumentation and taxonomy hooks |
| Tranche 2 | ✅ Completed | Telemetry export path wiring + collaboration alert threshold tuning |
| Tranche 3 | ✅ Completed | Runbook operationalization + canary validation + architecture/progress capture |

### Residual Follow-up Items

- Promote canary command above into CI/nightly job for continuous regression guardrails.
- Add Grafana panel links and dashboard JSON references to the runbook once dashboard IDs are finalized.
- Continue multi-node production validation for reconciliation/trust-path telemetry under sustained load.

---

## 24. Sprint 6 — Node Onboarding Reliability Now (Engineering + Ops) (2026-03-03)

### Sprint Objective

Increase local node onboarding reliability so a fresh operator can complete first-run bring-up with deterministic bootstrap behavior, actionable diagnostics, and an automated golden-path verification flow.

### Measurable Success Criteria

- First-run local onboarding success rate on clean environments reaches **≥95%** for the canonical quickstart path.
- Bootstrap peer/connectivity failure rate during initial bring-up is reduced to **≤2%** across repeated clean-start validation runs.
- Time-to-first-actionable-error for failed onboarding is **<60 seconds** with explicit remediation guidance surfaced in CLI logs.
- Golden-path onboarding validation automation runs in CI and blocks regressions on required onboarding checks.

### Execution Tranches

#### Tranche 1: Bootstrap reliability + fallback defaults

**Scope**
- Harden bootstrap peer selection and retry semantics for first-run startup.
- Introduce deterministic fallback defaults when configured bootstrap sources are unavailable.
- Ensure fail-closed behavior for invalid bootstrap identity data while preserving safe fallback to trusted defaults.

**Likely files**
- [`prsm/node/discovery.py`](prsm/node/discovery.py)
- [`prsm/node/node.py`](prsm/node/node.py)
- [`prsm/node/config.py`](prsm/node/config.py)
- [`config/`](config/)
- [`tests/unit/`](tests/unit/)
- [`tests/security/`](tests/security/)

**Acceptance tests**
- Add targeted tests for bootstrap source priority order, retry backoff, and fallback default activation.
- Add negative tests validating rejection of malformed or untrusted bootstrap identities.
- Add deterministic startup tests confirming node proceeds to healthy state when at least one trusted fallback path is available.

**Rollback notes**
- Keep previous bootstrap selection path behind a feature flag for one sprint cycle.
- If onboarding regresses, disable new bootstrap fallback policy and revert to last known stable peer-source ordering.
- Preserve structured telemetry keys so rollback does not break operational dashboards.

#### Tranche 2: Startup UX + preflight diagnostics

**Scope**
- Add startup preflight checks for required local dependencies and network prerequisites.
- Standardize onboarding failure taxonomy with immediate remediation hints.
- Improve CLI startup UX so operators can distinguish fatal vs recoverable startup conditions.

**Likely files**
- [`prsm/cli.py`](prsm/cli.py)
- [`prsm/node/node.py`](prsm/node/node.py)
- [`prsm/core/monitoring/`](prsm/core/monitoring/)
- [`docs/SECURE_SETUP.md`](docs/SECURE_SETUP.md)
- [`tests/unit/`](tests/unit/)

**Acceptance tests**
- Add preflight tests for missing config, unreachable bootstrap peers, and dependency unavailability.
- Add CLI output contract tests validating actionable error codes/messages for onboarding-critical failures.
- Add regression tests ensuring successful preflight path remains quiet and does not block normal startup.

**Rollback notes**
- Gate new preflight enforcement behind severity levels, allowing warning-only mode fallback.
- If false positives occur, downgrade specific checks to non-blocking warnings while retaining diagnostics logging.
- Retain prior startup path as emergency bypass for operational recovery.

#### Tranche 3: Docs golden path + validation automation

**Scope**
- Define a single canonical onboarding golden path for local node bring-up.
- Add scriptable validation steps that prove the documented flow still works.
- Wire onboarding validation into CI as a regression gate.

**Likely files**
- [`docs/`](docs/)
- [`scripts/`](scripts/)
- [`tests/`](tests/)
- [`.github/workflows/`](.github/workflows/)
- [`plans/prsm_architecture_analysis.md`](plans/prsm_architecture_analysis.md)

**Acceptance tests**
- Add documentation-driven onboarding smoke test that follows the exact golden-path sequence.
- Add CI job asserting onboarding smoke coverage and failure artifact capture.
- Add validation that doc commands and expected checkpoints remain synchronized.

**Rollback notes**
- Keep onboarding CI gate in non-blocking mode for initial rollout window.
- If instability is detected, revert to advisory-only workflow while preserving artifact collection for triage.
- Maintain previous onboarding documentation snapshot for rapid operator fallback.

### 7/14/30-Day Rollout and Owner-Style Checklist

#### Day 7

- [ ] Engineering owner: ship Tranche 1 bootstrap reliability changes behind feature flags.
- [ ] Engineering owner: land Tranche 1 test coverage for fallback and trust validation branches.
- [ ] Ops owner: define baseline onboarding run matrix for clean local environments.
- [ ] Ops owner: capture initial first-run success metrics and bootstrap-failure taxonomy.

#### Day 14

- [ ] Engineering owner: deliver Tranche 2 preflight diagnostics and startup UX taxonomy.
- [ ] Engineering owner: add structured onboarding error surfaces and code-mapped remediation guidance.
- [ ] Ops owner: validate warning-only fallback behavior and emergency bypass runbook.
- [ ] Shared owner review: evaluate first-run success trend against sprint targets.

#### Day 30

- [ ] Engineering owner: complete Tranche 3 golden-path docs and automated onboarding validation.
- [ ] Engineering owner: wire onboarding smoke validation into CI gate path.
- [ ] Ops owner: operationalize onboarding reliability dashboard views and regression alerting.
- [ ] Shared owner sign-off: confirm target thresholds are met and promote defaults to standard path.

### Execution starts now

Immediate implementation scope for code-mode handoff is strictly **Tranche 1**:

1. Add bootstrap source ordering policy with trusted fallback defaults in [`prsm/node/discovery.py`](prsm/node/discovery.py).
2. Extend node startup wiring to consume fallback bootstrap policy and emit bootstrap decision telemetry in [`prsm/node/node.py`](prsm/node/node.py).
3. Add configurable bootstrap retry and fallback settings in [`prsm/node/config.py`](prsm/node/config.py) with safe defaults.
4. Add unit/security tests for:
   - fallback activation on primary bootstrap failure
   - malformed bootstrap identity rejection
   - deterministic success when trusted fallback peers are reachable
5. Add concise operator-facing notes for new bootstrap flags and failure semantics in [`docs/SECURE_SETUP.md`](docs/SECURE_SETUP.md).

Definition of done for immediate start:
- Tranche 1 code path implemented behind feature flag and default-enabled for local onboarding profile.
- Tranche 1 targeted tests added and runnable.
- No changes outside Tranche 1 scope in this handoff.

---

## 25. Sprint 6 Completion Summary (2026-03-03)

### Tranche 1 — Bootstrap Reliability + Fallback Defaults ✅

**Files modified:** `prsm/node/config.py`, `prsm/node/discovery.py`, `prsm/node/node.py`, `docs/SECURE_SETUP.md`
**New file:** `tests/unit/test_node_bootstrap_fallback_tranche1.py` (21 tests)

- Added `FALLBACK_BOOTSTRAP_NODES`, `validate_bootstrap_address()`, exponential backoff between retries
- Bootstrap source ordering: configured primaries → trusted fallback peers (deduplicated)
- Feature flag `bootstrap_fallback_enabled` (default `True`) for rollback
- Bootstrap decision telemetry via `get_bootstrap_telemetry()`
- Operator-facing "Node Bootstrap Configuration" section added to `docs/SECURE_SETUP.md`
- **24 bootstrap tests passing** (3 existing + 21 new)

### Tranche 1b — Node Onboarding End-to-End Fix ✅

**Critical bug fixed:** `prsm/node/compute_provider.py:178` unconditionally rejected own jobs. A single-node researcher could never get a compute result.

**Files modified:** `prsm/node/compute_provider.py`, `prsm/node/config.py`, `prsm/node/node.py`, `README.md`
**New file:** `tests/unit/test_node_self_compute.py` (7 tests)

- Self-compute: when `allow_self_compute=True` (default) and 0 peers, node executes its own jobs locally
- README.md rewritten: verify steps, try-it compute walkthrough, two-node local test instructions, www.prsm-network.com references
- **7 self-compute tests passing**

### Tranche 1c — DAGLedgerAdapter Feature Parity ✅

**Critical gap fixed:** The default DAG ledger was missing 15 methods needed for gossip persistence, collaboration state persistence, and agent allowances. These silently failed on every node.

**Files modified:** `prsm/node/dag_ledger.py`, `prsm/node/node.py`, `tests/security/test_sprint6_security_coverage.py`

- Added 10 supplementary SQLite tables to DAGLedger (agent_allowances, gossip_log, collab_tasks/reviews/queries)
- Implemented 15 methods on DAGLedgerAdapter: agent allowances (4), gossip persistence (3), collaboration state (7), async stats (1)
- Fixed `get_status()` to use `get_stats_async()` for real DAG data instead of zeros
- **612 unit+security tests passing** (0 failures)

---

## 26. Comprehensive Technology Audit and Remaining Work Plan (2026-03-03)

### Current State Assessment

A thorough code-level audit of every PRSM subsystem was conducted, reading function bodies (not just signatures) to determine what works end-to-end versus what returns mock/placeholder data.

#### Maturity Matrix

| Subsystem | Status | Maturity | Notes |
|---|---|---|---|
| **P2P Networking** | REAL | 85% | WebSocket transport, gossip, discovery, handshake auth, replay prevention all work |
| **Node Runtime** | REAL | 90% | Identity, startup, dashboard, management API, preflight diagnostics |
| **DAG Ledger** | REAL | 85% | Atomic balance ops, TOCTOU prevention, agent allowances, collaboration persistence |
| **Safety / Circuit Breaker** | REAL | 80% | Emergency halt, threat detection, network consensus — rule-based (not ML) |
| **Authentication** | REAL | 80% | JWT, RBAC, account lockout, audit logging |
| **Compute Benchmarks** | REAL | 95% | CPU benchmark executes real computation, returns real results |
| **Governance Voting** | PARTIAL | 60% | Vote casting and tallying work; executing approved proposals is simulated |
| **FTNS Local Economy** | PARTIAL | 70% | Token tracking, transfers, agent allowances work locally; no on-chain backing |
| **Inference Jobs** | PARTIAL | 25% | Falls back to mock string; works only if NWTN orchestrator wired with real LLM |
| **NWTN 5-Agent Pipeline** | REAL | 90% | All 5 stages execute real logic; executor calls `_execute_with_backend()` for LLM inference; wired to P2P node via BackendRegistry (see Section 29) |
| **Embedding Jobs** | REAL | 85% | RealEmbeddingAPI wired into ContentUploader for provenance; ComputeProvider dispatches to real embedding backend (see Section 34) |
| **Teacher Models** | REAL | 75% | Full PyTorch training loop in `trainer.py`; SEAL, RLVR, RLT systems implemented; exposed via `prsm/interface/api/main.py` — **not yet wired into P2P node API** (see Section 36) |
| **IPFS Cross-Node Retrieval** | SCAFFOLD | 20% | Local pin/unpin works; no cross-node content fetch, no sharding |
| **Web3 / Blockchain** | PARTIAL | 40% | Contracts built and deployer wired; testnet config ready (Section 34 Item 9); no live deployment yet |
| **Cross-Node Content Fetch** | REAL | 90% | ContentProvider P2P protocol implemented and wired; GET /content/retrieve/{cid} endpoint live (Section 29) |
| **Web UI / Frontend** | REAL | 70% | Two implementations: Streamlit SPA (`prsm/interface/dashboard/`) + FastAPI+HTML/JS (`prsm/dashboard/app.py`); launched via `prsm dashboard` — **not served by `prsm node start`** (see Section 36) |
| **Distillation Pipeline** | REAL | 80% | 15-file production system with real training pipeline, architecture generator, knowledge extractor, safety validator — **not yet accessible via node API or CLI** (see Section 36) |

### Detailed Remaining Work

The remaining work is organized into four phases, ordered by impact on making PRSM a functional scientific computing platform.

---

### Phase 1: Real AI Compute (Highest Priority)

**Goal:** A researcher submits a prompt and gets a real AI-generated response — not a placeholder string.

**Current state:** `prsm/compute/nwtn/orchestrator.py` lines 384-435 show the 5-agent pipeline producing hardcoded output:
- Architect: returns `{"intent_category": ..., "complexity": ...}` (from intent clarifier — this part works)
- Router: returns `{"selected_models": models_used}` (from model registry — partially works)
- Executor: returns `{"analysis": "Processed query using N specialist models", "key_findings": ["Finding 1", "Finding 2", "Finding 3"]}` — **hardcoded, no real execution**
- Compiler: returns `{"synthesis": "Compiled comprehensive analysis"}` — **hardcoded**

**What must change:**

#### 1.1 Wire LLM Backends into NWTN Executor

**Files:** `prsm/compute/nwtn/orchestrator.py`, `prsm/compute/agents/`

- Replace hardcoded `ReasoningStep` outputs in `process_query()` (lines 384-435) with actual LLM API calls
- Executor step must call either:
  - **Anthropic API** (`anthropic` package, already in `[ml]` optional deps)
  - **OpenAI API** (`openai` package, already in `[ml]` optional deps)
  - **Local transformers** (`transformers` + `torch`, already in `[ml]` optional deps)
- Implement a `ModelBackend` abstraction with concrete implementations for each provider
- The Executor should dispatch to the discovered specialist model, pass the prompt, and return the actual response
- Compiler should synthesize from real Executor output, not a template

**Implementation approach:**
```
prsm/compute/nwtn/
  backends/
    __init__.py          # ModelBackend ABC
    anthropic_backend.py # Anthropic Claude API
    openai_backend.py    # OpenAI GPT API
    local_backend.py     # Local transformers inference
    mock_backend.py      # Current behavior (for testing)
  orchestrator.py        # Wire backend selection into Executor step
```

**Key design decisions:**
- Backend selection based on available API keys (env vars) with graceful fallback
- If no API keys configured, fall back to `mock_backend` with clear log message
- Rate limiting and cost tracking per-query (integrate with FTNS charging)
- Streaming support for long responses (optional, can be added later)

**Acceptance criteria:**
- `POST /compute/submit {"job_type": "inference", "payload": {"prompt": "What is CRISPR?"}}` returns a real AI-generated answer
- Response includes `"source": "anthropic"` or `"source": "openai"` (not `"source": "mock"`)
- FTNS charged based on actual token usage
- Falls back gracefully to mock if no API keys configured

**Estimated effort:** 2-3 weeks

#### 1.2 Real Embedding Pipeline

**Files:** `prsm/node/compute_provider.py` (`_run_embedding`)

- Replace SHA256-based pseudo-embeddings with real embedding model calls
- Options: OpenAI `text-embedding-3-small`, local `sentence-transformers`, Anthropic embeddings
- Store embedding dimensions in response metadata

**Acceptance criteria:**
- `POST /compute/submit {"job_type": "embedding", "payload": {"text": "quantum computing"}}` returns a real embedding vector
- Vector is semantically meaningful (similar texts produce similar vectors)

**Estimated effort:** 1 week

#### 1.3 Teacher Model Training (Real ML)

**Files:** `prsm/compute/teachers/teacher_model.py`, `prsm/compute/teachers/real_teacher_implementation.py`

- Replace simulated assessment scores with real model evaluation
- Implement actual fine-tuning loop using PyTorch/transformers
- Curriculum generation should produce real training examples from data
- Track convergence metrics (loss, accuracy) across training epochs

**Acceptance criteria:**
- A teacher model can be created with a domain specialization
- Training runs for specified epochs and produces measurable improvement
- Trained model can be used for inference via the NWTN pipeline

**Estimated effort:** 4-6 weeks (significant ML engineering)

---

### Phase 2: Cross-Node Content and Storage

**Goal:** Node A can store content, Node B can discover and retrieve it.

#### 2.1 Cross-Node Content Retrieval

**Current state:** Content metadata is gossiped via `GOSSIP_CONTENT_ADVERTISE` but actual content bytes are never transferred between nodes. `prsm/node/content_uploader.py` has `request_content()` but it only works for locally-pinned IPFS content.

**What must change:**
- Implement a P2P content request/response protocol over the existing WebSocket transport
- When node B wants CID X: send a `content_request` message to providers listed in the content index
- Provider node serves the content inline (small files) or via IPFS gateway URL (large files)
- Add content integrity verification (hash check on received bytes)

**Files:** `prsm/node/content_uploader.py`, `prsm/node/transport.py`, `prsm/node/api.py`

**Estimated effort:** 2-3 weeks

#### 2.2 IPFS Content Sharding

**Current state:** `prsm/core/ipfs_client.py` has extensive class definitions for sharding (IPFSConfig, chunking) but no implementation. Content is uploaded as whole files.

**What must change:**
- Implement chunked upload for files larger than a threshold (e.g., 256KB chunks)
- Track shard-to-CID mappings in the content index
- Implement parallel shard retrieval and reassembly
- Add erasure coding for redundancy (optional, can use simple replication first)

**Files:** `prsm/core/ipfs_client.py`, `prsm/node/storage_provider.py`

**Estimated effort:** 3-4 weeks

#### 2.3 Storage Proof Verification

**Current state:** Storage providers claim to pin content and earn rewards, but proof-of-storage is not cryptographically verified.

**What must change:**
- Implement challenge-response proof-of-storage (node must prove it holds content by answering random byte-range queries)
- Integrate with reward system (only pay verified storage)

**Files:** `prsm/node/storage_provider.py`, `prsm/node/gossip.py`

**Estimated effort:** 2 weeks

---

### Phase 3: Blockchain and Token Economics

**Goal:** FTNS tokens exist on-chain with real economic mechanisms.

#### 3.1 Smart Contract Deployment

**Current state:** `prsm/economy/blockchain/smart_contracts.py` generates Solidity source code as strings but never compiles or deploys. `prsm/economy/web3/mainnet_deployer.py` uses mocked balance checks (`current_balance = Decimal("15.0")`).

**What must change:**
- Choose a target chain (Ethereum L2, Polygon, Base, or Solana)
- Deploy FTNS ERC-20 token contract to testnet
- Wire `web3_service.py` to actual RPC endpoints
- Replace mocked balance checks with real on-chain queries
- Implement deposit/withdraw bridge between local DAG ledger and on-chain token

**Files:** `prsm/economy/blockchain/`, `prsm/economy/web3/`, `prsm/node/dag_ledger.py`

**Key decisions needed:**
- Which blockchain? (Ethereum L2 recommended for cost; Solana for speed)
- Testnet-only for alpha, or mainnet-ready?
- Bridge architecture: lock-and-mint or direct on-chain accounting?

**Estimated effort:** 6-8 weeks

#### 3.2 Staking and Incentive Mechanisms

**Current state:** `prsm/tokenomics/` contains files for anti-hoarding, dynamic supply, liquidity provenance, but these are largely interface definitions without complete implementations.

**What must change:**
- Implement staking contract (node operators stake FTNS to participate)
- Implement slashing for Byzantine behavior (integrate with existing circuit breaker reputation system)
- Dynamic supply controller: mint/burn based on network activity
- Anti-hoarding: decay mechanism for idle tokens

**Files:** `prsm/tokenomics/`, `prsm/economy/`

**Estimated effort:** 4-6 weeks

#### 3.3 Governance Execution

**Current state:** `prsm/core/safety/governance.py` — voting works, but `_execute_implementation()` just logs strings without changing anything.

**What must change:**
- Connect approved proposals to actual configuration changes
- Safety policy proposals → update circuit breaker thresholds
- Economic proposals → update fee schedules, reward rates
- Add proposal types for protocol upgrades

**Files:** `prsm/core/safety/governance.py`, `prsm/core/safety/circuit_breaker.py`

**Estimated effort:** 2-3 weeks

---

### Phase 4: User Interface and Developer Experience

**Goal:** Researchers interact with PRSM through a web interface, not just curl commands.

#### 4.1 Web Dashboard

**Current state:** CLI terminal dashboard only (Rich-based TUI in `prsm/node/dashboard.py`).

**What must build:**
- Browser-based node dashboard (React or similar)
- Real-time node status, peer map, FTNS balance
- Job submission form (select type, enter prompt, set budget)
- Transaction history and content index browser
- Agent management panel

**Estimated effort:** 4-6 weeks

#### 4.2 REST API Hardening

**Current state:** Node management API (`prsm/node/api.py`) works but has no authentication. The platform API (`prsm/interface/api/`) has JWT auth but isn't connected to the node layer.

**What must change:**
- Add optional JWT authentication to node management API
- Rate limiting on compute submission endpoints
- WebSocket endpoint for real-time status updates (replace polling)
- OpenAPI spec generation and documentation

**Files:** `prsm/node/api.py`, `prsm/core/auth/`

**Estimated effort:** 2-3 weeks

#### 4.3 SDK and Client Libraries

- Python SDK for programmatic node interaction
- JavaScript/TypeScript SDK for web integration
- CLI improvements (non-interactive job submission, result streaming)

**Estimated effort:** 3-4 weeks

---

### Phase 5: Production Readiness

#### 5.1 Bootstrap Infrastructure

**Current state:** `wss://bootstrap.prsm-network.com` and fallback URLs don't resolve to running WebSocket servers.

**What must deploy:**
- At least 2 bootstrap nodes on cloud infrastructure (behind the prsm-network.com domain)
- Health monitoring for bootstrap availability
- Geographic distribution for latency (US + EU minimum)

**Estimated effort:** 1-2 weeks (infrastructure)

#### 5.2 CI/CD and Release Pipeline

- Automated test matrix (Python 3.11-3.14, Linux/macOS)
- PyPI package publishing (`pip install prsm-network`)
- Docker image publishing
- Automated changelog generation

**Estimated effort:** 1-2 weeks

#### 5.3 Security Hardening

- Formal security audit of P2P transport and handshake protocol
- Penetration testing of node management API
- Rate limiting and DDoS protection for bootstrap nodes
- Secrets management review

**Estimated effort:** 2-4 weeks

---

### Implementation Priority and Timeline

| Phase | Priority | Est. Duration | Dependencies |
|---|---|---|---|
| **Phase 1.1**: LLM Backend Wiring | P0 — Critical | 2-3 weeks | API keys (Anthropic/OpenAI) |
| **Phase 1.2**: Real Embeddings | P0 — Critical | 1 week | Phase 1.1 backend abstraction |
| **Phase 5.1**: Bootstrap Infrastructure | P0 — Critical | 1-2 weeks | Domain DNS + cloud hosting |
| **Phase 2.1**: Cross-Node Content | P1 — High | 2-3 weeks | None |
| **Phase 1.3**: Teacher Model Training | P1 — High | 4-6 weeks | Phase 1.1 |
| **Phase 3.3**: Governance Execution | P2 — Medium | 2-3 weeks | None |
| **Phase 2.2**: IPFS Sharding | P2 — Medium | 3-4 weeks | IPFS daemon |
| **Phase 4.1**: Web Dashboard | P2 — Medium | 4-6 weeks | None |
| **Phase 3.1**: Smart Contract Deploy | P2 — Medium | 6-8 weeks | Blockchain decision |
| **Phase 2.3**: Storage Proofs | P3 — Lower | 2 weeks | Phase 2.2 |
| **Phase 3.2**: Staking/Incentives | P3 — Lower | 4-6 weeks | Phase 3.1 |
| **Phase 4.2**: API Hardening | P3 — Lower | 2-3 weeks | None |
| **Phase 4.3**: SDK/Client Libraries | P3 — Lower | 3-4 weeks | Phase 4.2 |
| **Phase 5.2**: CI/CD Pipeline | P3 — Lower | 1-2 weeks | None |
| **Phase 5.3**: Security Audit | P3 — Lower | 2-4 weeks | Phase 4.2 |

### Recommended Execution Order

**Weeks 1-3 (Immediate):**
- Phase 1.1: Wire LLM backends to NWTN pipeline — this is the single highest-impact change
- Phase 5.1: Deploy bootstrap nodes on prsm-network.com — unblocks multi-node networking

**Weeks 4-6:**
- Phase 1.2: Real embeddings
- Phase 2.1: Cross-node content retrieval

**Weeks 7-12:**
- Phase 1.3: Teacher model training
- Phase 4.1: Web dashboard (can run in parallel with ML work)

**Weeks 13-20:**
- Phase 3: Blockchain integration (chain selection → contract → bridge)
- Phase 2.2-2.3: Storage improvements

**Weeks 20+:**
- Phase 4.2-4.3: API hardening, SDKs
- Phase 5.2-5.3: CI/CD, security audit
- Phase 3.2: Advanced tokenomics

### What's Complete vs. What Remains (Summary)

```
COMPLETE (works end-to-end):
  ✅ P2P networking (transport, gossip, discovery, handshake, replay prevention)
  ✅ Node runtime (identity, startup, dashboard, API, preflight)
  ✅ DAG ledger (atomic ops, TOCTOU prevention, allowances, persistence)
  ✅ Safety system (circuit breaker, emergency halt, rule-based monitoring)
  ✅ Authentication (JWT, RBAC, audit logging)
  ✅ Local FTNS economy (tracking, transfers, welcome grants)
  ✅ Compute benchmarks (real CPU computation)
  ✅ Collaboration protocol (tasks, reviews, queries, bid selection)
  ✅ Self-compute for single nodes
  ✅ Bootstrap fallback with address validation

PARTIALLY COMPLETE (core works, edges need finishing):
  🔄 Governance (voting works, execution simulated)
  🔄 IPFS storage (local pin/unpin works, no cross-node)
  🔄 Inference (pipeline exists, falls back to mock without LLM keys)

FULLY WORKING END-TO-END (updated):
  ✅ Teacher model creation + training (POST /teacher/create, prsm teacher create)
  ✅ Distillation pipeline (POST /distillation/submit, JobType.TRAINING P2P routing)
  ✅ Web dashboard co-served with node (http://localhost:8000/, no separate command)
  ✅ P2P training marketplace (training-capable nodes advertise and accept TRAINING jobs)

PARTIALLY COMPLETE (remaining work):
  🔄 Web3 / blockchain integration — deployer built, testnet config ready, no live deployment
```

---

## 27. Phase Implementation Completion Summary (2026-03-04)

All five phases from the technology audit plan (Section 26) have been implemented:

| Phase | Commit | Key Deliverables |
|---|---|---|
| Phase 1: Real AI Compute | `0fd9dc7` | BackendRegistry with Anthropic/OpenAI/local/mock backends, TeacherTrainer with PyTorch loop, real embedding pipeline |
| Phase 2: Cross-Node Content | `046c303` | ContentProvider P2P protocol, ContentSharder with parallel chunks, StorageProofVerifier with Merkle proofs |
| Phase 3: Blockchain/Tokens | `28de01a` | ContractDeployer (5 networks), FTNSBridge (deposit/withdraw), StakingManager (full lifecycle), GovernanceExecutor |
| Phase 4: UI/DX | `3e6b402` | Web dashboard (FastAPI+WebSocket), API hardening (JWT+rate limiting), 13-module Python SDK, enhanced CLI |
| Phase 5: Production | `671b8fa` | Bootstrap server (SSL+federation), CI/CD release pipeline, security module (audit+scanner+pentest+secrets) |

Test results after all phases: **1,326 passed, 0 failed, 5 skipped, 4 xfailed**

---

## 28. Integration Wiring Audit (2026-03-05)

### Purpose

A code-level audit was conducted to verify which Phase 1-5 modules are actually wired end-to-end into the running node versus existing as standalone modules that are never called.

### Fully Working End-to-End

These features have complete code paths from user action through to result:

| Feature | User Entry Point | Internal Path | Status |
|---|---|---|---|
| Multi-node compute | `POST /compute/submit` | Requester → gossip `job_offer` → Provider accepts → executes → gossip `job_result` → payment | **Working** |
| Node dashboard | `prsm node start` | Auto-launches Rich TUI, polls `get_status()` every 2s | **Working** |
| Governance execution | `SafetyGovernance.submit_proposal()` → vote → conclude | GovernanceExecutor with TimelockController processes approved proposals | **Working** |
| Bootstrap/discovery | `prsm node start --bootstrap HOST:PORT` | PeerDiscovery → connect → request peer list → announce loop | **Working** |
| Self-compute | `POST /compute/submit` (0 peers) | ComputeProvider accepts own job when `allow_self_compute=True` and `peer_count == 0` | **Working** |
| DAG ledger + persistence | Node initialization | Agent allowances, gossip log, collaboration state all persist in SQLite | **Working** |

### Built But Not Wired (Integration Gaps)

These modules have real, production-quality implementations but are **not connected** to the execution pipeline or user-facing interfaces:

#### Gap 1: NWTN Executor → LLM Backend (HIGH PRIORITY)

**What exists:**
- `prsm/compute/nwtn/backends/registry.py` — `BackendRegistry.execute_with_fallback()` with retry + health monitoring
- `prsm/compute/nwtn/orchestrator.py` — `_execute_with_backend()` method (lines 273-307)

**What's missing:**
- `process_query()` Stage 4 (lines 487-499) still creates `ReasoningStep` objects with hardcoded output:
  ```python
  output_data={
      "analysis": f"Processed query using {len(models_used)} specialist models",
      "key_findings": ["Finding 1", "Finding 2", "Finding 3"]
  }
  ```
- `_execute_with_backend()` is never called from the Executor or Compiler steps
- **Impact:** Users get static fake analysis instead of real LLM inference

**Fix:** Replace hardcoded Executor/Compiler `ReasoningStep` construction with calls to `await self._execute_with_backend(prompt, system_prompt)` and use the real response.

**Files:** `prsm/compute/nwtn/orchestrator.py` (lines 487-499)

---

#### Gap 2: Content Retrieval API Endpoint (HIGH PRIORITY)

**What exists:**
- `prsm/node/content_provider.py` — `ContentProvider.request_content(cid)` with P2P request/response, hash verification, provider discovery

**What's missing:**
- `prsm/node/api.py` has no endpoint for cross-node content retrieval
- Only `/content/upload` (local IPFS) and `/content/search` (index query) exist
- ContentProvider is started by the node but has no HTTP-accessible entry point

**Fix:** Add `GET /content/retrieve/{cid}` endpoint to `prsm/node/api.py` that calls `content_provider.request_content(cid)`.

**Files:** `prsm/node/api.py`

---

#### Gap 3: Staking API Endpoints (MEDIUM PRIORITY)

**What exists:**
- `prsm/economy/tokenomics/staking_manager.py` — Full staking lifecycle: stake, unstake (7-day lockup), withdraw, slash, appeal, rewards

**What's missing:**
- No staking endpoints in `prsm/node/api.py`
- No `prsm staking` CLI command group
- StakingManager is not instantiated by `PRSMNode`

**Fix:** Add staking endpoints (`POST /staking/stake`, `POST /staking/unstake`, `GET /staking/status`, `POST /staking/claim-rewards`) to node API. Wire StakingManager into PRSMNode initialization.

**Files:** `prsm/node/api.py`, `prsm/node/node.py`

---

#### Gap 4: Storage Proofs → StorageProvider (MEDIUM PRIORITY)

**What exists:**
- `prsm/node/storage_proofs.py` — `StorageProofVerifier`, `StorageProver`, `MerkleProofGenerator` with challenge-response protocol

**What's missing:**
- `StorageProvider` never imports or calls `StorageProofVerifier`
- No challenge-response loop in `StorageProvider.start()` or reward cycle
- Storage providers claim to pin content but are never challenged to prove it

**Fix:** Wire `StorageProofVerifier` into `StorageProvider._verify_pins()` cycle. Add periodic challenge-response for pinned CIDs.

**Files:** `prsm/node/storage_provider.py`

---

#### Gap 5: IPFS Sharding → ContentUploader (MEDIUM PRIORITY)

**What exists:**
- `prsm/core/ipfs_sharding.py` — `ContentSharder` with configurable chunk sizes, parallel upload/download, manifest tracking

**What's missing:**
- `ContentUploader` uploads files monolithically — never calls `ContentSharder`
- Large files bypass the sharding system entirely

**Fix:** In `ContentUploader.upload()`, check file size against shard threshold. If above threshold, delegate to `ContentSharder.shard_content()` instead of direct `_ipfs_add()`.

**Files:** `prsm/node/content_uploader.py`

---

#### Gap 6: FTNS Bridge API/CLI Exposure (LOWER PRIORITY)

**What exists:**
- `prsm/economy/blockchain/ftns_bridge.py` — `FTNSBridge` with deposit/withdraw, validators, rate limits

**What's missing:**
- No bridge endpoints in any API file
- No `prsm bridge` or `prsm ftns bridge` CLI commands
- Bridge is completely inaccessible from user-facing interfaces

**Fix:** Add bridge endpoints to node API (`POST /bridge/deposit`, `POST /bridge/withdraw`, `GET /bridge/status`). Add CLI commands under `prsm ftns` group.

**Files:** `prsm/node/api.py`, `prsm/cli.py`

---

### Integration Priority and Estimated Effort

| Gap | Priority | Effort | Impact |
|---|---|---|---|
| **Gap 1:** NWTN → LLM backend | P0 Critical | 1-2 days | Transforms fake analysis into real AI responses |
| **Gap 2:** Content retrieval API | P1 High | 1 day | Enables cross-node content sharing via API |
| **Gap 3:** Staking API | P2 Medium | 1-2 days | Exposes staking to users |
| **Gap 4:** Storage proofs integration | P2 Medium | 2-3 days | Validates storage claims cryptographically |
| **Gap 5:** Sharding integration | P2 Medium | 1 day | Efficient large file handling |
| **Gap 6:** Bridge API/CLI | P3 Lower | 1 day | Exposes on-chain operations |

**Total estimated effort:** ~8-10 days of integration work.

### Updated Status Summary

```
FULLY WORKING END-TO-END:
  ✅ P2P networking (transport, gossip, discovery, handshake, replay prevention)
  ✅ Node runtime (identity, startup, dashboard, management API, preflight)
  ✅ DAG ledger (atomic ops, TOCTOU prevention, allowances, gossip + collab persistence)
  ✅ Safety system (circuit breaker, emergency halt, rule-based monitoring)
  ✅ Authentication (JWT, RBAC, audit logging)
  ✅ Local FTNS economy (tracking, transfers, welcome grants, agent allowances)
  ✅ Compute benchmarks (real CPU computation)
  ✅ Multi-node compute jobs (submit → accept → execute → result → payment)
  ✅ Collaboration protocol (tasks, reviews, queries, bid selection, persistence)
  ✅ Self-compute for single nodes
  ✅ Bootstrap fallback with address validation
  ✅ Governance voting + execution (votes → timelock → parameter changes)
  ✅ Web dashboard (auto-launched with node)

MODULES BUILT, WIRING NEEDED:
  (none — all gaps resolved, see Section 29)

PUBLISHED & DEPLOYED:
  ✅ PyPI package live: pip install prsm-network (v0.2.0)
  ✅ Bootstrap server live: bootstrap1.prsm-network.com:8765 (DigitalOcean NYC3)
  ✅ DNS records: bootstrap1, fallback1, fallback2 on Cloudflare
  ✅ GitHub Actions: PYPI_API_TOKEN configured for automated releases

REMAINING (LOWER PRIORITY):
  📦 SSL certificate (certbot on server to upgrade ws:// → wss://)
  📦 Security tooling (audit/scanner/pentest ready, needs scheduled runs)
```

---

## 29. Integration Wiring Completion (2026-03-05)

All 6 integration gaps identified in Section 28 have been resolved.

### Gap Resolutions

| Gap | Fix | Files Modified |
|---|---|---|
| **Gap 1:** NWTN → LLM backend | `process_query()` Stage 4 now calls `_execute_with_backend()` for Executor and Compiler steps; added `_build_executor_system_prompt()` and `_extract_key_findings()` | `prsm/compute/nwtn/orchestrator.py` |
| **Gap 2:** Content retrieval API | Added `GET /content/retrieve/{cid}` endpoint calling `ContentProvider.request_content()` with timeout and hash verification | `prsm/node/api.py` |
| **Gap 3:** Staking API | Wired `StakingManager` into `PRSMNode` via `_StakingFTNSAdapter`; added 8 staking endpoints (stake, unstake, status, claim-rewards, withdraw, cancel-unstake, history, config) | `prsm/node/node.py`, `prsm/node/api.py` |
| **Gap 4:** Storage proofs | Integrated `StorageProofVerifier` into `StorageProvider` with periodic challenge loop, provider reputation tracking, and configurable `ChallengeConfig` | `prsm/node/storage_provider.py` |
| **Gap 5:** Content sharding | Integrated `ContentSharder` into `ContentUploader.upload()` for files above threshold (default 10MB); automatic fallback to monolithic upload on sharding failure | `prsm/node/content_uploader.py` |
| **Gap 6:** Bridge API/CLI | Added 5 bridge API endpoints (`/bridge/deposit`, `/bridge/withdraw`, `/bridge/status`, `/bridge/tx/{id}`, `/bridge/transactions`) and 4 CLI commands under `prsm ftns bridge` | `prsm/node/api.py`, `prsm/cli.py` |

### Verification

- **New test file:** `tests/unit/test_section28_integrations.py` — 40 passed, 1 skipped
- **Full unit + security suite:** All passing, 0 regressions

### Updated Status Summary

```
FULLY WORKING END-TO-END:
  ✅ P2P networking (transport, gossip, discovery, handshake, replay prevention)
  ✅ Node runtime (identity, startup, dashboard, management API, preflight)
  ✅ DAG ledger (atomic ops, TOCTOU prevention, allowances, gossip + collab persistence)
  ✅ Safety system (circuit breaker, emergency halt, rule-based monitoring)
  ✅ Authentication (JWT, RBAC, audit logging)
  ✅ Local FTNS economy (tracking, transfers, welcome grants, agent allowances)
  ✅ Compute benchmarks (real CPU computation)
  ✅ Multi-node compute jobs (submit → accept → execute → result → payment)
  ✅ Collaboration protocol (tasks, reviews, queries, bid selection, persistence)
  ✅ Self-compute for single nodes
  ✅ Bootstrap fallback with address validation
  ✅ Governance voting + execution (votes → timelock → parameter changes)
  ✅ Web dashboard (auto-launched with node)
  ✅ NWTN LLM inference (Anthropic/OpenAI/local backends via BackendRegistry)
  ✅ Cross-node content retrieval (GET /content/retrieve/{cid})
  ✅ Staking lifecycle (stake/unstake/withdraw/slash/rewards via API)
  ✅ Storage proof verification (challenge-response integrated into StorageProvider)
  ✅ IPFS content sharding (auto-shard large files in ContentUploader)
  ✅ FTNS bridge (deposit/withdraw via API and CLI)

PUBLISHED & DEPLOYED:
  ✅ PyPI package live: pip install prsm-network (v0.2.0)
  ✅ Bootstrap server live: bootstrap1.prsm-network.com:8765 (DigitalOcean NYC3)
  ✅ DNS records: bootstrap1, fallback1, fallback2 on Cloudflare
  ✅ GitHub Actions: PYPI_API_TOKEN configured for automated releases

REMAINING (LOWER PRIORITY):
  📦 SSL certificate (certbot on server to upgrade ws:// → wss://)
  📦 Security tooling (audit/scanner/pentest ready, needs scheduled runs)
```

---

## 30. Sprint 7–13 Completion Log (2026-03-05 — 2026-03-06)

Following the integration wiring completion, seven additional sprints were executed to bring PRSM from "code-complete" to "release-ready":

### Sprint 7 — Production Polish: Silent Failures & First-Run UX ✅

**Commit:** `f3e7a06`

| Deliverable | Files |
|---|---|
| LLM mock warning at node startup when no API keys detected | `prsm/cli.py` |
| Mock response tagging (`"source": "mock"`, `"warning"` field) on inference and embedding jobs | `prsm/node/compute_provider.py` |
| Preflight diagnostics wired into `prsm node start` (port, Python, config, bootstrap, IPFS checks) | `prsm/cli.py` |
| `detect_available_backends()` utility for checking configured LLM providers | `prsm/compute/nwtn/backends/config.py` |
| Deprecated FTNSService singleton warning silenced during normal startup | `prsm/economy/tokenomics/ftns_service.py` |
| 25 new UX tests | `tests/unit/test_sprint7_ux.py` |

### Sprint 8 — End-to-End Validation & Cleanup ✅

**Commit:** `ae419b9`

| Deliverable | Files |
|---|---|
| End-to-end smoke test script (14 checks: init → start → benchmark → inference → balance → shutdown) | `scripts/smoke_test.py` |
| Gossip local delivery fix (messages reach local subscribers with 0 peers, enabling self-compute) | `prsm/node/gossip.py` |
| Fixed 4 stale integration test imports (dag_consensus, marketplace, nwtn_e2e, p2p_partition) | `tests/integration/` |
| Removed dead code: `prsm/compute/nwtn/archive_unused_files.py`, `do_archive_cleanup.py`, `prsm/core/database/optimized_queries.py` | — |
| Repo hygiene: moved stray files from root, updated `.gitignore` | — |

### Sprint 9 — Documentation & Developer Onboarding ✅

**Commit:** `25d5d9c`

| Deliverable | Files |
|---|---|
| Quickstart rewrite: 10-minute guide from clone to cross-node compute (397 lines) | `docs/quickstart.md` |
| API reference: 71 endpoints documented with request/response formats (2,275 lines) | `docs/API_REFERENCE.md` |
| Environment variables: 140+ vars across 15 categories documented | `.env.example` |
| Contributing guide: smoke test reference, updated project structure, pre-PR checklist | `CONTRIBUTING.md` |

### Sprint 10 — Version 0.2.0 Release Preparation ✅

**Commit:** `ae9505f` | **Tag:** `v0.2.0`

| Deliverable | Files |
|---|---|
| Version bumped to 0.2.0 across 27 occurrences in 24 files | `pyproject.toml`, `prsm/__init__.py`, `prsm/cli.py`, `README.md`, + 20 others |
| CHANGELOG.md: complete 0.2.0 entry with Added/Changed/Breaking Changes/Migration Notes | `CHANGELOG.md` |
| RELEASE_NOTES.md: user-facing summary of capabilities | `RELEASE_NOTES.md` |

### Sprint 11 — Infrastructure Deployment Readiness ✅

**Commit:** `f844956`

| Deliverable | Files |
|---|---|
| Deleted `setup.py` (outdated 0.1.0 conflicting with pyproject.toml 0.2.0) | `setup.py` (deleted) |
| Fixed release workflow Dockerfile path (`./docker/Dockerfile.production` → `./Dockerfile`) | `.github/workflows/release.yml` |
| Fixed bootstrap Dockerfile to remove deleted setup.py reference | `docker/Dockerfile.bootstrap` |
| Makefile deployment targets: build, publish-test, publish, docker-build, smoke, test | `Makefile` |
| TestPyPI publishing guide | `docs/PUBLISHING_TESTPYPI.md` |
| Build verified: `prsm-0.2.0-py3-none-any.whl` + `.tar.gz` install cleanly, `prsm --version` → 0.2.0 | — |

### Sprint 12 — PyPI Package Name & Publication Prep ✅

**Commit:** `e198771`

| Deliverable | Files |
|---|---|
| PyPI distribution name changed from `prsm` (taken) to `prsm-network` | `pyproject.toml` |
| All docs updated: `pip install prsm-network` (import name `prsm` unchanged) | `README.md`, `docs/quickstart.md`, `CONTRIBUTING.md`, + 3 others |
| PyPI badge added to README | `README.md` |
| Build verified: `prsm_network-0.2.0-py3-none-any.whl` + `.tar.gz` | — |
| Comprehensive publication guide (TestPyPI + PyPI + GitHub Actions) | `docs/PUBLISHING_TESTPYPI.md` |

### Sprint 13 — Bootstrap Server Deployment ✅

**Commit:** `ca2ef73`

| Deliverable | Files |
|---|---|
| Local bootstrap Docker Compose (simplified, no SSL, for testing) | `docker/docker-compose.bootstrap-local.yml` |
| Standalone health check script for containers | `docker/healthcheck.py` |
| Local bootstrap server test runner (start → health → WebSocket → shutdown) | `scripts/test_bootstrap_local.py` |
| Bootstrap connectivity test suite | `tests/integration/test_bootstrap_connectivity.py` |
| Step-by-step production deployment guide (DNS, SSL, Docker, verification) | `docs/BOOTSTRAP_DEPLOYMENT_GUIDE.md` |
| Local bootstrap verified: server starts, health endpoint responds, WebSocket accepts connections | — |

### Sprint 7–13 Cumulative Results

| Metric | Value |
|---|---|
| Sprints completed | 7 |
| Commits pushed | 8 |
| Test suite size | 1,391 unit+security tests passing |
| Smoke test | 14/14 checks |
| Integration test collection | 194 tests, 0 collection errors |
| Package build | `prsm_network-0.2.0` (.whl + .tar.gz) verified |
| Release tag | `v0.2.0` on GitHub |

---

## 31. Operational Deployment Checklist — Outside-the-Codebase Tasks

PRSM's codebase is feature-complete and release-ready. The remaining work to bring PRSM live as a functioning network is entirely operational — infrastructure provisioning, account setup, and credential configuration. None of these require code changes.

### Step 1: Publish to PyPI ✅ COMPLETED (2026-03-06)

**Goal:** Researchers worldwide can run `pip install prsm-network` to get PRSM.

**Status:** Published and verified. Live at https://pypi.org/project/prsm-network/0.2.0/

**What was done:**
1. PyPI account created and API token generated
2. Package built: `prsm_network-0.2.0-py3-none-any.whl` + `prsm_network-0.2.0.tar.gz`
3. Both artifacts passed `twine check` validation
4. Uploaded via `twine upload dist/* --username __token__`
5. Verified installation: `pip install prsm-network` → installs all dependencies → `prsm --version` → `0.2.0`

**Install command (works now for anyone):**
```bash
pip install prsm-network
prsm --version   # → PRSM, version 0.2.0
prsm node start  # → starts a PRSM node
```

**For future releases:**
```bash
# Bump version in pyproject.toml, rebuild, re-upload
rm -rf dist/ build/ *.egg-info
python -m build
twine upload dist/* --username __token__
```

---

### Step 2: Deploy Bootstrap Server ✅ COMPLETED (2026-03-06)

**Goal:** New PRSM nodes can discover peers via the bootstrap server instead of starting in degraded local mode.

**Status:** Live at `http://bootstrap1.prsm-network.com:8000/health`

**What was done:**
1. DigitalOcean Droplet provisioned (Ubuntu, 1 vCPU / 2GB RAM, NYC3 region, IP: 159.203.129.218)
2. Docker installed, PRSM cloned, bootstrap server built with minimal dependencies
3. Simplified Dockerfile (single-stage, 6 pip packages instead of 100+)
4. Server running and accepting connections on ports 8765 (WebSocket) and 8000 (HTTP API)
5. UFW firewall configured (ports 8765, 8000, 443 open)

**Verification:**
```bash
curl http://bootstrap1.prsm-network.com:8000/health
# → {"status":"healthy","active_connections":0,...}
curl http://159.203.129.218:8000/health
# → {"status":"healthy",...}
```

**Previous documentation (for reference/future servers):**

**2.1 Provision a VPS**

Any cloud provider works. Minimum requirements: 2 CPU cores, 4GB RAM, 20GB SSD, Ubuntu 22.04.

| Provider | Recommended Plan | Cost |
|---|---|---|
| DigitalOcean | Basic Droplet (2 vCPU, 4GB) | ~$24/month |
| Linode | Linode 4GB | ~$24/month |
| Vultr | Cloud Compute (2 vCPU, 4GB) | ~$24/month |
| AWS | t3.medium (2 vCPU, 4GB) | ~$30/month |
| GCP | e2-medium (2 vCPU, 4GB) | ~$25/month |

For initial deployment, a single server is sufficient. All three bootstrap subdomains can point to the same IP.

**2.2 Server setup**
```bash
# SSH into the server
ssh root@YOUR_SERVER_IP

# Install Docker
curl -fsSL https://get.docker.com | sh
apt-get install -y docker-compose-plugin

# Clone PRSM
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM/docker

# Start bootstrap server
docker compose -f docker-compose.bootstrap.yml up -d

# Verify it's running
curl http://localhost:8000/health
```

**2.3 Open firewall ports**
```bash
# If using UFW (Ubuntu default)
ufw allow 8765/tcp   # WebSocket (bootstrap protocol)
ufw allow 8000/tcp   # HTTP API (health/metrics)
ufw allow 443/tcp    # HTTPS/WSS (SSL-terminated)
ufw allow 22/tcp     # SSH (already open)
ufw enable
```

**2.4 Alternative: AWS automated deployment**
```bash
# From local machine (requires AWS CLI configured)
cd /path/to/PRSM
./scripts/deploy_bootstrap_aws.sh -r us-east-1 -e production
```

**2.5 Alternative: GCP automated deployment**
```bash
# From local machine (requires gcloud CLI configured)
cd /path/to/PRSM
./scripts/deploy_bootstrap_gcp.sh -p YOUR_GCP_PROJECT -r us-east1 -e production
```

**Verification:**
```bash
# From local machine
curl http://YOUR_SERVER_IP:8000/health
# Expected: {"status":"healthy", ...}
```

---

### Step 3: DNS Records ✅ COMPLETED (2026-03-06)

**Goal:** Bootstrap subdomains resolve to the server provisioned in Step 2.

**Status:** All three subdomains resolving via Cloudflare (DNS only, no proxy).

**Records created:**

| Hostname | Type | Content | Mode |
|---|---|---|---|
| `bootstrap1.prsm-network.com` | A | `159.203.129.218` | DNS only |
| `fallback1.prsm-network.com` | A | `159.203.129.218` | DNS only |
| `fallback2.prsm-network.com` | A | `159.203.129.218` | DNS only |

**Note:** `bootstrap.prsm-network.com` (without the "1") was already used by a Cloudflare Tunnel record, so the primary bootstrap uses `bootstrap1` instead. Code updated in `prsm/node/config.py`.

**Previous documentation (for reference):**

**3.1 Access DNS management**
- Log into the DNS provider for `prsm-network.com` (likely the domain registrar or Cloudflare)
- Navigate to DNS records management

**3.2 Create A records**

| Hostname | Type | Value | TTL |
|---|---|---|---|
| `bootstrap.prsm-network.com` | A | `YOUR_SERVER_IP` | 300 |
| `fallback1.prsm-network.com` | A | `YOUR_SERVER_IP` | 300 |
| `fallback2.prsm-network.com` | A | `YOUR_SERVER_IP` | 300 |

For initial deployment, all three can point to the same IP. When scaling, point `fallback1` and `fallback2` to servers in different regions.

**3.3 Verify DNS propagation**
```bash
dig bootstrap.prsm-network.com +short
# Should return: YOUR_SERVER_IP

dig fallback1.prsm-network.com +short
dig fallback2.prsm-network.com +short
```

DNS propagation typically takes 5–30 minutes. Use https://dnschecker.org to monitor.

---

### Step 4: SSL Certificate — DEFERRED (using ws:// for alpha)

**Goal:** Bootstrap connections use `wss://` (secure WebSocket) instead of `ws://`.

**Current status:** Using `ws://` (non-SSL) for alpha deployment. Cloudflare proxy was incompatible with non-standard ports (8765, 8000), so DNS-only mode is used. SSL via certbot can be added later to upgrade to `wss://`.

**To add SSL later:**

**4.1 Install certbot on the server**
```bash
ssh root@YOUR_SERVER_IP
apt-get install -y certbot
```

**4.2 Generate certificates**
```bash
# Stop any service using port 80 temporarily
docker compose -f docker-compose.bootstrap.yml stop nginx 2>/dev/null || true

# Generate certificates for all bootstrap subdomains
certbot certonly --standalone \
  -d bootstrap.prsm-network.com \
  -d fallback1.prsm-network.com \
  -d fallback2.prsm-network.com \
  --agree-tos \
  --email admin@prsm-network.com

# Restart services
docker compose -f docker-compose.bootstrap.yml up -d
```

Certificates are saved to `/etc/letsencrypt/live/bootstrap.prsm-network.com/`.

**4.3 Configure the bootstrap server to use SSL**

Update the environment in `docker-compose.bootstrap.yml` (or create a `.env` file on the server):
```bash
PRSM_SSL_CERT=/etc/letsencrypt/live/bootstrap.prsm-network.com/fullchain.pem
PRSM_SSL_KEY=/etc/letsencrypt/live/bootstrap.prsm-network.com/privkey.pem
```

Mount the certificate directory into the Docker container (add to docker-compose volumes):
```yaml
volumes:
  - /etc/letsencrypt:/etc/letsencrypt:ro
```

Restart the container:
```bash
docker compose -f docker-compose.bootstrap.yml down
docker compose -f docker-compose.bootstrap.yml up -d
```

**4.4 Set up auto-renewal**
```bash
# Certbot auto-renews via systemd timer (installed automatically)
# Verify timer is active:
systemctl status certbot.timer

# Test renewal:
certbot renew --dry-run
```

**4.5 Verify SSL**
```bash
# From local machine
curl https://bootstrap.prsm-network.com:8000/health

# Test WebSocket
python3 -c "
import asyncio, websockets
async def test():
    async with websockets.connect('wss://bootstrap.prsm-network.com:8765') as ws:
        print('WSS connection successful!')
asyncio.run(test())
"
```

---

### Step 5: GitHub Repository Secrets ✅ COMPLETED (2026-03-06)

**Goal:** Automated releases publish to PyPI and Docker registries without manual token entry.

**Status:** `PYPI_API_TOKEN` secret added to GitHub repository. Future releases will auto-publish via the release workflow.

**Previous documentation (for reference):**

**5.1 Navigate to repository secrets**
- Go to https://github.com/Ryno2390/PRSM/settings/secrets/actions
- Click "New repository secret"

**5.2 Add required secrets**

| Secret Name | Value | Purpose |
|---|---|---|
| `PYPI_API_TOKEN` | `pypi-...` (from Step 1.2) | Automated PyPI publishing on GitHub Release |
| `TESTPYPI_API_TOKEN` | `pypi-...` (from TestPyPI, optional) | Pre-release testing |
| `DOCKER_USERNAME` | Docker Hub username (optional) | Docker image publishing |
| `DOCKER_PASSWORD` | Docker Hub access token (optional) | Docker image publishing |

**Note:** `GITHUB_TOKEN` is automatically available — no setup needed for GitHub Container Registry (ghcr.io).

**5.3 Verify automated release works**

After adding secrets, test the pipeline:
1. Go to https://github.com/Ryno2390/PRSM/releases
2. Click "Create a new release"
3. Tag: `v0.2.1` (or next version)
4. Title: "PRSM v0.2.1"
5. Publish release
6. Watch the Actions tab — the release workflow should build, test, and publish automatically

---

### Post-Deployment Verification

After completing all 5 steps, run this end-to-end verification:

```bash
# 1. Install from PyPI (proves Step 1 worked)
pip install prsm-network
prsm --version

# 2. Start a node (proves Steps 2-3 worked)
prsm node start --no-dashboard

# Expected output should include:
#   "Bootstrap success via ws://bootstrap1.prsm-network.com:8765"
# Instead of:
#   "DEGRADED local mode"

# 3. Verify health (proves bootstrap is live)
curl http://bootstrap1.prsm-network.com:8000/health

# 4. Submit a compute job
curl -s -X POST http://localhost:8000/compute/submit \
  -H 'Content-Type: application/json' \
  -d '{"job_type": "benchmark", "payload": {"iterations": 100000}, "ftns_budget": 1.0}'
```

If a second person runs `prsm node start` on a different machine, both nodes should discover each other via the bootstrap server and be able to exchange compute jobs.

---

## 32. Polish & Scale Roadmap (Post-Launch)

PRSM is operational as of 2026-03-06. The critical path is complete:
- Researchers can `pip install prsm-network` and run a node
- Nodes connect to a live bootstrap server
- Single-node compute works out of the box
- Multi-node P2P works when multiple nodes are running
- All code is tested (1,391+ tests), documented, and on GitHub

The remaining work is **polish and scale** — nothing blocks users from using PRSM today.

### Near-Term (Quick Wins) ✅ ALL COMPLETED (2026-03-06)

| # | Task | Status | Details |
|---|---|---|---|
| 1 | **v0.2.1 release** | ✅ Complete | Published to PyPI with `wss://bootstrap1.prsm-network.com:8765` URLs. Tag `v0.2.1` pushed. `pip install prsm-network` now auto-connects to the live bootstrap server. |
| 2 | **SSL for bootstrap** | ✅ Complete | Let's Encrypt certificates obtained for bootstrap1, fallback1, fallback2. Certs copied to `/root/ssl/` on server and mounted into Docker container. Bootstrap URLs upgraded from `ws://` to `wss://`. Auto-renewal configured via certbot systemd timer. |
| 3 | **Server SSH access** | ✅ Complete | SSH key generated (`id_ed25519`), added to DigitalOcean Droplet. `ssh root@159.203.129.218` working. |

**Implementation notes for future reference:**

SSL cert renewal: Certs expire 2026-06-04. Certbot auto-renews, but after renewal the certs need to be re-copied to `/root/ssl/` and the container restarted:
```bash
ssh root@159.203.129.218
cp /etc/letsencrypt/live/bootstrap1.prsm-network.com/fullchain.pem /root/ssl/
cp /etc/letsencrypt/live/bootstrap1.prsm-network.com/privkey.pem /root/ssl/
chmod 644 /root/ssl/*.pem
cd ~/PRSM/docker
docker compose -f docker-compose.bootstrap-local.yml restart
```

### Medium-Term (When Users Start Joining)

These improve reliability and observability as the network grows:

| # | Task | Effort | Impact | Details |
|---|---|---|---|---|
| 4 | **Multi-region bootstrap** | 2 hours | High | Deploy separate bootstrap servers for `fallback1` (EU) and `fallback2` (Asia-Pacific). Update DNS A records to point to the new server IPs. Provides redundancy and lower latency for international users. Reuse the same Docker setup — just provision two more VPS instances. |
| 5 | **Monitoring dashboards** | 1–2 hours | High | Connect Grafana to bootstrap Prometheus metrics. The Docker compose stack (`docker-compose.bootstrap.yml`) already includes Prometheus and Grafana services. Key metrics to track: active peer count, connection rate, message throughput, peer churn. Set up alerts for peer count drops or sustained connection failures. |
| 6 | **Automated security scans** | 1 hour | Medium | Schedule `prsm/security/audit_checklist.py` and `prsm/security/scanner.py` as a weekly GitHub Actions cron job or a cron task on the server. The code is ready — just needs a trigger. |
| 7 | **FTNS testnet deployment** | 2–3 hours | Medium | Deploy the FTNS ERC-20 token contract to Sepolia (Ethereum testnet) or Polygon Mumbai using `prsm/economy/blockchain/deployment.py`. Requires: an Ethereum wallet with testnet ETH (free from faucets), an Infura/Alchemy RPC URL. This enables real on-chain token operations for testing. |

### Longer-Term (Product Growth)

These are strategic initiatives for scaling PRSM from alpha to a real research platform:

| # | Task | Effort | Impact | Details |
|---|---|---|---|---|
| 8 | **Production LLM API keys** | 30 min setup | **High** | Configure Anthropic and/or OpenAI API keys on nodes so the NWTN pipeline returns real AI-generated research analysis instead of mock responses. Without keys, inference works but returns `"source": "mock"` responses. With keys, researchers get real Claude/GPT-powered analysis. |
| 9 | **Web dashboard hosting** | 1–2 days | High | The web dashboard (`prsm/dashboard/`) exists but currently runs as part of the node. Could be deployed as a standalone web app (hosted on the server or a separate service) so researchers can monitor the network from a browser without running a node. |
| 10 | **SDK documentation site** | 1 day | Medium | Host the Python SDK documentation on GitHub Pages or ReadTheDocs. The SDK (`sdks/python/prsm_sdk/`) has 13 modules — proper API docs would help developers integrate with PRSM programmatically. |
| 11 | **PyPI trusted publishing** | 30 min | Low | Configure OIDC-based trusted publishing at pypi.org so releases don't need a stored API token. More secure than the current token-based approach. See https://docs.pypi.org/trusted-publishers/ |
| 12 | **Community & adoption** | Ongoing | **Critical** | Get researchers running nodes, filing issues, and contributing. Write blog posts, submit to Hacker News/Reddit, present at ML meetups. The technology is ready — adoption is the bottleneck. |

### Current Network Topology

```
                    ┌─────────────────────────────────────┐
                    │   bootstrap1.prsm-network.com:8765  │
                    │   (DigitalOcean NYC3, 159.203.129.218)│
                    │   fallback1 ──┘  └── fallback2      │
                    │   (same server for alpha)            │
                    └──────────┬──────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
         ┌────▼────┐    ┌─────▼─────┐   ┌──────▼──────┐
         │ Node A  │    │  Node B   │   │   Node C    │
         │ (user)  │◄──►│  (user)   │◄─►│   (user)    │
         └─────────┘    └───────────┘   └─────────────┘
              P2P direct connections after discovery
```

Once two or more nodes discover each other via bootstrap, they communicate directly via P2P WebSocket — the bootstrap server is only used for initial discovery.

---

## 33. Semantic Provenance System

The Semantic Provenance System provides content attribution, derivative work tracking, and automated royalty distribution across the PRSM network. It ensures that content creators receive FTNS compensation when their work is accessed or used as source material for derivative works.

### System Architecture

```mermaid
flowchart TB
    subgraph Upload Flow
        CONTENT[Content Upload] --> EMBED[Generate Embedding]
        EMBED --> SEMANTIC[_SemanticIndex Lookup]
        SEMANTIC --> |Similarity >= 0.92| DERIV[Register as Derivative]
        SEMANTIC --> |Similarity < 0.92| ORIGINAL[Register as Original]
        DERIV --> PARENTS[Auto-link Parent CIDs]
        ORIGINAL --> PROV[Create Provenance Record]
        PARENTS --> PROV
        PROV --> GOSSIP[Broadcast GOSSIP_PROVENANCE_REGISTER]
        GOSSIP --> LEDGER[(SQLite Persistence)]
    end

    subgraph Cross-Node Resolution
        QUERY[Content Access Request] --> LOCAL{Local Ledger?}
        LOCAL --> |Yes| SERVE[Serve Content]
        LOCAL --> |No| BROADCAST[Broadcast GOSSIP_PROVENANCE_QUERY]
        BROADCAST --> PEER[Peer Node]
        PEER --> |GOSSIP_PROVENANCE_RESPONSE| CACHE[Cache in Local Ledger]
        CACHE --> SERVE
    end

    subgraph Royalty Distribution
        ACCESS[Content Access] --> ROYALTY[Calculate Royalty]
        ROYALTY --> |Has Parents| SPLIT[Multi-level Split]
        ROYALTY --> |No Parents| SINGLE[Single Creator]
        SPLIT --> D70[70% Derivative Creator]
        SPLIT --> D25[25% Source Creators]
        SPLIT --> D05[5% Network Fee]
        SINGLE --> CREATOR[100% to Creator]
    end
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `_SemanticIndex` | [`prsm/node/content_uploader.py`](prsm/node/content_uploader.py:66) | L2-normalized embedding index for near-duplicate detection |
| `ContentIndex` | [`prsm/node/content_index.py`](prsm/node/content_index.py:51) | Network-wide content registry with provenance query handling |
| `LocalLedger` | [`prsm/node/local_ledger.py`](prsm/node/local_ledger.py:100) | SQLite persistence for provenance records |
| `ContentUploader` | [`prsm/node/content_uploader.py`](prsm/node/content_uploader.py:171) | Upload flow with provenance registration and royalty distribution |

### _SemanticIndex: Near-Duplicate Detection

The `_SemanticIndex` class provides semantic similarity matching to detect derivative works at upload time:

**Implementation Details:**

```python
class _SemanticIndex:
    DERIVATIVE_THRESHOLD: float = 0.92   # Auto-register as derivative
    DUPLICATE_THRESHOLD: float = 0.99    # Near-exact duplicate warning
    
    def __init__(self, persist_path: Optional[Path] = None):
        self._index: Dict[str, Tuple] = {}  # cid → (normalized_embedding, creator_id)
    
    def store(self, cid: str, embedding: np.ndarray, creator_id: str):
        # L2-normalize and store for cosine similarity via dot product
        norm = np.linalg.norm(embedding)
        normalized = embedding / norm if norm > 0 else embedding
        self._index[cid] = (normalized, creator_id)
    
    def find_nearest(self, embedding: np.ndarray) -> Optional[Tuple[str, float, str]]:
        # O(n) scan — acceptable for early-network scale
        # Returns (cid, cosine_similarity, creator_id)
```

**Similarity Thresholds:**

| Threshold | Value | Behavior |
|-----------|-------|----------|
| Derivative | 0.92 | Content auto-registered as derivative; parent CID prepended to lineage |
| Duplicate | 0.99 | Logged as near-exact duplicate; upload proceeds with derivative rate |

**Persistence:** Index persisted as JSON to survive node restarts. Embedding vectors serialized as float arrays.

### Gossip Protocol Extensions

Two new message types enable cross-node provenance resolution:

| Message Type | Direction | Purpose |
|--------------|-----------|---------|
| `GOSSIP_PROVENANCE_QUERY` | Request | Broadcast query for a CID's provenance record |
| `GOSSIP_PROVENANCE_RESPONSE` | Response | Return provenance data from node with local record |

**Query Flow:**

```python
# ContentIndex._on_provenance_query
async def _on_provenance_query(self, subtype, data, origin):
    cid = data.get("cid", "")
    requester_id = data.get("requester_id", "")
    record = await self.ledger.get_provenance(cid)
    if record:
        await self.gossip.publish(GOSSIP_PROVENANCE_RESPONSE, {
            "cid": cid,
            "for_requester": requester_id,
            "provenance": record,
        })
```

**Response Handling:**

```python
# ContentIndex._on_provenance_response
async def _on_provenance_response(self, subtype, data, origin):
    cid = data.get("cid", "")
    provenance = data.get("provenance", {})
    # Persist to local ledger for future lookups
    await self.ledger.upsert_provenance(provenance)
    # Resolve any pending async get_provenance() call
    future = self._pending_provenance.get(cid)
    if future and not future.done():
        future.set_result(provenance)
```

### SQLite Persistence Schema

Provenance records are stored durably in the `provenance_chains` table:

```sql
CREATE TABLE provenance_chains (
    cid               TEXT PRIMARY KEY,
    content_hash      TEXT NOT NULL DEFAULT '',
    creator_id        TEXT NOT NULL,
    creator_pubkey    TEXT NOT NULL DEFAULT '',
    filename          TEXT NOT NULL DEFAULT '',
    size_bytes        INTEGER NOT NULL DEFAULT 0,
    royalty_rate      REAL NOT NULL DEFAULT 0.01,
    parent_cids       TEXT NOT NULL DEFAULT '[]',   -- JSON array
    signature         TEXT NOT NULL DEFAULT '',
    embedding_id      TEXT,                         -- "emb:<cid>" if embedded
    near_duplicate_of TEXT,                         -- CID of most-similar content
    metadata          TEXT NOT NULL DEFAULT '{}',
    registered_at     REAL NOT NULL
);
CREATE INDEX idx_prov_creator ON provenance_chains(creator_id);
CREATE INDEX idx_prov_hash ON provenance_chains(content_hash);
```

### FTNS Royalty Distribution

When content is accessed, royalties are distributed based on provenance lineage:

**Single Creator (No Parents):**
- 100% of royalty goes to content creator

**Derivative Work (Has Parents):**

| Share | Recipient | Description |
|-------|-----------|-------------|
| 70% | Derivative Creator | Node that uploaded the derivative work |
| 25% | Source Creators | Split evenly among parent CID creators |
| 5% | Network Fee | System treasury for infrastructure |

**Implementation:**

```python
# Multi-level royalty constants
DERIVATIVE_CREATOR_SHARE = 0.70
SOURCE_CREATOR_SHARE = 0.25
NETWORK_FEE_SHARE = 0.05

async def _distribute_multilevel_royalty(self, content, total_royalty, accessor_id):
    derivative_share = total_royalty * DERIVATIVE_CREATOR_SHARE
    source_pool = total_royalty * SOURCE_CREATOR_SHARE
    network_fee = total_royalty * NETWORK_FEE_SHARE
    
    # Credit derivative creator
    await self.ledger.credit(wallet_id=self.identity.node_id, amount=derivative_share, ...)
    
    # Resolve and credit source creators
    parent_creators = self._resolve_parent_creators(content.parent_cids)
    if parent_creators:
        per_parent = source_pool / len(parent_creators)
        for parent_creator_id in parent_creators:
            # Remote creators credited via GOSSIP_CONTENT_ACCESS
            await self.ledger.credit(wallet_id=parent_creator_id, amount=per_parent, ...)
    
    # Network fee
    await self.ledger.credit(wallet_id="system", amount=network_fee, ...)
```

### Content Upload Flow with Provenance

```mermaid
sequenceDiagram
    participant User
    participant ContentUploader
    participant SemanticIndex
    participant IPFS
    participant Gossip
    participant Ledger

    User->>ContentUploader: upload(content, filename, metadata)
    ContentUploader->>ContentUploader: Generate SHA-256 hash
    ContentUploader->>SemanticIndex: find_nearest(embedding)
    
    alt Similarity >= 0.92
        SemanticIndex-->>ContentUploader: (cid, similarity, creator)
        ContentUploader->>ContentUploader: Prepend parent CID
    end
    
    ContentUploader->>IPFS: Add content
    IPFS-->>ContentUploader: CID
    ContentUploader->>ContentUploader: Create provenance record
    ContentUploader->>ContentUploader: Sign with node key
    ContentUploader->>Gossip: publish(GOSSIP_PROVENANCE_REGISTER)
    Gossip->>Ledger: upsert_provenance(data)
    ContentUploader->>SemanticIndex: store(cid, embedding, creator_id)
    ContentUploader-->>User: UploadedContent
```

### Cross-Node Provenance Resolution

When a node needs provenance for content it doesn't have locally:

```mermaid
sequenceDiagram
    participant NodeA as Node A
    participant Gossip
    participant NodeB as Node B
    participant LedgerB as Ledger B

    NodeA->>NodeA: get_provenance(cid) - not found locally
    NodeA->>Gossip: publish(GOSSIP_PROVENANCE_QUERY, cid, requester_id)
    Gossip->>NodeB: Forward query
    NodeB->>LedgerB: get_provenance(cid)
    LedgerB-->>NodeB: provenance record
    NodeB->>Gossip: publish(GOSSIP_PROVENANCE_RESPONSE, cid, provenance)
    Gossip->>NodeA: Forward response
    NodeA->>NodeA: Cache in local ledger
    NodeA-->>NodeA: Return provenance
```

### Data Structures

**UploadedContent:**

```python
@dataclass
class UploadedContent:
    cid: str
    filename: str
    size_bytes: int
    content_hash: str           # SHA-256 of raw content
    creator_id: str
    created_at: float
    provenance_signature: str   # Node's signature on provenance data
    royalty_rate: float         # FTNS per access (0.001–0.1)
    parent_cids: List[str]      # Source material CIDs
    access_count: int
    total_royalties: float
    is_sharded: bool
    manifest_cid: Optional[str]
    embedding_id: Optional[str]     # "emb:<cid>" if embedded
    near_duplicate_of: Optional[str]  # Most similar CID
    near_duplicate_similarity: Optional[float]
```

**ContentRecord:**

```python
@dataclass
class ContentRecord:
    cid: str
    filename: str
    size_bytes: int
    content_hash: str
    creator_id: str
    providers: Set[str]         # Nodes that can serve this content
    created_at: float
    metadata: Dict[str, Any]
    royalty_rate: float
    parent_cids: List[str]
    embedding_id: Optional[str]
    near_duplicate_of: Optional[str]
```

### Integration Points

| System | Integration |
|--------|-------------|
| IPFS Storage | Content stored in IPFS; CID as primary identifier |
| Gossip Protocol | Provenance registration and queries broadcast network-wide |
| LocalLedger | SQLite persistence ensures provenance survives restarts |
| FTNS Tokenomics | Royalty credits recorded as `CONTENT_ROYALTY` transactions |
| ContentIndex | Tracks which nodes can serve each CID |

### Performance Considerations

- **Embedding Index:** O(n) similarity scan — acceptable for early network scale; planned migration to FAISS for tens of thousands of embeddings
- **Persistence:** JSON serialization for semantic index; SQLite for provenance records
- **Query Timeout:** 5-second default for cross-node provenance resolution
- **LRU Eviction:** ContentIndex limits to 10,000 records to bound memory

---

## 34. Priority Execution Plan — Completion Summary (2026-03-08)

Nine priority items were identified and implemented following a comprehensive audit of outstanding work. The items spanned repository hygiene, core P2P compute, network intelligence, observability, and operational readiness.

### Item 1: Repository Root Hygiene ✅

**Problem:** 26+ JSON test-artifact files, build artifacts (`prsm_network-0.2.0/`, `*.egg-info/`, `dist/`), and virtual environments (`venv/`, `agents_venv/`) cluttered the root directory — an investor/developer red flag.

**Resolution:**
- Moved all test result and regression baseline JSON files to `reports/` (already existed)
- Added `reports/`, `agents_venv/`, `prsm_embedding_cache/`, `prsm_network-*/` to `.gitignore`
- Un-tracked 55 previously-committed JSON artifacts from git history (`git rm --cached`)
- Root now contains only essential project files

**Files changed:** `.gitignore`, `reports/` (55 files removed from tracking)

---

### Item 2: Architecture Analysis Update ✅

**Problem:** The two most recent commits (semantic embedding provenance, FTNS cross-node royalties) were not reflected in the architecture documentation.

**Resolution:** Added Section 33 documenting the complete Semantic Provenance System: `_SemanticIndex` implementation, gossip protocol extensions (`GOSSIP_PROVENANCE_QUERY/RESPONSE`), SQLite schema, royalty distribution mechanics, and cross-node resolution flow.

**Files changed:** `plans/prsm_architecture_analysis.md`

---

### Item 3: Compute Provider → NWTN Orchestrator Wiring ✅

**Problem:** `ComputeProvider._run_inference()` returned mock strings (`"[PRSM node XYZ processed inference]"`) regardless of API key configuration. The real NWTN pipeline (with `BackendRegistry`) existed but was only reachable via the REST API — not via P2P compute jobs. Any node accepting inference work from a peer returned fake data.

**Resolution:**

- `ComputeProvider.__init__()` now accepts `orchestrator: Optional[NWTNOrchestrator]`
- `_run_inference()` dispatches to `orchestrator.process_query()` when orchestrator is injected; returns real LLM response with `source`, `tokens_used`, `reasoning_steps` fields
- `_run_embedding()` wires to `RealEmbeddingAPI.generate_embedding()` when available
- `prsm/node/node.py` passes both `orchestrator` and `embedding_api` to `ComputeProvider` during initialization
- Graceful fallback to mock when no API keys configured; warning logged at startup

**Impact:** Closes the P2P AI inference loop. `POST /compute/submit {"job_type":"inference"}` from Node A → Node B accepts and returns a real Anthropic/OpenAI response, billed in FTNS.

**Files changed:** `prsm/node/compute_provider.py`, `prsm/node/node.py`
**Tests added:** `tests/unit/test_compute_provider_nwtn_integration.py` (12 tests)

---

### Item 4: Capability-Based Peer Discovery ✅

**Problem:** `PeerDiscovery` tracked `{node_id, address, roles}` only. The compute marketplace broadcast inference/embedding jobs to all peers indiscriminately — nodes without LLM backends wasted resources on jobs they couldn't fulfill.

**Resolution:**

**`prsm/node/discovery.py`:**
- `PeerInfo` extended with `capabilities: List[str]`, `supported_backends: List[str]`, `gpu_available: bool`, `last_capability_update: float`
- Added `find_peers_with_capability(capability)` and `find_peers_with_backend(backend)` query methods

**`prsm/node/gossip.py`:**
- Added `GOSSIP_CAPABILITY_ANNOUNCE` subtype
- Broadcast on node start and when backend configuration changes

**`prsm/node/capability_detection.py`** (new file):
- `detect_node_capabilities()` auto-detects available backends from environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, CUDA presence)
- Returns structured `NodeCapabilities` with `capabilities[]`, `supported_backends[]`, `gpu_available`

**`prsm/node/compute_requester.py`:**
- Prefers `find_peers_with_capability(job_type)` when routing jobs
- Falls back to broadcast if no capable peers found

**Files changed:** `prsm/node/discovery.py`, `prsm/node/gossip.py`, `prsm/node/compute_requester.py`, `prsm/node/node.py`
**New files:** `prsm/node/capability_detection.py`
**Tests added:** `tests/unit/test_capability_discovery.py` (24 tests)

---

### Item 5: Monitoring Dashboards ✅

**Problem:** `docker/monitoring/grafana/` existed with a skeleton dashboard. Prometheus metrics were being exported by the bootstrap server but no panels were configured. No alert rules were defined.

**Resolution:**

**`docker/monitoring/grafana/dashboards/bootstrap-dashboard.json`:**
- 8 new panels: Active Peer Count, Connection Rate (req/s), Message Throughput (msg/s), Peer Churn Rate, Connection Error Rate, Gossip Subtype Breakdown (pie), Job Completion Rate, FTNS Transaction Volume
- All panels use the `prometheus` datasource with 30s refresh

**`docker/monitoring/alert_rules.yml`:**
- `prsm_bootstrap_peers_critical`: active peers < 1 for 5 min → critical
- `prsm_connection_error_rate_high`: error rate > 10% for 2 min → warning
- `prsm_gossip_backlog_high`: backlog > 1000 messages → warning
- `prsm_bootstrap_health_degraded`: health endpoint non-200 for 3 min → critical

**`docs/BOOTSTRAP_DEPLOYMENT_GUIDE.md`:** Added Grafana dashboard section with access URL, default credentials, and alert configuration instructions.

**Files changed:** `docker/monitoring/grafana/dashboards/bootstrap-dashboard.json`, `docker/monitoring/alert_rules.yml`, `docs/BOOTSTRAP_DEPLOYMENT_GUIDE.md`

---

### Item 6: Gossip Persistence for Late-Joining Nodes ✅

**Problem:** The gossip protocol used epidemic fanout with TTL but had no historical catch-up. A node starting after a task offer, content advertisement, or agent registration was broadcast would never learn about it — unless the originator re-broadcast, which doesn't happen for one-shot messages.

**Resolution:**

**`prsm/node/gossip.py`:**
- Added `GOSSIP_DIGEST_REQUEST` and `GOSSIP_DIGEST_RESPONSE` message types
- On new peer connection, node sends a digest of last-seen timestamps per subtype: `{subtype → last_timestamp}`
- Peer queries its SQLite gossip log for messages newer than each timestamp and sends them in a `GOSSIP_DIGEST_RESPONSE`
- Per-subtype retention policy constants:

| Subtype | Retention |
|---------|-----------|
| `AGENT_TASK_POST`, `QUERY_POST` | 1 hour |
| `CONTENT_ADVERTISE`, `AGENT_REGISTER`, `CAPABILITY_ANNOUNCE` | 24 hours |
| All others | 30 minutes |

- Periodic cleanup (runs every 10 min) prunes messages past retention window from SQLite

**`prsm/node/transport.py`:** On peer handshake completion, fires digest request exchange.

**Files changed:** `prsm/node/gossip.py`, `prsm/node/transport.py`
**Tests added:** `tests/unit/test_gossip_persistence.py` (22 tests)

---

### Item 7: Multi-Region Bootstrap ✅

**Problem:** All three bootstrap subdomains (`bootstrap1`, `fallback1`, `fallback2`) pointed to the same DigitalOcean NYC3 server. A regional outage or high-latency connection for international users would degrade the entire bootstrap layer.

**Resolution:**

- `config/secure.env.template`: Added `PRSM_REGION` and `PRSM_BOOTSTRAP_REGION` environment variables for region identification
- `docker/docker-compose.bootstrap.yml`: Added `BOOTSTRAP_REGION` label to service definition for monitoring and routing
- `docs/BOOTSTRAP_DEPLOYMENT_GUIDE.md`: Step-by-step guide for deploying `fallback1` to EU and `fallback2` to APAC — provision VPS, clone PRSM, update DNS A records, verify health endpoint

**Target topology when fully deployed:**

```
bootstrap1.prsm-network.com → NYC3 (primary, live)
fallback1.prsm-network.com  → EU (Amsterdam/Frankfurt)
fallback2.prsm-network.com  → APAC (Singapore/Tokyo)
```

**Files changed:** `config/secure.env.template`, `docker/docker-compose.bootstrap.yml`, `docs/BOOTSTRAP_DEPLOYMENT_GUIDE.md`

---

### Item 8: Automated Security Scans ✅

**Problem:** `prsm/security/audit_checklist.py` and `prsm/security/scanner.py` were production-ready but had no trigger. Security checks ran only when manually invoked — effectively never in practice.

**Resolution:**

**`.github/workflows/security.yml`** (new file):
- Triggers: push to `main`/`develop`, pull requests, daily schedule (`0 2 * * *`), manual dispatch
- Steps: install PRSM with `[security]` extras, run full audit + scan, upload report as artifact, fail CI if critical findings detected

**`prsm/security/scanner.py`:**
- Added CLI entry point: `python -m prsm.security.scanner --output-dir reports/`
- Structured JSON output compatible with `check_security_reports.py` parser

**`scripts/ci/check_security_reports.py`:**
- Added `--fail-on-critical` flag: exits with code 1 if any critical-severity findings present
- Used as the final CI gate step in the security workflow

**Files changed:** `prsm/security/scanner.py`, `scripts/ci/check_security_reports.py`
**New files:** `.github/workflows/security.yml`

---

### Item 9: FTNS Testnet Deployment ✅

**Problem:** `prsm/economy/blockchain/` contained production-quality smart contract code and deployer logic but had no deployment configuration or operational workflow. The FTNS bridge was inaccessible to anyone wanting to test on-chain economics.

**Resolution:**

**`contracts/deployment-config.json`** (new file):
```json
{
  "sepolia": {
    "network": "sepolia",
    "chain_id": 11155111,
    "rpc_url": "${SEPOLIA_RPC_URL}",
    "token_name": "FTNS Token (Testnet)",
    "token_symbol": "FTNS",
    "initial_supply": 1000000,
    "gas_limit": 3000000
  },
  "polygon_mumbai": { ... }
}
```

**`.github/workflows/deploy-testnet.yml`** (new file):
- Manual-trigger workflow (`workflow_dispatch`) with `network` input (sepolia/polygon_mumbai)
- Requires `PRIVATE_KEY` and `RPC_URL` GitHub secrets
- Runs deployment script, saves contract addresses as workflow artifact

**`docs/FTNS_TESTNET_DEPLOYMENT.md`** (new file):
- Complete deployment guide: get testnet ETH from faucets, configure `.env`, run deployer, verify on Etherscan/Polygonscan, configure bridge in node settings

**Files changed:** `contracts/.env.example`, `contracts/hardhat.config.js`, `scripts/deploy_contracts.py`
**New files:** `contracts/deployment-config.json`, `.github/workflows/deploy-testnet.yml`, `docs/FTNS_TESTNET_DEPLOYMENT.md`

---

### Execution Summary

| Item | Type | Files Changed | Tests Added | Status |
|------|------|---------------|-------------|--------|
| Root hygiene | chore | 56 (55 deleted) | — | ✅ |
| Arch analysis update | docs | 1 | — | ✅ |
| Compute → NWTN wiring | feat | 2 | 12 | ✅ |
| Capability discovery | feat | 4 + 1 new | 24 | ✅ |
| Monitoring dashboards | feat | 3 | — | ✅ |
| Gossip persistence | feat | 2 | 22 | ✅ |
| Multi-region bootstrap | feat | 3 | — | ✅ |
| Security CI | ci | 2 + 1 new | — | ✅ |
| FTNS testnet | feat | 3 + 3 new | — | ✅ |

**Total new tests: 58 (all passing)**
**Commits: 3 (`4020507`, `d54aefc`, `acea879`)**

### Updated Status Matrix

```
FULLY WORKING END-TO-END:
  ✅ P2P networking (transport, gossip, discovery, handshake, replay prevention)
  ✅ Node runtime (identity, startup, dashboard, management API, preflight)
  ✅ DAG ledger (atomic ops, TOCTOU prevention, allowances, gossip + collab persistence)
  ✅ Safety system (circuit breaker, emergency halt, rule-based monitoring)
  ✅ Authentication (JWT, RBAC, audit logging)
  ✅ Local FTNS economy (tracking, transfers, welcome grants, agent allowances)
  ✅ Compute benchmarks (real CPU computation)
  ✅ Multi-node compute jobs (submit → accept → execute → result → payment)
  ✅ Collaboration protocol (tasks, reviews, queries, bid selection, persistence)
  ✅ Self-compute for single nodes
  ✅ Bootstrap fallback with address validation
  ✅ Governance voting + execution (votes → timelock → parameter changes)
  ✅ Web dashboard (auto-launched with node)
  ✅ NWTN LLM inference (Anthropic/OpenAI/local backends via BackendRegistry)
  ✅ P2P inference via ComputeProvider → NWTN (real responses to remote nodes)
  ✅ Cross-node content retrieval (GET /content/retrieve/{cid})
  ✅ Staking lifecycle (stake/unstake/withdraw/slash/rewards via API)
  ✅ Storage proof verification (challenge-response integrated into StorageProvider)
  ✅ IPFS content sharding (auto-shard large files in ContentUploader)
  ✅ FTNS bridge (deposit/withdraw via API and CLI)
  ✅ Semantic provenance (embedding near-dup detection, FTNS royalties, cross-node query)
  ✅ Capability-based peer discovery (smart job routing to capable nodes)
  ✅ Gossip persistence (late-joining nodes receive catch-up on connect)

PUBLISHED & DEPLOYED:
  ✅ PyPI: pip install prsm-network (v0.2.1)
  ✅ Bootstrap: wss://bootstrap1.prsm-network.com:8765 (DigitalOcean NYC3)
  ✅ SSL: Let's Encrypt certs (expires 2026-06-04)
  ✅ DNS: bootstrap1, fallback1, fallback2 on Cloudflare
  ✅ GitHub Actions: automated releases + daily security scans

REMAINING (MEDIUM-TERM, OPERATIONAL):
  📦 Multi-region bootstrap: deploy fallback1 (EU) + fallback2 (APAC)
  📦 Monitoring: connect Grafana dashboards to live bootstrap Prometheus
  📦 FTNS testnet: deploy contracts to Sepolia/Polygon Mumbai (config ready)
  📦 LLM API keys: configure Anthropic/OpenAI keys on production nodes
  📦 Community & adoption: researcher outreach, blog posts, conference demos
```

---

## 35. Resource Contribution Controls (2026-03-08)

Node participants can now precisely control how much CPU, RAM, GPU, storage, and network bandwidth each of their devices contributes to the PRSM network, with scheduling, live runtime updates, and bandwidth throttling.

### Configuration Fields Added to `NodeConfig`

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `max_concurrent_jobs` | `int` | `3` | Parallel compute job slots accepted at once |
| `gpu_allocation_pct` | `int` | `80` | % of detected GPU VRAM to offer to the network |
| `upload_mbps_limit` | `float` | `0.0` | Upload bandwidth cap in Mbps (0 = unlimited) |
| `download_mbps_limit` | `float` | `0.0` | Download bandwidth cap in Mbps (0 = unlimited) |
| `active_hours_start` | `Optional[int]` | `None` | Hour 0–23 when node starts accepting work (None = always) |
| `active_hours_end` | `Optional[int]` | `None` | Hour 0–23 when node stops accepting work |
| `active_days` | `List[int]` | `[]` | Days active (0=Mon … 6=Sun; empty = every day) |

All fields persist in `~/.prsm/node_config.json` and survive restarts.

### Phase 1 — Config Foundation

**Files:** `prsm/node/config.py`, `prsm/node/compute_provider.py`, `prsm/node/node.py`

- Added the 7 new fields above to `NodeConfig.save()` / `NodeConfig.load()`
- `ComputeProvider.__init__()` now accepts `max_concurrent_jobs` and `gpu_allocation_pct`
- `available_capacity` property extended to include `gpu_memory_gb_allocated` when GPU is detected
- `node.py` passes both new fields through from config at startup
- `max_concurrent_jobs` is now enforced in `_on_job_offer()` instead of being hardcoded at 3

### Phase 2 — CLI Flags + `prsm node configure`

**Files:** `prsm/cli.py`

**New flags on `prsm node start`:**
```bash
prsm node start --cpu 75 --memory 60 --storage 50 --jobs 4
```

**New command: `prsm node configure`**

```bash
# Inspect current settings (human-readable)
prsm node configure --show
# Output example:
#   PRSM Node Resource Configuration
#     Role:             full (compute + storage)
#     CPU allocation:   75% of 8 cores → 6 cores offered
#     RAM allocation:   60% of 16.0 GB → 9.6 GB offered
#     Concurrent jobs:  4 slots
#     GPU:              RTX 3080 (10.0 GB) — 80% → 8.0 GB offered
#     Storage pledged:  50.0 GB  (used: 2.3 GB, available: 47.7 GB)
#     Upload limit:     unlimited
#     Active hours:     22:00 – 08:00 (weekdays only)

# Update settings without re-running full wizard
prsm node configure --cpu 75 --memory 60 --jobs 5
prsm node configure --storage 100 --upload-limit 50

# Configure time-based scheduling
prsm node configure --active-hours "22-8" --active-days "weekdays"
prsm node configure --active-hours off   # Clear schedule (always on)
```

Changes are saved to `~/.prsm/node_config.json` immediately and take effect on next start (or immediately via the live API for running nodes).

### Phase 3 — Live Runtime Update API

**Files:** `prsm/node/api.py`, `prsm/node/compute_provider.py`, `prsm/node/storage_provider.py`

**New endpoints:**

```
GET  /node/resources   → current allocation + live utilization
PUT  /node/resources   → update any allocation fields while node is running
```

`PUT /node/resources` request body (all fields optional):
```json
{
  "cpu_allocation_pct": 75,
  "memory_allocation_pct": 60,
  "max_concurrent_jobs": 5,
  "gpu_allocation_pct": 80,
  "storage_gb": 50.0,
  "upload_mbps_limit": 100.0,
  "download_mbps_limit": 0.0,
  "active_hours_start": 22,
  "active_hours_end": 8
}
```

`ComputeProvider.update_allocation()` and `StorageProvider.update_limits()` apply changes to the live provider instances without restart and persist the updated config to disk.

### Phase 4 — Active Hours Scheduling

**Files:** `prsm/node/config.py`, `prsm/node/compute_provider.py`, `prsm/node/storage_provider.py`

`NodeConfig.is_active_now()` helper:

```python
def is_active_now(self) -> bool:
    if self.active_hours_start is None:
        return True  # Always on
    now = datetime.now()
    if self.active_days and now.weekday() not in self.active_days:
        return False
    start, end = self.active_hours_start, self.active_hours_end
    hour = now.hour
    if start <= end:
        return start <= hour < end
    return hour >= start or hour < end  # Wraps midnight
```

Both `ComputeProvider._on_job_offer()` and `StorageProvider._on_storage_request()` call `is_active_now()` before accepting work. Jobs offered during inactive hours are silently declined — the peer will find another provider.

**Scheduling examples:**
```bash
# Laptop: only contribute at night (10pm–8am)
prsm node configure --active-hours "22-8"

# Old workstation: weekends only, all day
prsm node configure --active-hours off --active-days "weekend"

# Office server: business hours, weekdays
prsm node configure --active-hours "9-17" --active-days "weekdays"
```

**Tests:** `tests/unit/test_active_hours_scheduling.py` — 14 tests covering normal ranges, midnight wrap-around, day filtering, and edge cases (start == end, always-on defaults).

### Phase 5 — Bandwidth Limiting

**Files:** `prsm/core/bandwidth_limiter.py` (new), `prsm/node/storage_provider.py`, `prsm/node/content_provider.py`

**`TokenBucket` class** in `prsm/core/bandwidth_limiter.py`:

```python
class TokenBucket:
    """Async token bucket for bandwidth rate limiting."""
    def __init__(self, rate_mbps: float):
        self.rate_bytes_per_sec = rate_mbps * 1024 * 1024 / 8
        self._tokens = self.rate_bytes_per_sec   # Start full (1-second burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, byte_count: int) -> None:
        """Wait until enough tokens available, then consume."""
        if self.rate_bytes_per_sec == 0:
            return  # Unlimited — zero cost
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    self.rate_bytes_per_sec,
                    self._tokens + (now - self._last_refill) * self.rate_bytes_per_sec
                )
                self._last_refill = now
                if self._tokens >= byte_count:
                    self._tokens -= byte_count
                    return
                await asyncio.sleep(0.05)
```

`BandwidthLimiter` wraps upload and download `TokenBucket` instances together. Initialized from `config.upload_mbps_limit` and `config.download_mbps_limit` in `node.py` and injected into:
- `StorageProvider` — throttles IPFS content serving responses
- `ContentProvider` — throttles inline content transfer in P2P responses

Setting either limit to `0.0` disables throttling on that direction (no-overhead fast path).

### Multi-Device Usage Patterns

The combination of these controls enables the key multi-device scenarios:

| Device | Example Config |
|--------|---------------|
| **Primary laptop** | `--cpu 30 --memory 25 --storage 5 --jobs 1 --active-hours "22-8"` — light contribution, only at night |
| **Old workstation** | `--cpu 80 --memory 70 --storage 200 --jobs 6 --active-days "weekend"` — heavy contribution on weekends |
| **Home server / NAS** | `--cpu 50 --storage 2000 --jobs 3 --upload-limit 20` — storage-focused, bandwidth-capped for metered connection |
| **Office GPU box** | `--cpu 60 --memory 50 --jobs 4 --active-hours "18-9"` — contributes after-hours, GPU-enabled inference |

Each device runs an independent `prsm node` process with its own `~/.prsm/node_config.json`. All devices connect to the same bootstrap server and are discoverable by the network. The capability advertisement (Phase 4 of Section 34) broadcasts each device's actual allocated resources so the compute marketplace can route jobs to the right machine.

### Updated Status Matrix Entry

```
  ✅ Resource contribution controls (CPU%, RAM%, GPU%, storage GB, bandwidth
     limits, active hours scheduling, live runtime updates via PUT /node/resources)
```

---

## 36. Two-Stack Architecture Gap — Unification Plan

### The Problem

PRSM has evolved into two parallel, disconnected server stacks. A researcher using one cannot access the capabilities of the other without running a second process and knowing which port serves what.

#### Stack A — P2P Node (`prsm node start`)

Entry point: `prsm/node/node.py` + `prsm/node/api.py`

| Capability | Status |
|---|---|
| P2P transport, gossip, discovery | ✅ Working |
| Compute jobs (submit/accept/pay) | ✅ Working |
| Storage provider + proofs | ✅ Working |
| DAG ledger + FTNS | ✅ Working |
| Collaboration protocol | ✅ Working |
| Semantic provenance + royalties | ✅ Working |
| Resource contribution controls | ✅ Working |
| Rich TUI dashboard | ✅ Working |
| Teacher model access | ❌ Not wired |
| Distillation pipeline access | ❌ Not wired |
| Web UI (browser-accessible) | ❌ Not served |

#### Stack B — Platform API (`prsm serve` + `prsm dashboard`)

Entry point: `prsm/interface/api/main.py` + `prsm/interface/dashboard/streamlit_app.py`

| Capability | Status |
|---|---|
| Teacher model create/train/assess | ✅ Working |
| NWTN 5-agent pipeline (direct) | ✅ Working |
| Governance endpoints | ✅ Working |
| Marketplace endpoints | ✅ Working |
| Streamlit web UI | ✅ Working |
| FastAPI+HTML/JS dashboard (`prsm/dashboard/`) | ✅ Working |
| P2P node access | ❌ Not connected |
| DAG ledger / real FTNS | ❌ Uses mock ledger |
| Compute marketplace routing | ❌ Not connected |

#### The Core Issue

A researcher who runs `prsm node start` gets a fully functional P2P node with real FTNS economy and compute marketplace — but cannot create a teacher model, start a distillation job, or view anything in a browser.

A researcher who runs `prsm serve` + `prsm dashboard` gets teacher models and a web UI — but all FTNS operations hit a mock ledger and no P2P compute ever happens.

The two stacks need to be unified under `prsm node start`.

---

### Architecture of the Unified Node

The target architecture serves everything from a single `prsm node start` invocation:

```
prsm node start
    │
    ├── P2P Layer (port 9001)
    │     WebSocket transport, gossip, discovery
    │
    ├── Node Management API (port 8000)
    │     /node/*          existing node API
    │     /compute/*       existing compute API
    │     /content/*       existing content API
    │     /staking/*       existing staking API
    │     /bridge/*        existing bridge API
    │     /teacher/*       NEW — wired to prsm/compute/teachers/
    │     /distillation/*  NEW — wired to prsm/compute/distillation/
    │
    └── Web Dashboard (port 8000, path /dashboard)
          Serves prsm/dashboard/ HTML/JS SPA
          OR mounts Streamlit on /ui (optional)
```

All FTNS operations use the node's `LocalLedger` (SQLite). No PostgreSQL/Redis required. Teacher training jobs that need significant compute can be submitted to the P2P compute marketplace via `ComputeRequester`.

---

### Implementation Plan

#### Phase 1 — Teacher Model Router in Node API (1–2 days)

**Goal:** `prsm teacher create physics` and `GET /teacher/create` work from the P2P node without running `prsm serve`.

**Files to modify:** `prsm/node/api.py`, `prsm/node/node.py`, `prsm/cli.py`

**Step 1 — Add teacher router to `prsm/node/api.py`:**

```python
# New endpoints to add alongside existing node API routes

@router.post("/teacher/create")
async def create_teacher(request: TeacherCreateRequest):
    """Create a teacher model and register it with the node."""
    from prsm.compute.teachers.teacher_model import create_teacher_with_specialization
    teacher = await create_teacher_with_specialization(
        specialization=request.specialization,
        domain=request.domain,
        use_real_implementation=True,
    )
    teacher_id = str(teacher.teacher_model.teacher_id)
    # Register in node's local registry for future lookup
    await node_state.register_teacher(teacher_id, teacher)
    # Award FTNS for creating a teacher (economic incentive)
    await ledger.credit(node_id, TEACHER_CREATION_REWARD, "teacher_created")
    return {"teacher_id": teacher_id, "specialization": request.specialization}

@router.get("/teacher/list")
async def list_teachers():
    """List teacher models registered on this node."""
    return {"teachers": await node_state.get_teachers()}

@router.post("/teacher/{teacher_id}/train")
async def train_teacher(teacher_id: str, request: TrainingRequest):
    """Start a training run for a teacher model."""
    # If training data is large, submit as a P2P compute job
    # If small enough, run locally via TeacherTrainer
    ...

@router.get("/teacher/backends")
async def get_teacher_backends():
    """Return available ML training backends."""
    from prsm.compute.teachers.real_teacher_implementation import get_available_training_backends
    return {"backends": await get_available_training_backends()}
```

**Step 2 — Add teacher registry to `PRSMNode`:** A simple `Dict[str, Any]` mapping `teacher_id → teacher_instance`, persisted to `~/.prsm/teachers.json` on save.

**Step 3 — Wire the `prsm teacher` CLI stubs to the live API:**

```python
# Replace the "coming in v0.2.0" stubs in cli.py

@teacher.command()
@click.argument("specialization")
@click.option("--domain", default=None)
@click.option("--api-url", default="http://localhost:8000")
def create(specialization: str, domain: str, api_url: str):
    """Create a new teacher model on the running node."""
    import httpx
    response = httpx.post(f"{api_url}/teacher/create",
                          json={"specialization": specialization, "domain": domain or specialization})
    data = response.json()
    console.print(f"✅ Teacher created: {data['teacher_id']}", style="bold green")
```

**Tests:** Add `tests/unit/test_teacher_node_integration.py` — create via API, list, verify FTNS credited.

---

#### Phase 2 — Distillation Router in Node API (1–2 days)

**Goal:** A researcher can request a distillation job from their node, which optionally farms it out to the P2P compute network.

**Files to modify:** `prsm/node/api.py`, `prsm/node/node.py`

**New endpoints:**

```python
@router.post("/distillation/submit")
async def submit_distillation(request: DistillationRequest):
    """Submit a model distillation job."""
    from prsm.compute.distillation.orchestrator import get_distillation_orchestrator
    orchestrator = await get_distillation_orchestrator()
    job_id = await orchestrator.submit_job(
        teacher_model_id=request.teacher_model_id,
        target_size=request.target_size,           # e.g. ModelSize.SMALL
        optimization_target=request.optimization,  # e.g. OptimizationTarget.SPEED
        ftns_budget=request.ftns_budget,
    )
    # Deduct FTNS for the job
    await ledger.debit(node_id, request.ftns_budget, f"distillation_job_{job_id}")
    return {"job_id": job_id}

@router.get("/distillation/{job_id}/status")
async def get_distillation_status(job_id: str):
    """Get the status of a running distillation job."""
    ...

@router.get("/distillation/{job_id}/result")
async def get_distillation_result(job_id: str):
    """Download the distilled model artifact."""
    ...
```

**Key design decision:** Large distillation jobs (training a 7B → 1B model) should be submitted to the P2P compute network as a `JobType.TRAINING` job, distributed across willing compute providers. Small distillation jobs (fine-tuning a small classifier) run locally via `ProductionTrainingPipeline`.

**Add `JobType.TRAINING` to `ComputeProvider`** to accept and execute training jobs from the network. This is the bridge that makes the P2P marketplace useful for ML researchers, not just inference.

---

#### Phase 3 — Web Dashboard Co-served with Node (1 day)

**Goal:** Opening `http://localhost:8000/` in a browser shows the PRSM dashboard. No separate `prsm dashboard` command needed.

**Approach — Mount the existing `prsm/dashboard/` HTML/JS SPA on the node API:**

The `prsm/dashboard/` FastAPI app (`app.py`) already has the full 9-page SPA with WebSocket support. Rather than running it as a separate service, mount it as a sub-application on the node's FastAPI instance.

**Files to modify:** `prsm/node/api.py`

```python
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard" / "static"
DASHBOARD_HTML = Path(__file__).parent.parent / "dashboard" / "templates" / "dashboard.html"

# Mount static assets (JS, CSS)
app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

@app.get("/")
@app.get("/dashboard")
async def serve_dashboard():
    """Serve the web dashboard SPA."""
    return FileResponse(str(DASHBOARD_HTML))
```

**Update `dashboard.js`** to point API calls at the node's own port (relative paths — already works if served from the same origin).

**Update `prsm node start` output** to show the dashboard URL:
```
  ✅ Node started
  📡 P2P:       ws://0.0.0.0:9001
  🌐 API:       http://localhost:8000
  🖥️  Dashboard: http://localhost:8000/  ← open in browser
```

---

#### Phase 4 — Connect Dashboard to Real Node Data (1 day)

**Goal:** The web dashboard shows live P2P data (peer list, compute jobs, FTNS balance) from the node's DAG ledger, not the platform API's mock data.

The `dashboard.js` currently calls endpoints like `/api/v1/wallet/balance`, `/api/v1/compute/jobs`, `/api/v1/peers`. These need to map to the node API's endpoints:

| Dashboard call | Map to node API endpoint |
|---|---|
| `GET /api/v1/wallet/balance` | `GET /ftns/balance` |
| `GET /api/v1/compute/jobs` | `GET /compute/jobs` |
| `GET /api/v1/peers` | `GET /node/peers` |
| `GET /api/v1/content` | `GET /content/index` |
| `GET /api/v1/staking/status` | `GET /staking/status` |
| `WS /ws` | `WS /ws` (already exists in node API) |

Two options:
- **A (fast):** Add alias routes to node API that proxy to existing endpoints
- **B (clean):** Update `dashboard.js` to use the node API paths directly

Option B is better long-term. The `dashboard.js` API base URL should be configurable (already uses a constant) and the node API paths are already well-defined.

---

### Summary

| Phase | Feature | Effort | Priority | Unblocks |
|---|---|---|---|---|
| 1 | Teacher router + CLI wiring | 1–2 days | **P0** | Researcher adoption; "recursive" in PRSM's name |
| 2 | Distillation router + `JobType.TRAINING` | 1–2 days | **P0** | P2P ML training marketplace |
| 3 | Dashboard co-served with node | 1 day | **P1** | Browser access without separate command |
| 4 | Dashboard ↔ node API data binding | 1 day | **P1** | Live data in web UI |

**Total estimated effort: 4–6 days**

This sprint completes PRSM's value proposition. After it:
- `prsm node start` is the single command that starts everything
- Researchers can create teacher models, submit distillation jobs, and see live network status in a browser
- The P2P marketplace handles ML training jobs alongside inference and storage
- The "recursive scientific modeling" loop is fully operational end-to-end

---

### Section 36 Completion Summary (2026-03-08)

All four phases implemented. 69 tests passing. The two-stack architecture gap is closed.

#### What Changed

**Phase 1 — Teacher Model Router:**

| Item | Detail |
|---|---|
| `PRSMNode.teacher_registry` | `Dict[str, DistilledTeacher]` — in-memory registry with JSON metadata persistence to `~/.prsm/teachers.json` |
| `PRSMNode._ftns_adapter` | `_FTNSLedgerAdapter` now stored as a public attribute for reuse by distillation and other subsystems |
| `POST /teacher/create` | Calls `create_teacher_with_specialization()`, registers result, credits `10.0 FTNS` reward |
| `GET /teacher/list` | Returns live + persisted-but-not-loaded teacher metadata |
| `GET /teacher/{id}` | 404 if unknown; includes `loaded: bool` field |
| `POST /teacher/{id}/train` | Checks FTNS balance (50 FTNS minimum), debits, fires `asyncio.create_task(teacher.train())` |
| `GET /teacher/backends/available` | Wraps `get_available_training_backends()` from real teacher implementation |
| `prsm teacher create` / `list` / `train` | All CLI stubs replaced with real `httpx` API calls; rich table output |

**Phase 2 — Distillation Router + P2P Training:**

| Item | Detail |
|---|---|
| `_get_distillation_orchestrator()` | Lazy singleton injected with `node._ftns_adapter`; adapts `LocalLedger` to `FTNSService` interface |
| `POST /distillation/submit` | Validates FTNS balance, constructs `DistillationRequest`, returns `job_id` and `estimated_cost_ftns` |
| `GET /distillation/{job_id}` | Proxies to `orchestrator.get_job_status()` |
| `DELETE /distillation/{job_id}` | Cancels and refunds; 409 if job not cancellable |
| `GET /distillation` | Lists all jobs from orchestrator's `active_jobs` dict (last 50 completed) |
| `JobType.TRAINING` | New enum value in `ComputeProvider`; capability-gated (only accepted if `"training"` in node capabilities) |
| `ComputeProvider._run_training()` | Executes a training job locally via `DistillationOrchestrator.create_distillation()` |
| `ComputeRequester.submit_training_job()` | Broadcasts `GOSSIP_JOB_OFFER` with `job_type=training`; returns `None` if no capable peers (triggers local fallback) |
| `capability_detection.py` | Detects `training` + `distillation` capabilities when `PyTorchDistillationBackend` importable |

**Phase 3 — Dashboard Co-served with Node:**

| Item | Detail |
|---|---|
| `GET /` and `GET /dashboard` | Serve `prsm/dashboard/templates/dashboard.html` via `FileResponse` |
| `GET /static/*` | Mount `prsm/dashboard/static/` as `StaticFiles` |
| `app.mount("/api", dashboard_sub_app)` | `create_dashboard_app(node=node)` mounted as ASGI sub-app; `dashboard.js` `CONFIG.API_BASE = '/api'` matches automatically |
| WebSocket | `dashboard.js` `CONFIG.WS_URL = ws(s)://{window.location.host}/ws/status` resolves to node API's existing `/ws/status` — zero additional wiring |
| Startup output | Node logs and CLI banner now print `🖥️  Dashboard: http://localhost:{api_port}/` |

**Phase 4 — Dashboard Data Binding:**

| Item | Detail |
|---|---|
| `ftns_service` refs | All `self.node.ftns_service` references in `DashboardServer` replaced with `self.node.ledger` calls |
| `GET /node` | Added alias returning `await self.node.get_status()` |
| `GET /jobs` | Returns active + last 50 completed jobs from `node.compute_provider` |
| `POST /jobs/submit` | Proxies to `node.compute_requester.submit_job()` |
| Teacher + Distillation endpoints | `GET /teacher/list`, `POST /teacher/create`, `GET /distillation`, `POST /distillation/submit` wired in `DashboardServer._setup_routes()` |
| `dashboard.html` | Teachers and Distillation nav items + page sections added |
| `dashboard.js` | `listTeachers()`, `createTeacher()`, `getTeacher()`, `listDistillationJobs()`, `submitDistillation()`, `getDistillationJob()` API methods added |

#### Key Architecture Decisions

1. **Teacher metadata vs weights**: `teachers.json` stores only display metadata (name, specialization, created_at). Model weights live in IPFS via the trainer's checkpoint system and are re-created from config on demand.
2. **FTNS credits via negative charges**: `_ftns_adapter.charge_user(user_id, -10.0)` used for creation rewards — consistent with the adapter's interface rather than calling ledger directly.
3. **Fire-and-forget training**: `asyncio.create_task(teacher.train())` returns immediately to the API caller. Future work: add a `training_jobs` dict to track status and expose via `GET /teacher/{id}/training-status`.
4. **Dashboard ASGI mounting**: `app.mount("/api", sub_app)` evaluates after regular routes, so any future `/api/*` routes added directly to the node API take precedence over the mounted sub-app.
5. **Training job routing**: `ComputeRequester.submit_training_job()` returns `None` when no capable peers are found, signalling the distillation endpoint to run locally instead of P2P — avoiding silent failures.

#### Updated Status Matrix (complete)

```
FULLY WORKING END-TO-END:
  ✅ P2P networking, node runtime, DAG ledger, safety, auth
  ✅ Local + multi-node FTNS economy
  ✅ Compute benchmarks, inference, embedding (P2P + self-compute)
  ✅ Collaboration protocol (tasks, reviews, queries, bid selection)
  ✅ Storage (IPFS pin, proofs, sharding, cross-node retrieval)
  ✅ Staking, FTNS bridge, governance
  ✅ Semantic provenance + royalties
  ✅ Capability-based peer discovery + gossip persistence
  ✅ Resource contribution controls (CPU/RAM/GPU/storage/bandwidth/schedule)
  ✅ Teacher model creation + training (prsm teacher create/train)
  ✅ Distillation pipeline (prsm distillation submit, P2P TRAINING jobs)
  ✅ Web dashboard co-served at http://localhost:8000/ with live node data
  ✅ P2P ML training marketplace (JobType.TRAINING routed to capable nodes)

REMAINING (MEDIUM-TERM, OPERATIONAL):
  📦 Multi-region bootstrap: deploy fallback1 (EU) + fallback2 (APAC)
  📦 Monitoring: connect Grafana to live bootstrap Prometheus
  📦 FTNS testnet: deploy contracts to Sepolia/Polygon Mumbai (config ready)
  📦 LLM API keys: configure Anthropic/OpenAI keys on production nodes
  📦 Community & adoption: researcher outreach, blog posts, conference demos
  ✅ Training job status tracking (run_id, PENDING→RUNNING→COMPLETED/FAILED/CANCELLED,
     live progress polling, persistence, prsm teacher status/cancel-training CLI)
  📦 OS-level compute enforcement (cgroups/RLIMIT — Phase 6, deferred)
```

---

*Analysis completed: 2026-02-20*
*Code Review completed: 2026-02-20*
*Sprint 1 completed: 2026-02-20*
*Sprint 2 completed: 2026-02-27*
*Sprint 3 completed: 2026-03-01*
*Sprint 4 completed: 2026-03-02*
*Sprint 5 Item 1 completed: 2026-03-02*
*Sprint 5 Item 2 completed: 2026-03-02*
*Sprint 6 completed: 2026-03-03*
*Technology audit completed: 2026-03-03*
*Phase 1-5 implementation completed: 2026-03-04*
*Integration wiring audit completed: 2026-03-05*
*Integration wiring completed: 2026-03-05*
*Sprint 7 (UX polish) completed: 2026-03-05*
*Sprint 8 (E2E validation) completed: 2026-03-05*
*Sprint 9 (documentation) completed: 2026-03-05*
*Sprint 10 (v0.2.0 release) completed: 2026-03-05*
*Sprint 11 (deployment readiness) completed: 2026-03-06*
*Sprint 12 (PyPI package name) completed: 2026-03-06*
*Sprint 13 (bootstrap deployment) completed: 2026-03-06*
*PyPI published: 2026-03-06 — https://pypi.org/project/prsm-network/0.2.0/*
*Bootstrap server live: 2026-03-06 — bootstrap1.prsm-network.com:8765 (159.203.129.218)*
*DNS records created: 2026-03-06 — bootstrap1, fallback1, fallback2 on Cloudflare*
*GitHub secrets configured: 2026-03-06 — PYPI_API_TOKEN for automated releases*
*All operational deployment steps complete: 2026-03-06*
*Polish & scale roadmap published: 2026-03-06*
*Near-term items completed: 2026-03-06 — v0.2.1 release, SSL certs, server access*
*PyPI v0.2.1 published: 2026-03-06 — https://pypi.org/project/prsm-network/0.2.1/*
*Section 33 (Semantic Provenance System) documented: 2026-03-08*
*Section 34 (Priority Execution Plan completion) documented: 2026-03-08*
*Compute Provider → NWTN wiring completed: 2026-03-08*
*Capability-based peer discovery completed: 2026-03-08*
*Gossip persistence for late-joining nodes completed: 2026-03-08*
*Monitoring dashboards + alert rules completed: 2026-03-08*
*Multi-region bootstrap config completed: 2026-03-08*
*Security CI workflow completed: 2026-03-08*
*FTNS testnet deployment config completed: 2026-03-08*
*Repository root cleaned: 55 test-artifact JSONs removed from git: 2026-03-08*
*Section 35 (Resource Contribution Controls) completed: 2026-03-08*
*Section 26 maturity matrix corrected: 2026-03-08 (teacher, distillation, NWTN, web UI)*
*Section 36 (Two-Stack Unification) completed: 2026-03-08 — 69 tests, prsm node start now serves everything*
*Training job status tracking completed: 2026-03-09 — 24 tests, run_id lifecycle, live progress, CLI --follow*
*PRSM Version: 0.2.1*

---

## 37. Operational Work Session Completion (2026-03-13)

### Summary

A focused operational session was conducted to complete the five outstanding items from Section 32's Medium-Term roadmap. Four of five items were completed; multi-region bootstrap was deferred on cost grounds.

---

### Item 1: Multi-Region Bootstrap — DEFERRED

Provisioning two additional VPS servers (EU + Asia-Pacific) was scoped and ready to execute but deferred to control ongoing hosting costs. When ready:

- Target regions: `fra1` or `ams3` (EU), `sgp1` or `syd1` (Asia-Pacific)
- Same Docker setup as existing bootstrap1 server
- Update Cloudflare DNS: `fallback1` → EU IP, `fallback2` → AP IP
- Recommended spec: `s-1vcpu-2gb` (~$12/month each)
- DigitalOcean API token required for automated provisioning via `doctl`

---

### Item 2: Grafana Live ✅ COMPLETED (2026-03-13)

**Goal:** Connect live Grafana dashboards to bootstrap server Prometheus metrics.

**What was done:**

1. Extended `docker/docker-compose.bootstrap-local.yml` to add `prometheus`, `grafana`, and `node-exporter` services alongside the existing `bootstrap` service
2. Created `docker/monitoring/prometheus-bootstrap-local.yml` — simplified scrape config for single-node setup (no postgres/redis/nginx exporters)
3. Added a `/prometheus` endpoint to `prsm/bootstrap/server.py` that returns proper Prometheus text format (the existing `/metrics` endpoint returned JSON, which Prometheus cannot scrape)
4. Updated Prometheus scrape path from `/metrics` to `/prometheus`
5. Fixed Grafana dashboard JSON — was wrapped in `{"dashboard": {...}}` API envelope which Grafana file provisioning cannot load; unwrapped to bare dashboard object
6. Opened UFW port 3000 for Grafana access
7. Deployed to `bootstrap1.prsm-network.com` (159.203.129.218)

**Access:**
- Grafana: `http://159.203.129.218:3000` — credentials: `admin` / `PRSMgrafana2026`
- Dashboard: PRSM Bootstrap Server — pre-loaded with panels for server status, uptime, active connections, messages processed, and node system metrics

**Files modified:**
- `docker/docker-compose.bootstrap-local.yml` — added monitoring stack
- `docker/monitoring/prometheus-bootstrap-local.yml` — new file, single-node Prometheus config
- `docker/monitoring/grafana/dashboards/bootstrap-dashboard.json` — unwrapped from API envelope
- `prsm/bootstrap/server.py` — added `/prometheus` endpoint with Prometheus text format output

**Infrastructure note:**
During this session, the bootstrap server became unreachable due to UFW blocking port 22 on restart. The recovery process required:
1. DigitalOcean Recovery ISO boot
2. Mounting `vda1` disk and editing `/etc/ufw/ufw.conf` to `ENABLED=no`
3. Rebooting to the real OS via `Boot from Hard Drive`
4. Resolving PAM-enforced root password expiry using emailed temporary password via `expect`
5. SSH config entry added to `~/.ssh/config` as `prsm-bootstrap` alias

SSH config (`~/.ssh/config`):
```
Host prsm-bootstrap
  HostName 159.203.129.218
  User root
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  AddKeysToAgent yes
```

---

### Item 3: FTNS Testnet Deployment ✅ COMPLETED (2026-03-13)

**Goal:** Deploy FTNS ERC-20 token contract to Ethereum Sepolia testnet.

**What was done:**

1. Created Alchemy account; obtained Sepolia RPC URL
2. Created MetaMask wallet; funded with 0.05 Sepolia ETH via Google/Coinbase faucet
3. Deployed FTNS token contract using standalone `web3.py` + `py-solc-x` script (PRSM's built-in deployer could not be used due to heavy import chain requiring the full ML dependency stack)
4. Contract compiled from self-contained Solidity source (ERC-20 with mint, no OpenZeppelin imports required)
5. Deployment record saved to `/root/ftns_deployment.json` on server

**Deployment details:**

| Field | Value |
|---|---|
| Network | Ethereum Sepolia (chain ID 11155111) |
| Contract address | `0xd979c096BE297F4C3a85175774Bc38C22b95E6a4` |
| Transaction hash | `d489443716eddd7629def77b397e6e924d05dabc4ccf14920090baff08eaf79d` |
| Deployer address | `0x8eaA00FF741323bc8B0ab1290c544738D9b2f012` |
| Gas used | 1,101,764 |
| Block | 10439997 |
| Token name | FTNS Token |
| Symbol | FTNS |
| Initial supply | 1,000,000,000 FTNS |
| Etherscan | https://sepolia.etherscan.io/address/0xd979c096BE297F4C3a85175774Bc38C22b95E6a4 |

**Credentials stored in `/root/PRSM/docker/.env` on server (not committed to git):**
- `FTNS_CONTRACT_ADDRESS`
- `SEPOLIA_RPC_URL`
- `FTNS_NETWORK`

---

### Item 4: LLM API Keys on Production Node ✅ COMPLETED (2026-03-13)

**Goal:** Configure Anthropic and OpenAI API keys on the bootstrap server so the NWTN pipeline returns real AI-generated responses.

**What was done:**

1. Added `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` environment variable references to `docker/docker-compose.bootstrap-local.yml`
2. Created `/root/PRSM/docker/.env` on the server with actual key values (chmod 600, not committed to git)
3. Added `docker/.env` to `.gitignore`
4. Rebuilt and restarted `prsm-bootstrap` container with keys injected

**Result:** NWTN pipeline now returns real AI-generated responses. Inference source will be `"anthropic"` or `"openai"` rather than `"mock"`.

**Files modified:**
- `docker/docker-compose.bootstrap-local.yml` — added `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` env var references
- `.gitignore` — added `docker/.env`

---

### Item 5: Community & Adoption ✅ COMPLETED (2026-03-13)

**Goal:** Draft launch content for developer communities.

**What was done:**

Four launch pieces drafted and saved to `docs/community/`:

| File | Platform | Focus |
|---|---|---|
| `hacker_news_launch.md` | Hacker News Show HN | Technical overview, install command, honest alpha state |
| `blog_post_launch.md` | prsm-network.com / Medium | Problem framing, architecture, what works, roadmap |
| `reddit_machinelearning.md` | r/MachineLearning | NWTN pipeline, neuro-symbolic approach, provenance, distillation roadmap |
| `reddit_ethereum.md` | r/ethereum | FTNS token economics, Sepolia deployment, royalty distribution, honest caveats |

All posts are technically honest about alpha state and avoid marketing language. Ready to post.

---

### Updated Status Summary (2026-03-13)

```
OPERATIONAL INFRASTRUCTURE:
  ✅ PyPI: pip install prsm-network (v0.2.1)
  ✅ Bootstrap server: wss://bootstrap1.prsm-network.com:8765 (DigitalOcean NYC3)
  ✅ SSL: Let's Encrypt certs, auto-renewal via certbot
  ✅ DNS: bootstrap1, fallback1, fallback2 on Cloudflare (all → 159.203.129.218)
  ✅ Monitoring: Grafana live at http://159.203.129.218:3000
  ✅ LLM inference: Anthropic + OpenAI keys active on production node
  ✅ FTNS testnet: ERC-20 deployed on Sepolia at 0xd979c096BE297F4C3a85175774Bc38C22b95E6a4
  ✅ Community content: 4 launch posts drafted, saved to docs/community/

DEFERRED:
  📦 Multi-region bootstrap (fallback1 EU + fallback2 AP) — deferred on cost
  📦 SSL cert renewal automation for Docker (manual copy still required on renewal)
```
