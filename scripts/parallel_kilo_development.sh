#!/bin/bash
# =============================================================================
# PRSM Parallel Development - Launch 3 Kilo Instances
# =============================================================================
# This script opens a tmux session with 3 panes, each running a Kilo instance
# focused on a different PRSM subsystem.
#
# Usage: ./scripts/parallel_kilo_development.sh
# =============================================================================

PROJECT_DIR="/Users/ryneschultz/Documents/GitHub/PRSM"
SESSION_NAME="prsm-dev"

# Task descriptions for each instance
TASK_INSTANCE_1='# PRSM Core Infrastructure & Security Hardening

## Your Focus
Complete core infrastructure tasks to unblock 21 tests:

### 1. PostgreSQL Test Infrastructure
- Create docker-compose.test.yml with PostgreSQL 16
- Update .github/workflows/ for PostgreSQL service
- Document DATABASE_URL setup in README

### 2. Module Exports (6 tests)
- Export ftns_service singleton and constants from prsm/economy/tokenomics/ftns_service.py
- Create prsm/interface/auth.py for FastAPI dependencies (JWT token extraction)
- Add AgentTask Pydantic model to prsm/core/models.py

### 3. Performance Module (3 tests)
Create prsm/performance/ package with:
- instrumentation.py - Timing decorators, memory profiling context managers
- optimization.py - Caching strategies, batch processing optimization
- benchmark_orchestrator.py - Test orchestration, result aggregation, reporting

### 4. Collaboration Module (2 tests)
Create prsm/collaboration/ package with:
- __init__.py
- collaboration_manager.py - Multi-agent collaboration coordination

Reference: docs/development/TEST_SKIP_RESOLUTION_ROADMAP.md Phase 1, 2, 5'

TASK_INSTANCE_2='# PRSM NWTN Reasoning Engine Completion

## Your Focus
Complete the NWTN subsystem to unblock 22 tests:

### 1. Orchestrator (Critical - blocks 16 tests)
Create prsm/compute/nwtn/orchestrator.py with NWTNOrchestrator class:

```python
class NWTNOrchestrator:
    """Central orchestrator for NWTN reasoning pipeline"""
    
    async def process_query(
        self, 
        query: str, 
        context: Optional[Dict] = None
    ) -> NWTNResponse:
        # Route through reasoning pipeline
        # Return structured response
        pass
```

Key integration points:
- Use existing prsm/compute/nwtn/breakthrough_reasoning_coordinator.py
- Connect to reasoning/ modules
- Handle context allocation

### 2. Meta-Reasoning Engine (4 tests)
Create prsm/compute/nwtn/meta_reasoning_engine.py:
- System 1 (fast path) - Quick responses for simple queries
- System 2 (deep reasoning) - Multi-step reasoning for complex queries
- Use existing: convergence_analyzer, meta_generation_engine

### 3. Complete System Facade (1 test)
Create prsm/compute/nwtn/complete_system.py:
- Entry point combining orchestrator + provenance + content grounding
- Simple API for end-to-end NWTN queries

### 4. External Storage Config (1 test)
Create prsm/compute/nwtn/external_storage_config.py:
- IPFS configuration
- External storage backends
- Content retrieval settings

Reference: docs/development/TEST_SKIP_RESOLUTION_ROADMAP.md Phase 3
Existing code: prsm/compute/nwtn/ has extensive modules to integrate with'

TASK_INSTANCE_3='# PRSM Marketplace & Economy Features

## Your Focus
Complete marketplace functionality to unblock 20 tests:

### 1. RealMarketplaceService CRUD Methods (13 tests)
File: prsm/economy/marketplace/real_marketplace_service.py

Implement these methods on the RealMarketplaceService class:

```python
async def create_resource_listing(
    self, 
    resource_type: ResourceType, 
    data: Dict[str, Any]
) -> ResourceListing:
    """Create a generic resource listing"""
    pass

async def create_ai_model_listing(
    self, 
    model_data: AIModelListingData
) -> AIModelListing:
    """Create AI model-specific listing"""
    pass

async def create_dataset_listing(
    self, 
    dataset_data: DatasetListingData
) -> DatasetListing:
    """Create dataset-specific listing"""
    pass

async def create_agent_listing(
    self, 
    agent_data: AgentListingData
) -> AgentListing:
    """Create agent-specific listing"""
    pass

async def create_tool_listing(
    self, 
    tool_data: ToolListingData
) -> ToolListing:
    """Create tool-specific listing"""
    pass

async def search_resources(
    self, 
    filters: ResourceSearchFilters
) -> List[ResourceListing]:
    """Filtered search across all listing types"""
    pass

async def get_comprehensive_stats(self) -> MarketplaceStats:
    """Aggregate marketplace statistics"""
    pass
```

### 2. ModelProvider Enum Update
Add PRSM = "prsm" to the ModelProvider enum in prsm/economy/marketplace/models.py

### 3. Advanced FTNS Features (4 tests)
Complete advanced tokenomics in prsm/economy/tokenomics/:
- Dividend distribution mechanism
- Staking rewards calculation
- Advanced token economics

### 4. Governance Integration (2 tests)
Complete prsm/economy/governance/:
- Proposal workflow (create, vote, execute)
- Voting weight based on FTNS stake
- Proposal execution system

Reference: docs/development/TEST_SKIP_RESOLUTION_ROADMAP.md Phase 4'

# =============================================================================
# Script execution starts here
# =============================================================================

echo "🚀 Starting PRSM Parallel Development Session"
echo "=============================================="
echo ""
echo "This will create a tmux session with 3 panes:"
echo "  - Pane 1: Core Infrastructure (Instance 1)"
echo "  - Pane 2: NWTN Reasoning (Instance 2)"  
echo "  - Pane 3: Marketplace & Economy (Instance 3)"
echo ""
echo "Press Ctrl+B then D to detach from the session"
echo "Reattach with: tmux attach -t $SESSION_NAME"
echo ""

# Kill existing session if present
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session with first pane (Instance 1)
tmux new-session -d -s $SESSION_NAME -c "$PROJECT_DIR"

# Set up pane 1 (Instance 1 - Core Infrastructure)
tmux send-keys -t $SESSION_NAME:0.0 "cd $PROJECT_DIR" Enter
tmux send-keys -t $SESSION_NAME:0.0 "echo '🔵 INSTANCE 1: Core Infrastructure & Security Hardening'" Enter
tmux send-keys -t $SESSION_NAME:0.0 "echo ''" Enter
tmux send-keys -t $SESSION_NAME:0.0 "kilo run --auto \"$TASK_INSTANCE_1\"" Enter

# Split horizontally for second pane
tmux split-window -h -t $SESSION_NAME:0

# Set up pane 2 (Instance 2 - NWTN Reasoning)
tmux send-keys -t $SESSION_NAME:0.1 "cd $PROJECT_DIR" Enter
tmux send-keys -t $SESSION_NAME:0.1 "echo '🟢 INSTANCE 2: NWTN Reasoning Engine'" Enter
tmux send-keys -t $SESSION_NAME:0.1 "echo ''" Enter
tmux send-keys -t $SESSION_NAME:0.1 "kilo run --auto \"$TASK_INSTANCE_2\"" Enter

# Split pane 1 vertically for third pane
tmux split-window -v -t $SESSION_NAME:0.0

# Set up pane 3 (Instance 3 - Marketplace)
tmux send-keys -t $SESSION_NAME:0.2 "cd $PROJECT_DIR" Enter
tmux send-keys -t $SESSION_NAME:0.2 "echo '🟡 INSTANCE 3: Marketplace & Economy'" Enter
tmux send-keys -t $SESSION_NAME:0.2 "echo ''" Enter
tmux send-keys -t $SESSION_NAME:0.2 "kilo run --auto \"$TASK_INSTANCE_3\"" Enter

# Select pane 1 as active
tmux select-pane -t $SESSION_NAME:0.0

# Attach to the session
tmux attach -t $SESSION_NAME
