# PRSM User Onboarding Guide

## Table of Contents

1. [Welcome to PRSM](#welcome-to-prsm)
2. [User Types & Paths](#user-types--paths)
3. [Account Setup](#account-setup)
4. [Getting Started by Role](#getting-started-by-role)
5. [Essential Concepts](#essential-concepts)
6. [First Steps Tutorial](#first-steps-tutorial)
7. [Common Workflows](#common-workflows)
8. [Troubleshooting](#troubleshooting)
9. [Community & Support](#community--support)

## Welcome to PRSM

Welcome to the Protocol for Recursive Scientific Modeling (PRSM) - the world's first production-ready decentralized AI platform for scientific research. This comprehensive onboarding guide will help you get started regardless of your background or intended use of the platform.

### What PRSM Offers

- **🧠 Recursive AI Orchestration**: Break complex problems into coordinated agent workflows
- **🌐 Decentralized Research Network**: Join a global community of researchers and AI contributors
- **💰 Token Economy**: Earn FTNS tokens by contributing models, data, and computational resources
- **🗳️ Democratic Governance**: Participate in platform decisions through token-weighted voting
- **🛡️ Built-in Safety**: Comprehensive safety systems protect against misuse
- **🔒 Privacy Protection**: Anonymous participation with encrypted communications

## User Types & Paths

Choose your path based on your primary role and interests:

### 🎓 Academic Researcher
**Goal**: Use AI to accelerate scientific discovery and collaboration

**Quick Start Path**:
1. [Account Setup](#academic-researcher-setup) → 
2. [Submit Research Query](#first-research-query) → 
3. [Explore Marketplace](#marketplace-exploration) → 
4. [Join Research Groups](#research-collaboration)

### 🏢 Enterprise User
**Goal**: Leverage PRSM for R&D, innovation, and business applications

**Quick Start Path**:
1. [Enterprise Account Setup](#enterprise-setup) → 
2. [API Integration](#api-integration) → 
3. [Scale Resources](#resource-scaling) → 
4. [Governance Participation](#enterprise-governance)

### 🤖 AI Developer
**Goal**: Build, train, and monetize AI models on the platform

**Quick Start Path**:
1. [Developer Setup](#developer-setup) → 
2. [Model Training](#model-development) → 
3. [Marketplace Listing](#model-monetization) → 
4. [Platform Contribution](#platform-development)

### 🌍 Community Contributor
**Goal**: Support the network through hosting, governance, or content creation

**Quick Start Path**:
1. [Basic Setup](#community-setup) → 
2. [Choose Contribution Type](#contribution-options) → 
3. [Start Contributing](#active-contribution) → 
4. [Earn Rewards](#reward-optimization)

### 📊 Data Scientist
**Goal**: Analyze data, build insights, and create specialized research tools

**Quick Start Path**:
1. [Technical Setup](#data-science-setup) → 
2. [Data Analysis Workflows](#data-workflows) → 
3. [Custom Model Creation](#specialized-models) → 
4. [Results Publication](#research-publication)

## Account Setup

### Basic Requirements

**System Requirements**:
- Python 3.9+ (3.11+ recommended)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Stable internet connection

**Skills Recommended**:
- Basic command line familiarity
- Understanding of AI/ML concepts (helpful but not required)
- Research domain expertise (for domain-specific applications)

### Academic Researcher Setup

1. **Create Account**:
   ```bash
   # Clone PRSM repository
   git clone https://github.com/PRSM-AI/PRSM.git
   cd PRSM
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Initialize system
   python -m prsm.cli init --user-type researcher
   ```

2. **Configure Research Profile**:
   ```python
   from prsm.auth.models import User
   from prsm.core.database import get_database
   
   # Set up research profile
   user = User.create(
       username="dr_researcher",
       email="researcher@university.edu",
       role="researcher",
       organization="University Research Lab",
       research_domains=["climate_science", "machine_learning"],
       orcid_id="0000-0000-0000-0000"  # Optional but recommended
   )
   ```

3. **Initial FTNS Grant**:
   ```python
   # New researchers receive 5000 FTNS tokens
   from prsm.tokenomics.ftns_service import FTNSService
   
   ftns = FTNSService()
   balance = await ftns.get_balance("dr_researcher")
   print(f"Starting balance: {balance} FTNS")
   ```

### Enterprise Setup

1. **Enterprise Account Registration**:
   ```bash
   # Enterprise initialization with enhanced features
   python -m prsm.cli init --user-type enterprise --tier premium
   ```

2. **API Key Configuration**:
   ```bash
   # Configure external AI services
   python scripts/manage_credentials.py add --platform openai --key YOUR_OPENAI_KEY
   python scripts/manage_credentials.py add --platform anthropic --key YOUR_ANTHROPIC_KEY
   python scripts/manage_credentials.py add --platform huggingface --key YOUR_HF_TOKEN
   ```

3. **Enterprise Features Setup**:
   ```python
   from prsm.teams.service import TeamService
   
   # Create enterprise team
   team_service = TeamService()
   enterprise_team = await team_service.create_team(
       name="Enterprise Research Division",
       team_type="enterprise",
       max_members=100,
       features=["priority_support", "custom_models", "dedicated_compute"]
   )
   ```

### Developer Setup

1. **Development Environment**:
   ```bash
   # Full development setup with testing tools
   git clone https://github.com/PRSM-AI/PRSM.git
   cd PRSM
   pip install -r requirements-dev.txt
   
   # Enable development mode
   python -m prsm.cli init --user-type developer --development-mode
   ```

2. **Local Development Server**:
   ```bash
   # Start local PRSM instance
   python -m prsm.api.main --port 8000 --reload
   
   # Start background services
   docker-compose up -d postgres redis ipfs
   ```

3. **Development Verification**:
   ```bash
   # Run development tests
   python test_foundation.py
   python test_agent_framework.py
   
   # Verify API access
   curl http://localhost:8000/health
   ```

### Community Setup

1. **Basic Community Participation**:
   ```bash
   # Lightweight setup for community participation
   python -m prsm.cli init --user-type community --lightweight
   ```

2. **Choose Contribution Focus**:
   ```python
   from prsm.community.onboarding import CommunityOnboarding
   
   onboarding = CommunityOnboarding()
   
   # Select contribution types
   await onboarding.set_contribution_preferences(
       user_id="community_member",
       preferences=[
           "node_hosting",        # Run network nodes
           "data_contribution",   # Share datasets
           "governance",          # Participate in voting
           "content_moderation"   # Help with safety
       ]
   )
   ```

### Data Science Setup

1. **Scientific Computing Environment**:
   ```bash
   # Install scientific computing dependencies
   pip install -r requirements.txt
   pip install jupyter pandas numpy scipy matplotlib scikit-learn
   
   # Initialize with data science focus
   python -m prsm.cli init --user-type data-scientist --include-notebooks
   ```

2. **Jupyter Integration**:
   ```python
   # PRSM Jupyter magic commands
   %load_ext prsm_magic
   
   # Connect to PRSM in notebook
   %prsm_connect --user data_scientist --auto-auth
   
   # Quick query from notebook
   %prsm_query "Analyze correlation between climate variables and crop yields"
   ```

## Essential Concepts

### FTNS Token Economy

**What are FTNS Tokens?**
FTNS (Fungible Tokens for Node Support) are the native currency that powers PRSM's economy:

- **Utility**: Pay for AI compute, context allocation, and model access
- **Rewards**: Earn tokens by contributing to the network
- **Governance**: Vote on platform improvements and policies
- **Staking**: Stake tokens for enhanced benefits and rewards

**How to Earn FTNS**:
```python
from prsm.tokenomics.ftns_service import FTNSService

ftns = FTNSService()

# Different ways to earn tokens
earning_methods = {
    "data_upload": "Share research datasets (10-100 FTNS per dataset)",
    "model_hosting": "Host AI models (1-10 FTNS per hour)",
    "node_operation": "Run network nodes (5-50 FTNS per day)",
    "governance": "Participate in voting (1-5 FTNS per vote)",
    "research_publication": "Publish research results (50-500 FTNS per paper)",
    "model_training": "Train and share models (100-1000 FTNS per model)"
}

for method, description in earning_methods.items():
    print(f"💰 {method}: {description}")
```

### Agent Framework

**The 5-Layer Agent System**:
1. **Architect AI**: Breaks down complex problems into manageable subtasks
2. **Prompter AI**: Optimizes prompts for maximum effectiveness
3. **Router AI**: Matches tasks to the best available AI models
4. **Executor AI**: Coordinates model execution and resource management
5. **Compiler AI**: Synthesizes results into coherent research outputs

**Example Usage**:
```python
from prsm.agents import AgentOrchestrator

orchestrator = AgentOrchestrator()

# Submit complex research query
result = await orchestrator.process_research_query(
    query="Design a sustainable battery technology for electric vehicles",
    methodology="comprehensive_analysis",
    max_iterations=5
)

print(f"Agents used: {result.agents_involved}")
print(f"Reasoning steps: {len(result.reasoning_trace)}")
print(f"Cost: {result.cost_ftns} FTNS")
```

### Governance System

**How Decisions Are Made**:
- **Proposal Submission**: Community members propose improvements
- **Discussion Period**: 7-day discussion and amendment phase
- **Voting**: Token-weighted voting with anti-plutocracy measures
- **Implementation**: Approved proposals are automatically implemented

**Participation Example**:
```python
from prsm.governance.voting import GovernanceSystem

gov = GovernanceSystem()

# View active proposals
proposals = await gov.get_active_proposals()
for proposal in proposals:
    print(f"📝 {proposal.title}: {proposal.description[:100]}...")

# Submit a new proposal
new_proposal = await gov.submit_proposal(
    title="Improve Climate Model Integration",
    description="Add support for CMIP6 climate models in the marketplace",
    category="technical_enhancement",
    budget_request=10000  # FTNS
)

# Vote on proposals
await gov.cast_vote(
    proposal_id=new_proposal.id,
    vote_type="approve",
    reasoning="This would benefit climate researchers globally"
)
```

## First Steps Tutorial

### Tutorial 1: Your First Research Query

**Goal**: Submit and track a scientific research query

```python
import asyncio
from prsm.nwtn.orchestrator import NWTNOrchestrator
from prsm.core.models import PRSMSession

async def first_research_query():
    # Step 1: Initialize orchestrator
    orchestrator = NWTNOrchestrator()
    
    # Step 2: Create research session
    session = PRSMSession(
        user_id="your_username",
        nwtn_context_allocation=500,  # Allocate 500 FTNS tokens
        research_domain="environmental_science"
    )
    
    # Step 3: Submit query
    query = """
    What are the most promising carbon capture technologies 
    currently in development, and what are their scalability prospects?
    """
    
    print("🔬 Submitting research query...")
    response = await orchestrator.process_query(
        user_input=query,
        session=session
    )
    
    # Step 4: Review results
    print(f"✅ Query completed!")
    print(f"📊 Summary: {response.content[:200]}...")
    print(f"💰 Cost: {response.context_used} FTNS")
    print(f"🧠 Reasoning steps: {len(response.reasoning_trace)}")
    
    # Step 5: Access detailed results
    for i, step in enumerate(response.reasoning_trace[:3]):
        print(f"Step {i+1}: {step.operation}")
    
    return response

# Run the tutorial
result = asyncio.run(first_research_query())
```

### Tutorial 2: Model Marketplace Exploration

**Goal**: Browse and interact with the AI model marketplace

```python
from prsm.marketplace.marketplace_service import MarketplaceService

async def marketplace_tutorial():
    marketplace = MarketplaceService()
    
    # Step 1: Browse available models
    print("🛒 Browsing marketplace...")
    models = await marketplace.list_models(
        category="scientific",
        featured=True,
        limit=10
    )
    
    print(f"Found {len(models)} featured scientific models:")
    for model in models[:3]:
        print(f"  🤖 {model.name} - {model.description[:50]}...")
        print(f"     💰 Cost: {model.pricing['ftns_per_request']} FTNS per request")
        print(f"     ⭐ Rating: {model.stats['rating']}/5.0")
    
    # Step 2: Rent a model for research
    if models:
        chosen_model = models[0]
        print(f"\n🎯 Renting model: {chosen_model.name}")
        
        rental = await marketplace.rent_model(
            user_id="your_username",
            model_id=chosen_model.id,
            duration_hours=24,
            max_requests=100
        )
        
        print(f"✅ Model rented! Rental ID: {rental.id}")
        print(f"📅 Valid until: {rental.expires_at}")
    
    # Step 3: View your rentals
    my_rentals = await marketplace.get_user_rentals("your_username")
    print(f"\n📋 Your active rentals: {len(my_rentals)}")
    
    return models

# Run marketplace tutorial
models = asyncio.run(marketplace_tutorial())
```

### Tutorial 3: Contributing to the Network

**Goal**: Make your first contribution and earn FTNS tokens

```python
from prsm.tokenomics.ftns_service import FTNSService
from prsm.community.early_adopters import EarlyAdopterProgram

async def contribution_tutorial():
    ftns = FTNSService()
    early_adopter = EarlyAdopterProgram()
    
    # Step 1: Check starting balance
    initial_balance = await ftns.get_balance("your_username")
    print(f"💰 Starting balance: {initial_balance} FTNS")
    
    # Step 2: Make a data contribution
    print("\n📊 Contributing a sample dataset...")
    contribution_reward = await ftns.reward_contribution(
        user_id="your_username",
        contribution_type="data_upload",
        value=50.0,  # Dataset value
        description="Sample climate data for tutorial"
    )
    
    print(f"✅ Earned {contribution_reward} FTNS for data contribution!")
    
    # Step 3: Participate in governance
    print("\n🗳️ Participating in governance...")
    governance_reward = await ftns.reward_contribution(
        user_id="your_username",
        contribution_type="governance_participation",
        value=5.0,
        description="Voted on community proposal"
    )
    
    print(f"✅ Earned {governance_reward} FTNS for governance participation!")
    
    # Step 4: Check early adopter status
    adopter_status = await early_adopter.check_status("your_username")
    if adopter_status.is_early_adopter:
        print(f"\n🌟 Early Adopter Status: {adopter_status.tier}")
        print(f"🎁 Bonus multiplier: {adopter_status.reward_multiplier}x")
    
    # Step 5: Final balance
    final_balance = await ftns.get_balance("your_username")
    print(f"\n💰 Final balance: {final_balance} FTNS")
    print(f"📈 Earned: {final_balance - initial_balance} FTNS")
    
    return final_balance - initial_balance

# Run contribution tutorial
earnings = asyncio.run(contribution_tutorial())
```

## Common Workflows

### Research Workflow: Literature Analysis

**Use Case**: Analyze recent literature on a specific topic

```python
async def literature_analysis_workflow():
    from prsm.nwtn.orchestrator import NWTNOrchestrator
    from prsm.integrations.connectors.github_connector import GitHubConnector
    
    orchestrator = NWTNOrchestrator()
    github = GitHubConnector()
    
    # 1. Define research question
    research_question = """
    What are the recent advances in transformer architectures 
    for scientific text processing, particularly for chemistry and biology?
    """
    
    # 2. Gather recent papers and code
    print("📚 Gathering recent literature...")
    
    # Search academic databases (simulated)
    session = PRSMSession(
        user_id="researcher",
        nwtn_context_allocation=1000,
        research_domain="computational_linguistics"
    )
    
    # 3. Process with PRSM
    analysis = await orchestrator.process_query(
        user_input=research_question,
        session=session,
        include_citations=True,
        methodology="systematic_review"
    )
    
    # 4. Generate summary report
    print("📄 Analysis complete:")
    print(f"Key findings: {len(analysis.key_findings)}")
    print(f"Citations: {len(analysis.citations)}")
    print(f"Recommendations: {len(analysis.recommendations)}")
    
    return analysis

# Execute workflow
analysis = asyncio.run(literature_analysis_workflow())
```

### Development Workflow: Model Training

**Use Case**: Train a specialized model for your research domain

```python
async def model_training_workflow():
    from prsm.distillation.orchestrator import DistillationOrchestrator
    from prsm.distillation.models import DistillationRequest, ModelSize, TrainingStrategy
    
    orchestrator = DistillationOrchestrator()
    
    # 1. Define training requirements
    training_request = DistillationRequest(
        user_id="ai_developer",
        teacher_model="gpt-4",
        domain="biomedical_research",
        target_size=ModelSize.MEDIUM,
        optimization_target="accuracy",
        training_strategy=TrainingStrategy.PROGRESSIVE,
        budget_ftns=2000,
        quality_threshold=0.85
    )
    
    print("🎯 Starting model training...")
    
    # 2. Submit training job
    job = await orchestrator.create_distillation(training_request)
    print(f"📋 Training job created: {job.job_id}")
    print(f"🧠 Using backend: {job.backend}")
    
    # 3. Monitor progress
    while True:
        status = await orchestrator.get_job_status(job.job_id)
        print(f"📊 Progress: {status.progress}% - {status.current_stage}")
        
        if status.status == "completed":
            print("✅ Training completed!")
            break
        elif status.status == "failed":
            print("❌ Training failed!")
            break
        
        await asyncio.sleep(30)  # Check every 30 seconds
    
    # 4. Deploy to marketplace
    if status.status == "completed":
        model_info = await orchestrator.get_model_info(job.job_id)
        print(f"🚀 Model ready: {model_info.model_path}")
        print(f"⚡ Framework: {model_info.framework}")
        
        # List on marketplace
        from prsm.marketplace.marketplace_service import MarketplaceService
        marketplace = MarketplaceService()
        
        listing = await marketplace.create_model_listing(
            model_id=model_info.model_id,
            name="Custom Biomedical Research Assistant",
            description="Specialized model for biomedical literature analysis",
            category="scientific",
            pricing={"ftns_per_request": 10},
            tags=["biomedical", "research", "literature"]
        )
        
        print(f"🛒 Listed on marketplace: {listing.id}")
    
    return job

# Execute training workflow
job = asyncio.run(model_training_workflow())
```

### Enterprise Workflow: Team Collaboration

**Use Case**: Set up team-based research collaboration

```python
async def enterprise_collaboration_workflow():
    from prsm.teams.service import TeamService
    from prsm.teams.governance import TeamGovernance
    from prsm.teams.wallet import TeamWallet
    
    team_service = TeamService()
    team_governance = TeamGovernance()
    team_wallet = TeamWallet()
    
    # 1. Create research team
    print("👥 Creating research team...")
    team = await team_service.create_team(
        name="Climate Research Division",
        description="Collaborative climate change research team",
        team_type="research",
        max_members=25
    )
    
    print(f"✅ Team created: {team.id}")
    
    # 2. Set up team wallet and budget
    wallet = await team_wallet.create_team_wallet(
        team_id=team.id,
        initial_budget_ftns=10000,
        spending_policy="consensus_required"
    )
    
    print(f"💰 Team wallet created with 10,000 FTNS")
    
    # 3. Add team members
    members = [
        {"username": "climate_scientist", "role": "lead_researcher"},
        {"username": "data_analyst", "role": "data_specialist"},
        {"username": "ml_engineer", "role": "model_developer"},
        {"username": "domain_expert", "role": "advisor"}
    ]
    
    for member in members:
        await team_service.add_member(
            team_id=team.id,
            username=member["username"],
            role=member["role"]
        )
        print(f"👤 Added {member['username']} as {member['role']}")
    
    # 4. Set up team governance
    governance = await team_governance.setup_team_governance(
        team_id=team.id,
        voting_mechanism="weighted_by_contribution",
        decision_threshold=0.6,
        proposal_system=True
    )
    
    print("🗳️ Team governance configured")
    
    # 5. Create first team project
    project = await team_service.create_project(
        team_id=team.id,
        name="Arctic Ice Sheet Analysis",
        description="AI-powered analysis of Arctic ice sheet changes",
        budget_allocation=3000,  # FTNS
        timeline_weeks=8
    )
    
    print(f"📋 Project created: {project.name}")
    
    return team, project

# Execute enterprise workflow
team, project = asyncio.run(enterprise_collaboration_workflow())
```

## Troubleshooting

### Common Issues for New Users

#### Issue: "Module not found" errors
**Solution**:
```bash
# Ensure proper installation
pip install -r requirements.txt
pip install -e .

# Verify Python path
python -c "import prsm; print(prsm.__file__)"
```

#### Issue: Authentication failures
**Solution**:
```python
# Reset authentication
from prsm.auth.auth_manager import AuthManager

auth = AuthManager()
user = await auth.create_user(
    username="your_username",
    email="your_email@example.com",
    password="secure_password"
)

# Get new token
token = await auth.authenticate(username="your_username", password="secure_password")
print(f"New token: {token}")
```

#### Issue: Insufficient FTNS balance
**Solution**:
```python
# Check earning opportunities
from prsm.community.early_adopters import EarlyAdopterProgram

program = EarlyAdopterProgram()
opportunities = await program.get_earning_opportunities("your_username")

for opportunity in opportunities:
    print(f"💰 {opportunity.type}: {opportunity.potential_reward} FTNS")
    print(f"   {opportunity.description}")
```

#### Issue: Slow query responses
**Solution**:
```python
# Optimize query settings
session = PRSMSession(
    user_id="your_username",
    nwtn_context_allocation=200,  # Reduce allocation for faster response
    optimization_mode="speed",    # Prioritize speed over depth
    max_iterations=3             # Limit iterations
)
```

### Getting Additional Help

**Community Resources**:
- GitHub Discussions: Ask questions and share experiences
- Discord Channel: Real-time community support
- Wiki: Community-maintained documentation
- Stack Overflow: Tag questions with `prsm`

**Professional Support**:
- Email: support@prsm.org
- Enterprise Support: enterprise@prsm.org
- Security Issues: security@prsm.org

## Community & Support

### Joining the Community

**1. Community Channels**:
```markdown
- 💬 Discord: https://discord.gg/prsm-community
- 🐦 Twitter: @PRSM_AI
- 📖 Reddit: r/PRSM
- 📧 Newsletter: newsletter@prsm.org
```

**2. Contributing Back**:
```python
# Example: Share your successful workflow
from prsm.community.sharing import WorkflowSharing

sharing = WorkflowSharing()

await sharing.submit_workflow(
    title="Protein Structure Analysis Pipeline",
    description="Complete workflow for analyzing protein structures using PRSM",
    code_example=workflow_code,
    tags=["biochemistry", "protein", "analysis"],
    difficulty="intermediate"
)
```

**3. Mentorship Program**:
```python
# Connect with experienced users
from prsm.community.mentorship import MentorshipProgram

mentorship = MentorshipProgram()

# Request a mentor
mentor_request = await mentorship.request_mentor(
    user_id="your_username",
    interests=["AI development", "scientific research"],
    experience_level="beginner"
)

print(f"🎓 Mentor matched: {mentor_request.mentor_username}")
```

### Success Metrics

**Track Your Progress**:
```python
from prsm.community.achievements import AchievementSystem

achievements = AchievementSystem()

# Check your achievements
user_achievements = await achievements.get_user_achievements("your_username")

print("🏆 Your Achievements:")
for achievement in user_achievements:
    print(f"  ✅ {achievement.name}: {achievement.description}")
    
# See progress toward next achievements
progress = await achievements.get_progress("your_username")
print(f"\n📊 Next milestone: {progress.next_achievement}")
print(f"📈 Progress: {progress.completion_percentage}%")
```

---

## Welcome to Your PRSM Journey!

Congratulations on joining the PRSM community! You're now part of a revolutionary platform that's transforming scientific research through decentralized AI collaboration.

### Your Next Steps:
1. ✅ Complete your role-specific setup
2. 🚀 Run through the first steps tutorial
3. 💰 Start earning FTNS tokens through contributions
4. 🤝 Connect with the community
5. 🔬 Begin your first research project

### Remember:
- **Start Small**: Begin with simple queries and gradually explore advanced features
- **Community First**: Engage with other users, share experiences, and learn together
- **Safety Always**: Follow safety guidelines and report any issues
- **Contribute Back**: Share your knowledge and help others succeed

**Welcome to the future of collaborative scientific research!** 🌟

---

## Document Information

**Version**: 1.0  
**Last Updated**: June 11, 2025  
**Target Audience**: New PRSM users across all roles  
**Contact**: onboarding@prsm.org  

**Related Guides**:
- [Quick Start Guide](quickstart.md) - Fast 15-minute introduction
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md) - Common issues and solutions
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute to PRSM

---

*This guide is continuously updated based on user feedback. Help us improve by sharing your onboarding experience at feedback@prsm.org*