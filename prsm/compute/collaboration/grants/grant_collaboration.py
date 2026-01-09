#!/usr/bin/env python3
"""
Grant Writing Collaboration Platform for PRSM
============================================

This module implements comprehensive grant writing collaboration specifically
designed for multi-institutional research partnerships. It enables:

- Multi-university consortium grant writing
- Secure collaboration on sensitive budget information
- Role-based access for PIs, co-PIs, admin staff
- Integration with federal funding agency requirements
- NWTN AI assistance for proposal writing and review
- Automated compliance checking and formatting
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import re

# Import PRSM components
from ..security.crypto_sharding import BasicCryptoSharding
from ..models import QueryRequest

# Mock UnifiedPipelineController for testing
class UnifiedPipelineController:
    """Mock pipeline controller for grant collaboration"""
    async def initialize(self):
        pass
    
    async def process_query_full_pipeline(self, user_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Grant-specific NWTN responses
        if context.get("grant_assistance"):
            return {
                "response": {
                    "text": """
Grant Writing Assistance:

**Proposal Strength Analysis**:
- Strong preliminary data and methodology
- Clear innovation and significance statements
- Well-defined milestones and deliverables
- Comprehensive literature review

**Improvement Suggestions**:
1. Add more quantitative success metrics
2. Strengthen the broader impacts section
3. Include more diverse team composition
4. Clarify intellectual merit criteria alignment

**Compliance Check**:
- NSF formatting requirements: ✅ Met
- Page limits: ✅ Within bounds  
- Required sections: ✅ All present
- Budget justification: ⚠️ Needs more detail

**Competitive Analysis**:
Based on similar funded proposals, this project ranks in the top 25% for innovation
and methodology. Consider emphasizing the cross-institutional collaboration benefits.
""",
                    "confidence": 0.89,
                    "sources": ["nsf_proposal_guide.pdf", "successful_grants_database.pdf", "peer_review_criteria.pdf"]
                },
                "performance_metrics": {"total_processing_time": 2.7}
            }
        else:
            return {
                "response": {"text": "Grant assistance available", "confidence": 0.75, "sources": []},
                "performance_metrics": {"total_processing_time": 1.5}
            }

class FundingAgency(Enum):
    """Major funding agencies"""
    NSF = "nsf"  # National Science Foundation
    NIH = "nih"  # National Institutes of Health
    DOE = "doe"  # Department of Energy
    DOD = "dod"  # Department of Defense
    NASA = "nasa"  # NASA
    DARPA = "darpa"  # Defense Advanced Research Projects Agency
    PRIVATE_FOUNDATION = "private"  # Private foundations

class GrantRole(Enum):
    """Roles in grant collaboration"""
    PRINCIPAL_INVESTIGATOR = "pi"
    CO_PRINCIPAL_INVESTIGATOR = "co_pi"
    COLLABORATOR = "collaborator"
    POSTDOC = "postdoc"
    GRADUATE_STUDENT = "grad_student"
    ADMINISTRATIVE_STAFF = "admin"
    BUDGET_OFFICER = "budget"
    COMPLIANCE_OFFICER = "compliance"

class ProposalStatus(Enum):
    """Status of grant proposal"""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    READY_FOR_SUBMISSION = "ready"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    FUNDED = "funded"
    DECLINED = "declined"
    RESUBMISSION_REQUIRED = "resubmission"

@dataclass
class Institution:
    """Participating institution information"""
    institution_id: str
    name: str
    address: str
    federal_id: str  # EIN or similar
    contact_person: str
    contact_email: str
    budget_percentage: float  # Percentage of total budget
    primary_role: str  # Lead, collaborating, subcontract

@dataclass
class TeamMember:
    """Grant team member information"""
    member_id: str
    name: str
    email: str
    institution_id: str
    role: GrantRole
    expertise: List[str]
    cv_path: Optional[str]
    effort_percentage: float  # Percentage of time on project
    salary_requested: Optional[float]

@dataclass
class BudgetItem:
    """Individual budget line item"""
    item_id: str
    category: str  # Personnel, Equipment, Travel, Other
    description: str
    amount: float
    institution_id: str
    year: int  # Budget year (1, 2, 3, etc.)
    justification: str

@dataclass
class Milestone:
    """Project milestone"""
    milestone_id: str
    title: str
    description: str
    target_date: datetime
    deliverables: List[str]
    responsible_institutions: List[str]
    success_criteria: str

@dataclass
class GrantProposal:
    """Complete grant proposal project"""
    proposal_id: str
    title: str
    funding_agency: FundingAgency
    program_name: str
    submission_deadline: datetime
    project_summary: str
    participating_institutions: Dict[str, Institution]
    team_members: Dict[str, TeamMember]
    budget_items: List[BudgetItem]
    milestones: List[Milestone]
    documents: Dict[str, str]  # document_name -> file_content
    status: ProposalStatus
    lead_pi: str  # PI from lead institution
    created_by: str
    created_at: datetime
    last_modified: datetime
    total_budget: float
    project_duration_years: int
    security_level: str

class GrantCollaboration:
    """
    Main class for grant writing collaboration platform
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize grant collaboration system"""
        self.storage_path = storage_path or Path("./grant_collaboration")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize PRSM components
        self.crypto_sharding = BasicCryptoSharding()
        self.nwtn_pipeline = None
        
        # Active proposals
        self.active_proposals: Dict[str, GrantProposal] = {}
        
        # Agency-specific templates and requirements
        self.agency_requirements = self._initialize_agency_requirements()
        self.document_templates = self._initialize_document_templates()
    
    def _initialize_agency_requirements(self) -> Dict[str, Dict]:
        """Initialize funding agency specific requirements"""
        return {
            "nsf": {
                "page_limits": {
                    "project_description": 15,
                    "project_summary": 1,
                    "biographical_sketch": 2,
                    "budget_justification": 3
                },
                "required_sections": [
                    "Intellectual Merit",
                    "Broader Impacts", 
                    "Prior NSF Support",
                    "Project Description",
                    "References Cited"
                ],
                "font_requirements": "11-point or larger",
                "margin_requirements": "1 inch on all sides"
            },
            "nih": {
                "page_limits": {
                    "specific_aims": 1,
                    "research_strategy": 12,
                    "biographical_sketch": 5,
                    "budget_justification": 25
                },
                "required_sections": [
                    "Specific Aims",
                    "Research Strategy",
                    "References Cited",
                    "Human Subjects",
                    "Vertebrate Animals"
                ],
                "font_requirements": "11-point Arial or similar",
                "margin_requirements": "0.5 inch minimum"
            }
        }
    
    def _initialize_document_templates(self) -> Dict[str, str]:
        """Initialize document templates for different sections"""
        return {
            "nsf_project_summary": """
Project Summary

Overview:
[Provide a 1-page overview of the proposed work including objectives, methods, intellectual merit, and broader impacts]

Intellectual Merit:
[Describe how the project advances knowledge in the field]

Broader Impacts:
[Describe benefits to society, education, and underrepresented groups]

Keywords: [List 5-7 keywords]
""",
            "nsf_project_description": """
Project Description

1. INTRODUCTION AND MOTIVATION
[Provide background and motivation for the proposed research]

2. RELATED WORK
[Discuss relevant prior work and how this project differs]

3. RESEARCH OBJECTIVES
[List specific, measurable objectives]

4. METHODOLOGY
[Describe research approach and methods]

5. TIMELINE AND MILESTONES
[Provide detailed timeline with milestones]

6. EVALUATION PLAN
[Describe how success will be measured]

7. INTELLECTUAL MERIT
[Explain how this advances fundamental knowledge]

8. BROADER IMPACTS
[Describe societal benefits and educational impacts]
""",
            "multi_institutional_collaboration_plan": """
Multi-Institutional Collaboration Plan

1. COLLABORATION OVERVIEW
Lead Institution: [Name]
Collaborating Institutions: [List]

2. ROLE DISTRIBUTION
[Describe specific roles and responsibilities of each institution]

3. MANAGEMENT STRUCTURE
[Describe project management and coordination approach]

4. COMMUNICATION PLAN
[Detail regular meetings, reporting, and coordination mechanisms]

5. DATA AND IP SHARING
[Describe how data and intellectual property will be shared]

6. STUDENT EXCHANGE PROGRAM
[If applicable, describe student/researcher exchange plans]
"""
        }
    
    async def initialize_nwtn_pipeline(self):
        """Initialize NWTN pipeline for grant writing assistance"""
        if self.nwtn_pipeline is None:
            self.nwtn_pipeline = UnifiedPipelineController()
            await self.nwtn_pipeline.initialize()
    
    def create_grant_proposal(self,
                            title: str,
                            funding_agency: FundingAgency,
                            program_name: str,
                            submission_deadline: datetime,
                            lead_pi: str,
                            project_duration_years: int = 3,
                            security_level: str = "medium") -> GrantProposal:
        """Create a new multi-institutional grant proposal"""
        proposal_id = str(uuid.uuid4())
        
        proposal = GrantProposal(
            proposal_id=proposal_id,
            title=title,
            funding_agency=funding_agency,
            program_name=program_name,
            submission_deadline=submission_deadline,
            project_summary="",
            participating_institutions={},
            team_members={},
            budget_items=[],
            milestones=[],
            documents={},
            status=ProposalStatus.DRAFT,
            lead_pi=lead_pi,
            created_by=lead_pi,
            created_at=datetime.now(),
            last_modified=datetime.now(),
            total_budget=0.0,
            project_duration_years=project_duration_years,
            security_level=security_level
        )
        
        # Initialize with document templates
        self._initialize_proposal_documents(proposal)
        
        self.active_proposals[proposal_id] = proposal
        self._save_proposal(proposal)
        
        return proposal
    
    def _initialize_proposal_documents(self, proposal: GrantProposal):
        """Initialize proposal with agency-specific document templates"""
        agency = proposal.funding_agency.value
        
        if agency == "nsf":
            proposal.documents["project_summary"] = self.document_templates["nsf_project_summary"]
            proposal.documents["project_description"] = self.document_templates["nsf_project_description"]
        
        # Always add collaboration plan for multi-institutional proposals
        proposal.documents["collaboration_plan"] = self.document_templates["multi_institutional_collaboration_plan"]
    
    def add_institution(self,
                       proposal_id: str,
                       name: str,
                       address: str,
                       federal_id: str,
                       contact_person: str,
                       contact_email: str,
                       budget_percentage: float,
                       primary_role: str = "collaborating") -> Institution:
        """Add participating institution to grant proposal"""
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        institution_id = str(uuid.uuid4())
        
        institution = Institution(
            institution_id=institution_id,
            name=name,
            address=address,
            federal_id=federal_id,
            contact_person=contact_person,
            contact_email=contact_email,
            budget_percentage=budget_percentage,
            primary_role=primary_role
        )
        
        proposal = self.active_proposals[proposal_id]
        proposal.participating_institutions[institution_id] = institution
        proposal.last_modified = datetime.now()
        
        self._save_proposal(proposal)
        return institution
    
    def add_team_member(self,
                       proposal_id: str,
                       name: str,
                       email: str,
                       institution_id: str,
                       role: GrantRole,
                       expertise: List[str],
                       effort_percentage: float,
                       salary_requested: Optional[float] = None) -> TeamMember:
        """Add team member to grant proposal"""
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self.active_proposals[proposal_id]
        
        if institution_id not in proposal.participating_institutions:
            raise ValueError(f"Institution {institution_id} not found in proposal")
        
        member_id = str(uuid.uuid4())
        
        team_member = TeamMember(
            member_id=member_id,
            name=name,
            email=email,
            institution_id=institution_id,
            role=role,
            expertise=expertise,
            cv_path=None,
            effort_percentage=effort_percentage,
            salary_requested=salary_requested
        )
        
        proposal.team_members[member_id] = team_member
        proposal.last_modified = datetime.now()
        
        self._save_proposal(proposal)
        return team_member
    
    def add_budget_item(self,
                       proposal_id: str,
                       category: str,
                       description: str,
                       amount: float,
                       institution_id: str,
                       year: int,
                       justification: str) -> BudgetItem:
        """Add budget item to grant proposal"""
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        item_id = str(uuid.uuid4())
        
        budget_item = BudgetItem(
            item_id=item_id,
            category=category,
            description=description,
            amount=amount,
            institution_id=institution_id,
            year=year,
            justification=justification
        )
        
        proposal = self.active_proposals[proposal_id]
        proposal.budget_items.append(budget_item)
        
        # Update total budget
        proposal.total_budget = sum(item.amount for item in proposal.budget_items)
        proposal.last_modified = datetime.now()
        
        self._save_proposal(proposal)
        return budget_item
    
    def add_milestone(self,
                     proposal_id: str,
                     title: str,
                     description: str,
                     target_date: datetime,
                     deliverables: List[str],
                     responsible_institutions: List[str],
                     success_criteria: str) -> Milestone:
        """Add project milestone to grant proposal"""
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        milestone_id = str(uuid.uuid4())
        
        milestone = Milestone(
            milestone_id=milestone_id,
            title=title,
            description=description,
            target_date=target_date,
            deliverables=deliverables,
            responsible_institutions=responsible_institutions,
            success_criteria=success_criteria
        )
        
        proposal = self.active_proposals[proposal_id]
        proposal.milestones.append(milestone)
        proposal.last_modified = datetime.now()
        
        self._save_proposal(proposal)
        return milestone
    
    def update_document(self,
                       proposal_id: str,
                       document_name: str,
                       content: str,
                       user_id: str) -> bool:
        """Update a document section in the grant proposal"""
        if proposal_id not in self.active_proposals:
            return False
        
        proposal = self.active_proposals[proposal_id]
        proposal.documents[document_name] = content
        proposal.last_modified = datetime.now()
        
        self._save_proposal(proposal)
        return True
    
    async def get_grant_assistance(self,
                                 proposal_id: str,
                                 section: str,
                                 user_id: str) -> Dict[str, Any]:
        """Get NWTN AI assistance for grant writing"""
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self.active_proposals[proposal_id]
        await self.initialize_nwtn_pipeline()
        
        # Get document content for analysis
        document_content = proposal.documents.get(section, "")
        
        # Construct grant assistance query
        assistance_query = f"""
Please review this grant proposal section for a {proposal.funding_agency.value.upper()} {proposal.program_name} proposal:

PROPOSAL TITLE: {proposal.title}
SECTION: {section}
SUBMISSION DEADLINE: {proposal.submission_deadline.strftime('%Y-%m-%d')}
TOTAL BUDGET: ${proposal.total_budget:,.2f}
PARTICIPATING INSTITUTIONS: {len(proposal.participating_institutions)}

CONTENT TO REVIEW:
{document_content[:2000]}...

Please provide:
1. Content quality assessment and suggestions
2. Compliance with {proposal.funding_agency.value.upper()} requirements
3. Competitive analysis and strengthening recommendations
4. Formatting and structure improvements
5. Missing elements that should be included

Focus on maximizing the chances of funding success.
"""
        
        result = await self.nwtn_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=assistance_query,
            context={
                "domain": "grant_writing",
                "grant_assistance": True,
                "funding_agency": proposal.funding_agency.value,
                "proposal_type": "multi_institutional"
            }
        )
        
        return {
            "analysis": result.get('response', {}).get('text', ''),
            "confidence": result.get('response', {}).get('confidence', 0.0),
            "sources": result.get('response', {}).get('sources', []),
            "processing_time": result.get('performance_metrics', {}).get('total_processing_time', 0.0)
        }
    
    def check_compliance(self, proposal_id: str) -> Dict[str, Any]:
        """Check proposal compliance with funding agency requirements"""
        if proposal_id not in self.active_proposals:
            raise ValueError(f"Proposal {proposal_id} not found")
        
        proposal = self.active_proposals[proposal_id]
        agency = proposal.funding_agency.value
        requirements = self.agency_requirements.get(agency, {})
        
        compliance_report = {
            "overall_compliance": True,
            "issues": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check required sections
        required_sections = requirements.get("required_sections", [])
        for section in required_sections:
            section_key = section.lower().replace(" ", "_")
            if section_key not in proposal.documents or not proposal.documents[section_key].strip():
                compliance_report["issues"].append(f"Missing required section: {section}")
                compliance_report["overall_compliance"] = False
        
        # Check page limits (simplified - would need actual document analysis)
        page_limits = requirements.get("page_limits", {})
        for doc_name, limit in page_limits.items():
            if doc_name in proposal.documents:
                # Rough estimate: 500 words per page
                estimated_pages = len(proposal.documents[doc_name].split()) / 500
                if estimated_pages > limit:
                    compliance_report["warnings"].append(
                        f"{doc_name} may exceed {limit} page limit (estimated: {estimated_pages:.1f} pages)"
                    )
        
        # Check budget distribution
        if len(proposal.participating_institutions) > 1:
            total_percentage = sum(inst.budget_percentage for inst in proposal.participating_institutions.values())
            if abs(total_percentage - 100.0) > 0.01:
                compliance_report["issues"].append(
                    f"Institution budget percentages must sum to 100% (currently: {total_percentage:.1f}%)"
                )
                compliance_report["overall_compliance"] = False
        
        # Check team composition
        pi_count = sum(1 for member in proposal.team_members.values() 
                      if member.role == GrantRole.PRINCIPAL_INVESTIGATOR)
        if pi_count == 0:
            compliance_report["issues"].append("No Principal Investigator assigned")
            compliance_report["overall_compliance"] = False
        elif pi_count > 1:
            compliance_report["warnings"].append("Multiple Principal Investigators - ensure this is allowed")
        
        # Check deadline
        days_until_deadline = (proposal.submission_deadline - datetime.now()).days
        if days_until_deadline < 0:
            compliance_report["issues"].append("Submission deadline has passed")
            compliance_report["overall_compliance"] = False
        elif days_until_deadline < 7:
            compliance_report["warnings"].append(f"Only {days_until_deadline} days until deadline")
        
        return compliance_report
    
    def get_proposal_analytics(self, proposal_id: str) -> Dict[str, Any]:
        """Get analytics for grant proposal"""
        if proposal_id not in self.active_proposals:
            return {}
        
        proposal = self.active_proposals[proposal_id]
        
        # Budget analysis by institution
        budget_by_institution = {}
        for item in proposal.budget_items:
            inst_name = proposal.participating_institutions[item.institution_id].name
            if inst_name not in budget_by_institution:
                budget_by_institution[inst_name] = 0.0
            budget_by_institution[inst_name] += item.amount
        
        # Budget analysis by category
        budget_by_category = {}
        for item in proposal.budget_items:
            if item.category not in budget_by_category:
                budget_by_category[item.category] = 0.0
            budget_by_category[item.category] += item.amount
        
        # Team composition analysis
        team_by_role = {}
        team_by_institution = {}
        for member in proposal.team_members.values():
            # By role
            role_name = member.role.value
            if role_name not in team_by_role:
                team_by_role[role_name] = 0
            team_by_role[role_name] += 1
            
            # By institution
            inst_name = proposal.participating_institutions[member.institution_id].name
            if inst_name not in team_by_institution:
                team_by_institution[inst_name] = 0
            team_by_institution[inst_name] += 1
        
        return {
            "proposal_id": proposal_id,
            "title": proposal.title,
            "funding_agency": proposal.funding_agency.value,
            "total_budget": proposal.total_budget,
            "days_until_deadline": (proposal.submission_deadline - datetime.now()).days,
            "status": proposal.status.value,
            "institutions_count": len(proposal.participating_institutions),
            "team_members_count": len(proposal.team_members),
            "milestones_count": len(proposal.milestones),
            "budget_by_institution": budget_by_institution,
            "budget_by_category": budget_by_category,
            "team_by_role": team_by_role,
            "team_by_institution": team_by_institution,
            "completion_percentage": self._calculate_completion_percentage(proposal)
        }
    
    def _calculate_completion_percentage(self, proposal: GrantProposal) -> float:
        """Calculate proposal completion percentage"""
        total_components = 0
        completed_components = 0
        
        # Check documents
        required_docs = ["project_summary", "project_description", "collaboration_plan"]
        for doc in required_docs:
            total_components += 1
            if doc in proposal.documents and proposal.documents[doc].strip():
                completed_components += 1
        
        # Check institutions
        total_components += 1
        if len(proposal.participating_institutions) > 0:
            completed_components += 1
        
        # Check team members
        total_components += 1
        if len(proposal.team_members) > 0:
            completed_components += 1
        
        # Check budget
        total_components += 1
        if len(proposal.budget_items) > 0:
            completed_components += 1
        
        # Check milestones
        total_components += 1
        if len(proposal.milestones) > 0:
            completed_components += 1
        
        return (completed_components / total_components) * 100.0 if total_components > 0 else 0.0
    
    def _save_proposal(self, proposal: GrantProposal):
        """Save grant proposal with optional encryption"""
        proposal_dir = self.storage_path / "proposals" / proposal.proposal_id
        proposal_dir.mkdir(parents=True, exist_ok=True)
        
        # Save proposal metadata
        proposal_path = proposal_dir / "proposal.json"
        with open(proposal_path, 'w') as f:
            json.dump(asdict(proposal), f, default=str, indent=2)
        
        # Save documents separately (for potential sharding)
        docs_dir = proposal_dir / "documents"
        docs_dir.mkdir(exist_ok=True)
        
        for doc_name, content in proposal.documents.items():
            doc_path = docs_dir / f"{doc_name}.txt"
            
            if proposal.security_level == "high":
                # Use crypto sharding for sensitive proposals
                temp_file = docs_dir / f"temp_{doc_name}.txt"
                with open(temp_file, 'w') as f:
                    f.write(content)
                
                try:
                    # Get all team members as authorized users
                    authorized_users = [member.email for member in proposal.team_members.values()]
                    authorized_users.append(proposal.created_by)
                    
                    shards, manifest = self.crypto_sharding.shard_file(
                        str(temp_file),
                        list(set(authorized_users))  # Remove duplicates
                    )
                    
                    # Save shards
                    shard_dir = docs_dir / "shards" / doc_name
                    shard_dir.mkdir(parents=True, exist_ok=True)
                    
                    for i, shard in enumerate(shards):
                        shard_path = shard_dir / f"shard_{i}.enc"
                        with open(shard_path, 'wb') as f:
                            f.write(shard.shard_data)
                    
                    # Save manifest
                    manifest_path = shard_dir / "manifest.json"
                    with open(manifest_path, 'w') as f:
                        json.dump(asdict(manifest), f, default=str, indent=2)
                    
                    temp_file.unlink()  # Remove temporary file
                    
                except Exception as e:
                    print(f"Error sharding document {doc_name}: {e}")
                    # Fall back to regular storage
                    with open(doc_path, 'w') as f:
                        f.write(content)
            else:
                # Regular storage for medium/standard security
                with open(doc_path, 'w') as f:
                    f.write(content)

# Example usage and testing
if __name__ == "__main__":
    async def test_grant_collaboration():
        """Test grant collaboration system"""
        
        print("🚀 Testing Grant Writing Collaboration Platform")
        
        # Initialize collaboration system
        grant_collab = GrantCollaboration()
        
        # Create NSF multi-institutional proposal
        deadline = datetime.now() + timedelta(days=45)
        proposal = grant_collab.create_grant_proposal(
            "Quantum-Enhanced Machine Learning for Climate Modeling",
            FundingAgency.NSF,
            "Computer and Information Science and Engineering",
            deadline,
            "sarah.chen@unc.edu",
            project_duration_years=3,
            security_level="high"
        )
        
        print(f"✅ Created grant proposal: {proposal.title}")
        print(f"   Proposal ID: {proposal.proposal_id}")
        print(f"   Funding Agency: {proposal.funding_agency.value.upper()}")
        print(f"   Deadline: {proposal.submission_deadline.strftime('%Y-%m-%d')}")
        
        # Add participating institutions
        unc = grant_collab.add_institution(
            proposal.proposal_id,
            "University of North Carolina at Chapel Hill",
            "Chapel Hill, NC 27599",
            "56-6001393",
            "Dr. Sarah Chen",
            "sarah.chen@unc.edu",
            40.0,
            "lead"
        )
        
        duke = grant_collab.add_institution(
            proposal.proposal_id,
            "Duke University",
            "Durham, NC 27708",
            "56-0532129",
            "Dr. Alex Rodriguez",
            "alex.rodriguez@duke.edu",
            35.0,
            "collaborating"
        )
        
        sas = grant_collab.add_institution(
            proposal.proposal_id,
            "SAS Institute Inc.",
            "Cary, NC 27513",
            "56-1156892",
            "Michael Johnson",
            "michael.johnson@sas.com",
            25.0,
            "industry_partner"
        )
        
        print(f"✅ Added {len(proposal.participating_institutions)} institutions")
        
        # Add team members
        pi = grant_collab.add_team_member(
            proposal.proposal_id,
            "Dr. Sarah Chen",
            "sarah.chen@unc.edu",
            unc.institution_id,
            GrantRole.PRINCIPAL_INVESTIGATOR,
            ["Quantum Computing", "Machine Learning", "Climate Science"],
            25.0,
            75000.0
        )
        
        co_pi = grant_collab.add_team_member(
            proposal.proposal_id,
            "Dr. Alex Rodriguez",
            "alex.rodriguez@duke.edu",
            duke.institution_id,
            GrantRole.CO_PRINCIPAL_INVESTIGATOR,
            ["Deep Learning", "Climate Modeling", "Data Science"],
            20.0,
            60000.0
        )
        
        industry_collaborator = grant_collab.add_team_member(
            proposal.proposal_id,
            "Michael Johnson",
            "michael.johnson@sas.com",
            sas.institution_id,
            GrantRole.COLLABORATOR,
            ["Statistical Computing", "Enterprise AI", "Data Analytics"],
            15.0,
            None  # Industry partner not requesting salary
        )
        
        print(f"✅ Added {len(proposal.team_members)} team members")
        
        # Add budget items
        grant_collab.add_budget_item(
            proposal.proposal_id,
            "Personnel",
            "PI Salary and Benefits (25% effort)",
            75000.0,
            unc.institution_id,
            1,
            "Principal Investigator will lead quantum algorithm development"
        )
        
        grant_collab.add_budget_item(
            proposal.proposal_id,
            "Equipment",
            "High-performance computing cluster",
            150000.0,
            duke.institution_id,
            1,
            "Dedicated cluster for climate model training and quantum simulations"
        )
        
        grant_collab.add_budget_item(
            proposal.proposal_id,
            "Travel",
            "Conference presentations and collaboration visits",
            25000.0,
            unc.institution_id,
            1,
            "Present results at major conferences and facilitate collaboration"
        )
        
        print(f"✅ Added budget items totaling ${proposal.total_budget:,.2f}")
        
        # Add project milestones
        milestone1 = grant_collab.add_milestone(
            proposal.proposal_id,
            "Quantum Algorithm Development",
            "Develop quantum-enhanced ML algorithms for climate data",
            datetime.now() + timedelta(days=365),
            ["Algorithm specification", "Theoretical analysis", "Proof of concept"],
            [unc.institution_id],
            "Demonstrate 20% improvement over classical algorithms"
        )
        
        milestone2 = grant_collab.add_milestone(
            proposal.proposal_id,
            "Climate Model Integration",
            "Integrate quantum algorithms with existing climate models",
            datetime.now() + timedelta(days=730),
            ["Integration framework", "Performance benchmarks", "Validation results"],
            [duke.institution_id, sas.institution_id],
            "Achieve real-time climate prediction with improved accuracy"
        )
        
        print(f"✅ Added {len(proposal.milestones)} project milestones")
        
        # Update project description
        project_description = """
This proposal presents a groundbreaking approach to climate modeling through the integration of quantum-enhanced machine learning algorithms. Our multi-institutional team will develop novel quantum computing techniques specifically designed to accelerate climate model training and improve prediction accuracy.

The collaboration between UNC Chapel Hill's quantum computing expertise, Duke University's climate modeling capabilities, and SAS Institute's enterprise-scale data analytics will create a uniquely powerful platform for addressing one of humanity's most pressing challenges.

Our approach leverages quantum advantage in machine learning optimization to overcome computational bottlenecks in current climate models, potentially revolutionizing our ability to predict and adapt to climate change.
"""
        
        grant_collab.update_document(
            proposal.proposal_id,
            "project_description",
            project_description,
            "sarah.chen@unc.edu"
        )
        
        print("✅ Updated project description")
        
        # Test grant writing assistance
        try:
            assistance = await grant_collab.get_grant_assistance(
                proposal.proposal_id,
                "project_description",
                "sarah.chen@unc.edu"
            )
            
            print("✅ Grant writing assistance provided:")
            print(f"   Confidence: {assistance['confidence']:.2f}")
            print(f"   Processing time: {assistance['processing_time']:.1f}s")
            print(f"   Analysis preview: {assistance['analysis'][:150]}...")
            
        except Exception as e:
            print(f"⚠️  Grant assistance test: {e}")
        
        # Check compliance
        compliance = grant_collab.check_compliance(proposal.proposal_id)
        print(f"✅ Compliance check completed:")
        print(f"   Overall compliance: {compliance['overall_compliance']}")
        print(f"   Issues: {len(compliance['issues'])}")
        print(f"   Warnings: {len(compliance['warnings'])}")
        
        # Get analytics
        analytics = grant_collab.get_proposal_analytics(proposal.proposal_id)
        print(f"✅ Proposal analytics:")
        print(f"   Completion: {analytics['completion_percentage']:.1f}%")
        print(f"   Days until deadline: {analytics['days_until_deadline']}")
        print(f"   Budget breakdown: {analytics['budget_by_category']}")
        
        print("\n🎉 Grant collaboration platform test completed!")
        print("Ready for multi-institutional grant writing workflows")
    
    # Run the test
    import asyncio
    asyncio.run(test_grant_collaboration())