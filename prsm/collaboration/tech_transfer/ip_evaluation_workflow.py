#!/usr/bin/env python3
"""
Technology Transfer IP Evaluation Workflow for PRSM
===================================================

This module implements secure IP evaluation workflows specifically designed for
university-industry partnerships. It enables:

- Secure sharing of research assets for commercial evaluation
- Granular access controls for different stakeholders
- Audit trails for compliance and legal requirements
- Integration with university tech transfer offices
- NWTN AI analysis for IP valuation and market assessment
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib

# Import PRSM components
from ..security.crypto_sharding import BasicCryptoSharding
from ..models import QueryRequest

# Mock UnifiedPipelineController for testing
class UnifiedPipelineController:
    """Mock pipeline controller for integration testing"""
    async def initialize(self):
        pass
    
    async def process_query_full_pipeline(self, user_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # This will be replaced with actual NWTN integration
        return {
            "response": {"text": "Mock NWTN response for IP analysis", "confidence": 0.87, "sources": []},
            "performance_metrics": {"total_processing_time": 2.1}
        }

class EvaluationStatus(Enum):
    """Status of IP evaluation process"""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    EVALUATING = "evaluating"
    NEGOTIATING = "negotiating"
    LICENSED = "licensed"
    DECLINED = "declined"
    EXPIRED = "expired"

class StakeholderRole(Enum):
    """Roles in the IP evaluation process"""
    RESEARCHER = "researcher"
    TECH_TRANSFER_OFFICER = "tech_transfer_officer"
    INDUSTRY_EVALUATOR = "industry_evaluator"
    LEGAL_COUNSEL = "legal_counsel"
    ADMINISTRATOR = "administrator"

@dataclass
class ResearchAsset:
    """Represents a research asset for evaluation"""
    asset_id: str
    name: str
    description: str
    asset_type: str  # 'paper', 'dataset', 'code', 'prototype', 'patent'
    file_path: Optional[str]
    metadata: Dict[str, Any]
    security_classification: str  # 'public', 'confidential', 'proprietary'
    created_by: str
    created_at: datetime
    
@dataclass
class AccessPermission:
    """Defines access permissions for stakeholders"""
    user_id: str
    role: StakeholderRole
    permissions: List[str]  # 'view', 'download', 'comment', 'modify'
    expiry_date: Optional[datetime]
    access_conditions: Dict[str, Any]

@dataclass
class EvaluationProject:
    """Complete IP evaluation project"""
    project_id: str
    title: str
    description: str
    university: str
    research_assets: List[ResearchAsset]
    status: EvaluationStatus
    stakeholders: List[AccessPermission]
    created_by: str
    created_at: datetime
    last_modified: datetime
    evaluation_timeline: Dict[str, datetime]
    nda_required: bool
    compliance_requirements: List[str]

@dataclass
class EvaluationActivity:
    """Activity log for evaluation process"""
    activity_id: str
    project_id: str
    user_id: str
    activity_type: str
    description: str
    timestamp: datetime
    metadata: Dict[str, Any]

class TechTransferWorkflow:
    """
    Main class for technology transfer IP evaluation workflows
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the tech transfer workflow system"""
        self.storage_path = storage_path or Path("./tech_transfer")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize PRSM components
        self.crypto_sharding = BasicCryptoSharding()
        self.nwtn_pipeline = None
        
        # Active projects and activities
        self.active_projects: Dict[str, EvaluationProject] = {}
        self.activity_log: List[EvaluationActivity] = []
        
        # Compliance templates
        self.compliance_templates = {
            "HIPAA": ["health_data_handling", "patient_privacy", "audit_logging"],
            "ITAR": ["export_control", "foreign_person_access", "technical_data_classification"],
            "CUI": ["controlled_unclassified_info", "marking_requirements", "dissemination_controls"],
            "FERPA": ["educational_records", "student_privacy", "disclosure_authorization"]
        }
    
    async def initialize_nwtn_pipeline(self):
        """Initialize NWTN pipeline for AI-powered IP analysis"""
        if self.nwtn_pipeline is None:
            self.nwtn_pipeline = UnifiedPipelineController()
            await self.nwtn_pipeline.initialize()
    
    def create_evaluation_project(self,
                                title: str,
                                description: str,
                                university: str,
                                created_by: str,
                                nda_required: bool = True,
                                compliance_requirements: Optional[List[str]] = None) -> EvaluationProject:
        """Create a new IP evaluation project"""
        project_id = str(uuid.uuid4())
        
        project = EvaluationProject(
            project_id=project_id,
            title=title,
            description=description,
            university=university,
            research_assets=[],
            status=EvaluationStatus.DRAFT,
            stakeholders=[],
            created_by=created_by,
            created_at=datetime.now(),
            last_modified=datetime.now(),
            evaluation_timeline={
                "created": datetime.now(),
                "submission_deadline": datetime.now() + timedelta(days=30),
                "evaluation_deadline": datetime.now() + timedelta(days=90)
            },
            nda_required=nda_required,
            compliance_requirements=compliance_requirements or []
        )
        
        self.active_projects[project_id] = project
        
        # Log project creation
        self._log_activity(
            project_id=project_id,
            user_id=created_by,
            activity_type="project_created",
            description=f"IP evaluation project '{title}' created"
        )
        
        return project
    
    def add_research_asset(self,
                          project_id: str,
                          name: str,
                          description: str,
                          asset_type: str,
                          file_path: Optional[str] = None,
                          security_classification: str = "confidential",
                          metadata: Optional[Dict[str, Any]] = None,
                          created_by: str = "system") -> ResearchAsset:
        """Add a research asset to an evaluation project"""
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found")
        
        asset_id = str(uuid.uuid4())
        
        asset = ResearchAsset(
            asset_id=asset_id,
            name=name,
            description=description,
            asset_type=asset_type,
            file_path=file_path,
            metadata=metadata or {},
            security_classification=security_classification,
            created_by=created_by,
            created_at=datetime.now()
        )
        
        project = self.active_projects[project_id]
        project.research_assets.append(asset)
        project.last_modified = datetime.now()
        
        # If file provided, secure it with crypto sharding
        if file_path and Path(file_path).exists():
            self._secure_research_asset(project_id, asset, file_path)
        
        # Log asset addition
        self._log_activity(
            project_id=project_id,
            user_id=created_by,
            activity_type="asset_added",
            description=f"Research asset '{name}' added to project",
            metadata={"asset_id": asset_id, "asset_type": asset_type}
        )
        
        return asset
    
    def add_stakeholder(self,
                       project_id: str,
                       user_id: str,
                       role: StakeholderRole,
                       permissions: List[str],
                       expiry_date: Optional[datetime] = None,
                       access_conditions: Optional[Dict[str, Any]] = None) -> AccessPermission:
        """Add a stakeholder to an evaluation project"""
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found")
        
        # Set default expiry to 6 months for industry evaluators
        if expiry_date is None and role == StakeholderRole.INDUSTRY_EVALUATOR:
            expiry_date = datetime.now() + timedelta(days=180)
        
        permission = AccessPermission(
            user_id=user_id,
            role=role,
            permissions=permissions,
            expiry_date=expiry_date,
            access_conditions=access_conditions or {}
        )
        
        project = self.active_projects[project_id]
        project.stakeholders.append(permission)
        project.last_modified = datetime.now()
        
        # Log stakeholder addition
        self._log_activity(
            project_id=project_id,
            user_id="system",
            activity_type="stakeholder_added",
            description=f"Stakeholder {user_id} added with role {role.value}",
            metadata={"stakeholder_role": role.value, "permissions": permissions}
        )
        
        return permission
    
    def _secure_research_asset(self,
                              project_id: str,
                              asset: ResearchAsset,
                              file_path: str):
        """Secure a research asset using cryptographic sharding"""
        project = self.active_projects[project_id]
        
        # Get all stakeholders who can access this asset
        authorized_users = [
            perm.user_id for perm in project.stakeholders 
            if "view" in perm.permissions or "download" in perm.permissions
        ]
        
        # Add project creator
        authorized_users.append(project.created_by)
        
        try:
            # Shard the file with appropriate security level
            security_level = "high" if asset.security_classification == "proprietary" else "medium"
            num_shards = 7 if security_level == "high" else 5
            required_shards = 5 if security_level == "high" else 3
            
            shards, manifest = self.crypto_sharding.shard_file(
                file_path,
                authorized_users,
                num_shards=num_shards,
                required_shards=required_shards
            )
            
            # Store shards and manifest
            self._store_asset_shards(project_id, asset.asset_id, shards, manifest)
            
            # Update asset metadata
            asset.metadata.update({
                "sharded": True,
                "shard_count": num_shards,
                "required_shards": required_shards,
                "security_level": security_level
            })
            
        except Exception as e:
            print(f"Error securing asset {asset.asset_id}: {e}")
            asset.metadata["sharding_error"] = str(e)
    
    def _store_asset_shards(self,
                           project_id: str,
                           asset_id: str,
                           shards: List,
                           manifest: Any):
        """Store encrypted asset shards"""
        asset_dir = self.storage_path / "projects" / project_id / "assets" / asset_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        # Store each shard
        for i, shard in enumerate(shards):
            shard_path = asset_dir / f"shard_{i}.enc"
            with open(shard_path, 'wb') as f:
                f.write(shard.shard_data)
        
        # Store manifest
        manifest_path = asset_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(asdict(manifest), f, default=str, indent=2)
    
    async def analyze_ip_with_nwtn(self,
                                 project_id: str,
                                 asset_id: str,
                                 user_id: str) -> Dict[str, Any]:
        """Use NWTN AI to analyze IP for market potential and valuation"""
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.active_projects[project_id]
        asset = next((a for a in project.research_assets if a.asset_id == asset_id), None)
        
        if not asset:
            raise ValueError(f"Asset {asset_id} not found")
        
        # Check user permissions
        if not self._check_user_permission(project_id, user_id, "view"):
            raise PermissionError(f"User {user_id} does not have permission to analyze this IP")
        
        # Initialize NWTN pipeline
        await self.initialize_nwtn_pipeline()
        
        # Construct analysis query
        analysis_query = f"""
Analyze the following intellectual property for commercial potential and market value:

**Asset Name**: {asset.name}
**Type**: {asset.asset_type}
**Description**: {asset.description}
**University**: {project.university}
**Classification**: {asset.security_classification}

Please provide:
1. Market potential assessment
2. Competitive landscape analysis
3. Technology readiness level (TRL) estimation
4. Potential licensing value range
5. Key commercial applications
6. Risk factors and mitigation strategies
7. Recommended next steps for commercialization

Focus on actionable insights for technology transfer and licensing decisions.
"""
        
        # Get NWTN analysis
        nwtn_result = await self.nwtn_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=analysis_query,
            context={
                "domain": "technology_transfer",
                "ip_analysis": True,
                "university": project.university,
                "asset_type": asset.asset_type
            }
        )
        
        # Structure the analysis results
        analysis_result = {
            "asset_id": asset_id,
            "project_id": project_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "nwtn_analysis": {
                "market_assessment": nwtn_result.get('response', {}).get('text', ''),
                "confidence_score": nwtn_result.get('response', {}).get('confidence', 0.0),
                "sources": nwtn_result.get('response', {}).get('sources', []),
                "processing_time": nwtn_result.get('performance_metrics', {}).get('total_processing_time', 0.0)
            },
            "automated_insights": self._extract_automated_insights(nwtn_result),
            "analyzed_by": user_id
        }
        
        # Log the analysis
        self._log_activity(
            project_id=project_id,
            user_id=user_id,
            activity_type="ip_analyzed",
            description=f"NWTN AI analysis performed on asset '{asset.name}'",
            metadata={
                "asset_id": asset_id,
                "confidence_score": analysis_result["nwtn_analysis"]["confidence_score"]
            }
        )
        
        return analysis_result
    
    def _extract_automated_insights(self, nwtn_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured insights from NWTN analysis"""
        response_text = nwtn_result.get('response', {}).get('text', '').lower()
        
        # Simple keyword-based insight extraction
        insights = {
            "market_potential": "unknown",
            "trl_level": "unknown",
            "risk_level": "medium",
            "commercial_readiness": "unknown"
        }
        
        # Market potential indicators
        if any(term in response_text for term in ["high potential", "significant market", "strong demand"]):
            insights["market_potential"] = "high"
        elif any(term in response_text for term in ["moderate potential", "niche market"]):
            insights["market_potential"] = "medium"
        elif any(term in response_text for term in ["limited potential", "small market"]):
            insights["market_potential"] = "low"
        
        # TRL level extraction
        for i in range(1, 10):
            if f"trl {i}" in response_text or f"level {i}" in response_text:
                insights["trl_level"] = f"TRL {i}"
                break
        
        # Risk assessment
        if any(term in response_text for term in ["high risk", "significant challenges", "major barriers"]):
            insights["risk_level"] = "high"
        elif any(term in response_text for term in ["low risk", "minimal barriers", "straightforward"]):
            insights["risk_level"] = "low"
        
        return insights
    
    def update_project_status(self,
                            project_id: str,
                            new_status: EvaluationStatus,
                            user_id: str,
                            notes: str = "") -> bool:
        """Update the status of an evaluation project"""
        if project_id not in self.active_projects:
            return False
        
        project = self.active_projects[project_id]
        old_status = project.status
        project.status = new_status
        project.last_modified = datetime.now()
        
        # Log status change
        self._log_activity(
            project_id=project_id,
            user_id=user_id,
            activity_type="status_changed",
            description=f"Status changed from {old_status.value} to {new_status.value}",
            metadata={"old_status": old_status.value, "new_status": new_status.value, "notes": notes}
        )
        
        return True
    
    def _check_user_permission(self,
                              project_id: str,
                              user_id: str,
                              required_permission: str) -> bool:
        """Check if user has required permission for project"""
        if project_id not in self.active_projects:
            return False
        
        project = self.active_projects[project_id]
        
        # Project creator has all permissions
        if project.created_by == user_id:
            return True
        
        # Check stakeholder permissions
        for permission in project.stakeholders:
            if permission.user_id == user_id:
                # Check if permission is expired
                if permission.expiry_date and datetime.now() > permission.expiry_date:
                    continue
                
                return required_permission in permission.permissions
        
        return False
    
    def _log_activity(self,
                     project_id: str,
                     user_id: str,
                     activity_type: str,
                     description: str,
                     metadata: Optional[Dict[str, Any]] = None):
        """Log an activity for audit trail"""
        activity = EvaluationActivity(
            activity_id=str(uuid.uuid4()),
            project_id=project_id,
            user_id=user_id,
            activity_type=activity_type,
            description=description,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.activity_log.append(activity)
    
    def get_project_analytics(self, project_id: str) -> Dict[str, Any]:
        """Get analytics for an evaluation project"""
        if project_id not in self.active_projects:
            return {}
        
        project = self.active_projects[project_id]
        project_activities = [a for a in self.activity_log if a.project_id == project_id]
        
        # Calculate analytics
        analytics = {
            "project_id": project_id,
            "created_date": project.created_at.isoformat(),
            "days_active": (datetime.now() - project.created_at).days,
            "current_status": project.status.value,
            "total_assets": len(project.research_assets),
            "total_stakeholders": len(project.stakeholders),
            "total_activities": len(project_activities),
            "asset_breakdown": {},
            "stakeholder_breakdown": {},
            "activity_timeline": []
        }
        
        # Asset breakdown by type
        for asset in project.research_assets:
            asset_type = asset.asset_type
            if asset_type not in analytics["asset_breakdown"]:
                analytics["asset_breakdown"][asset_type] = 0
            analytics["asset_breakdown"][asset_type] += 1
        
        # Stakeholder breakdown by role
        for stakeholder in project.stakeholders:
            role = stakeholder.role.value
            if role not in analytics["stakeholder_breakdown"]:
                analytics["stakeholder_breakdown"][role] = 0
            analytics["stakeholder_breakdown"][role] += 1
        
        # Activity timeline (last 30 days)
        recent_activities = [
            a for a in project_activities 
            if (datetime.now() - a.timestamp).days <= 30
        ]
        
        analytics["activity_timeline"] = [
            {
                "date": a.timestamp.date().isoformat(),
                "activity_type": a.activity_type,
                "user_id": a.user_id,
                "description": a.description
            }
            for a in sorted(recent_activities, key=lambda x: x.timestamp, reverse=True)[:10]
        ]
        
        return analytics

# University-specific implementations
class UNCTechTransfer(TechTransferWorkflow):
    """UNC-specific tech transfer workflow"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        super().__init__(storage_path)
        self.university_name = "University of North Carolina at Chapel Hill"
        self.tech_transfer_office = "UNC Office of Technology Commercialization"
    
    def create_unc_project(self,
                          title: str,
                          principal_investigator: str,
                          department: str,
                          funding_source: str = "NIH",
                          created_by: str = "system") -> EvaluationProject:
        """Create UNC-specific evaluation project"""
        description = f"""
University of North Carolina at Chapel Hill Technology Transfer Project

**Principal Investigator**: {principal_investigator}
**Department**: {department}
**Funding Source**: {funding_source}
**Tech Transfer Office**: {self.tech_transfer_office}

This project involves the evaluation of intellectual property developed at UNC-Chapel Hill
for potential commercialization and licensing to industry partners.
"""
        
        # UNC-specific compliance requirements
        compliance_reqs = ["UNC_IP_Policy", "NIH_Funding_Requirements"]
        if funding_source in ["NIH", "NSF", "DOD"]:
            compliance_reqs.append("Federal_Funding_Compliance")
        
        project = self.create_evaluation_project(
            title=title,
            description=description,
            university=self.university_name,
            created_by=created_by,
            nda_required=True,
            compliance_requirements=compliance_reqs
        )
        
        # Add UNC tech transfer officer as stakeholder
        self.add_stakeholder(
            project.project_id,
            "tech.transfer@unc.edu",
            StakeholderRole.TECH_TRANSFER_OFFICER,
            ["view", "download", "comment", "modify"]
        )
        
        return project

# Example usage and testing
if __name__ == "__main__":
    async def test_tech_transfer_workflow():
        """Test the technology transfer workflow"""
        
        print("🚀 Testing Technology Transfer IP Evaluation Workflow")
        
        # Initialize UNC-specific workflow
        unc_workflow = UNCTechTransfer()
        
        # Create a sample project
        project = unc_workflow.create_unc_project(
            title="Quantum Error Correction Algorithm for Noisy Intermediate-Scale Quantum Devices",
            principal_investigator="Dr. Sarah Chen",
            department="Physics and Astronomy",
            funding_source="NSF",
            created_by="sarah.chen@unc.edu"
        )
        
        print(f"✅ Created project: {project.title}")
        print(f"   Project ID: {project.project_id}")
        print(f"   University: {project.university}")
        print(f"   Status: {project.status.value}")
        
        # Add research assets
        paper_asset = unc_workflow.add_research_asset(
            project.project_id,
            "Quantum Error Correction Research Paper",
            "Comprehensive analysis of novel error correction techniques for NISQ devices",
            "paper",
            security_classification="confidential",
            metadata={"journal": "Nature Physics", "impact_factor": 19.684},
            created_by="sarah.chen@unc.edu"
        )
        
        algorithm_asset = unc_workflow.add_research_asset(
            project.project_id,
            "Proprietary Error Correction Algorithm",
            "Python implementation of novel quantum error correction algorithm",
            "code",
            security_classification="proprietary",
            metadata={"language": "Python", "performance_improvement": "40%"},
            created_by="sarah.chen@unc.edu"
        )
        
        print(f"✅ Added research assets:")
        print(f"   - {paper_asset.name} ({paper_asset.security_classification})")
        print(f"   - {algorithm_asset.name} ({algorithm_asset.security_classification})")
        
        # Add industry stakeholder
        industry_permission = unc_workflow.add_stakeholder(
            project.project_id,
            "michael.johnson@sas.com",
            StakeholderRole.INDUSTRY_EVALUATOR,
            ["view", "comment"],
            expiry_date=datetime.now() + timedelta(days=90)
        )
        
        print(f"✅ Added industry stakeholder: {industry_permission.user_id}")
        print(f"   Role: {industry_permission.role.value}")
        print(f"   Permissions: {industry_permission.permissions}")
        
        # Update project status
        unc_workflow.update_project_status(
            project.project_id,
            EvaluationStatus.EVALUATING,
            "sarah.chen@unc.edu",
            "Project ready for industry evaluation"
        )
        
        print(f"✅ Updated project status to: {project.status.value}")
        
        # Perform NWTN IP analysis
        try:
            analysis_result = await unc_workflow.analyze_ip_with_nwtn(
                project.project_id,
                algorithm_asset.asset_id,
                "sarah.chen@unc.edu"
            )
            
            print("✅ NWTN IP Analysis completed:")
            print(f"   Confidence Score: {analysis_result['nwtn_analysis']['confidence_score']:.2f}")
            print(f"   Market Potential: {analysis_result['automated_insights']['market_potential']}")
            print(f"   Risk Level: {analysis_result['automated_insights']['risk_level']}")
            
        except Exception as e:
            print(f"⚠️  NWTN analysis not available: {e}")
        
        # Get project analytics
        analytics = unc_workflow.get_project_analytics(project.project_id)
        
        print("✅ Project Analytics:")
        print(f"   Days Active: {analytics['days_active']}")
        print(f"   Total Assets: {analytics['total_assets']}")
        print(f"   Total Stakeholders: {analytics['total_stakeholders']}")
        print(f"   Total Activities: {analytics['total_activities']}")
        print(f"   Asset Breakdown: {analytics['asset_breakdown']}")
        
        print("\n🎉 Technology Transfer Workflow test completed!")
        print("Ready for integration with PRSM collaboration platform")
    
    # Run the test
    import asyncio
    asyncio.run(test_tech_transfer_workflow())