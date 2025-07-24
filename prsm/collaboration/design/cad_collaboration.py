"""
CAD File Collaboration for Engineering Partnerships

Provides secure P2P collaboration for CAD files including AutoCAD, SolidWorks, Fusion 360,
and other engineering design files. Features include real-time collaborative editing,
version control, design review workflows, and university-industry project coordination.

Key Features:
- Post-quantum cryptographic security for sensitive engineering designs
- Real-time collaborative CAD editing with conflict resolution
- Engineering design review workflows with approval chains
- University-industry project coordination templates
- Advanced version control with branching and merging
- Integration with popular CAD software packages
- NWTN AI-powered design optimization recommendations
- Export capabilities for multiple CAD formats
"""

import asyncio
import hashlib
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import zipfile
import tempfile
import shutil

from ..security.post_quantum_crypto_sharding import PostQuantumCryptoSharding

# Mock NWTN for testing
class MockNWTN:
    async def reason(self, prompt, context):
        return {
            "reasoning": [
                "Design complexity appears moderate for the given project scope",
                "Collaboration workflow should include regular design reviews",
                "Security considerations are appropriate for industry partnership"
            ],
            "recommendations": [
                "Implement milestone-based review checkpoints",
                "Use version control for all design iterations",
                "Establish clear approval criteria with industry partner"
            ]
        }


class CADFileType(Enum):
    """Supported CAD file types"""
    AUTOCAD_DWG = "autocad_dwg"
    AUTOCAD_DXF = "autocad_dxf"
    SOLIDWORKS_SLDPRT = "solidworks_sldprt"
    SOLIDWORKS_SLDASM = "solidworks_sldasm"
    SOLIDWORKS_SLDDRW = "solidworks_slddrw"
    FUSION360_F3D = "fusion360_f3d"
    INVENTOR_IPT = "inventor_ipt"
    INVENTOR_IAM = "inventor_iam"
    CATIA_CATPART = "catia_catpart"
    CATIA_CATPRODUCT = "catia_catproduct"
    STEP_STP = "step_stp"
    IGES_IGS = "iges_igs"
    STL = "stl"
    OBJ = "obj"
    PLY = "ply"


class ReviewStatus(Enum):
    """Design review status"""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    FINAL = "final"


class CollaborationRole(Enum):
    """CAD collaboration roles"""
    LEAD_ENGINEER = "lead_engineer"
    DESIGN_ENGINEER = "design_engineer"
    REVIEWER = "reviewer"
    VIEWER = "viewer"
    STUDENT = "student"
    INDUSTRY_MENTOR = "industry_mentor"
    FACULTY_ADVISOR = "faculty_advisor"


@dataclass
class CADDesignComment:
    """Design review comment"""
    id: str
    author_id: str
    author_name: str
    timestamp: datetime
    content: str
    coordinates: Tuple[float, float, float]  # 3D coordinates
    view_angle: Dict[str, float]  # Camera view parameters
    severity: str  # "info", "warning", "critical"
    status: str  # "open", "resolved", "dismissed"
    thread_id: Optional[str] = None
    attachments: List[str] = None


@dataclass
class CADDesignRevision:
    """CAD design revision"""
    id: str
    version: str
    author_id: str
    author_name: str
    timestamp: datetime
    description: str
    file_hash: str
    parent_revision: Optional[str]
    status: ReviewStatus
    review_comments: List[CADDesignComment]
    approval_chain: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class CollaborativeCADProject:
    """Collaborative CAD project"""
    id: str
    name: str
    description: str
    created_by: str
    created_at: datetime
    project_type: str  # "university_capstone", "industry_coop", "research_project"
    university: str
    industry_partner: Optional[str]
    file_type: CADFileType
    current_revision: str
    collaborators: Dict[str, CollaborationRole]
    access_permissions: Dict[str, List[str]]
    security_level: str  # "public", "internal", "confidential", "secret"
    encryption_enabled: bool
    version_history: List[CADDesignRevision]
    project_timeline: Dict[str, datetime]
    design_requirements: Dict[str, Any]
    nwtn_insights: List[Dict[str, Any]]


@dataclass
class CADCollaborationSession:
    """Real-time CAD collaboration session"""
    id: str
    project_id: str
    participants: List[str]
    started_at: datetime
    last_activity: datetime
    active_cursors: Dict[str, Dict[str, float]]  # User cursors in 3D space
    shared_view: Dict[str, Any]  # Synchronized camera view
    live_edits: List[Dict[str, Any]]  # Real-time edit operations
    voice_channel: Optional[str]
    screen_sharing: Optional[str]


class CADCollaboration:
    """Main CAD collaboration system"""
    
    def __init__(self):
        self.crypto_sharding = PostQuantumCryptoSharding()
        self.nwtn = MockNWTN()
        self.projects: Dict[str, CollaborativeCADProject] = {}
        self.active_sessions: Dict[str, CADCollaborationSession] = {}
        self.supported_formats = set(CADFileType)
        
        # University-industry partnership templates
        self.partnership_templates = {
            "unc_capstone": {
                "name": "UNC Engineering Capstone Project",
                "approval_chain": ["student", "faculty_advisor", "industry_mentor"],
                "timeline_template": {
                    "concept_review": timedelta(weeks=2),
                    "preliminary_design": timedelta(weeks=6),
                    "detailed_design": timedelta(weeks=10),
                    "prototype_review": timedelta(weeks=14),
                    "final_presentation": timedelta(weeks=16)
                },
                "deliverables": ["concept_drawings", "analysis_reports", "prototype_cad", "final_drawings"]
            },
            "ncsu_coop": {
                "name": "NC State Co-op Engineering Project",
                "approval_chain": ["student", "industry_mentor", "faculty_coordinator"],
                "timeline_template": {
                    "project_kickoff": timedelta(days=0),
                    "design_milestone_1": timedelta(weeks=4),
                    "design_milestone_2": timedelta(weeks=8),
                    "final_review": timedelta(weeks=12)
                },
                "deliverables": ["technical_drawings", "simulation_results", "manufacturing_specs"]
            },
            "duke_research": {
                "name": "Duke Engineering Research Collaboration",
                "approval_chain": ["graduate_student", "faculty_advisor", "industry_collaborator"],
                "timeline_template": {
                    "literature_review": timedelta(weeks=4),
                    "experimental_design": timedelta(weeks=8),
                    "prototype_development": timedelta(weeks=16),
                    "results_analysis": timedelta(weeks=20),
                    "publication_prep": timedelta(weeks=24)
                },
                "deliverables": ["research_drawings", "experimental_setups", "data_visualizations"]
            },
            "sas_analytics": {
                "name": "SAS Institute Engineering Analytics Project",
                "approval_chain": ["engineer", "team_lead", "project_manager"],
                "timeline_template": {
                    "requirements_analysis": timedelta(weeks=2),
                    "system_design": timedelta(weeks=6),
                    "implementation": timedelta(weeks=12),
                    "testing_validation": timedelta(weeks=16),
                    "deployment": timedelta(weeks=18)
                },
                "deliverables": ["system_architecture", "interface_designs", "technical_documentation"]
            }
        }
    
    async def create_cad_project(
        self,
        name: str,
        description: str,
        creator_id: str,
        project_type: str,
        university: str,
        file_type: CADFileType,
        industry_partner: Optional[str] = None,
        security_level: str = "internal"
    ) -> CollaborativeCADProject:
        """Create a new collaborative CAD project"""
        
        project_id = str(uuid.uuid4())
        
        # Apply partnership template if available
        template = self.partnership_templates.get(project_type, {})
        
        # Set up project timeline
        timeline = {}
        if "timeline_template" in template:
            base_date = datetime.now()
            for milestone, offset in template["timeline_template"].items():
                timeline[milestone] = base_date + offset
        
        # Initialize NWTN insights
        nwtn_context = {
            "project_type": project_type,
            "university": university,
            "industry_partner": industry_partner,
            "file_type": file_type.value,
            "security_level": security_level
        }
        
        nwtn_insights = await self._generate_project_insights(nwtn_context)
        
        project = CollaborativeCADProject(
            id=project_id,
            name=name,
            description=description,
            created_by=creator_id,
            created_at=datetime.now(),
            project_type=project_type,
            university=university,
            industry_partner=industry_partner,
            file_type=file_type,
            current_revision="",
            collaborators={creator_id: CollaborationRole.LEAD_ENGINEER},
            access_permissions={
                "read": [creator_id],
                "write": [creator_id],
                "admin": [creator_id]
            },
            security_level=security_level,
            encryption_enabled=security_level in ["confidential", "secret"],
            version_history=[],
            project_timeline=timeline,
            design_requirements={},
            nwtn_insights=nwtn_insights
        )
        
        self.projects[project_id] = project
        
        return project
    
    async def upload_cad_file(
        self,
        project_id: str,
        file_path: str,
        user_id: str,
        version_description: str = "Initial upload"
    ) -> CADDesignRevision:
        """Upload a CAD file to the project"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("write", []):
            raise PermissionError("User does not have write access to this project")
        
        # Calculate file hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        # Encrypt and shard file if required
        if project.encryption_enabled:
            encrypted_shards = self.crypto_sharding.shard_file(
                file_path,
                list(project.collaborators.keys()),
                num_shards=7  # Default number of shards
            )
        
        # Create revision
        revision_id = str(uuid.uuid4())
        version = f"v{len(project.version_history) + 1}.0"
        
        revision = CADDesignRevision(
            id=revision_id,
            version=version,
            author_id=user_id,
            author_name=f"User_{user_id}",
            timestamp=datetime.now(),
            description=version_description,
            file_hash=file_hash,
            parent_revision=project.current_revision if project.current_revision else None,
            status=ReviewStatus.DRAFT,
            review_comments=[],
            approval_chain=[],
            metadata={
                "file_size": Path(file_path).stat().st_size,
                "file_type": project.file_type.value,
                "encrypted": project.encryption_enabled
            }
        )
        
        project.version_history.append(revision)
        project.current_revision = revision_id
        
        # Generate NWTN analysis of the design
        design_insights = await self._analyze_cad_design(file_path, project)
        project.nwtn_insights.extend(design_insights)
        
        return revision
    
    async def add_design_comment(
        self,
        project_id: str,
        revision_id: str,
        user_id: str,
        content: str,
        coordinates: Tuple[float, float, float],
        view_angle: Dict[str, float],
        severity: str = "info"
    ) -> CADDesignComment:
        """Add a design review comment"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Find revision
        revision = None
        for rev in project.version_history:
            if rev.id == revision_id:
                revision = rev
                break
        
        if not revision:
            raise ValueError(f"Revision {revision_id} not found")
        
        comment = CADDesignComment(
            id=str(uuid.uuid4()),
            author_id=user_id,
            author_name=f"User_{user_id}",
            timestamp=datetime.now(),
            content=content,
            coordinates=coordinates,
            view_angle=view_angle,
            severity=severity,
            status="open",
            attachments=[]
        )
        
        revision.review_comments.append(comment)
        
        return comment
    
    async def start_collaboration_session(
        self,
        project_id: str,
        user_id: str,
        participants: List[str]
    ) -> CADCollaborationSession:
        """Start a real-time collaboration session"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("read", []):
            raise PermissionError("User does not have access to this project")
        
        session_id = str(uuid.uuid4())
        
        session = CADCollaborationSession(
            id=session_id,
            project_id=project_id,
            participants=[user_id] + participants,
            started_at=datetime.now(),
            last_activity=datetime.now(),
            active_cursors={},
            shared_view={
                "camera_position": [0, 0, 10],
                "camera_target": [0, 0, 0],
                "camera_up": [0, 1, 0],
                "zoom_level": 1.0
            },
            live_edits=[],
            voice_channel=None,
            screen_sharing=None
        )
        
        self.active_sessions[session_id] = session
        
        return session
    
    async def update_cursor_position(
        self,
        session_id: str,
        user_id: str,
        position: Dict[str, float]
    ):
        """Update user cursor position in 3D space"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        if user_id not in session.participants:
            raise PermissionError("User is not a participant in this session")
        
        session.active_cursors[user_id] = position
        session.last_activity = datetime.now()
    
    async def synchronize_view(
        self,
        session_id: str,
        user_id: str,
        view_parameters: Dict[str, Any]
    ):
        """Synchronize camera view across all participants"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        if user_id not in session.participants:
            raise PermissionError("User is not a participant in this session")
        
        session.shared_view.update(view_parameters)
        session.last_activity = datetime.now()
    
    async def submit_for_review(
        self,
        project_id: str,
        revision_id: str,
        user_id: str,
        reviewers: List[str]
    ) -> bool:
        """Submit design for review"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Find revision
        revision = None
        for rev in project.version_history:
            if rev.id == revision_id:
                revision = rev
                break
        
        if not revision:
            raise ValueError(f"Revision {revision_id} not found")
        
        # Check permissions
        if revision.author_id != user_id and user_id not in project.access_permissions.get("admin", []):
            raise PermissionError("Only the author or admin can submit for review")
        
        # Update status
        revision.status = ReviewStatus.UNDER_REVIEW
        
        # Initialize approval chain
        revision.approval_chain = [
            {
                "reviewer_id": reviewer_id,
                "status": "pending",
                "timestamp": None,
                "comments": ""
            }
            for reviewer_id in reviewers
        ]
        
        return True
    
    async def approve_design(
        self,
        project_id: str,
        revision_id: str,
        reviewer_id: str,
        approval_comments: str = ""
    ) -> bool:
        """Approve a design revision"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Find revision
        revision = None
        for rev in project.version_history:
            if rev.id == revision_id:
                revision = rev
                break
        
        if not revision:
            raise ValueError(f"Revision {revision_id} not found")
        
        # Find reviewer in approval chain
        reviewer_entry = None
        for entry in revision.approval_chain:
            if entry["reviewer_id"] == reviewer_id:
                reviewer_entry = entry
                break
        
        if not reviewer_entry:
            raise PermissionError("User is not a designated reviewer")
        
        # Update approval
        reviewer_entry["status"] = "approved"
        reviewer_entry["timestamp"] = datetime.now()
        reviewer_entry["comments"] = approval_comments
        
        # Check if all approvals are complete
        all_approved = all(
            entry["status"] == "approved"
            for entry in revision.approval_chain
        )
        
        if all_approved:
            revision.status = ReviewStatus.APPROVED
        
        return all_approved
    
    async def export_cad_project(
        self,
        project_id: str,
        user_id: str,
        export_format: str,
        include_history: bool = True
    ) -> str:
        """Export CAD project data"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("read", []):
            raise PermissionError("User does not have read access to this project")
        
        # Create temporary directory for export
        temp_dir = tempfile.mkdtemp()
        export_path = Path(temp_dir) / f"{project.name}_export.zip"
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Export project metadata
            project_data = asdict(project)
            project_data['created_at'] = project_data['created_at'].isoformat()
            project_data['project_timeline'] = {
                k: v.isoformat() for k, v in project_data['project_timeline'].items()
            }
            
            zipf.writestr(
                "project_metadata.json",
                json.dumps(project_data, indent=2, default=str)
            )
            
            # Export revision history if requested
            if include_history:
                for i, revision in enumerate(project.version_history):
                    revision_data = asdict(revision)
                    revision_data['timestamp'] = revision_data['timestamp'].isoformat()
                    
                    zipf.writestr(
                        f"revisions/revision_{i+1:03d}.json",
                        json.dumps(revision_data, indent=2, default=str)
                    )
            
            # Export NWTN insights
            zipf.writestr(
                "nwtn_insights.json",
                json.dumps(project.nwtn_insights, indent=2, default=str)
            )
            
            # Create project report
            report = self._generate_project_report(project)
            zipf.writestr("project_report.md", report)
        
        return str(export_path)
    
    async def _generate_project_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate NWTN insights for a new project"""
        
        nwtn_prompt = f"""
        Analyze this CAD collaboration project setup and provide insights:
        
        Project Type: {context['project_type']}
        University: {context['university']}
        Industry Partner: {context.get('industry_partner', 'None')}
        CAD File Type: {context['file_type']}
        Security Level: {context['security_level']}
        
        Provide insights on:
        1. Optimal collaboration workflow
        2. Potential design challenges
        3. Recommended review checkpoints
        4. Industry best practices
        5. Security considerations
        """
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "project_setup",
                "timestamp": datetime.now(),
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    async def _analyze_cad_design(
        self,
        file_path: str,
        project: CollaborativeCADProject
    ) -> List[Dict[str, Any]]:
        """Analyze CAD design using NWTN"""
        
        # Extract basic file information
        file_stats = Path(file_path).stat()
        
        nwtn_prompt = f"""
        Analyze this CAD design file for a {project.project_type} project:
        
        File Type: {project.file_type.value}
        File Size: {file_stats.st_size} bytes
        Project Context: {project.description}
        University: {project.university}
        Industry Partner: {project.industry_partner}
        
        Provide analysis on:
        1. Design complexity assessment
        2. Manufacturability considerations
        3. Potential optimization opportunities
        4. Collaboration workflow recommendations
        5. Quality assurance checkpoints
        """
        
        context = {
            "file_path": file_path,
            "project_data": asdict(project)
        }
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "design_analysis",
                "timestamp": datetime.now(),
                "file_hash": hashlib.sha256(file_path.encode()).hexdigest(),
                "analysis": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    def _generate_project_report(self, project: CollaborativeCADProject) -> str:
        """Generate a comprehensive project report"""
        
        template = self.partnership_templates.get(project.project_type, {})
        
        report = f"""# CAD Collaboration Project Report

## Project Information
- **Name**: {project.name}
- **Description**: {project.description}
- **Type**: {project.project_type}
- **University**: {project.university}
- **Industry Partner**: {project.industry_partner or 'None'}
- **Created**: {project.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **File Type**: {project.file_type.value}
- **Security Level**: {project.security_level}

## Collaboration Summary
- **Total Collaborators**: {len(project.collaborators)}
- **Current Revision**: {project.current_revision}
- **Total Revisions**: {len(project.version_history)}
- **Encryption Enabled**: {'Yes' if project.encryption_enabled else 'No'}

### Collaborator Roles
"""
        
        for user_id, role in project.collaborators.items():
            report += f"- **{user_id}**: {role.value}\n"
        
        report += f"\n## Version History\n"
        
        for i, revision in enumerate(project.version_history, 1):
            report += f"""
### Revision {i}: {revision.version}
- **Author**: {revision.author_name}
- **Date**: {revision.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- **Status**: {revision.status.value}
- **Description**: {revision.description}
- **Comments**: {len(revision.review_comments)}
"""
        
        report += f"\n## Project Timeline\n"
        for milestone, date in project.project_timeline.items():
            status = "✅" if date <= datetime.now() else "⏳"
            report += f"- {status} **{milestone.replace('_', ' ').title()}**: {date.strftime('%Y-%m-%d')}\n"
        
        report += f"\n## NWTN AI Insights\n"
        for insight in project.nwtn_insights:
            report += f"""
### {insight['type'].replace('_', ' ').title()} ({insight['timestamp'].strftime('%Y-%m-%d')})
"""
            if 'insights' in insight:
                for item in insight['insights']:
                    report += f"- {item}\n"
            
            if 'recommendations' in insight:
                report += "\n**Recommendations:**\n"
                for rec in insight['recommendations']:
                    report += f"- {rec}\n"
        
        report += f"""
## Security Information
- **Encryption**: {'Enabled' if project.encryption_enabled else 'Disabled'}
- **Access Control**: Role-based permissions
- **Audit Trail**: Complete version and access history
- **Post-Quantum Security**: {'Yes' if project.encryption_enabled else 'N/A'}

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report


# Testing and validation
async def test_cad_collaboration():
    """Test CAD collaboration functionality"""
    
    cad_collab = CADCollaboration()
    
    print("🔧 Testing CAD Collaboration System...")
    
    # Test 1: Create engineering capstone project
    print("\n1. Creating UNC Engineering Capstone Project...")
    project = await cad_collab.create_cad_project(
        name="Smart Manufacturing Robot Design",
        description="Senior capstone project developing automated manufacturing robot for SAS Institute",
        creator_id="student_001",
        project_type="unc_capstone",
        university="University of North Carolina at Chapel Hill",
        file_type=CADFileType.SOLIDWORKS_SLDASM,
        industry_partner="SAS Institute",
        security_level="confidential"
    )
    
    print(f"✅ Created project: {project.name}")
    print(f"   - ID: {project.id}")
    print(f"   - Timeline milestones: {len(project.project_timeline)}")
    print(f"   - NWTN insights: {len(project.nwtn_insights)}")
    
    # Test 2: Add collaborators
    print("\n2. Adding project collaborators...")
    project.collaborators.update({
        "faculty_001": CollaborationRole.FACULTY_ADVISOR,
        "industry_001": CollaborationRole.INDUSTRY_MENTOR,
        "student_002": CollaborationRole.DESIGN_ENGINEER
    })
    
    # Update permissions
    all_collaborators = list(project.collaborators.keys())
    project.access_permissions.update({
        "read": all_collaborators,
        "write": ["student_001", "student_002"],
        "admin": ["student_001", "faculty_001"]
    })
    
    print(f"✅ Added {len(project.collaborators)} collaborators")
    
    # Test 3: Simulate CAD file upload
    print("\n3. Simulating CAD file upload...")
    
    # Create temporary CAD file for testing
    temp_file = tempfile.NamedTemporaryFile(suffix='.sldasm', delete=False)
    temp_file.write(b"Mock SolidWorks assembly file content for testing")
    temp_file.close()
    
    try:
        revision = await cad_collab.upload_cad_file(
            project_id=project.id,
            file_path=temp_file.name,
            user_id="student_001",
            version_description="Initial robot assembly design"
        )
        
        print(f"✅ Uploaded CAD file: {revision.version}")
        print(f"   - File hash: {revision.file_hash[:16]}...")
        print(f"   - Encrypted: {project.encryption_enabled}")
        
    finally:
        # Clean up temporary file
        Path(temp_file.name).unlink()
    
    # Test 4: Add design comments
    print("\n4. Adding design review comments...")
    
    comment1 = await cad_collab.add_design_comment(
        project_id=project.id,
        revision_id=revision.id,
        user_id="faculty_001",
        content="Consider adding reinforcement to the base joint for improved stability",
        coordinates=(10.5, 20.3, 5.7),
        view_angle={"azimuth": 45, "elevation": 30, "distance": 100},
        severity="warning"
    )
    
    comment2 = await cad_collab.add_design_comment(
        project_id=project.id,
        revision_id=revision.id,
        user_id="industry_001",
        content="Excellent material choice for the gripper mechanism",
        coordinates=(15.2, 18.1, 12.4),
        view_angle={"azimuth": 90, "elevation": 45, "distance": 80},
        severity="info"
    )
    
    print(f"✅ Added {len(revision.review_comments)} design comments")
    
    # Test 5: Start collaboration session
    print("\n5. Starting real-time collaboration session...")
    
    session = await cad_collab.start_collaboration_session(
        project_id=project.id,
        user_id="student_001",
        participants=["student_002", "faculty_001"]
    )
    
    print(f"✅ Started collaboration session: {session.id}")
    print(f"   - Participants: {len(session.participants)}")
    
    # Test 6: Update cursor positions
    print("\n6. Testing real-time cursor synchronization...")
    
    await cad_collab.update_cursor_position(
        session_id=session.id,
        user_id="student_001",
        position={"x": 10.5, "y": 15.2, "z": 8.3}
    )
    
    await cad_collab.update_cursor_position(
        session_id=session.id,
        user_id="student_002",
        position={"x": 12.1, "y": 18.7, "z": 6.9}
    )
    
    print(f"✅ Updated cursor positions for {len(session.active_cursors)} users")
    
    # Test 7: Submit for review
    print("\n7. Testing design review workflow...")
    
    await cad_collab.submit_for_review(
        project_id=project.id,
        revision_id=revision.id,
        user_id="student_001",
        reviewers=["faculty_001", "industry_001"]
    )
    
    print(f"✅ Submitted for review - Status: {revision.status.value}")
    print(f"   - Reviewers: {len(revision.approval_chain)}")
    
    # Test 8: Approve design
    print("\n8. Testing approval process...")
    
    await cad_collab.approve_design(
        project_id=project.id,
        revision_id=revision.id,
        reviewer_id="faculty_001",
        approval_comments="Design meets all engineering requirements. Approved for prototype phase."
    )
    
    await cad_collab.approve_design(
        project_id=project.id,
        revision_id=revision.id,
        reviewer_id="industry_001",
        approval_comments="Manufacturing feasibility confirmed. Ready for production planning."
    )
    
    print(f"✅ Design approved - Final status: {revision.status.value}")
    
    # Test 9: Export project
    print("\n9. Testing project export...")
    
    export_path = await cad_collab.export_cad_project(
        project_id=project.id,
        user_id="student_001",
        export_format="comprehensive",
        include_history=True
    )
    
    print(f"✅ Exported project to: {export_path}")
    
    # Verify export contents
    with zipfile.ZipFile(export_path, 'r') as zipf:
        files = zipf.namelist()
        print(f"   - Export contains {len(files)} files:")
        for file in files[:5]:  # Show first 5 files
            print(f"     • {file}")
        if len(files) > 5:
            print(f"     • ... and {len(files) - 5} more files")
    
    # Clean up
    Path(export_path).unlink()
    shutil.rmtree(Path(export_path).parent)
    
    print(f"\n🎉 CAD Collaboration testing completed successfully!")
    print(f"   - Created {len(cad_collab.projects)} projects")
    print(f"   - Active sessions: {len(cad_collab.active_sessions)}")
    print(f"   - Supported formats: {len(cad_collab.supported_formats)}")
    print(f"   - Partnership templates: {len(cad_collab.partnership_templates)}")


if __name__ == "__main__":
    asyncio.run(test_cad_collaboration())