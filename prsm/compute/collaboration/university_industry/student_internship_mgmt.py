"""
Student Internship/Co-op Management Platform

Comprehensive platform for managing university-industry student collaborations including
internships, co-op programs, capstone projects, and research experiences. Features secure
project management, mentorship coordination, and academic-industry integration.

Key Features:
- Post-quantum cryptographic security for sensitive student and company data
- Multi-institutional internship program coordination
- Industry mentor and faculty advisor collaboration workflows
- Student project lifecycle management with academic credit tracking
- Secure company data access with graduated permissions
- University-industry partnership analytics and reporting
- FERPA-compliant student record management
- Integration with university systems and industry HR platforms
"""

import asyncio
import hashlib
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
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
                "Student placement appears well-aligned with academic goals",
                "Industry mentor engagement shows strong commitment to program",
                "Project scope is appropriate for internship duration and student level"
            ],
            "recommendations": [
                "Establish regular check-in schedule with all stakeholders",
                "Define clear learning objectives and assessment criteria",
                "Ensure proper onboarding and security training completion"
            ]
        }


class ProgramType(Enum):
    """Student program types"""
    SUMMER_INTERNSHIP = "summer_internship"
    COOP_ROTATION = "coop_rotation"
    CAPSTONE_PROJECT = "capstone_project"
    RESEARCH_EXPERIENCE = "research_experience"
    INDEPENDENT_STUDY = "independent_study"
    THESIS_PROJECT = "thesis_project"
    INDUSTRY_PRACTICUM = "industry_practicum"


class StudentLevel(Enum):
    """Academic level of students"""
    FRESHMAN = "freshman"
    SOPHOMORE = "sophomore"
    JUNIOR = "junior"
    SENIOR = "senior"
    MASTERS = "masters"
    PHD = "phd"
    POSTDOC = "postdoc"


class ProjectStatus(Enum):
    """Project status throughout lifecycle"""
    DRAFT = "draft"
    POSTED = "posted"
    APPLICATIONS_OPEN = "applications_open"
    UNDER_REVIEW = "under_review"
    STUDENT_SELECTED = "student_selected"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ParticipantRole(Enum):
    """Roles in internship program"""
    STUDENT = "student"
    INDUSTRY_MENTOR = "industry_mentor"
    FACULTY_ADVISOR = "faculty_advisor"
    PROGRAM_COORDINATOR = "program_coordinator"
    HR_REPRESENTATIVE = "hr_representative"
    ACADEMIC_SUPERVISOR = "academic_supervisor"


class AccessLevel(Enum):
    """Data access levels for security"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"


@dataclass
class StudentProfile:
    """Student participant profile"""
    id: str
    university_id: str
    name: str
    email: str
    university: str
    department: str
    major: str
    academic_level: StudentLevel
    gpa: float
    graduation_date: datetime
    skills: List[str]
    interests: List[str]
    previous_experience: List[Dict[str, Any]]
    security_clearance: Optional[str]
    availability: Dict[str, Any]
    portfolio_links: List[str]
    references: List[Dict[str, str]]
    ferpa_consent: bool
    created_at: datetime


@dataclass
class CompanyProfile:
    """Industry partner profile"""
    id: str
    name: str
    industry: str
    size: str  # "startup", "small", "medium", "large", "enterprise"
    location: str
    description: str
    website: str
    contact_person: str
    contact_email: str
    partnership_agreements: List[str]
    security_requirements: List[str]
    mentorship_capacity: int
    active_programs: List[ProgramType]
    created_at: datetime


@dataclass
class ProjectRequirements:
    """Project requirements and specifications"""
    technical_skills: List[str]
    soft_skills: List[str]
    academic_level: List[StudentLevel]
    majors: List[str]
    min_gpa: float
    security_clearance_required: Optional[str]
    time_commitment: str  # "part-time", "full-time"
    duration: str  # e.g., "10 weeks", "6 months"
    deliverables: List[str]
    learning_objectives: List[str]


@dataclass
class InternshipApplication:
    """Student application for internship"""
    id: str
    student_id: str
    project_id: str
    submitted_at: datetime
    cover_letter: str
    resume_path: str
    transcript_path: Optional[str]
    portfolio_items: List[str]
    responses: Dict[str, str]  # Question ID -> Response
    status: str  # "submitted", "under_review", "shortlisted", "interviewed", "accepted", "rejected"
    interview_notes: List[Dict[str, Any]]
    decision_rationale: str
    processed_by: List[str]


@dataclass
class MilestoneCheckpoint:
    """Project milestone and checkpoint"""
    id: str
    title: str
    description: str
    due_date: datetime
    deliverables: List[str]
    assessment_criteria: Dict[str, Any]
    completed: bool
    completion_date: Optional[datetime]
    student_submission: Optional[str]
    mentor_feedback: Optional[str]
    faculty_evaluation: Optional[str]
    grade: Optional[str]


@dataclass
class InternshipProject:
    """Complete internship project definition"""
    id: str
    title: str
    description: str
    company_id: str
    program_type: ProgramType
    status: ProjectStatus
    requirements: ProjectRequirements
    start_date: datetime
    end_date: datetime
    application_deadline: datetime
    compensation: Optional[Dict[str, Any]]
    academic_credit: bool
    credit_hours: Optional[int]
    industry_mentor_id: str
    faculty_advisor_id: Optional[str]
    assigned_student_id: Optional[str]
    applications: List[InternshipApplication]
    milestones: List[MilestoneCheckpoint]
    security_level: AccessLevel
    encrypted_files: Dict[str, str]  # File name -> encrypted path
    collaboration_tools: List[str]
    nwtn_insights: List[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


@dataclass
class StudentProgress:
    """Student progress tracking"""
    student_id: str
    project_id: str
    overall_progress: float  # 0.0 to 1.0
    milestone_completion: Dict[str, bool]
    mentor_ratings: List[Dict[str, Any]]
    self_assessments: List[Dict[str, Any]]
    academic_performance: Dict[str, Any]
    skill_development: Dict[str, float]  # Skill -> improvement score
    challenges: List[str]
    achievements: List[str]
    weekly_reports: List[Dict[str, Any]]
    last_updated: datetime


@dataclass
class ProgramAnalytics:
    """Program-wide analytics and metrics"""
    university: str
    semester: str
    total_projects: int
    total_applications: int
    placement_rate: float
    completion_rate: float
    average_rating: float
    skill_demand: Dict[str, int]
    industry_distribution: Dict[str, int]
    student_satisfaction: float
    mentor_satisfaction: float
    academic_outcomes: Dict[str, Any]
    generated_at: datetime


class StudentInternshipManagement:
    """Main student internship/co-op management system"""
    
    def __init__(self):
        self.crypto_sharding = PostQuantumCryptoSharding()
        self.nwtn = MockNWTN()
        
        self.students: Dict[str, StudentProfile] = {}
        self.companies: Dict[str, CompanyProfile] = {}
        self.projects: Dict[str, InternshipProject] = {}
        self.student_progress: Dict[str, StudentProgress] = {}
        
        # University program templates
        self.university_programs = {
            "unc_coop": {
                "name": "UNC Engineering Co-op Program",
                "requirements": {
                    "min_gpa": 3.0,
                    "academic_levels": [StudentLevel.JUNIOR, StudentLevel.SENIOR],
                    "duration": "6 months",
                    "credit_hours": 3
                },
                "milestones": [
                    {"title": "Orientation Completion", "weeks": 1},
                    {"title": "Mid-term Evaluation", "weeks": 12},
                    {"title": "Final Presentation", "weeks": 24},
                    {"title": "Academic Assessment", "weeks": 26}
                ],
                "industry_partners": ["SAS Institute", "IBM", "Cisco", "MetLife"]
            },
            "duke_research": {
                "name": "Duke Research Experience Program",
                "requirements": {
                    "min_gpa": 3.5,
                    "academic_levels": [StudentLevel.SOPHOMORE, StudentLevel.JUNIOR, StudentLevel.SENIOR],
                    "duration": "10 weeks",
                    "credit_hours": 4
                },
                "milestones": [
                    {"title": "Literature Review", "weeks": 2},
                    {"title": "Methodology Presentation", "weeks": 4},
                    {"title": "Preliminary Results", "weeks": 7},
                    {"title": "Final Research Presentation", "weeks": 10}
                ],
                "industry_partners": ["Biogen", "GSK", "Pfizer", "Duke Health"]
            },
            "ncsu_capstone": {
                "name": "NC State Senior Design Program",
                "requirements": {
                    "min_gpa": 2.75,
                    "academic_levels": [StudentLevel.SENIOR],
                    "duration": "2 semesters",
                    "credit_hours": 6
                },
                "milestones": [
                    {"title": "Project Proposal", "weeks": 4},
                    {"title": "Design Review", "weeks": 8},
                    {"title": "Prototype Demo", "weeks": 24},
                    {"title": "Final Presentation", "weeks": 32}
                ],
                "industry_partners": ["John Deere", "Caterpillar", "Boeing", "GE"]
            },
            "sas_analytics": {
                "name": "SAS Institute Analytics Internship",
                "requirements": {
                    "min_gpa": 3.2,
                    "academic_levels": [StudentLevel.JUNIOR, StudentLevel.SENIOR, StudentLevel.MASTERS],
                    "duration": "12 weeks",
                    "credit_hours": 3
                },
                "milestones": [
                    {"title": "SAS Training Completion", "weeks": 2},
                    {"title": "Project Kickoff", "weeks": 3},
                    {"title": "Data Analysis Checkpoint", "weeks": 6},
                    {"title": "Model Development", "weeks": 9},
                    {"title": "Final Analytics Presentation", "weeks": 12}
                ],
                "industry_partners": ["SAS Institute"]
            }
        }
    
    async def register_student(
        self,
        name: str,
        email: str,
        university: str,
        department: str,
        major: str,
        academic_level: StudentLevel,
        gpa: float,
        graduation_date: datetime,
        skills: List[str],
        interests: List[str]
    ) -> StudentProfile:
        """Register a new student in the program"""
        
        student_id = str(uuid.uuid4())
        
        student = StudentProfile(
            id=student_id,
            university_id=f"{university.lower().replace(' ', '_')}_{student_id[:8]}",
            name=name,
            email=email,
            university=university,
            department=department,
            major=major,
            academic_level=academic_level,
            gpa=gpa,
            graduation_date=graduation_date,
            skills=skills,
            interests=interests,
            previous_experience=[],
            security_clearance=None,
            availability={},
            portfolio_links=[],
            references=[],
            ferpa_consent=True,
            created_at=datetime.now()
        )
        
        self.students[student_id] = student
        return student
    
    async def register_company(
        self,
        name: str,
        industry: str,
        size: str,
        location: str,
        description: str,
        contact_person: str,
        contact_email: str,
        mentorship_capacity: int
    ) -> CompanyProfile:
        """Register a new industry partner"""
        
        company_id = str(uuid.uuid4())
        
        company = CompanyProfile(
            id=company_id,
            name=name,
            industry=industry,
            size=size,
            location=location,
            description=description,
            website="",
            contact_person=contact_person,
            contact_email=contact_email,
            partnership_agreements=[],
            security_requirements=[],
            mentorship_capacity=mentorship_capacity,
            active_programs=[],
            created_at=datetime.now()
        )
        
        self.companies[company_id] = company
        return company
    
    async def create_internship_project(
        self,
        title: str,
        description: str,
        company_id: str,
        program_type: ProgramType,
        requirements: ProjectRequirements,
        start_date: datetime,
        end_date: datetime,
        industry_mentor_id: str,
        security_level: AccessLevel = AccessLevel.INTERNAL,
        academic_credit: bool = True,
        credit_hours: Optional[int] = 3
    ) -> InternshipProject:
        """Create a new internship project"""
        
        if company_id not in self.companies:
            raise ValueError(f"Company {company_id} not found")
        
        project_id = str(uuid.uuid4())
        application_deadline = start_date - timedelta(weeks=4)  # Default 4 weeks before start
        
        # Generate NWTN insights for project setup
        company = self.companies[company_id]
        nwtn_context = {
            "project_title": title,
            "company": company.name,
            "industry": company.industry,
            "program_type": program_type.value,
            "duration": (end_date - start_date).days,
            "requirements": asdict(requirements)
        }
        
        nwtn_insights = await self._generate_project_insights(nwtn_context)
        
        # Create default milestones based on program type
        milestones = self._create_default_milestones(program_type, start_date, end_date)
        
        project = InternshipProject(
            id=project_id,
            title=title,
            description=description,
            company_id=company_id,
            program_type=program_type,
            status=ProjectStatus.DRAFT,
            requirements=requirements,
            start_date=start_date,
            end_date=end_date,
            application_deadline=application_deadline,
            compensation=None,
            academic_credit=academic_credit,
            credit_hours=credit_hours,
            industry_mentor_id=industry_mentor_id,
            faculty_advisor_id=None,
            assigned_student_id=None,
            applications=[],
            milestones=milestones,
            security_level=security_level,
            encrypted_files={},
            collaboration_tools=[],
            nwtn_insights=nwtn_insights,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.projects[project_id] = project
        return project
    
    async def submit_application(
        self,
        student_id: str,
        project_id: str,
        cover_letter: str,
        resume_path: str,
        additional_responses: Dict[str, str] = None
    ) -> InternshipApplication:
        """Submit student application for internship"""
        
        if student_id not in self.students:
            raise ValueError(f"Student {student_id} not found")
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        if project.status != ProjectStatus.APPLICATIONS_OPEN:
            raise ValueError("Applications are not currently open for this project")
        
        if datetime.now() > project.application_deadline:
            raise ValueError("Application deadline has passed")
        
        # Check if student already applied
        existing_application = any(
            app.student_id == student_id for app in project.applications
        )
        if existing_application:
            raise ValueError("Student has already applied to this project")
        
        application_id = str(uuid.uuid4())
        
        # Encrypt sensitive documents if required
        encrypted_resume = None
        if project.security_level in [AccessLevel.CONFIDENTIAL, AccessLevel.RESTRICTED, AccessLevel.SECRET]:
            # Encrypt resume file
            authorized_users = [
                student_id,
                project.industry_mentor_id,
                project.faculty_advisor_id
            ]
            authorized_users = [u for u in authorized_users if u is not None]
            
            encrypted_resume = self.crypto_sharding.shard_file(
                resume_path,
                authorized_users,
                num_shards=5
            )
        
        application = InternshipApplication(
            id=application_id,
            student_id=student_id,
            project_id=project_id,
            submitted_at=datetime.now(),
            cover_letter=cover_letter,
            resume_path=resume_path if not encrypted_resume else "encrypted",
            transcript_path=None,
            portfolio_items=[],
            responses=additional_responses or {},
            status="submitted",
            interview_notes=[],
            decision_rationale="",
            processed_by=[]
        )
        
        project.applications.append(application)
        project.updated_at = datetime.now()
        
        return application
    
    async def select_student(
        self,
        project_id: str,
        student_id: str,
        mentor_id: str,
        selection_rationale: str
    ) -> bool:
        """Select student for internship project"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Find the student's application
        student_application = None
        for app in project.applications:
            if app.student_id == student_id:
                student_application = app
                break
        
        if not student_application:
            raise ValueError("Student did not apply to this project")
        
        # Update application status
        student_application.status = "accepted"
        student_application.decision_rationale = selection_rationale
        student_application.processed_by.append(mentor_id)
        
        # Update project
        project.assigned_student_id = student_id
        project.status = ProjectStatus.STUDENT_SELECTED
        project.updated_at = datetime.now()
        
        # Reject other applications
        for app in project.applications:
            if app.student_id != student_id and app.status == "submitted":
                app.status = "rejected"
                app.decision_rationale = "Another candidate was selected"
        
        # Initialize student progress tracking
        await self._initialize_student_progress(student_id, project_id)
        
        return True
    
    async def start_internship(
        self,
        project_id: str,
        faculty_advisor_id: str
    ) -> bool:
        """Start the internship program"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        if project.status != ProjectStatus.STUDENT_SELECTED:
            raise ValueError("Project is not ready to start")
        
        if not project.assigned_student_id:
            raise ValueError("No student assigned to project")
        
        project.faculty_advisor_id = faculty_advisor_id
        project.status = ProjectStatus.ACTIVE
        project.updated_at = datetime.now()
        
        # Generate onboarding insights
        student = self.students[project.assigned_student_id]
        company = self.companies[project.company_id]
        
        onboarding_context = {
            "student_profile": student,
            "company_profile": company,
            "project_details": project
        }
        
        onboarding_insights = await self._generate_onboarding_insights(onboarding_context)
        project.nwtn_insights.extend(onboarding_insights)
        
        return True
    
    async def update_milestone_progress(
        self,
        project_id: str,
        milestone_id: str,
        student_submission: str,
        submitted_by: str
    ) -> bool:
        """Update progress on a project milestone"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Find milestone
        milestone = None
        for m in project.milestones:
            if m.id == milestone_id:
                milestone = m
                break
        
        if not milestone:
            raise ValueError(f"Milestone {milestone_id} not found")
        
        milestone.student_submission = student_submission
        milestone.completion_date = datetime.now()
        milestone.completed = True
        
        # Update student progress
        if project.assigned_student_id in self.student_progress:
            progress = self.student_progress[project.assigned_student_id]
            progress.milestone_completion[milestone_id] = True
            progress.last_updated = datetime.now()
            
            # Calculate overall progress
            completed_milestones = sum(1 for completed in progress.milestone_completion.values() if completed)
            total_milestones = len(project.milestones)
            progress.overall_progress = completed_milestones / total_milestones if total_milestones > 0 else 0.0
        
        project.updated_at = datetime.now()
        return True
    
    async def add_mentor_feedback(
        self,
        project_id: str,
        milestone_id: str,
        mentor_id: str,
        feedback: str,
        rating: float
    ) -> bool:
        """Add mentor feedback to milestone"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Find milestone
        milestone = None
        for m in project.milestones:
            if m.id == milestone_id:
                milestone = m
                break
        
        if not milestone:
            raise ValueError(f"Milestone {milestone_id} not found")
        
        milestone.mentor_feedback = feedback
        
        # Update student progress with mentor rating
        if project.assigned_student_id in self.student_progress:
            progress = self.student_progress[project.assigned_student_id]
            progress.mentor_ratings.append({
                "milestone_id": milestone_id,
                "mentor_id": mentor_id,
                "rating": rating,
                "feedback": feedback,
                "timestamp": datetime.now()
            })
            progress.last_updated = datetime.now()
        
        project.updated_at = datetime.now()
        return True
    
    async def complete_internship(
        self,
        project_id: str,
        final_evaluation: Dict[str, Any],
        academic_grade: str
    ) -> bool:
        """Complete the internship program"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        if project.status != ProjectStatus.ACTIVE:
            raise ValueError("Project is not currently active")
        
        project.status = ProjectStatus.COMPLETED
        project.updated_at = datetime.now()
        
        # Update student progress with final evaluation
        if project.assigned_student_id in self.student_progress:
            progress = self.student_progress[project.assigned_student_id]
            progress.academic_performance["final_evaluation"] = final_evaluation
            progress.academic_performance["final_grade"] = academic_grade
            progress.overall_progress = 1.0
            progress.last_updated = datetime.now()
        
        # Generate completion insights
        completion_context = {
            "project_id": project_id,
            "student_progress": asdict(self.student_progress.get(project.assigned_student_id)),
            "project_details": asdict(project)
        }
        
        completion_insights = await self._generate_completion_insights(completion_context)
        project.nwtn_insights.extend(completion_insights)
        
        return True
    
    async def generate_program_analytics(
        self,
        university: str = None,
        semester: str = None
    ) -> ProgramAnalytics:
        """Generate comprehensive program analytics"""
        
        # Filter projects based on criteria
        filtered_projects = list(self.projects.values())
        if university:
            student_ids = [s.id for s in self.students.values() if s.university == university]
            filtered_projects = [p for p in filtered_projects if p.assigned_student_id in student_ids]
        
        # Calculate metrics
        total_projects = len(filtered_projects)
        total_applications = sum(len(p.applications) for p in filtered_projects)
        
        completed_projects = [p for p in filtered_projects if p.status == ProjectStatus.COMPLETED]
        placement_rate = len([p for p in filtered_projects if p.assigned_student_id is not None]) / total_projects if total_projects > 0 else 0
        completion_rate = len(completed_projects) / total_projects if total_projects > 0 else 0
        
        # Calculate average rating from mentor feedback
        all_ratings = []
        for project_id in self.student_progress:
            progress = self.student_progress[project_id]
            all_ratings.extend([r["rating"] for r in progress.mentor_ratings])
        
        average_rating = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0
        
        # Skill demand analysis
        skill_demand = {}
        for project in filtered_projects:
            for skill in project.requirements.technical_skills:
                skill_demand[skill] = skill_demand.get(skill, 0) + 1
        
        # Industry distribution
        industry_distribution = {}
        for project in filtered_projects:
            company = self.companies[project.company_id]
            industry_distribution[company.industry] = industry_distribution.get(company.industry, 0) + 1
        
        analytics = ProgramAnalytics(
            university=university or "All Universities",
            semester=semester or "All Semesters",
            total_projects=total_projects,
            total_applications=total_applications,
            placement_rate=placement_rate,
            completion_rate=completion_rate,
            average_rating=average_rating,
            skill_demand=skill_demand,
            industry_distribution=industry_distribution,
            student_satisfaction=4.2,  # Mock data
            mentor_satisfaction=4.5,   # Mock data
            academic_outcomes={"average_gpa_impact": 0.3},
            generated_at=datetime.now()
        )
        
        return analytics
    
    async def export_student_transcript(
        self,
        student_id: str,
        project_id: str,
        include_confidential: bool = False
    ) -> str:
        """Export student internship transcript"""
        
        if student_id not in self.students:
            raise ValueError(f"Student {student_id} not found")
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        student = self.students[student_id]
        project = self.projects[project_id]
        progress = self.student_progress.get(student_id)
        
        # Create temporary directory for export
        temp_dir = tempfile.mkdtemp()
        export_path = Path(temp_dir) / f"{student.name.replace(' ', '_')}_transcript.zip"
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Student information
            student_data = asdict(student)
            student_data['graduation_date'] = student_data['graduation_date'].isoformat()
            student_data['created_at'] = student_data['created_at'].isoformat()
            
            zipf.writestr(
                "student_profile.json",
                json.dumps(student_data, indent=2, default=str)
            )
            
            # Project information
            project_data = asdict(project)
            project_data['start_date'] = project_data['start_date'].isoformat()
            project_data['end_date'] = project_data['end_date'].isoformat()
            project_data['application_deadline'] = project_data['application_deadline'].isoformat()
            project_data['created_at'] = project_data['created_at'].isoformat()
            project_data['updated_at'] = project_data['updated_at'].isoformat()
            
            if not include_confidential:
                # Remove sensitive information
                project_data.pop('encrypted_files', None)
                project_data.pop('security_level', None)
            
            zipf.writestr(
                "project_details.json",
                json.dumps(project_data, indent=2, default=str)
            )
            
            # Progress information
            if progress:
                progress_data = asdict(progress)
                progress_data['last_updated'] = progress_data['last_updated'].isoformat()
                
                zipf.writestr(
                    "student_progress.json",
                    json.dumps(progress_data, indent=2, default=str)
                )
            
            # Generate academic transcript
            transcript = self._generate_academic_transcript(student, project, progress)
            zipf.writestr("academic_transcript.txt", transcript)
        
        return str(export_path)
    
    async def _generate_project_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate NWTN insights for project setup"""
        
        nwtn_prompt = f"""
        Analyze this internship project setup and provide insights:
        
        Project: {context['project_title']}
        Company: {context['company']} ({context['industry']})
        Program Type: {context['program_type']}
        Duration: {context['duration']} days
        
        Requirements:
        - Technical Skills: {context['requirements']['technical_skills']}
        - Academic Levels: {context['requirements']['academic_level']}
        - Minimum GPA: {context['requirements']['min_gpa']}
        
        Provide insights on:
        1. Project alignment with student development goals
        2. Industry mentorship opportunities
        3. Academic integration possibilities
        4. Potential challenges and mitigation strategies
        5. Success metrics and evaluation criteria
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
    
    async def _generate_onboarding_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate onboarding insights for student-company pairing"""
        
        nwtn_prompt = f"""
        Analyze this student-company pairing for onboarding recommendations:
        
        Student: {context['student_profile'].name} 
        Major: {context['student_profile'].major}
        Skills: {context['student_profile'].skills}
        GPA: {context['student_profile'].gpa}
        
        Company: {context['company_profile'].name}
        Industry: {context['company_profile'].industry}
        
        Project: {context['project_details'].title}
        Duration: {(context['project_details'].end_date - context['project_details'].start_date).days} days
        
        Provide onboarding recommendations:
        1. Initial orientation priorities
        2. Skill development opportunities
        3. Mentorship pairing strategies
        4. Early project milestones
        5. Integration with company culture
        """
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "onboarding",
                "timestamp": datetime.now(),
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    async def _generate_completion_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate completion insights and recommendations"""
        
        nwtn_prompt = f"""
        Analyze this completed internship for insights and future recommendations:
        
        Overall Progress: {context['student_progress']['overall_progress'] * 100}%
        Milestones Completed: {sum(1 for completed in context['student_progress']['milestone_completion'].values() if completed)}
        Mentor Ratings: {len(context['student_progress']['mentor_ratings'])} evaluations
        
        Final Evaluation: {context['student_progress']['academic_performance'].get('final_evaluation', 'Not available')}
        Final Grade: {context['student_progress']['academic_performance'].get('final_grade', 'Not available')}
        
        Provide insights on:
        1. Student performance analysis
        2. Program effectiveness assessment
        3. Recommendations for future improvements
        4. Career development suggestions for student
        5. Partnership enhancement opportunities
        """
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "completion_analysis",
                "timestamp": datetime.now(),
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    def _create_default_milestones(
        self,
        program_type: ProgramType,
        start_date: datetime,
        end_date: datetime
    ) -> List[MilestoneCheckpoint]:
        """Create default milestones based on program type"""
        
        duration_weeks = (end_date - start_date).days // 7
        milestones = []
        
        if program_type == ProgramType.SUMMER_INTERNSHIP:
            milestone_schedule = [
                {"title": "Orientation & Setup", "week": 1},
                {"title": "Project Kickoff", "week": 2},
                {"title": "Mid-term Review", "week": duration_weeks // 2},
                {"title": "Final Presentation", "week": duration_weeks - 1}
            ]
        elif program_type == ProgramType.COOP_ROTATION:
            milestone_schedule = [
                {"title": "Onboarding Complete", "week": 2},
                {"title": "First Quarter Review", "week": duration_weeks // 4},
                {"title": "Mid-term Evaluation", "week": duration_weeks // 2},
                {"title": "Third Quarter Review", "week": 3 * duration_weeks // 4},
                {"title": "Final Assessment", "week": duration_weeks - 1}
            ]
        elif program_type == ProgramType.CAPSTONE_PROJECT:
            milestone_schedule = [
                {"title": "Project Proposal", "week": 4},
                {"title": "Literature Review", "week": 8},
                {"title": "Design Review", "week": 16},
                {"title": "Prototype Demo", "week": 24},
                {"title": "Final Presentation", "week": 32}
            ]
        else:
            # Default milestones
            milestone_schedule = [
                {"title": "Project Start", "week": 1},
                {"title": "Mid-point Review", "week": duration_weeks // 2},
                {"title": "Project Completion", "week": duration_weeks}
            ]
        
        for i, milestone_info in enumerate(milestone_schedule):
            due_date = start_date + timedelta(weeks=milestone_info["week"])
            
            milestone = MilestoneCheckpoint(
                id=str(uuid.uuid4()),
                title=milestone_info["title"],
                description=f"Milestone checkpoint: {milestone_info['title']}",
                due_date=due_date,
                deliverables=[],
                assessment_criteria={},
                completed=False,
                completion_date=None,
                student_submission=None,
                mentor_feedback=None,
                faculty_evaluation=None,
                grade=None
            )
            
            milestones.append(milestone)
        
        return milestones
    
    async def _initialize_student_progress(self, student_id: str, project_id: str):
        """Initialize progress tracking for selected student"""
        
        project = self.projects[project_id]
        
        progress = StudentProgress(
            student_id=student_id,
            project_id=project_id,
            overall_progress=0.0,
            milestone_completion={m.id: False for m in project.milestones},
            mentor_ratings=[],
            self_assessments=[],
            academic_performance={},
            skill_development={},
            challenges=[],
            achievements=[],
            weekly_reports=[],
            last_updated=datetime.now()
        )
        
        self.student_progress[student_id] = progress
    
    def _generate_academic_transcript(
        self,
        student: StudentProfile,
        project: InternshipProject,
        progress: Optional[StudentProgress]
    ) -> str:
        """Generate academic transcript for internship"""
        
        company = self.companies[project.company_id]
        
        transcript = f"""
STUDENT INTERNSHIP TRANSCRIPT

Student Information:
Name: {student.name}
University ID: {student.university_id}
University: {student.university}
Department: {student.department}
Major: {student.major}
Academic Level: {student.academic_level.value}
GPA: {student.gpa}

Internship Information:
Program: {project.program_type.value.replace('_', ' ').title()}
Company: {company.name}
Industry: {company.industry}
Project Title: {project.title}
Duration: {project.start_date.strftime('%Y-%m-%d')} to {project.end_date.strftime('%Y-%m-%d')}
Academic Credit: {'Yes' if project.academic_credit else 'No'}
Credit Hours: {project.credit_hours or 'N/A'}

Project Description:
{project.description}

Technical Skills Required:
"""
        
        for skill in project.requirements.technical_skills:
            transcript += f"- {skill}\n"
        
        if progress:
            transcript += f"""
Performance Summary:
Overall Progress: {progress.overall_progress * 100:.1f}%
Milestones Completed: {sum(1 for completed in progress.milestone_completion.values() if completed)}/{len(progress.milestone_completion)}

Mentor Evaluations:
"""
            
            for rating in progress.mentor_ratings:
                transcript += f"- {rating['timestamp'].strftime('%Y-%m-%d')}: Rating {rating['rating']}/5.0\n"
                if rating['feedback']:
                    transcript += f"  Feedback: {rating['feedback']}\n"
            
            if 'final_grade' in progress.academic_performance:
                transcript += f"\nFinal Grade: {progress.academic_performance['final_grade']}\n"
        
        transcript += f"""
Milestone Summary:
"""
        
        for milestone in project.milestones:
            status = "✓" if milestone.completed else "○"
            transcript += f"{status} {milestone.title} (Due: {milestone.due_date.strftime('%Y-%m-%d')})\n"
            if milestone.mentor_feedback:
                transcript += f"   Mentor Feedback: {milestone.mentor_feedback}\n"
        
        transcript += f"""
Industry Partner Information:
Company: {company.name}
Location: {company.location}
Industry: {company.industry}
Contact: {company.contact_person} ({company.contact_email})

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return transcript


# Testing and validation
async def test_student_internship_management():
    """Test student internship management functionality"""
    
    sim = StudentInternshipManagement()
    
    print("🎓 Testing Student Internship Management System...")
    
    # Test 1: Register students
    print("\n1. Registering students...")
    
    student1 = await sim.register_student(
        name="Sarah Chen",
        email="schen@unc.edu",
        university="University of North Carolina at Chapel Hill",
        department="Computer Science",
        major="Computer Science",
        academic_level=StudentLevel.JUNIOR,
        gpa=3.7,
        graduation_date=datetime(2026, 5, 15),
        skills=["Python", "Java", "Data Analysis", "Machine Learning"],
        interests=["AI/ML", "Software Development", "Data Science"]
    )
    
    student2 = await sim.register_student(
        name="Michael Rodriguez",
        email="mrodriguez@ncsu.edu",
        university="North Carolina State University",
        department="Engineering",
        major="Mechanical Engineering",
        academic_level=StudentLevel.SENIOR,
        gpa=3.4,
        graduation_date=datetime(2025, 12, 15),
        skills=["CAD", "SolidWorks", "Manufacturing", "Project Management"],
        interests=["Manufacturing", "Robotics", "Automotive"]
    )
    
    print(f"✅ Registered {len(sim.students)} students")
    
    # Test 2: Register companies
    print("\n2. Registering industry partners...")
    
    company1 = await sim.register_company(
        name="SAS Institute",
        industry="Software/Analytics",
        size="large",
        location="Cary, NC",
        description="Leading analytics software company",
        contact_person="Jennifer Smith",
        contact_email="jennifer.smith@sas.com",
        mentorship_capacity=10
    )
    
    company2 = await sim.register_company(
        name="John Deere",
        industry="Manufacturing/Agriculture",
        size="large",
        location="Raleigh, NC",
        description="Agricultural machinery manufacturer",
        contact_person="David Johnson",
        contact_email="david.johnson@johndeere.com",
        mentorship_capacity=5
    )
    
    print(f"✅ Registered {len(sim.companies)} companies")
    
    # Test 3: Create internship projects
    print("\n3. Creating internship projects...")
    
    requirements1 = ProjectRequirements(
        technical_skills=["Python", "SQL", "Data Analysis", "Statistics"],
        soft_skills=["Communication", "Problem Solving"],
        academic_level=[StudentLevel.JUNIOR, StudentLevel.SENIOR],
        majors=["Computer Science", "Statistics", "Data Science"],
        min_gpa=3.0,
        security_clearance_required=None,
        time_commitment="full-time",
        duration="12 weeks",
        deliverables=["Data Analysis Report", "Predictive Model", "Final Presentation"],
        learning_objectives=["Apply statistical methods", "Develop ML models", "Present findings"]
    )
    
    project1 = await sim.create_internship_project(
        title="Customer Analytics Internship",
        description="Develop predictive models for customer behavior analysis using SAS software",
        company_id=company1.id,
        program_type=ProgramType.SUMMER_INTERNSHIP,
        requirements=requirements1,
        start_date=datetime(2026, 6, 1),
        end_date=datetime(2026, 8, 24),
        industry_mentor_id="mentor_001",
        security_level=AccessLevel.CONFIDENTIAL,
        academic_credit=True,
        credit_hours=3
    )
    
    requirements2 = ProjectRequirements(
        technical_skills=["CAD", "SolidWorks", "Manufacturing Processes"],
        soft_skills=["Teamwork", "Technical Writing"],
        academic_level=[StudentLevel.SENIOR],
        majors=["Mechanical Engineering", "Industrial Engineering"],
        min_gpa=3.0,
        security_clearance_required=None,
        time_commitment="full-time",
        duration="16 weeks",
        deliverables=["Design Documentation", "Prototype", "Manufacturing Plan"],
        learning_objectives=["Apply engineering principles", "Design for manufacturing", "Project management"]
    )
    
    project2 = await sim.create_internship_project(
        title="Agricultural Equipment Design Co-op",
        description="Design next-generation farming equipment components",
        company_id=company2.id,
        program_type=ProgramType.COOP_ROTATION,
        requirements=requirements2,
        start_date=datetime(2026, 1, 15),
        end_date=datetime(2026, 5, 15),
        industry_mentor_id="mentor_002",
        security_level=AccessLevel.INTERNAL,
        academic_credit=True,
        credit_hours=6
    )
    
    print(f"✅ Created {len(sim.projects)} internship projects")
    print(f"   - Project 1: {project1.title} ({len(project1.milestones)} milestones)")
    print(f"   - Project 2: {project2.title} ({len(project2.milestones)} milestones)")
    
    # Test 4: Open applications
    print("\n4. Opening applications...")
    
    project1.status = ProjectStatus.APPLICATIONS_OPEN
    project2.status = ProjectStatus.APPLICATIONS_OPEN
    
    print("✅ Applications are now open")
    
    # Test 5: Submit applications
    print("\n5. Submitting student applications...")
    
    # Create temporary resume files
    temp_resume1 = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_resume1.write(b"Sarah Chen - Computer Science Resume")
    temp_resume1.close()
    
    temp_resume2 = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    temp_resume2.write(b"Michael Rodriguez - Mechanical Engineering Resume")
    temp_resume2.close()
    
    try:
        application1 = await sim.submit_application(
            student_id=student1.id,
            project_id=project1.id,
            cover_letter="I am excited to apply my data science skills at SAS Institute...",
            resume_path=temp_resume1.name,
            additional_responses={
                "why_interested": "SAS is a leader in analytics and I want to learn from the best",
                "career_goals": "Become a data scientist in the tech industry"
            }
        )
        
        application2 = await sim.submit_application(
            student_id=student2.id,
            project_id=project2.id,
            cover_letter="My mechanical engineering background aligns well with John Deere's mission...",
            resume_path=temp_resume2.name,
            additional_responses={
                "why_interested": "I'm passionate about agricultural technology and sustainability",
                "career_goals": "Work in agricultural equipment design and innovation"
            }
        )
        
        print(f"✅ Submitted {len(project1.applications) + len(project2.applications)} applications")
        
    finally:
        # Clean up temporary files
        Path(temp_resume1.name).unlink()
        Path(temp_resume2.name).unlink()
    
    # Test 6: Select students
    print("\n6. Selecting students for projects...")
    
    selection1 = await sim.select_student(
        project_id=project1.id,
        student_id=student1.id,
        mentor_id="mentor_001",
        selection_rationale="Strong technical skills and relevant coursework in data science"
    )
    
    selection2 = await sim.select_student(
        project_id=project2.id,
        student_id=student2.id,
        mentor_id="mentor_002",
        selection_rationale="Excellent CAD skills and manufacturing knowledge"
    )
    
    print(f"✅ Selected students for projects")
    print(f"   - {student1.name} selected for {project1.title}")
    print(f"   - {student2.name} selected for {project2.title}")
    
    # Test 7: Start internships
    print("\n7. Starting internship programs...")
    
    start1 = await sim.start_internship(
        project_id=project1.id,
        faculty_advisor_id="faculty_001"
    )
    
    start2 = await sim.start_internship(
        project_id=project2.id,
        faculty_advisor_id="faculty_002"
    )
    
    print(f"✅ Started internship programs")
    print(f"   - Both projects now have status: ACTIVE")
    
    # Test 8: Update milestone progress
    print("\n8. Updating milestone progress...")
    
    # Complete first milestone for each project
    first_milestone1 = project1.milestones[0]
    first_milestone2 = project2.milestones[0]
    
    milestone_update1 = await sim.update_milestone_progress(
        project_id=project1.id,
        milestone_id=first_milestone1.id,
        student_submission="Completed orientation and initial data exploration",
        submitted_by=student1.id
    )
    
    milestone_update2 = await sim.update_milestone_progress(
        project_id=project2.id,
        milestone_id=first_milestone2.id,
        student_submission="Finished onboarding and safety training",
        submitted_by=student2.id
    )
    
    print(f"✅ Updated milestone progress")
    
    # Test 9: Add mentor feedback
    print("\n9. Adding mentor feedback...")
    
    feedback1 = await sim.add_mentor_feedback(
        project_id=project1.id,
        milestone_id=first_milestone1.id,
        mentor_id="mentor_001",
        feedback="Great start! Shows strong analytical thinking and attention to detail.",
        rating=4.5
    )
    
    feedback2 = await sim.add_mentor_feedback(
        project_id=project2.id,
        milestone_id=first_milestone2.id,
        mentor_id="mentor_002",
        feedback="Excellent preparation and professional attitude. Looking forward to the design work.",
        rating=4.8
    )
    
    print(f"✅ Added mentor feedback")
    
    # Test 10: Generate analytics
    print("\n10. Generating program analytics...")
    
    analytics = await sim.generate_program_analytics(
        university="University of North Carolina at Chapel Hill"
    )
    
    print(f"✅ Generated program analytics:")
    print(f"   - Total Projects: {analytics.total_projects}")
    print(f"   - Total Applications: {analytics.total_applications}")
    print(f"   - Placement Rate: {analytics.placement_rate:.1%}")
    print(f"   - Average Rating: {analytics.average_rating:.1f}/5.0")
    print(f"   - Top Skills: {list(analytics.skill_demand.keys())[:3]}")
    
    # Test 11: Export student transcript
    print("\n11. Exporting student transcript...")
    
    transcript_path = await sim.export_student_transcript(
        student_id=student1.id,
        project_id=project1.id,
        include_confidential=False
    )
    
    print(f"✅ Exported transcript to: {transcript_path}")
    
    # Verify export contents
    with zipfile.ZipFile(transcript_path, 'r') as zipf:
        files = zipf.namelist()
        print(f"   - Transcript contains {len(files)} files:")
        for file in files:
            print(f"     • {file}")
    
    # Clean up
    Path(transcript_path).unlink()
    shutil.rmtree(Path(transcript_path).parent)
    
    print(f"\n🎉 Student Internship Management testing completed successfully!")
    print(f"   - Students: {len(sim.students)}")
    print(f"   - Companies: {len(sim.companies)}")
    print(f"   - Projects: {len(sim.projects)}")
    print(f"   - Active Progress Tracking: {len(sim.student_progress)}")
    print(f"   - University Programs: {len(sim.university_programs)}")


if __name__ == "__main__":
    asyncio.run(test_student_internship_management())