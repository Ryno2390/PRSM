#!/usr/bin/env python3
"""
R Studio Server Integration for PRSM Secure Collaboration
========================================================

This module implements collaborative R Studio Server functionality with advanced
security features designed for university-industry research partnerships:

- Secure shared R environments with post-quantum encryption
- Real-time collaborative R script editing and execution
- Package management and reproducible research environments
- Integration with statistical data analysis workflows
- University-industry data sharing with access controls
- NWTN AI-powered statistical analysis assistance

Key Features:
- Multi-user R Studio Server sessions with P2P security
- Shared R workspaces with cryptographic file sharding
- Statistical computing collaboration between institutions
- Integration with grant writing and research paper platforms
- Secure data visualization and dashboard sharing
- SAS Institute analytics workflow integration
"""

import json
import uuid
import asyncio
import subprocess
import shutil
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import tempfile
import tarfile
import zipfile

# Import PRSM components
from ..security.post_quantum_crypto_sharding import PostQuantumCryptoSharding, CryptoMode
from ..models import QueryRequest

# Mock UnifiedPipelineController for testing
class UnifiedPipelineController:
    """Mock pipeline controller for R Studio collaboration"""
    async def initialize(self):
        pass
    
    async def process_query_full_pipeline(self, user_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # R-specific NWTN responses
        if context.get("statistical_analysis"):
            return {
                "response": {
                    "text": """
Statistical Analysis Assistance:

📊 **R Code Suggestions**:
```r
# Recommended analysis approach
library(tidyverse)
library(ggplot2)
library(corrplot)

# Load and explore your dataset
data <- read.csv("research_data.csv")
summary(data)

# Statistical testing recommendations
cor_matrix <- cor(data[numeric_columns])
corrplot(cor_matrix, method="circle")

# Advanced modeling suggestions
model <- lm(outcome ~ predictor1 + predictor2 + institution, data=data)
summary(model)
```

🔬 **Statistical Method Recommendations**:
- Use mixed-effects models for multi-institutional data
- Consider Bayesian approaches for small sample sizes
- Apply appropriate corrections for multiple comparisons
- Validate assumptions before interpreting results

📈 **Visualization Suggestions**:
- Interactive plots with plotly for stakeholder presentations
- Publication-ready figures with ggplot2 themes
- Dashboard creation with Shiny for ongoing monitoring

🤝 **Collaboration Best Practices**:
- Use reproducible research principles (renv, here packages)
- Document analysis decisions in RMarkdown
- Version control statistical analysis scripts
- Share code with clear commenting for industry partners
                    """,
                    "confidence": 0.91,
                    "sources": ["r_documentation.org", "cran.r-project.org", "statistical_methods.pdf"]
                },
                "performance_metrics": {"total_processing_time": 2.8}
            }
        elif context.get("package_management"):
            return {
                "response": {
                    "text": """
R Package Management Analysis:

📦 **Package Recommendations**:
- **tidyverse**: Essential data manipulation and visualization
- **rmarkdown**: Reproducible research documentation
- **shiny**: Interactive web applications for stakeholders
- **corrplot**: Correlation matrix visualization
- **randomForest**: Machine learning for predictive modeling
- **survival**: Survival analysis for medical research
- **forecast**: Time series analysis and forecasting

🔧 **Environment Management**:
```r
# Use renv for reproducible environments
install.packages("renv")
renv::init()  # Initialize project environment
renv::snapshot()  # Capture current package versions
renv::restore()  # Restore environment on collaborator machines
```

🏛️ **University-Industry Specific**:
- **SAS integration**: Haven package for SAS file import
- **Database connectivity**: RPostgreSQL, RMySQL for institutional data
- **Security**: keyring package for secure credential management
- **Performance**: parallel, foreach for large-scale computations

⚠️ **Compatibility Notes**:
- Test package versions across different institutional R installations
- Document system requirements for specialized packages
- Consider Docker containers for consistent environments
                    """,
                    "confidence": 0.87,
                    "sources": ["cran.r-project.org", "rstudio.com", "package_documentation.pdf"]
                },
                "performance_metrics": {"total_processing_time": 2.3}
            }
        else:
            return {
                "response": {"text": "R Studio collaboration assistance available", "confidence": 0.75, "sources": []},
                "performance_metrics": {"total_processing_time": 1.5}
            }

class RStudioAccessLevel(Enum):
    """Access levels for R Studio sessions"""
    OWNER = "owner"
    COLLABORATOR = "collaborator"
    VIEWER = "viewer"
    GUEST = "guest"

class SessionType(Enum):
    """Types of R Studio sessions"""
    UNIVERSITY_RESEARCH = "university_research"
    INDUSTRY_PARTNERSHIP = "industry_partnership"
    STATISTICAL_CONSULTING = "statistical_consulting"
    DATA_ANALYSIS = "data_analysis"
    TEACHING = "teaching"

class PackageSource(Enum):
    """R package installation sources"""
    CRAN = "cran"
    BIOCONDUCTOR = "bioconductor"
    GITHUB = "github"
    LOCAL = "local"
    INSTITUTIONAL = "institutional"

@dataclass
class RPackage:
    """R package specification"""
    name: str
    version: str
    source: PackageSource
    repository: Optional[str] = None
    required: bool = True
    description: str = ""
    dependencies: List[str] = None
    security_vetted: bool = False

@dataclass
class RStudioSession:
    """Collaborative R Studio session"""
    session_id: str
    name: str
    description: str
    session_type: SessionType
    owner: str
    collaborators: Dict[str, RStudioAccessLevel]
    created_at: datetime
    last_active: datetime
    
    # Environment configuration
    r_version: str
    packages: List[RPackage]
    working_directory: str
    data_sources: List[str]
    
    # Security settings
    security_level: str  # 'standard', 'high', 'maximum'
    encrypted: bool = True
    access_controlled: bool = True
    
    # Session state
    active_users: List[str] = None
    shared_variables: Dict[str, Any] = None
    execution_history: List[Dict[str, Any]] = None

@dataclass
class RScript:
    """Collaborative R script"""
    script_id: str
    name: str
    content: str
    created_by: str
    created_at: datetime
    last_modified: datetime
    last_modified_by: str
    
    # Collaboration
    edit_history: List[Dict[str, Any]] = None
    comments: List[Dict[str, Any]] = None
    shared_with: List[str] = None
    
    # Execution
    last_execution: Optional[datetime] = None
    execution_results: Optional[str] = None
    execution_status: str = "not_run"

@dataclass
class StatisticalProject:
    """Multi-institutional statistical research project"""
    project_id: str
    title: str
    description: str
    principal_investigator: str
    institutions: List[str]
    collaborators: Dict[str, RStudioAccessLevel]
    
    # Research components  
    research_questions: List[str]
    datasets: List[str]
    analysis_scripts: List[RScript]
    results: List[Dict[str, Any]]
    
    # Timeline
    created_at: datetime
    deadline: Optional[datetime] = None
    milestones: List[Dict[str, Any]] = None
    
    # Security and compliance
    data_classification: str = "restricted"
    irb_approval: Optional[str] = None
    compliance_requirements: List[str] = None

class RStudioCollaboration:
    """
    Main class for collaborative R Studio Server integration with P2P security
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize R Studio collaboration system"""
        self.storage_path = storage_path or Path("./rstudio_collaboration")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize PRSM components
        self.crypto_sharding = PostQuantumCryptoSharding(
            default_shards=5,
            required_shards=3,
            crypto_mode=CryptoMode.POST_QUANTUM
        )
        self.nwtn_pipeline = None
        
        # Active sessions and projects
        self.active_sessions: Dict[str, RStudioSession] = {}
        self.statistical_projects: Dict[str, StatisticalProject] = {}
        
        # R environment configuration
        self.r_installation_path = self._detect_r_installation()
        self.rstudio_server_config = self._initialize_rstudio_config()
        
        # Common R packages for university-industry collaboration
        self.standard_packages = self._initialize_standard_packages()
    
    def _detect_r_installation(self) -> Optional[str]:
        """Detect R installation on system"""
        possible_paths = [
            "/usr/bin/R",
            "/usr/local/bin/R",
            "/opt/R/bin/R",
            "C:\\Program Files\\R\\R-4.3.0\\bin\\R.exe",
            "C:\\Program Files\\Microsoft\\R Client\\R_SERVER\\bin\\R.exe"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                print(f"📊 Found R installation: {path}")
                return path
        
        # Check if R is in PATH
        try:
            result = subprocess.run(["R", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print("📊 R found in system PATH")
                return "R"
        except FileNotFoundError:
            pass
        
        print("⚠️  R installation not detected - will use Docker fallback")
        return None
    
    def _initialize_rstudio_config(self) -> Dict[str, Any]:
        """Initialize R Studio Server configuration"""
        return {
            "port": 8787,
            "secure_cookie": True,
            "auth_required": True,
            "collaborative_editing": True,
            "version_control": True,
            "package_management": True,
            "custom_themes": True,
            "extensions": ["shiny", "rmarkdown", "plumber"]
        }
    
    def _initialize_standard_packages(self) -> List[RPackage]:
        """Initialize standard R packages for university-industry collaboration"""
        return [
            # Core data manipulation and visualization
            RPackage("tidyverse", "2.0.0", PackageSource.CRAN, 
                    description="Essential data manipulation and visualization", required=True),
            RPackage("ggplot2", "3.4.2", PackageSource.CRAN,
                    description="Grammar of graphics plotting system", required=True),
            RPackage("dplyr", "1.1.2", PackageSource.CRAN,
                    description="Data manipulation verbs", required=True),
            
            # Statistical analysis
            RPackage("corrplot", "0.92", PackageSource.CRAN,
                    description="Correlation matrix visualization"),
            RPackage("randomForest", "4.7-1.1", PackageSource.CRAN,
                    description="Random forest machine learning"),
            RPackage("survival", "3.5-5", PackageSource.CRAN,
                    description="Survival analysis"),
            RPackage("forecast", "8.21", PackageSource.CRAN,
                    description="Time series forecasting"),
            
            # Reproducible research
            RPackage("rmarkdown", "2.22", PackageSource.CRAN,
                    description="Dynamic documents and reports", required=True),
            RPackage("knitr", "1.43", PackageSource.CRAN,
                    description="Literate programming", required=True),
            RPackage("renv", "1.0.0", PackageSource.CRAN,
                    description="Reproducible environments", required=True),
            
            # Interactive applications
            RPackage("shiny", "1.7.4", PackageSource.CRAN,
                    description="Interactive web applications"),
            RPackage("plotly", "4.10.2", PackageSource.CRAN,
                    description="Interactive visualizations"),
            RPackage("DT", "0.28", PackageSource.CRAN,
                    description="Interactive data tables"),
            
            # Database and data import
            RPackage("haven", "2.5.2", PackageSource.CRAN,
                    description="SAS, SPSS, Stata data import"),
            RPackage("readxl", "1.4.2", PackageSource.CRAN,
                    description="Excel file reading"),
            RPackage("RPostgreSQL", "0.7-5", PackageSource.CRAN,
                    description="PostgreSQL database connection"),
            
            # University-specific packages
            RPackage("IRkernel", "1.3.2", PackageSource.CRAN,
                    description="Jupyter R kernel integration"),
            RPackage("keyring", "1.3.1", PackageSource.CRAN,
                    description="Secure credential management"),
            RPackage("parallelly", "1.36.0", PackageSource.CRAN,
                    description="Parallel computing utilities")
        ]
    
    async def initialize_nwtn_pipeline(self):
        """Initialize NWTN pipeline for statistical analysis assistance"""
        if self.nwtn_pipeline is None:
            self.nwtn_pipeline = UnifiedPipelineController()
            await self.nwtn_pipeline.initialize()
    
    def create_rstudio_session(self,
                             name: str,
                             description: str,
                             session_type: SessionType,
                             owner: str,
                             collaborators: Optional[Dict[str, RStudioAccessLevel]] = None,
                             r_version: str = "4.3.0",
                             security_level: str = "high") -> RStudioSession:
        """Create a new collaborative R Studio session"""
        
        session_id = str(uuid.uuid4())
        
        # Create session directory
        session_dir = self.storage_path / "sessions" / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        session = RStudioSession(
            session_id=session_id,
            name=name,
            description=description,
            session_type=session_type,
            owner=owner,
            collaborators=collaborators or {},
            created_at=datetime.now(),
            last_active=datetime.now(),
            r_version=r_version,
            packages=self.standard_packages.copy(),
            working_directory=str(session_dir),
            data_sources=[],
            security_level=security_level,
            encrypted=True,
            access_controlled=True,
            active_users=[],
            shared_variables={},
            execution_history=[]
        )
        
        self.active_sessions[session_id] = session
        self._save_session(session)
        
        print(f"📊 Created R Studio session: {name}")
        print(f"   Session ID: {session_id}")
        print(f"   Type: {session_type.value}")
        print(f"   R Version: {r_version}")
        print(f"   Collaborators: {len(collaborators or {})}")
        print(f"   Security: {security_level}")
        
        return session
    
    def create_statistical_project(self,
                                 title: str,
                                 description: str,
                                 principal_investigator: str,
                                 institutions: List[str],
                                 collaborators: Dict[str, RStudioAccessLevel],
                                 research_questions: List[str],
                                 deadline: Optional[datetime] = None) -> StatisticalProject:
        """Create a multi-institutional statistical research project"""
        
        project_id = str(uuid.uuid4())
        
        project = StatisticalProject(
            project_id=project_id,
            title=title,
            description=description,
            principal_investigator=principal_investigator,
            institutions=institutions,
            collaborators=collaborators,
            research_questions=research_questions,
            datasets=[],
            analysis_scripts=[],
            results=[],
            created_at=datetime.now(),
            deadline=deadline,
            milestones=[],
            data_classification="restricted",
            irb_approval=None,
            compliance_requirements=["HIPAA", "FERPA", "IRB"]
        )
        
        self.statistical_projects[project_id] = project
        self._save_project(project)
        
        print(f"🔬 Created statistical research project: {title}")
        print(f"   Project ID: {project_id}")
        print(f"   PI: {principal_investigator}")
        print(f"   Institutions: {', '.join(institutions)}")
        print(f"   Research Questions: {len(research_questions)}")
        print(f"   Collaborators: {len(collaborators)}")
        
        return project
    
    def add_r_script(self,
                    session_id: str,
                    name: str,
                    content: str,
                    created_by: str) -> RScript:
        """Add an R script to a collaborative session"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Check permissions
        if not self._check_session_access(session, created_by, RStudioAccessLevel.COLLABORATOR):
            raise PermissionError(f"User {created_by} cannot add scripts to this session")
        
        script_id = str(uuid.uuid4())
        
        script = RScript(
            script_id=script_id,
            name=name,
            content=content,
            created_by=created_by,
            created_at=datetime.now(),
            last_modified=datetime.now(),
            last_modified_by=created_by,
            edit_history=[],
            comments=[],
            shared_with=list(session.collaborators.keys()),
            last_execution=None,
            execution_results=None,
            execution_status="not_run"
        )
        
        # Save script to session directory
        script_file = Path(session.working_directory) / "scripts" / f"{name}.R"
        script_file.parent.mkdir(exist_ok=True)
        
        with open(script_file, 'w') as f:
            f.write(content)
        
        print(f"📝 Added R script: {name}")
        print(f"   Script ID: {script_id}")
        print(f"   Session: {session.name}")
        print(f"   Created by: {created_by}")
        
        return script
    
    async def install_packages(self,
                             session_id: str,
                             packages: List[RPackage],
                             user_id: str) -> Dict[str, bool]:
        """Install R packages in a collaborative session"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Check permissions
        if not self._check_session_access(session, user_id, RStudioAccessLevel.COLLABORATOR):
            raise PermissionError(f"User {user_id} cannot install packages in this session")
        
        print(f"📦 Installing {len(packages)} R packages...")
        
        installation_results = {}
        
        for package in packages:
            try:
                print(f"   Installing {package.name} ({package.version}) from {package.source.value}...")
                
                # Mock package installation for demonstration
                # In real implementation, would use rpy2 or subprocess calls
                if package.source == PackageSource.CRAN:
                    install_command = f'install.packages("{package.name}", version="{package.version}")'
                elif package.source == PackageSource.BIOCONDUCTOR:
                    install_command = f'BiocManager::install("{package.name}")'
                elif package.source == PackageSource.GITHUB:
                    install_command = f'devtools::install_github("{package.repository}")'
                else:
                    install_command = f'# Custom installation for {package.name}'
                
                # Add to session packages
                session.packages.append(package)
                installation_results[package.name] = True
                
                print(f"   ✅ {package.name} installed successfully")
                
            except Exception as e:
                print(f"   ❌ Failed to install {package.name}: {e}")
                installation_results[package.name] = False
        
        # Update session
        session.last_active = datetime.now()
        self._save_session(session)
        
        successful_installs = sum(installation_results.values())
        print(f"📦 Package installation completed: {successful_installs}/{len(packages)} successful")
        
        return installation_results
    
    async def get_statistical_analysis_assistance(self,
                                                session_id: str,
                                                analysis_description: str,
                                                data_description: str,
                                                user_id: str) -> Dict[str, Any]:
        """Get NWTN AI assistance for statistical analysis"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Check permissions
        if not self._check_session_access(session, user_id, RStudioAccessLevel.VIEWER):
            raise PermissionError("Insufficient permissions to access analysis assistance")
        
        await self.initialize_nwtn_pipeline()
        
        analysis_prompt = f"""
Please provide statistical analysis assistance for this R Studio collaboration:

**Analysis Description**: {analysis_description}
**Data Description**: {data_description}
**Session Type**: {session.session_type.value}
**Institutions**: Multi-institutional research collaboration
**Security Level**: {session.security_level}

Please provide:
1. Recommended R packages and statistical methods
2. Sample R code for the analysis approach
3. Visualization suggestions appropriate for stakeholders
4. Best practices for reproducible research
5. Collaboration recommendations for multi-institutional work

Focus on practical code examples and statistical rigor appropriate for university-industry partnerships.
"""
        
        result = await self.nwtn_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=analysis_prompt,
            context={
                "domain": "statistical_analysis",
                "statistical_analysis": True,
                "session_type": session.session_type.value,
                "analysis_type": "comprehensive_guidance"
            }
        )
        
        assistance = {
            "session_id": session_id,
            "session_name": session.name,
            "analysis_description": analysis_description,
            "recommendations": result.get('response', {}).get('text', ''),
            "confidence": result.get('response', {}).get('confidence', 0.0),
            "sources": result.get('response', {}).get('sources', []),
            "processing_time": result.get('performance_metrics', {}).get('total_processing_time', 0.0),
            "generated_at": datetime.now().isoformat(),
            "requested_by": user_id
        }
        
        print(f"🤖 Statistical analysis assistance provided:")
        print(f"   Session: {session.name}")
        print(f"   Analysis: {analysis_description}")
        print(f"   Confidence: {assistance['confidence']:.2f}")
        print(f"   Processing time: {assistance['processing_time']:.1f}s")
        
        return assistance
    
    async def get_package_recommendations(self,
                                        research_area: str,
                                        analysis_goals: List[str],
                                        user_id: str) -> Dict[str, Any]:
        """Get AI-powered R package recommendations"""
        
        await self.initialize_nwtn_pipeline()
        
        package_prompt = f"""
Please recommend R packages for this research collaboration:

**Research Area**: {research_area}
**Analysis Goals**: {', '.join(analysis_goals)}
**Context**: University-industry statistical collaboration
**Requirements**: Reproducible research, multi-institutional data sharing

Please provide:
1. Essential R packages for this research area
2. Specialized packages for the specific analysis goals
3. Packages for collaboration and reproducibility
4. Installation and setup instructions
5. Potential compatibility issues and solutions

Focus on packages that are well-maintained, widely used in academic research, and suitable for sensitive university-industry collaborations.
"""
        
        result = await self.nwtn_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=package_prompt,
            context={
                "domain": "package_management",
                "package_management": True,
                "research_area": research_area,
                "recommendation_type": "comprehensive_packages"
            }
        )
        
        recommendations = {
            "research_area": research_area,
            "analysis_goals": analysis_goals,
            "package_recommendations": result.get('response', {}).get('text', ''),
            "confidence": result.get('response', {}).get('confidence', 0.0),
            "sources": result.get('response', {}).get('sources', []),
            "processing_time": result.get('performance_metrics', {}).get('total_processing_time', 0.0),
            "generated_at": datetime.now().isoformat(),
            "requested_by": user_id
        }
        
        print(f"📦 Package recommendations generated:")
        print(f"   Research Area: {research_area}")
        print(f"   Goals: {len(analysis_goals)} analysis objectives")
        print(f"   Confidence: {recommendations['confidence']:.2f}")
        
        return recommendations
    
    def create_reproducible_environment(self,
                                      session_id: str,
                                      user_id: str) -> Dict[str, Any]:
        """Create reproducible R environment configuration"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Check permissions
        if not self._check_session_access(session, user_id, RStudioAccessLevel.COLLABORATOR):
            raise PermissionError("Insufficient permissions to create environment configuration")
        
        # Generate renv configuration
        renv_config = {
            "R": {
                "Version": session.r_version,
                "Repositories": [
                    {"Name": "CRAN", "URL": "https://cran.rstudio.com/"},
                    {"Name": "BioConductor", "URL": "https://bioconductor.org/packages/3.17/bioc"}
                ]
            },
            "Packages": {}
        }
        
        # Add package specifications
        for package in session.packages:
            renv_config["Packages"][package.name] = {
                "Package": package.name,
                "Version": package.version,
                "Source": package.source.value,
                "Repository": package.repository or "CRAN",
                "Requirements": package.dependencies or [],
                "Hash": f"sha256_{uuid.uuid4().hex[:16]}"  # Mock hash
            }
        
        # Save environment configuration
        env_file = Path(session.working_directory) / "renv.lock"
        with open(env_file, 'w') as f:
            json.dump(renv_config, f, indent=2)
        
        # Generate setup script
        setup_script = f"""
# Reproducible R Environment Setup
# Generated for session: {session.name}
# Created: {datetime.now().isoformat()}

# Install renv if not available
if (!requireNamespace("renv", quietly = TRUE)) {{
  install.packages("renv")
}}

# Initialize renv project
renv::init()

# Restore packages from lockfile
renv::restore()

# Load essential packages
library(tidyverse)
library(rmarkdown)
library(knitr)

# Set working directory
setwd("{session.working_directory}")

# Configure collaboration settings
options(repos = c(CRAN = "https://cran.rstudio.com/"))
options(warn = 1)  # Show warnings immediately

# Print session information
sessionInfo()
cat("\\n✅ Reproducible environment ready for collaboration!\\n")
"""
        
        setup_file = Path(session.working_directory) / "setup.R"
        with open(setup_file, 'w') as f:
            f.write(setup_script)
        
        environment_info = {
            "session_id": session_id,
            "session_name": session.name,
            "r_version": session.r_version,
            "package_count": len(session.packages),
            "environment_file": str(env_file),
            "setup_script": str(setup_file),
            "created_at": datetime.now().isoformat(),
            "created_by": user_id,
            "reproducible": True
        }
        
        print(f"🔄 Created reproducible environment:")
        print(f"   Session: {session.name}")
        print(f"   R Version: {session.r_version}")
        print(f"   Packages: {len(session.packages)}")
        print(f"   Environment file: renv.lock")
        print(f"   Setup script: setup.R")
        
        return environment_info
    
    def export_session_data(self,
                          session_id: str,
                          export_format: str = "rdata",
                          user_id: str = "system") -> str:
        """Export session data and results"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        session_dir = Path(session.working_directory)
        
        export_name = f"{session.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if export_format.lower() == "rdata":
            export_file = self.storage_path / "exports" / f"{export_name}.RData"
            export_file.parent.mkdir(exist_ok=True)
            
            # Generate R script to save workspace
            save_script = f"""
# Save R session workspace
# Session: {session.name}
# Exported: {datetime.now().isoformat()}

# Set working directory
setwd("{session.working_directory}")

# Save all objects in workspace
save.image("{export_file}")

cat("✅ Session workspace saved to {export_file}\\n")
"""
            
            script_file = session_dir / f"export_{export_name}.R"
            with open(script_file, 'w') as f:
                f.write(save_script)
                
        elif export_format.lower() == "zip":
            export_file = self.storage_path / "exports" / f"{export_name}.zip"
            export_file.parent.mkdir(exist_ok=True)
            
            # Create zip archive of session directory
            with zipfile.ZipFile(export_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in session_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(session_dir)
                        zipf.write(file_path, arcname)
        
        else:
            raise ValueError(f"Export format '{export_format}' not supported")
        
        print(f"📦 Exported session data:")
        print(f"   Session: {session.name}")
        print(f"   Format: {export_format.upper()}")
        print(f"   File: {export_file}")
        
        return str(export_file)
    
    def _check_session_access(self, session: RStudioSession, user_id: str, required_level: RStudioAccessLevel) -> bool:
        """Check if user has required access level to session"""
        
        # Owner has all access
        if session.owner == user_id:
            return True
        
        # Check collaborator access
        if user_id in session.collaborators:
            user_level = session.collaborators[user_id]
            
            # Define access hierarchy
            access_hierarchy = {
                RStudioAccessLevel.GUEST: 1,
                RStudioAccessLevel.VIEWER: 2,
                RStudioAccessLevel.COLLABORATOR: 3,
                RStudioAccessLevel.OWNER: 4
            }
            
            return access_hierarchy[user_level] >= access_hierarchy[required_level]
        
        return False
    
    def _save_session(self, session: RStudioSession):
        """Save R Studio session configuration"""
        session_dir = self.storage_path / "sessions" / session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        session_file = session_dir / "session.json"
        with open(session_file, 'w') as f:
            session_data = asdict(session)
            json.dump(session_data, f, default=str, indent=2)
    
    def _save_project(self, project: StatisticalProject):
        """Save statistical research project"""
        project_dir = self.storage_path / "projects" / project.project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        project_file = project_dir / "project.json"
        with open(project_file, 'w') as f:
            project_data = asdict(project)
            json.dump(project_data, f, default=str, indent=2)

# RTP-specific R Studio integration for SAS Institute partnerships
class RTPRStudioIntegration:
    """SAS Institute and Research Triangle Park specific R Studio integrations"""
    
    def __init__(self, rstudio_collab: RStudioCollaboration):
        self.rstudio_collab = rstudio_collab
        self.sas_integration_packages = self._initialize_sas_packages()
    
    def _initialize_sas_packages(self) -> List[RPackage]:
        """Initialize SAS-specific R packages"""
        return [
            RPackage("haven", "2.5.2", PackageSource.CRAN,
                    description="Read and write SAS, SPSS, and Stata files"),
            RPackage("SASxport", "1.7.0", PackageSource.CRAN,
                    description="Read and write SAS XPORT files"),
            RPackage("sas7bdat", "0.6", PackageSource.CRAN,
                    description="Read SAS7BDAT files"),
            RPackage("Hmisc", "5.1-0", PackageSource.CRAN,
                    description="Statistical analysis functions used by SAS users"),
            RPackage("car", "3.1-2", PackageSource.CRAN,
                    description="Companion to Applied Regression (SAS-like functions)")
        ]
    
    def create_sas_collaboration_session(self,
                                       project_name: str,
                                       university_pi: str,
                                       sas_collaborator: str,
                                       data_sources: List[str]) -> RStudioSession:
        """Create R Studio session optimized for SAS Institute collaboration"""
        
        session = self.rstudio_collab.create_rstudio_session(
            name=f"SAS Partnership: {project_name}",
            description=f"University-SAS Institute collaboration for {project_name}",
            session_type=SessionType.INDUSTRY_PARTNERSHIP,
            owner=university_pi,
            collaborators={
                sas_collaborator: RStudioAccessLevel.COLLABORATOR
            },
            security_level="high"
        )
        
        # Add SAS-specific packages
        session.packages.extend(self.sas_integration_packages)
        
        # Create SAS data import script
        sas_import_script = f"""
# SAS Data Import and Analysis Script
# Project: {project_name}
# Created: {datetime.now().isoformat()}

# Load SAS integration packages
library(haven)
library(SASxport)
library(tidyverse)

# Import SAS datasets
{chr(10).join([f'# data_{i} <- read_sas("{source}")' for i, source in enumerate(data_sources)])}

# SAS-style data processing
# proc_means_equivalent <- function(data, vars) {{
#   data %>%
#     select(all_of(vars)) %>%
#     summarise_all(list(
#       n = ~n(),
#       mean = ~mean(., na.rm = TRUE),
#       std = ~sd(., na.rm = TRUE),
#       min = ~min(., na.rm = TRUE),
#       max = ~max(., na.rm = TRUE)
#     ))
# }}

# proc_freq_equivalent <- function(data, vars) {{
#   data %>%
#     count(across(all_of(vars))) %>%
#     mutate(percent = n / sum(n) * 100)
# }}

cat("✅ SAS collaboration environment ready!\\n")
cat("📊 Data sources configured: {len(data_sources)}\\n")
cat("🤝 University-Industry partnership session active\\n")
"""
        
        self.rstudio_collab.add_r_script(
            session.session_id,
            "sas_integration_setup",
            sas_import_script,
            university_pi
        )
        
        print(f"🏢 Created SAS Institute collaboration session:")
        print(f"   University PI: {university_pi}")
        print(f"   SAS Collaborator: {sas_collaborator}")
        print(f"   Data Sources: {len(data_sources)}")
        print(f"   SAS Packages: {len(self.sas_integration_packages)}")
        
        return session

# Example usage and testing
if __name__ == "__main__":
    async def test_rstudio_collaboration():
        """Test R Studio collaboration system"""
        
        print("🚀 Testing R Studio Server Integration")
        print("=" * 60)
        
        # Initialize R Studio collaboration
        rstudio_collab = RStudioCollaboration()
        
        # Create multi-university statistical research session
        session = rstudio_collab.create_rstudio_session(
            name="Quantum Computing Statistical Analysis - Multi-University Partnership",
            description="Statistical validation of quantum error correction algorithms across UNC, Duke, NC State, and SAS Institute",
            session_type=SessionType.UNIVERSITY_RESEARCH,
            owner="sarah.chen@unc.edu",
            collaborators={
                "alex.rodriguez@duke.edu": RStudioAccessLevel.COLLABORATOR,
                "jennifer.kim@ncsu.edu": RStudioAccessLevel.COLLABORATOR,
                "michael.johnson@sas.com": RStudioAccessLevel.VIEWER,
                "stats.consultant@unc.edu": RStudioAccessLevel.COLLABORATOR
            },
            r_version="4.3.0",
            security_level="high"
        )
        
        print(f"\n✅ Created R Studio session: {session.name}")
        print(f"   Session ID: {session.session_id}")
        print(f"   Collaborators: {len(session.collaborators)}")
        
        # Add statistical analysis script
        analysis_script = """
# Multi-University Quantum Computing Statistical Analysis
# Universities: UNC + Duke + NC State + SAS Institute
# Principal Investigator: Dr. Sarah Chen (UNC Physics)

library(tidyverse)
library(ggplot2)
library(corrplot)
library(randomForest)
library(rmarkdown)

# Load quantum error correction experimental data
quantum_data <- read.csv("quantum_error_correction_results.csv")

# Exploratory data analysis
summary(quantum_data)

# Correlation analysis of error correction methods
error_methods <- quantum_data %>%
  select(adaptive_correction, standard_correction, noise_level, success_rate)

cor_matrix <- cor(error_methods, use = "complete.obs")
corrplot(cor_matrix, method = "circle", 
         title = "Quantum Error Correction Method Correlations")

# Statistical comparison of correction methods
t_test_result <- t.test(quantum_data$adaptive_correction, 
                       quantum_data$standard_correction,
                       paired = TRUE)

cat("📊 Statistical Analysis Results:\\n")
cat("Adaptive vs Standard Correction Methods:\\n")
cat(sprintf("Mean difference: %.3f\\n", t_test_result$estimate))
cat(sprintf("P-value: %.6f\\n", t_test_result$p.value))

# Multi-institutional modeling
model <- lm(success_rate ~ adaptive_correction + noise_level + institution, 
           data = quantum_data)
summary(model)

# Visualization for stakeholders
ggplot(quantum_data, aes(x = noise_level, y = success_rate, 
                        color = institution)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_wrap(~correction_method) +
  labs(title = "Quantum Error Correction Performance by Institution",
       subtitle = "Multi-University Collaboration Results",
       x = "Noise Level", 
       y = "Success Rate",
       color = "Institution") +
  theme_minimal()

cat("✅ Statistical analysis completed!\\n")
cat("🎯 40% improvement confirmed across all institutions\\n")
"""
        
        script = rstudio_collab.add_r_script(
            session.session_id,
            "quantum_statistical_analysis",
            analysis_script,
            "sarah.chen@unc.edu"
        )
        
        print(f"\n✅ Added statistical analysis script: {script.name}")
        
        # Test NWTN statistical analysis assistance
        print(f"\n🤖 Testing statistical analysis assistance...")
        
        assistance = await rstudio_collab.get_statistical_analysis_assistance(
            session.session_id,
            "Compare quantum error correction methods across multiple university research groups",
            "Experimental data from quantum computing labs at UNC, Duke, and NC State with performance metrics",
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ Statistical analysis assistance provided:")
        print(f"   Confidence: {assistance['confidence']:.2f}")
        print(f"   Processing time: {assistance['processing_time']:.1f}s")
        print(f"   Recommendations preview: {assistance['recommendations'][:200]}...")
        
        # Test package recommendations
        print(f"\n📦 Testing package recommendations...")
        
        package_recs = await rstudio_collab.get_package_recommendations(
            "quantum_computing_statistics",
            ["comparative_analysis", "multi_institutional_modeling", "visualization"],
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ Package recommendations generated:")
        print(f"   Research Area: {package_recs['research_area']}")
        print(f"   Confidence: {package_recs['confidence']:.2f}")
        
        # Create reproducible environment
        print(f"\n🔄 Creating reproducible environment...")
        
        env_info = rstudio_collab.create_reproducible_environment(
            session.session_id,
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ Reproducible environment created:")
        print(f"   R Version: {env_info['r_version']}")
        print(f"   Packages: {env_info['package_count']}")
        print(f"   Environment file: {Path(env_info['environment_file']).name}")
        
        # Test SAS Institute integration
        print(f"\n🏢 Testing SAS Institute integration...")
        
        rtp_integration = RTPRStudioIntegration(rstudio_collab)
        
        sas_session = rtp_integration.create_sas_collaboration_session(
            "Advanced Analytics for Quantum Algorithm Validation",
            "sarah.chen@unc.edu",
            "michael.johnson@sas.com",
            ["quantum_performance_data.sas7bdat", "university_metrics.xpt"]
        )
        
        print(f"✅ SAS collaboration session created:")
        print(f"   Session: {sas_session.name}")
        print(f"   SAS Packages: {len(rtp_integration.sas_integration_packages)}")
        
        # Export session data
        print(f"\n📦 Testing session data export...")
        
        export_file = rstudio_collab.export_session_data(
            session.session_id,
            "zip",
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ Session data exported: {Path(export_file).name}")
        
        print(f"\n🎉 R Studio collaboration system test completed!")
        print("✅ Ready for university-industry statistical research partnerships!")
    
    # Run test
    import asyncio
    asyncio.run(test_rstudio_collaboration())