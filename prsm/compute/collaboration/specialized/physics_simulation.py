"""
Physics Simulation Environments for Collaborative Research

Provides secure P2P collaboration for computational physics including molecular dynamics,
quantum simulations, finite element analysis, and high-performance computing workflows.
Features include shared simulation environments, result analysis, and university-industry
physics research partnerships with sensitive computational data protection.

Key Features:
- Post-quantum cryptographic security for sensitive simulation data and results
- Collaborative molecular dynamics simulations with shared parameter optimization
- Quantum mechanics calculations with distributed computing coordination
- Finite element analysis for engineering physics applications
- High-performance computing workflow management and job scheduling
- Multi-institutional physics research coordination and data sharing
- NWTN AI-powered simulation analysis and parameter optimization
- Integration with popular physics software (LAMMPS, GROMACS, VASP, etc.)
- Export capabilities for publications and computational reproducibility
"""

import asyncio
import hashlib
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
from pathlib import Path
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET

from ..security.post_quantum_crypto_sharding import PostQuantumCryptoSharding

# Mock NWTN for testing
class MockNWTN:
    async def reason(self, prompt, context):
        return {
            "reasoning": [
                "Simulation parameters appear well-chosen for the physical system under study",
                "Computational approach is appropriate for the scale and complexity of the problem",
                "Results show good convergence behavior and physical consistency"
            ],
            "recommendations": [
                "Consider increasing sampling frequency for better statistical accuracy",
                "Explore alternative algorithms for improved computational efficiency",
                "Validate results against experimental data or analytical solutions where available"
            ]
        }


class SimulationType(Enum):
    """Physics simulation types"""
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    MONTE_CARLO = "monte_carlo"
    QUANTUM_MECHANICS = "quantum_mechanics"
    DENSITY_FUNCTIONAL_THEORY = "density_functional_theory"
    FINITE_ELEMENT_ANALYSIS = "finite_element_analysis"
    FLUID_DYNAMICS = "fluid_dynamics"
    PARTICLE_PHYSICS = "particle_physics"
    ASTROPHYSICS = "astrophysics"
    CONDENSED_MATTER = "condensed_matter"
    PLASMA_PHYSICS = "plasma_physics"


class ComputationalMethod(Enum):
    """Computational physics methods"""
    CLASSICAL_MD = "classical_md"
    AB_INITIO_MD = "ab_initio_md"
    QUANTUM_MONTE_CARLO = "quantum_monte_carlo"
    DENSITY_FUNCTIONAL = "density_functional"
    HARTREE_FOCK = "hartree_fock"
    COUPLED_CLUSTER = "coupled_cluster"
    FINITE_DIFFERENCE = "finite_difference"
    FINITE_ELEMENT = "finite_element"
    SPECTRAL_METHODS = "spectral_methods"
    LATTICE_BOLTZMANN = "lattice_boltzmann"


class CollaborationRole(Enum):
    """Physics collaboration roles"""
    PRINCIPAL_INVESTIGATOR = "principal_investigator"
    THEORETICAL_PHYSICIST = "theoretical_physicist"
    COMPUTATIONAL_PHYSICIST = "computational_physicist"
    EXPERIMENTAL_PHYSICIST = "experimental_physicist"
    POSTDOC_RESEARCHER = "postdoc_researcher"
    GRADUATE_STUDENT = "graduate_student"
    HPC_SPECIALIST = "hpc_specialist"
    INDUSTRY_RESEARCHER = "industry_researcher"
    SOFTWARE_DEVELOPER = "software_developer"


@dataclass
class PhysicsSystem:
    """Physical system definition"""
    id: str
    name: str
    system_type: str  # "molecular", "solid_state", "fluid", "plasma", etc.
    description: str
    dimensions: int  # 1D, 2D, 3D
    particle_count: Optional[int]
    system_size: Tuple[float, float, float]  # Physical dimensions
    boundary_conditions: str  # "periodic", "fixed", "open"
    temperature: Optional[float]  # Kelvin
    pressure: Optional[float]   # atm or Pa
    composition: Dict[str, Any]  # Chemical/atomic composition
    initial_conditions: Dict[str, Any]
    force_field: Optional[str]  # For MD simulations
    created_by: str
    created_at: datetime
    validated: bool


@dataclass
class SimulationParameters:
    """Simulation configuration parameters"""
    id: str
    simulation_type: SimulationType
    method: ComputationalMethod
    time_step: Optional[float]  # fs for MD, arbitrary for other methods
    total_time: Optional[float]  # Total simulation time
    ensemble: Optional[str]     # NVE, NVT, NPT for MD
    integrator: Optional[str]   # Verlet, Leapfrog, etc.
    cutoff_radius: Optional[float]  # Angstroms
    grid_spacing: Optional[float]   # For grid-based methods
    convergence_criteria: Dict[str, float]
    output_frequency: int       # Steps between data output
    checkpoint_frequency: int   # Steps between checkpoints
    parallel_settings: Dict[str, Any]
    software_specific: Dict[str, Any]  # Software-specific parameters


@dataclass
class ComputationalResource:
    """HPC resource specification"""
    id: str
    name: str
    institution: str
    architecture: str  # "CPU", "GPU", "hybrid"
    node_count: int
    cores_per_node: int
    memory_per_node: int  # GB
    interconnect: str     # "Infiniband", "Ethernet"
    storage_capacity: int  # TB
    queue_system: str     # "SLURM", "PBS", "SGE"
    software_stack: List[str]
    access_permissions: List[str]
    cost_per_hour: Optional[float]
    available: bool


@dataclass
class SimulationJob:
    """Physics simulation job"""
    id: str
    name: str
    project_id: str
    system_id: str
    parameters_id: str
    resource_id: str
    status: str  # "queued", "running", "completed", "failed", "cancelled"
    submitted_by: str
    submitted_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    execution_time: Optional[float]  # hours
    job_script: str
    output_files: List[str]
    error_log: Optional[str]
    resource_usage: Dict[str, float]
    checkpoint_files: List[str]
    progress: float  # 0.0 to 1.0
    estimated_remaining: Optional[float]  # hours


@dataclass
class SimulationResult:
    """Physics simulation results"""
    id: str
    job_id: str
    result_type: str  # "trajectory", "energy", "structure", "spectrum"
    data_files: List[str]
    analysis_data: Dict[str, Any]
    statistical_metrics: Dict[str, float]
    visualizations: List[str]
    physical_properties: Dict[str, float]
    convergence_analysis: Dict[str, Any]
    validation_results: Dict[str, float]
    generated_at: datetime
    file_size: int  # bytes
    encrypted: bool
    peer_reviewed: bool


@dataclass
class PhysicsCollaborationSession:
    """Real-time physics collaboration session"""
    id: str
    project_id: str
    participants: List[str]
    active_simulation: Optional[str]
    shared_workspace: Dict[str, Any]
    parameter_discussions: List[Dict[str, Any]]
    result_annotations: List[Dict[str, Any]]
    code_sharing: List[Dict[str, Any]]
    started_at: datetime
    last_activity: datetime
    session_status: str


@dataclass
class PhysicsProject:
    """Physics research collaboration project"""
    id: str
    name: str
    description: str
    physics_domain: str  # "condensed_matter", "quantum", "astrophysics", etc.
    created_by: str
    created_at: datetime
    university: str
    department: str
    national_lab: Optional[str]
    industry_partner: Optional[str]
    funding_agency: Optional[str]
    collaborators: Dict[str, CollaborationRole]
    access_permissions: Dict[str, List[str]]
    systems: Dict[str, PhysicsSystem]
    simulation_parameters: Dict[str, SimulationParameters]
    computational_resources: Dict[str, ComputationalResource]
    jobs: Dict[str, SimulationJob]
    results: Dict[str, SimulationResult]
    active_sessions: List[str]
    security_level: str
    export_controlled: bool  # ITAR/EAR restrictions
    classification_level: str  # "unclassified", "cui", "confidential"
    timeline: Dict[str, datetime]
    deliverables: List[str]
    nwtn_insights: List[Dict[str, Any]]
    reproducibility_package: bool


class PhysicsSimulation:
    """Main physics simulation collaboration system"""
    
    def __init__(self):
        self.crypto_sharding = PostQuantumCryptoSharding()
        self.nwtn = MockNWTN()
        
        self.projects: Dict[str, PhysicsProject] = {}
        self.active_sessions: Dict[str, PhysicsCollaborationSession] = {}
        
        # Physics research templates
        self.research_templates = {
            "condensed_matter": {
                "name": "Condensed Matter Physics Research",
                "typical_methods": ["density_functional", "molecular_dynamics", "monte_carlo"],
                "common_software": ["VASP", "Quantum Espresso", "LAMMPS", "GROMACS"],
                "typical_systems": ["crystals", "surfaces", "nanoparticles", "liquids"],
                "computational_requirements": {"cores": 128, "memory_gb": 256, "storage_tb": 5}
            },
            "quantum_physics": {
                "name": "Quantum Physics Simulations",
                "typical_methods": ["hartree_fock", "coupled_cluster", "quantum_monte_carlo"],
                "common_software": ["Gaussian", "MOLPRO", "NWChem", "Qiskit"],
                "typical_systems": ["atoms", "molecules", "quantum_dots", "qubits"],
                "computational_requirements": {"cores": 64, "memory_gb": 512, "storage_tb": 2}
            },
            "fluid_dynamics": {
                "name": "Computational Fluid Dynamics",
                "typical_methods": ["finite_element", "finite_difference", "lattice_boltzmann"],
                "common_software": ["OpenFOAM", "ANSYS Fluent", "COMSOL", "SU2"],
                "typical_systems": ["flows", "turbulence", "heat_transfer", "multiphase"],
                "computational_requirements": {"cores": 256, "memory_gb": 128, "storage_tb": 10}
            },
            "astrophysics": {
                "name": "Astrophysical Simulations",
                "typical_methods": ["n_body", "hydrodynamics", "magnetohydrodynamics"],
                "common_software": ["GADGET", "FLASH", "Athena++", "PLUTO"],
                "typical_systems": ["galaxies", "stellar_systems", "black_holes", "cosmic_rays"],
                "computational_requirements": {"cores": 1024, "memory_gb": 1024, "storage_tb": 50}
            }
        }
        
        # University-national lab partnerships
        self.lab_partnerships = {
            "unc_ornl": {
                "name": "UNC - Oak Ridge National Laboratory Partnership",
                "focus_areas": ["materials_science", "neutron_scattering", "supercomputing"],
                "resources": ["Summit", "Spallation Neutron Source", "Center for Nanophase Materials"],
                "security_level": "cui_required"
            },
            "duke_slac": {
                "name": "Duke - SLAC National Accelerator Laboratory",
                "focus_areas": ["particle_physics", "accelerator_physics", "x_ray_science"],
                "resources": ["LCLS", "FACET", "Stanford Synchrotron"],
                "security_level": "standard"
            },
            "ncsu_sandia": {
                "name": "NC State - Sandia National Laboratories",
                "focus_areas": ["plasma_physics", "fusion_energy", "nuclear_engineering"],
                "resources": ["Z Machine", "Sandia HPC", "Combustion Research"],
                "security_level": "cui_required"
            }
        }
        
        # Common physics software packages
        self.physics_software = {
            "lammps": {
                "name": "Large-scale Atomic/Molecular Massively Parallel Simulator",
                "domain": "molecular_dynamics",
                "license": "GPL",
                "supported_systems": ["materials", "biomolecules", "polymers"],
                "input_format": "lammps_script",
                "output_formats": ["dump", "log", "restart"]
            },
            "vasp": {
                "name": "Vienna Ab initio Simulation Package",
                "domain": "density_functional_theory",
                "license": "Commercial",
                "supported_systems": ["crystals", "surfaces", "molecules"],
                "input_format": "POSCAR",
                "output_formats": ["OUTCAR", "vasprun.xml", "CHGCAR"]
            },
            "gromacs": {
                "name": "GROningen MAchine for Chemical Simulations",
                "domain": "molecular_dynamics",
                "license": "LGPL",
                "supported_systems": ["biomolecules", "polymers", "materials"],
                "input_format": "gro",
                "output_formats": ["xtc", "trr", "edr"]
            },
            "openfoam": {
                "name": "Open Source Field Operation and Manipulation",
                "domain": "computational_fluid_dynamics",
                "license": "GPL",
                "supported_systems": ["fluids", "heat_transfer", "turbulence"],
                "input_format": "case_directory",
                "output_formats": ["vtk", "ensight", "tecplot"]
            }
        }
    
    async def create_physics_project(
        self,
        name: str,
        description: str,
        physics_domain: str,
        creator_id: str,
        university: str,
        department: str,
        template: Optional[str] = None,
        national_lab: Optional[str] = None,
        industry_partner: Optional[str] = None,
        security_level: str = "standard"
    ) -> PhysicsProject:
        """Create a new physics research project"""
        
        project_id = str(uuid.uuid4())
        
        # Apply template if provided
        template_config = self.research_templates.get(physics_domain, {})
        lab_config = self.lab_partnerships.get(template, {})
        
        # Set security requirements
        export_controlled = physics_domain in ["plasma_physics", "nuclear_physics", "defense_physics"]
        classification = lab_config.get("security_level", "unclassified")
        
        # Generate NWTN insights for project setup
        nwtn_context = {
            "project_name": name,
            "physics_domain": physics_domain,
            "university": university,
            "department": department,
            "national_lab": national_lab,
            "industry_partner": industry_partner
        }
        
        nwtn_insights = await self._generate_project_insights(nwtn_context)
        
        # Set up timeline
        timeline = {
            "project_start": datetime.now(),
            "system_setup": datetime.now() + timedelta(weeks=2),
            "parameter_optimization": datetime.now() + timedelta(weeks=6),
            "production_runs": datetime.now() + timedelta(weeks=12),
            "analysis_phase": datetime.now() + timedelta(weeks=18),
            "manuscript_prep": datetime.now() + timedelta(weeks=24)
        }
        
        project = PhysicsProject(
            id=project_id,
            name=name,
            description=description,
            physics_domain=physics_domain,
            created_by=creator_id,
            created_at=datetime.now(),
            university=university,
            department=department,
            national_lab=national_lab,
            industry_partner=industry_partner,
            funding_agency=None,
            collaborators={creator_id: CollaborationRole.PRINCIPAL_INVESTIGATOR},
            access_permissions={
                "read": [creator_id],
                "write": [creator_id],
                "execute": [creator_id],
                "admin": [creator_id]
            },
            systems={},
            simulation_parameters={},
            computational_resources={},
            jobs={},
            results={},
            active_sessions=[],
            security_level=security_level,
            export_controlled=export_controlled,
            classification_level=classification,
            timeline=timeline,
            deliverables=template_config.get("typical_deliverables", []),
            nwtn_insights=nwtn_insights,
            reproducibility_package=True
        )
        
        self.projects[project_id] = project
        return project
    
    async def create_physics_system(
        self,
        project_id: str,
        system_name: str,
        system_type: str,
        description: str,
        dimensions: int,
        user_id: str,
        particle_count: Optional[int] = None,
        system_size: Tuple[float, float, float] = (10.0, 10.0, 10.0),
        temperature: Optional[float] = None
    ) -> PhysicsSystem:
        """Create a physics system definition"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("write", []):
            raise PermissionError("User does not have write access")
        
        system = PhysicsSystem(
            id=str(uuid.uuid4()),
            name=system_name,
            system_type=system_type,
            description=description,
            dimensions=dimensions,
            particle_count=particle_count,
            system_size=system_size,
            boundary_conditions="periodic",
            temperature=temperature,
            pressure=None,
            composition={},
            initial_conditions={},
            force_field=None,
            created_by=user_id,
            created_at=datetime.now(),
            validated=False
        )
        
        project.systems[system_name] = system
        
        # Generate system insights
        system_insights = await self._analyze_physics_system(system, project)
        project.nwtn_insights.extend(system_insights)
        
        return system
    
    async def create_simulation_parameters(
        self,
        project_id: str,
        params_name: str,
        simulation_type: SimulationType,
        method: ComputationalMethod,
        user_id: str,
        time_step: Optional[float] = None,
        total_time: Optional[float] = None,
        additional_params: Dict[str, Any] = None
    ) -> SimulationParameters:
        """Create simulation parameters"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("write", []):
            raise PermissionError("User does not have write access")
        
        # Set default parameters based on simulation type
        default_params = self._get_default_parameters(simulation_type, method)
        if additional_params:
            default_params.update(additional_params)
        
        parameters = SimulationParameters(
            id=str(uuid.uuid4()),
            simulation_type=simulation_type,
            method=method,
            time_step=time_step,
            total_time=total_time,
            ensemble=default_params.get("ensemble"),
            integrator=default_params.get("integrator"),
            cutoff_radius=default_params.get("cutoff_radius"),
            grid_spacing=default_params.get("grid_spacing"),
            convergence_criteria=default_params.get("convergence_criteria", {}),
            output_frequency=default_params.get("output_frequency", 1000),
            checkpoint_frequency=default_params.get("checkpoint_frequency", 10000),
            parallel_settings=default_params.get("parallel_settings", {}),
            software_specific=default_params.get("software_specific", {})
        )
        
        project.simulation_parameters[params_name] = parameters
        
        # Generate parameter insights
        param_insights = await self._analyze_simulation_parameters(parameters, project)
        project.nwtn_insights.extend(param_insights)
        
        return parameters
    
    async def add_computational_resource(
        self,
        project_id: str,
        resource_name: str,
        institution: str,
        architecture: str,
        node_count: int,
        cores_per_node: int,
        memory_per_node: int,
        user_id: str
    ) -> ComputationalResource:
        """Add computational resource to project"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("admin", []):
            raise PermissionError("User does not have admin access")
        
        resource = ComputationalResource(
            id=str(uuid.uuid4()),
            name=resource_name,
            institution=institution,
            architecture=architecture,
            node_count=node_count,
            cores_per_node=cores_per_node,
            memory_per_node=memory_per_node,
            interconnect="Infiniband",
            storage_capacity=1000,  # Default 1TB
            queue_system="SLURM",
            software_stack=["GCC", "Intel MPI", "CUDA"],
            access_permissions=[user_id],
            cost_per_hour=None,
            available=True
        )
        
        project.computational_resources[resource_name] = resource
        
        return resource
    
    async def submit_simulation_job(
        self,
        project_id: str,
        job_name: str,
        system_name: str,
        parameters_name: str,
        resource_name: str,
        user_id: str,
        job_script: str = None
    ) -> SimulationJob:
        """Submit physics simulation job"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("execute", []):
            raise PermissionError("User does not have execute access")
        
        # Validate components exist
        if system_name not in project.systems:
            raise ValueError(f"System {system_name} not found")
        if parameters_name not in project.simulation_parameters:
            raise ValueError(f"Parameters {parameters_name} not found")
        if resource_name not in project.computational_resources:
            raise ValueError(f"Resource {resource_name} not found")
        
        # Generate job script if not provided
        if not job_script:
            job_script = await self._generate_job_script(
                project.systems[system_name],
                project.simulation_parameters[parameters_name],
                project.computational_resources[resource_name]
            )
        
        job = SimulationJob(
            id=str(uuid.uuid4()),
            name=job_name,
            project_id=project_id,
            system_id=project.systems[system_name].id,
            parameters_id=project.simulation_parameters[parameters_name].id,
            resource_id=project.computational_resources[resource_name].id,
            status="queued",
            submitted_by=user_id,
            submitted_at=datetime.now(),
            started_at=None,
            completed_at=None,
            execution_time=None,
            job_script=job_script,
            output_files=[],
            error_log=None,
            resource_usage={},
            checkpoint_files=[],
            progress=0.0,
            estimated_remaining=None
        )
        
        project.jobs[job_name] = job
        
        # Start job simulation
        await self._simulate_job_execution(job)
        
        return job
    
    async def analyze_simulation_results(
        self,
        project_id: str,
        job_name: str,
        user_id: str,
        analysis_type: str = "standard"
    ) -> SimulationResult:
        """Analyze physics simulation results"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("read", []):
            raise PermissionError("User does not have read access")
        
        if job_name not in project.jobs:
            raise ValueError(f"Job {job_name} not found")
        
        job = project.jobs[job_name]
        
        if job.status != "completed":
            raise ValueError("Job must be completed for analysis")
        
        # Perform analysis simulation
        analysis_results = await self._simulate_result_analysis(job, analysis_type)
        
        # Encrypt results if required
        encrypted = False
        if project.security_level == "maximum" or project.export_controlled:
            # Encrypt result files
            for file_path in analysis_results["data_files"]:
                encrypted_shards = self.crypto_sharding.shard_file(
                    file_path,
                    list(project.collaborators.keys()),
                    num_shards=7
                )
            encrypted = True
        
        result = SimulationResult(
            id=str(uuid.uuid4()),
            job_id=job.id,
            result_type=analysis_results["type"],
            data_files=analysis_results["data_files"] if not encrypted else ["encrypted"],
            analysis_data=analysis_results["analysis"],
            statistical_metrics=analysis_results["statistics"],
            visualizations=analysis_results["plots"],
            physical_properties=analysis_results["properties"],
            convergence_analysis=analysis_results["convergence"],
            validation_results=analysis_results["validation"],
            generated_at=datetime.now(),
            file_size=sum(analysis_results["file_sizes"]),
            encrypted=encrypted,
            peer_reviewed=False
        )
        
        project.results[f"{job_name}_results"] = result
        
        # Generate result insights
        result_insights = await self._analyze_simulation_results_nwtn(result, job, project)
        project.nwtn_insights.extend(result_insights)
        
        return result
    
    async def start_collaboration_session(
        self,
        project_id: str,
        user_id: str,
        participants: List[str] = None
    ) -> PhysicsCollaborationSession:
        """Start real-time physics collaboration session"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("read", []):
            raise PermissionError("User does not have read access")
        
        session_id = str(uuid.uuid4())
        
        session = PhysicsCollaborationSession(
            id=session_id,
            project_id=project_id,
            participants=[user_id] + (participants or []),
            active_simulation=None,
            shared_workspace={
                "parameter_editor": {},
                "result_viewer": {},
                "code_editor": {},
                "visualization_panel": {}
            },
            parameter_discussions=[],
            result_annotations=[],
            code_sharing=[],
            started_at=datetime.now(),
            last_activity=datetime.now(),
            session_status="active"
        )
        
        self.active_sessions[session_id] = session
        project.active_sessions.append(session_id)
        
        return session
    
    async def optimize_parameters(
        self,
        project_id: str,
        system_name: str,
        parameters_name: str,
        optimization_target: str,
        user_id: str
    ) -> Dict[str, Any]:
        """AI-powered parameter optimization"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("execute", []):
            raise PermissionError("User does not have execute access")
        
        system = project.systems[system_name]
        parameters = project.simulation_parameters[parameters_name]
        
        # Generate optimization recommendations
        optimization_results = await self._generate_parameter_optimization(
            system, parameters, optimization_target, project
        )
        
        return optimization_results
    
    async def export_physics_project(
        self,
        project_id: str,
        user_id: str,
        include_results: bool = True,
        include_job_scripts: bool = True,
        reproducibility_package: bool = True
    ) -> str:
        """Export physics project data"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions and export restrictions
        if user_id not in project.access_permissions.get("read", []):
            raise PermissionError("User does not have read access")
        
        if project.export_controlled and project.classification_level != "unclassified":
            # Additional export control checks would be implemented here
            print("⚠️ Export control restrictions apply to this project")
        
        # Create export package
        temp_dir = tempfile.mkdtemp()
        export_path = Path(temp_dir) / f"{project.name}_physics_export.zip"
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Project metadata
            project_data = asdict(project)
            project_data['created_at'] = project_data['created_at'].isoformat()
            
            zipf.writestr(
                "project_metadata.json",
                json.dumps(project_data, indent=2, default=str)
            )
            
            # Physics systems
            systems_data = {}
            for name, system in project.systems.items():
                system_data = asdict(system)
                system_data['created_at'] = system_data['created_at'].isoformat()
                systems_data[name] = system_data
            
            zipf.writestr(
                "physics_systems.json",
                json.dumps(systems_data, indent=2, default=str)
            )
            
            # Simulation parameters
            params_data = {}
            for name, params in project.simulation_parameters.items():
                params_data[name] = asdict(params)
            
            zipf.writestr(
                "simulation_parameters.json",
                json.dumps(params_data, indent=2, default=str)
            )
            
            # Job scripts
            if include_job_scripts:
                for job_name, job in project.jobs.items():
                    zipf.writestr(
                        f"job_scripts/{job_name}.sh",
                        job.job_script
                    )
            
            # Results metadata
            if include_results:
                results_metadata = {}
                for name, result in project.results.items():
                    result_data = asdict(result)
                    result_data['generated_at'] = result_data['generated_at'].isoformat()
                    
                    # Remove actual data files for security
                    result_data.pop('data_files', None)
                    results_metadata[name] = result_data
                
                zipf.writestr(
                    "results_metadata.json",
                    json.dumps(results_metadata, indent=2, default=str)
                )
            
            # Reproducibility package
            if reproducibility_package:
                repro_info = self._generate_reproducibility_package(project)
                zipf.writestr("reproducibility_guide.md", repro_info)
            
            # NWTN insights
            zipf.writestr(
                "nwtn_insights.json",
                json.dumps(project.nwtn_insights, indent=2, default=str)
            )
            
            # Generate project summary
            summary_report = self._generate_physics_summary(project)
            zipf.writestr("project_summary.md", summary_report)
        
        return str(export_path)
    
    def _get_default_parameters(
        self,
        simulation_type: SimulationType,
        method: ComputationalMethod
    ) -> Dict[str, Any]:
        """Get default parameters for simulation type"""
        
        if simulation_type == SimulationType.MOLECULAR_DYNAMICS:
            return {
                "ensemble": "NVT",
                "integrator": "velocity_verlet",
                "cutoff_radius": 12.0,
                "output_frequency": 1000,
                "checkpoint_frequency": 10000,
                "convergence_criteria": {"energy": 1e-6, "force": 1e-4}
            }
        elif simulation_type == SimulationType.DENSITY_FUNCTIONAL_THEORY:
            return {
                "cutoff_energy": 500.0,  # eV
                "k_point_mesh": [8, 8, 8],
                "convergence_criteria": {"energy": 1e-6, "force": 1e-3},
                "mixing_parameter": 0.7,
                "electronic_steps": 100
            }
        elif simulation_type == SimulationType.FINITE_ELEMENT_ANALYSIS:
            return {
                "element_type": "tetrahedral",
                "mesh_refinement": 3,
                "convergence_criteria": {"displacement": 1e-6},
                "solver_type": "direct",
                "output_frequency": 1
            }
        else:
            return {
                "output_frequency": 1000,
                "convergence_criteria": {"default": 1e-6}
            }
    
    async def _generate_job_script(
        self,
        system: PhysicsSystem,
        parameters: SimulationParameters,
        resource: ComputationalResource
    ) -> str:
        """Generate HPC job submission script"""
        
        total_cores = resource.node_count * resource.cores_per_node
        
        if parameters.simulation_type == SimulationType.MOLECULAR_DYNAMICS:
            script = f"""#!/bin/bash
#SBATCH --job-name={system.name}_MD
#SBATCH --nodes={resource.node_count}
#SBATCH --ntasks-per-node={resource.cores_per_node}
#SBATCH --time=24:00:00
#SBATCH --partition=compute

# Load modules
module load gcc/11.2.0
module load openmpi/4.1.0
module load lammps/29Oct2020

# Set environment
export OMP_NUM_THREADS=1

# Run simulation
mpirun -np {total_cores} lmp_mpi -in input.lammps > output.log 2>&1

echo "Simulation completed at $(date)"
"""
        elif parameters.simulation_type == SimulationType.DENSITY_FUNCTIONAL_THEORY:
            script = f"""#!/bin/bash
#SBATCH --job-name={system.name}_DFT
#SBATCH --nodes={resource.node_count}
#SBATCH --ntasks-per-node={resource.cores_per_node}
#SBATCH --time=48:00:00
#SBATCH --partition=compute

# Load modules
module load intel/2021.4
module load intel-mpi/2021.4
module load vasp/6.3.0

# Set environment
export I_MPI_PIN_DOMAIN=omp
export OMP_NUM_THREADS=1

# Run calculation
mpirun -np {total_cores} vasp_std > OUTCAR 2>&1

echo "DFT calculation completed at $(date)"
"""
        else:
            script = f"""#!/bin/bash
#SBATCH --job-name={system.name}_{parameters.simulation_type.value}
#SBATCH --nodes={resource.node_count}
#SBATCH --ntasks-per-node={resource.cores_per_node}
#SBATCH --time=12:00:00
#SBATCH --partition=compute

# Generic physics simulation script
echo "Starting {parameters.simulation_type.value} simulation at $(date)"
echo "System: {system.name}"
echo "Cores: {total_cores}"

# Simulation would run here
sleep 10  # Placeholder for actual simulation

echo "Simulation completed at $(date)"
"""
        
        return script
    
    async def _simulate_job_execution(self, job: SimulationJob):
        """Simulate job execution progress"""
        
        # Simulate job lifecycle
        await asyncio.sleep(0.1)  # Queue time
        job.status = "running"
        job.started_at = datetime.now()
        
        # Simulate execution time based on job complexity
        execution_time = np.random.uniform(0.5, 2.0)  # hours
        job.estimated_remaining = execution_time
        
        # Update progress periodically (simulated)
        for progress in [0.2, 0.4, 0.6, 0.8, 1.0]:
            await asyncio.sleep(0.02)  # Simulate time passage
            job.progress = progress
            job.estimated_remaining = execution_time * (1 - progress)
        
        # Complete job
        job.status = "completed"
        job.completed_at = datetime.now()
        job.execution_time = execution_time
        job.progress = 1.0
        job.estimated_remaining = 0.0
        
        # Generate output files
        job.output_files = [
            "simulation.out",
            "trajectory.xyz",
            "energy.dat",
            "final_structure.pdb"
        ]
        
        job.resource_usage = {
            "cpu_hours": execution_time * 128,  # Assuming 128 cores
            "memory_gb_hours": execution_time * 256,
            "storage_gb": 15.4
        }
    
    async def _simulate_result_analysis(
        self,
        job: SimulationJob,
        analysis_type: str
    ) -> Dict[str, Any]:
        """Simulate physics result analysis"""
        
        if analysis_type == "molecular_dynamics":
            return {
                "type": "trajectory",
                "data_files": ["trajectory.xtc", "energy.xvg", "rdf.xvg"],
                "file_sizes": [1024*1024*50, 1024*100, 1024*50],  # bytes
                "analysis": {
                    "total_frames": 10000,
                    "simulation_time": "10 ns",
                    "temperature_avg": 298.15,
                    "temperature_std": 2.1,
                    "pressure_avg": 1.01325,
                    "density_avg": 0.997
                },
                "statistics": {
                    "rmsd": 2.3,  # Angstroms
                    "rg": 15.6,   # Radius of gyration
                    "sasa": 1234.5  # Solvent accessible surface area
                },
                "plots": ["energy_vs_time.png", "rmsd_vs_time.png", "rdf.png"],
                "properties": {
                    "diffusion_coefficient": 2.1e-5,  # cm²/s
                    "viscosity": 0.89,  # cP
                    "compressibility": 4.5e-5  # bar⁻¹
                },
                "convergence": {
                    "energy_converged": True,
                    "temperature_equilibrated": True,
                    "pressure_equilibrated": True
                },
                "validation": {
                    "energy_conservation": 0.01,  # %
                    "momentum_conservation": 0.001
                }
            }
        elif analysis_type == "quantum_mechanics":
            return {
                "type": "electronic_structure",
                "data_files": ["wavefunction.cube", "density.cube", "eigenvalues.dat"],
                "file_sizes": [1024*1024*100, 1024*1024*100, 1024*10],
                "analysis": {
                    "total_energy": -1234.567,  # Hartree
                    "homo_energy": -0.234,      # Hartree
                    "lumo_energy": 0.123,       # Hartree
                    "band_gap": 0.357,          # Hartree
                    "dipole_moment": 2.34       # Debye
                },
                "statistics": {
                    "scf_cycles": 25,
                    "convergence_energy": 1e-8,
                    "max_force": 1e-4
                },
                "plots": ["dos.png", "band_structure.png", "electron_density.png"],
                "properties": {
                    "ionization_potential": 0.234,  # Hartree
                    "electron_affinity": 0.123,     # Hartree
                    "polarizability": 45.6          # a.u.
                },
                "convergence": {
                    "scf_converged": True,
                    "geometry_optimized": True,
                    "forces_converged": True
                },
                "validation": {
                    "energy_accuracy": 1e-6,
                    "force_accuracy": 1e-4
                }
            }
        else:
            return {
                "type": "general",
                "data_files": ["output.dat"],
                "file_sizes": [1024*1024*10],
                "analysis": {"completed": True},
                "statistics": {"success_rate": 1.0},
                "plots": ["results.png"],
                "properties": {"computed": True},
                "convergence": {"achieved": True},
                "validation": {"passed": True}
            }
    
    async def _generate_parameter_optimization(
        self,
        system: PhysicsSystem,
        parameters: SimulationParameters,
        target: str,
        project: PhysicsProject
    ) -> Dict[str, Any]:
        """Generate AI-powered parameter optimization"""
        
        # Simulate optimization analysis
        current_params = {
            "time_step": parameters.time_step,
            "cutoff_radius": parameters.cutoff_radius,
            "output_frequency": parameters.output_frequency
        }
        
        optimized_params = {}
        improvements = {}
        
        for param, value in current_params.items():
            if value is not None:
                # Simulate optimization
                if param == "time_step":
                    new_value = value * 0.8  # Suggest smaller timestep
                    improvement = 15.2  # % accuracy improvement
                elif param == "cutoff_radius":
                    new_value = value * 1.1  # Suggest larger cutoff
                    improvement = 8.7   # % efficiency improvement
                else:
                    new_value = value
                    improvement = 0.0
                
                optimized_params[param] = new_value
                improvements[param] = improvement
        
        return {
            "optimization_target": target,
            "current_parameters": current_params,
            "optimized_parameters": optimized_params,
            "expected_improvements": improvements,
            "confidence": 0.87,
            "optimization_method": "bayesian_optimization",
            "recommendations": [
                "Reduce time step for better numerical stability",
                "Increase cutoff radius for more accurate interactions",
                "Consider adaptive time stepping algorithms"
            ],
            "estimated_speedup": 1.23,
            "accuracy_improvement": 12.5
        }
    
    async def _generate_project_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate NWTN insights for project setup"""
        
        nwtn_prompt = f"""
        Analyze this physics research project setup:
        
        Project: {context['project_name']}
        Physics Domain: {context['physics_domain']}
        University: {context['university']}
        Department: {context['department']}
        National Lab: {context.get('national_lab', 'None')}
        Industry Partner: {context.get('industry_partner', 'None')}
        
        Provide insights on:
        1. Appropriate computational methods and software
        2. HPC resource requirements and optimization
        3. Collaboration strategies with national labs/industry
        4. Data management and reproducibility considerations
        5. Publication and dissemination strategies
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
    
    async def _analyze_physics_system(
        self,
        system: PhysicsSystem,
        project: PhysicsProject
    ) -> List[Dict[str, Any]]:
        """Analyze physics system using NWTN"""
        
        nwtn_prompt = f"""
        Analyze this physics system for computational modeling:
        
        System: {system.name}
        Type: {system.system_type}
        Dimensions: {system.dimensions}D
        Particles: {system.particle_count}
        Size: {system.system_size}
        Temperature: {system.temperature} K
        
        Physics Domain: {project.physics_domain}
        
        Provide analysis on:
        1. Appropriate simulation methods and approximations
        2. Computational complexity and resource requirements
        3. Physical accuracy and validation strategies
        4. Potential computational challenges
        5. Optimization opportunities for performance
        """
        
        context = {
            "system": asdict(system),
            "project": asdict(project)
        }
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "system_analysis",
                "timestamp": datetime.now(),
                "system_id": system.id,
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    async def _analyze_simulation_parameters(
        self,
        parameters: SimulationParameters,
        project: PhysicsProject
    ) -> List[Dict[str, Any]]:
        """Analyze simulation parameters using NWTN"""
        
        nwtn_prompt = f"""
        Analyze these simulation parameters:
        
        Simulation Type: {parameters.simulation_type.value}
        Method: {parameters.method.value}
        Time Step: {parameters.time_step}
        Total Time: {parameters.total_time}
        Cutoff Radius: {parameters.cutoff_radius}
        
        Physics Domain: {project.physics_domain}
        
        Provide analysis on:
        1. Parameter appropriateness for the physical system
        2. Numerical stability and accuracy considerations
        3. Computational efficiency optimization
        4. Convergence and validation strategies
        5. Sensitivity analysis recommendations
        """
        
        context = {
            "parameters": asdict(parameters),
            "project": asdict(project)
        }
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "parameter_analysis",
                "timestamp": datetime.now(),
                "parameters_id": parameters.id,
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    async def _analyze_simulation_results_nwtn(
        self,
        result: SimulationResult,
        job: SimulationJob,
        project: PhysicsProject
    ) -> List[Dict[str, Any]]:
        """Analyze simulation results using NWTN"""
        
        nwtn_prompt = f"""
        Analyze these physics simulation results:
        
        Result Type: {result.result_type}
        Physical Properties: {result.physical_properties}
        Statistical Metrics: {result.statistical_metrics}
        Convergence: {result.convergence_analysis}
        File Size: {result.file_size / (1024*1024):.1f} MB
        
        Project Context: {project.physics_domain}
        
        Provide analysis on:
        1. Physical interpretation of results
        2. Statistical significance and uncertainty
        3. Validation against theory or experiment
        4. Further analysis recommendations
        5. Publication and dissemination strategy
        """
        
        context = {
            "result": asdict(result),
            "job": asdict(job),
            "project": asdict(project)
        }
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "result_analysis",
                "timestamp": datetime.now(),
                "result_id": result.id,
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    def _generate_reproducibility_package(self, project: PhysicsProject) -> str:
        """Generate reproducibility guide"""
        
        guide = f"""# Computational Physics Reproducibility Guide

## Project: {project.name}

### System Requirements
- **Physics Domain**: {project.physics_domain}
- **Security Classification**: {project.classification_level}
- **Export Control**: {'Yes' if project.export_controlled else 'No'}

### Software Stack
"""
        
        # Add software requirements based on project type
        template = self.research_templates.get(project.physics_domain, {})
        software_list = template.get("common_software", [])
        
        for software in software_list:
            if software.lower() in self.physics_software:
                sw_info = self.physics_software[software.lower()]
                guide += f"""
#### {sw_info['name']}
- **Domain**: {sw_info['domain']}
- **License**: {sw_info['license']}
- **Input Format**: {sw_info['input_format']}
- **Output Formats**: {', '.join(sw_info['output_formats'])}
"""
        
        guide += f"""
### Computational Resources
"""
        
        for name, resource in project.computational_resources.items():
            guide += f"""
#### {name}
- **Institution**: {resource.institution}
- **Architecture**: {resource.architecture}
- **Nodes**: {resource.node_count}
- **Cores per Node**: {resource.cores_per_node}
- **Memory per Node**: {resource.memory_per_node} GB
- **Queue System**: {resource.queue_system}
"""
        
        guide += f"""
### Simulation Workflows

#### Systems Studied
"""
        
        for name, system in project.systems.items():
            guide += f"""
**{name}**:
- Type: {system.system_type}
- Dimensions: {system.dimensions}D
- Particles: {system.particle_count or 'N/A'}
- Size: {system.system_size}
- Temperature: {system.temperature or 'N/A'} K
"""
        
        guide += f"""
### Reproducibility Instructions

1. **Environment Setup**
   - Install required software packages as listed above
   - Configure HPC environment with appropriate modules
   - Verify computational resource access

2. **Data Preparation**
   - Download system configuration files
   - Verify input parameter files
   - Set up directory structure as specified

3. **Job Submission**
   - Use provided job scripts with appropriate modifications
   - Adjust resource requirements based on available systems
   - Monitor job progress and checkpoint files

4. **Analysis Pipeline**
   - Follow analysis scripts in order
   - Verify intermediate results against provided benchmarks
   - Generate final visualizations and reports

### Data Availability Statement

Simulation input files, job scripts, and analysis code are available through the PRSM secure collaboration platform. Raw simulation data may be subject to export control restrictions.

### Contact Information

For questions about reproducing these results, contact the project collaborators through the PRSM platform.

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return guide
    
    def _generate_physics_summary(self, project: PhysicsProject) -> str:
        """Generate physics project summary"""
        
        summary = f"""# Physics Project Summary: {project.name}

## Project Information
- **Project ID**: {project.id}
- **Created**: {project.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **Physics Domain**: {project.physics_domain}
- **University**: {project.university}
- **Department**: {project.department}
- **National Lab**: {project.national_lab or 'None'}
- **Industry Partner**: {project.industry_partner or 'None'}
- **Funding Agency**: {project.funding_agency or 'None'}

## Security and Compliance
- **Security Level**: {project.security_level}
- **Export Controlled**: {'Yes' if project.export_controlled else 'No'}
- **Classification**: {project.classification_level}

## Computational Assets
- **Physics Systems**: {len(project.systems)}
- **Parameter Sets**: {len(project.simulation_parameters)}
- **Computational Resources**: {len(project.computational_resources)}
- **Simulation Jobs**: {len(project.jobs)}
- **Analysis Results**: {len(project.results)}
- **NWTN Insights**: {len(project.nwtn_insights)}

## Collaboration
- **Total Collaborators**: {len(project.collaborators)}
- **Active Sessions**: {len(project.active_sessions)}

### Collaborator Roles
"""
        
        for user_id, role in project.collaborators.items():
            summary += f"- **{user_id}**: {role.value}\n"
        
        summary += f"""
## Timeline Progress
"""
        
        for milestone, date in project.timeline.items():
            status = "✅" if date <= datetime.now() else "⏳"
            summary += f"- {status} **{milestone.replace('_', ' ').title()}**: {date.strftime('%Y-%m-%d')}\n"
        
        summary += f"""
## Physics Systems
"""
        
        for name, system in list(project.systems.items())[:3]:  # Show first 3
            summary += f"""
### {name}
- **Type**: {system.system_type}
- **Dimensions**: {system.dimensions}D
- **Particles**: {system.particle_count or 'N/A'}
- **Temperature**: {system.temperature or 'N/A'} K
- **Validated**: {'Yes' if system.validated else 'No'}
"""
        
        if len(project.systems) > 3:
            summary += f"\n*... and {len(project.systems) - 3} more systems*\n"
        
        summary += f"""
## Computational Resources
"""
        
        total_cores = sum(r.node_count * r.cores_per_node for r in project.computational_resources.values())
        total_memory = sum(r.node_count * r.memory_per_node for r in project.computational_resources.values())
        
        summary += f"- **Total Cores Available**: {total_cores:,}\n"
        summary += f"- **Total Memory Available**: {total_memory:,} GB\n"
        summary += f"- **Resource Locations**: {len(set(r.institution for r in project.computational_resources.values()))}\n"
        
        summary += f"""
## Job Execution Summary
"""
        
        completed_jobs = [j for j in project.jobs.values() if j.status == "completed"]
        total_cpu_hours = sum(j.execution_time or 0 for j in completed_jobs) * 128  # Assuming avg 128 cores
        
        summary += f"- **Total Jobs**: {len(project.jobs)}\n"
        summary += f"- **Completed Jobs**: {len(completed_jobs)}\n"
        summary += f"- **Total CPU Hours**: {total_cpu_hours:,.1f}\n"
        summary += f"- **Average Job Runtime**: {np.mean([j.execution_time or 0 for j in completed_jobs]):.1f} hours\n"
        
        summary += f"""
---
*Summary generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return summary


# Testing and validation
async def test_physics_simulation():
    """Test physics simulation collaboration"""
    
    physics_sim = PhysicsSimulation()
    
    print("⚛️ Testing Physics Simulation Collaboration...")
    
    # Test 1: Create condensed matter physics project
    print("\n1. Creating UNC-ORNL Condensed Matter Physics Project...")
    
    project = await physics_sim.create_physics_project(
        name="2D Materials Electronic Structure Investigation",
        description="First-principles study of electronic properties in transition metal dichalcogenides",
        physics_domain="condensed_matter",
        creator_id="pi_physics_001",
        university="University of North Carolina at Chapel Hill",
        department="Department of Physics and Astronomy",
        template="unc_ornl",
        national_lab="Oak Ridge National Laboratory",
        security_level="cui_required"
    )
    
    print(f"✅ Created project: {project.name}")
    print(f"   - ID: {project.id}")
    print(f"   - Physics Domain: {project.physics_domain}")
    print(f"   - National Lab: {project.national_lab}")
    print(f"   - Export Controlled: {project.export_controlled}")
    print(f"   - Classification: {project.classification_level}")
    print(f"   - NWTN Insights: {len(project.nwtn_insights)}")
    
    # Test 2: Add collaborators
    print("\n2. Adding research collaborators...")
    
    project.collaborators.update({
        "theorist_001": CollaborationRole.THEORETICAL_PHYSICIST,
        "compphys_001": CollaborationRole.COMPUTATIONAL_PHYSICIST,
        "postdoc_001": CollaborationRole.POSTDOC_RESEARCHER,
        "student_001": CollaborationRole.GRADUATE_STUDENT,
        "hpc_001": CollaborationRole.HPC_SPECIALIST,
        "ornl_001": CollaborationRole.INDUSTRY_RESEARCHER
    })
    
    # Update permissions
    all_collaborators = list(project.collaborators.keys())
    project.access_permissions.update({
        "read": all_collaborators,
        "write": ["pi_physics_001", "theorist_001", "compphys_001", "postdoc_001"],
        "execute": ["pi_physics_001", "compphys_001", "hpc_001", "ornl_001"],
        "admin": ["pi_physics_001"]
    })
    
    print(f"✅ Added {len(project.collaborators)} collaborators")
    
    # Test 3: Create physics system
    print("\n3. Creating 2D material physics system...")
    
    system = await physics_sim.create_physics_system(
        project_id=project.id,
        system_name="MoS2_monolayer",
        system_type="2d_material",
        description="Molybdenum disulfide monolayer with hexagonal crystal structure",
        dimensions=2,
        user_id="theorist_001",
        particle_count=12,  # Unit cell atoms
        system_size=(6.15, 6.15, 20.0),  # Angstroms with vacuum
        temperature=300.0
    )
    
    print(f"✅ Created physics system: {system.name}")
    print(f"   - Type: {system.system_type}")
    print(f"   - Dimensions: {system.dimensions}D")
    print(f"   - Particles: {system.particle_count}")
    print(f"   - Size: {system.system_size} Å")
    print(f"   - Temperature: {system.temperature} K")
    
    # Test 4: Create simulation parameters
    print("\n4. Creating DFT simulation parameters...")
    
    parameters = await physics_sim.create_simulation_parameters(
        project_id=project.id,
        params_name="DFT_PBE_parameters",
        simulation_type=SimulationType.DENSITY_FUNCTIONAL_THEORY,
        method=ComputationalMethod.DENSITY_FUNCTIONAL,
        user_id="compphys_001",
        additional_params={
            "functional": "PBE",
            "cutoff_energy": 520.0,  # eV
            "k_point_mesh": [12, 12, 1],
            "smearing": "gaussian",
            "smearing_width": 0.02
        }
    )
    
    print(f"✅ Created simulation parameters: {parameters.simulation_type.value}")
    print(f"   - Method: {parameters.method.value}")
    print(f"   - Convergence criteria: {len(parameters.convergence_criteria)}")
    print(f"   - Software-specific params: {len(parameters.software_specific)}")
    
    # Test 5: Add computational resource
    print("\n5. Adding Oak Ridge Summit supercomputer resource...")
    
    resource = await physics_sim.add_computational_resource(
        project_id=project.id,
        resource_name="Summit_ORNL",
        institution="Oak Ridge National Laboratory",
        architecture="GPU",
        node_count=64,
        cores_per_node=42,
        memory_per_node=512,  # GB
        user_id="pi_physics_001"
    )
    
    print(f"✅ Added computational resource: {resource.name}")
    print(f"   - Architecture: {resource.architecture}")
    print(f"   - Total cores: {resource.node_count * resource.cores_per_node:,}")
    print(f"   - Total memory: {resource.node_count * resource.memory_per_node:,} GB")
    print(f"   - Queue system: {resource.queue_system}")
    
    # Test 6: Submit simulation job
    print("\n6. Submitting DFT calculation job...")
    
    job = await physics_sim.submit_simulation_job(
        project_id=project.id,
        job_name="MoS2_electronic_structure",
        system_name="MoS2_monolayer",
        parameters_name="DFT_PBE_parameters",
        resource_name="Summit_ORNL",
        user_id="compphys_001"
    )
    
    print(f"✅ Submitted simulation job: {job.name}")
    print(f"   - Status: {job.status}")
    print(f"   - Progress: {job.progress * 100:.1f}%")
    print(f"   - Execution time: {job.execution_time:.2f} hours")
    print(f"   - Output files: {len(job.output_files)}")
    print(f"   - CPU hours used: {job.resource_usage.get('cpu_hours', 0):,.1f}")
    
    # Test 7: Analyze simulation results
    print("\n7. Analyzing simulation results...")
    
    analysis_result = await physics_sim.analyze_simulation_results(
        project_id=project.id,
        job_name="MoS2_electronic_structure",
        user_id="compphys_001",
        analysis_type="quantum_mechanics"
    )
    
    print(f"✅ Completed result analysis: {analysis_result.result_type}")
    print(f"   - Data files: {len(analysis_result.data_files) if not analysis_result.encrypted else 'encrypted'}")
    print(f"   - File size: {analysis_result.file_size / (1024*1024):.1f} MB")
    print(f"   - Physical properties: {len(analysis_result.physical_properties)}")
    print(f"   - Visualizations: {len(analysis_result.visualizations)}")
    print(f"   - Band gap: {analysis_result.analysis_data.get('band_gap', 'N/A')} Hartree")
    
    # Test 8: Start collaboration session
    print("\n8. Starting physics collaboration session...")
    
    session = await physics_sim.start_collaboration_session(
        project_id=project.id,
        user_id="pi_physics_001",
        participants=["theorist_001", "compphys_001", "postdoc_001"]
    )
    
    print(f"✅ Started collaboration session: {session.id}")
    print(f"   - Participants: {len(session.participants)}")
    print(f"   - Workspace components: {len(session.shared_workspace)}")
    print(f"   - Status: {session.session_status}")
    
    # Test 9: AI-powered parameter optimization
    print("\n9. Running AI-powered parameter optimization...")
    
    optimization = await physics_sim.optimize_parameters(
        project_id=project.id,
        system_name="MoS2_monolayer",
        parameters_name="DFT_PBE_parameters",
        optimization_target="accuracy",
        user_id="compphys_001"
    )
    
    print(f"✅ Completed parameter optimization:")
    print(f"   - Target: {optimization['optimization_target']}")
    print(f"   - Confidence: {optimization['confidence']:.2f}")
    print(f"   - Expected speedup: {optimization['estimated_speedup']:.2f}x")
    print(f"   - Accuracy improvement: {optimization['accuracy_improvement']:.1f}%")
    print(f"   - Optimization method: {optimization['optimization_method']}")
    print(f"   - Recommendations: {len(optimization['recommendations'])}")
    
    # Test 10: Export physics project
    print("\n10. Exporting physics project with reproducibility package...")
    
    export_path = await physics_sim.export_physics_project(
        project_id=project.id,
        user_id="pi_physics_001",
        include_results=True,
        include_job_scripts=True,
        reproducibility_package=True
    )
    
    print(f"✅ Exported project to: {export_path}")
    
    # Verify export contents
    with zipfile.ZipFile(export_path, 'r') as zipf:
        files = zipf.namelist()
        print(f"   - Export contains {len(files)} files:")
        for file in files[:8]:  # Show first 8 files
            print(f"     • {file}")
        if len(files) > 8:
            print(f"     • ... and {len(files) - 8} more files")
    
    # Clean up export
    Path(export_path).unlink()
    shutil.rmtree(Path(export_path).parent)
    
    print(f"\n🎉 Physics Simulation testing completed successfully!")
    print(f"   - Projects: {len(physics_sim.projects)}")
    print(f"   - Active Sessions: {len(physics_sim.active_sessions)}")
    print(f"   - Research Templates: {len(physics_sim.research_templates)}")
    print(f"   - Lab Partnerships: {len(physics_sim.lab_partnerships)}")
    print(f"   - Physics Software: {len(physics_sim.physics_software)}")


if __name__ == "__main__":
    asyncio.run(test_physics_simulation())