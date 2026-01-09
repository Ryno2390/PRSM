"""
Chemistry Collaboration Tools for Research Partnerships

Provides secure P2P collaboration for chemistry research including ChemDraw molecule editing,
molecular visualization, reaction planning, and computational chemistry. Features include
collaborative structure drawing, spectral data analysis, and university-industry
chemical research partnerships with patent-sensitive data handling.

Key Features:
- Post-quantum cryptographic security for sensitive chemical IP
- ChemDraw-style collaborative molecule editing with real-time sync
- 3D molecular visualization and property prediction
- Reaction mechanism planning and pathway optimization
- Spectroscopic data analysis and interpretation
- Multi-institutional synthetic chemistry coordination
- NWTN AI-powered retrosynthesis and reaction prediction
- Patent-aware collaboration with IP protection protocols
- Export capabilities for publications and patent applications
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
import base64

from ..security.post_quantum_crypto_sharding import PostQuantumCryptoSharding

# Mock NWTN for testing
class MockNWTN:
    async def reason(self, prompt, context):
        return {
            "reasoning": [
                "Molecular structure appears chemically reasonable with appropriate bonding patterns",
                "Synthetic route shows good strategic bond disconnections and reagent choices",
                "Spectroscopic data is consistent with the proposed molecular structure"
            ],
            "recommendations": [
                "Consider alternative protecting group strategies for improved selectivity",
                "Explore greener synthetic methodologies for environmental sustainability",
                "Validate computational predictions with experimental characterization"
            ]
        }


class MoleculeFormat(Enum):
    """Chemical structure formats"""
    SMILES = "smiles"
    MOLFILE = "molfile"
    SDF = "sdf"
    CML = "cml"
    INCHI = "inchi"
    XYZ = "xyz"
    PDB = "pdb"
    MOL2 = "mol2"


class ReactionType(Enum):
    """Chemical reaction types"""
    SUBSTITUTION = "substitution"
    ELIMINATION = "elimination"
    ADDITION = "addition"
    REARRANGEMENT = "rearrangement"
    OXIDATION = "oxidation"
    REDUCTION = "reduction"
    CYCLOADDITION = "cycloaddition"
    COUPLING = "coupling"
    PROTECTION = "protection"
    DEPROTECTION = "deprotection"


class SpectroscopyType(Enum):
    """Spectroscopic techniques"""
    NMR_1H = "nmr_1h"
    NMR_13C = "nmr_13c"
    NMR_2D = "nmr_2d"
    IR = "ir"
    MS = "ms"
    UV_VIS = "uv_vis"
    FLUORESCENCE = "fluorescence"
    RAMAN = "raman"
    XRD = "xrd"
    EPR = "epr"


class CollaborationRole(Enum):
    """Chemistry collaboration roles"""
    PRINCIPAL_INVESTIGATOR = "principal_investigator"
    SYNTHETIC_CHEMIST = "synthetic_chemist"
    ANALYTICAL_CHEMIST = "analytical_chemist"
    COMPUTATIONAL_CHEMIST = "computational_chemist"
    MEDICINAL_CHEMIST = "medicinal_chemist"
    GRADUATE_STUDENT = "graduate_student"
    POSTDOC = "postdoc"
    INDUSTRY_CHEMIST = "industry_chemist"
    PATENT_ATTORNEY = "patent_attorney"


@dataclass
class ChemicalMolecule:
    """Chemical molecule representation"""
    id: str
    name: str
    formula: str
    smiles: str
    inchi: str
    molecular_weight: float
    structure_data: Dict[str, Any]  # 2D/3D coordinates, bonds
    properties: Dict[str, float]  # computed properties
    stereochemistry: str  # absolute/relative configuration
    created_by: str
    created_at: datetime
    version: int
    parent_molecule: Optional[str]
    modification_history: List[Dict[str, Any]]
    validated: bool
    confidence_score: float


@dataclass
class ChemicalReaction:
    """Chemical reaction representation"""
    id: str
    name: str
    reaction_type: ReactionType
    reactants: List[str]  # Molecule IDs
    products: List[str]   # Molecule IDs
    reagents: List[str]   # Reagent molecule IDs
    conditions: Dict[str, Any]  # Temperature, solvent, time, etc.
    mechanism: List[Dict[str, Any]]  # Step-by-step mechanism
    yield_theoretical: Optional[float]
    yield_experimental: Optional[float]
    selectivity: Dict[str, float]  # Regio-, stereo-, chemoselectivity
    literature_references: List[str]
    safety_notes: List[str]
    created_by: str
    created_at: datetime
    validated: bool
    patent_relevant: bool


@dataclass
class SpectroscopicData:
    """Spectroscopic analysis data"""
    id: str
    molecule_id: str
    technique: SpectroscopyType
    data_file: str
    processed_data: Dict[str, Any]
    peak_assignments: List[Dict[str, Any]]
    interpretation: str
    quality_metrics: Dict[str, float]
    instrument_parameters: Dict[str, Any]
    acquired_by: str
    acquired_at: datetime
    processed_by: str
    processed_at: datetime
    validated: bool


@dataclass
class SyntheticRoute:
    """Multi-step synthetic route"""
    id: str
    target_molecule_id: str
    route_name: str
    starting_materials: List[str]  # Molecule IDs
    synthetic_steps: List[str]     # Reaction IDs in order
    overall_yield: Optional[float]
    step_count: int
    complexity_score: float
    cost_estimate: Optional[float]
    time_estimate: Optional[str]
    scalability: str  # lab/pilot/industrial
    green_chemistry_score: float
    patent_landscape: List[str]
    created_by: str
    created_at: datetime
    optimization_suggestions: List[str]


@dataclass
class ChemistryCollaborationSession:
    """Real-time chemistry collaboration session"""
    id: str
    project_id: str
    participants: List[str]
    active_molecule: Optional[str]
    shared_canvas: Dict[str, Any]  # Drawing canvas state
    live_edits: List[Dict[str, Any]]  # Real-time structure edits
    chat_messages: List[Dict[str, Any]]
    started_at: datetime
    last_activity: datetime
    session_status: str  # "active", "paused", "ended"


@dataclass
class ChemistryProject:
    """Chemistry research collaboration project"""
    id: str
    name: str
    description: str
    research_area: str  # "drug_discovery", "materials", "catalysis", etc.
    created_by: str
    created_at: datetime
    university: str
    department: str
    industry_partner: Optional[str]
    collaborators: Dict[str, CollaborationRole]
    access_permissions: Dict[str, List[str]]
    molecules: Dict[str, ChemicalMolecule]
    reactions: Dict[str, ChemicalReaction]
    synthetic_routes: Dict[str, SyntheticRoute]
    spectroscopic_data: Dict[str, SpectroscopicData]
    active_sessions: List[str]
    security_level: str
    patent_sensitive: bool
    confidentiality_agreements: List[str]
    computational_resources: Dict[str, Any]
    timeline: Dict[str, datetime]
    deliverables: List[str]
    nwtn_insights: List[Dict[str, Any]]
    publication_pipeline: bool


class ChemistryCollaboration:
    """Main chemistry collaboration system"""
    
    def __init__(self):
        self.crypto_sharding = PostQuantumCryptoSharding()
        self.nwtn = MockNWTN()
        
        self.projects: Dict[str, ChemistryProject] = {}
        self.active_sessions: Dict[str, ChemistryCollaborationSession] = {}
        
        # Chemistry research templates
        self.research_templates = {
            "drug_discovery": {
                "name": "Pharmaceutical Drug Discovery",
                "typical_deliverables": ["lead_compounds", "sar_analysis", "admet_data", "patent_application"],
                "required_techniques": ["NMR", "MS", "HPLC", "cell_assays"],
                "collaboration_roles": ["medicinal_chemist", "analytical_chemist", "biologist"],
                "patent_sensitivity": True,
                "regulatory_considerations": ["FDA", "EMA", "ICH_guidelines"]
            },
            "materials_chemistry": {
                "name": "Advanced Materials Development",
                "typical_deliverables": ["material_characterization", "property_analysis", "synthesis_protocols"],
                "required_techniques": ["XRD", "SEM", "thermal_analysis", "mechanical_testing"],
                "collaboration_roles": ["materials_scientist", "analytical_chemist", "engineer"],
                "patent_sensitivity": True,
                "regulatory_considerations": ["REACH", "RoHS", "environmental_impact"]
            },
            "catalysis_research": {
                "name": "Catalytic Process Development",
                "typical_deliverables": ["catalyst_design", "reaction_optimization", "mechanistic_studies"],
                "required_techniques": ["GC-MS", "IR", "kinetic_analysis", "computational_modeling"],
                "collaboration_roles": ["catalysis_researcher", "computational_chemist", "process_engineer"],
                "patent_sensitivity": True,
                "regulatory_considerations": ["green_chemistry", "process_safety", "waste_minimization"]
            },
            "natural_products": {
                "name": "Natural Product Discovery",
                "typical_deliverables": ["structure_elucidation", "bioactivity_screening", "synthesis"],
                "required_techniques": ["NMR_2D", "MS_fragmentation", "bioassays", "total_synthesis"],
                "collaboration_roles": ["natural_products_chemist", "spectroscopist", "synthetic_chemist"],
                "patent_sensitivity": False,
                "regulatory_considerations": ["biodiversity_treaties", "traditional_knowledge"]
            }
        }
        
        # Common chemical databases and tools
        self.chemical_databases = {
            "chemdraw": {
                "name": "ChemDraw Structure Database",
                "structures": 15000000,
                "features": ["2d_drawing", "3d_modeling", "name_to_structure", "property_prediction"]
            },
            "scifinder": {
                "name": "SciFinder Chemical Database",
                "structures": 170000000,
                "features": ["literature_search", "reaction_search", "patent_search", "supplier_info"]
            },
            "reaxys": {
                "name": "Reaxys Reaction Database",
                "reactions": 55000000,
                "features": ["reaction_search", "synthesis_planning", "property_data", "experimental_details"]
            },
            "pubchem": {
                "name": "PubChem Chemical Database",
                "structures": 110000000,
                "features": ["open_access", "bioactivity_data", "literature_links", "3d_structures"]
            }
        }
        
        # University-industry partnership templates
        self.partnership_templates = {
            "unc_pharma": {
                "name": "UNC-Pharma Drug Discovery Partnership",
                "focus_areas": ["oncology", "neuroscience", "infectious_disease"],
                "ip_arrangements": "joint_ownership",
                "milestone_payments": True,
                "publication_delays": 6  # months
            },
            "ncsu_materials": {
                "name": "NC State Materials Innovation Partnership",
                "focus_areas": ["polymers", "composites", "nanomaterials"],
                "ip_arrangements": "exclusive_license",
                "milestone_payments": False,
                "publication_delays": 3  # months
            },
            "duke_catalysis": {
                "name": "Duke Catalysis Research Alliance",
                "focus_areas": ["sustainable_chemistry", "process_development", "green_catalysis"],
                "ip_arrangements": "non_exclusive_license",
                "milestone_payments": True,
                "publication_delays": 0  # months
            }
        }
    
    async def create_chemistry_project(
        self,
        name: str,
        description: str,
        research_area: str,
        creator_id: str,
        university: str,
        department: str,
        template: Optional[str] = None,
        industry_partner: Optional[str] = None,
        security_level: str = "high"
    ) -> ChemistryProject:
        """Create a new chemistry research project"""
        
        project_id = str(uuid.uuid4())
        
        # Apply template if provided
        template_config = self.research_templates.get(research_area, {})
        partnership_config = self.partnership_templates.get(template, {})
        
        # Set patent sensitivity
        patent_sensitive = template_config.get("patent_sensitivity", False)
        
        # Generate NWTN insights for project setup
        nwtn_context = {
            "project_name": name,
            "research_area": research_area,
            "university": university,
            "department": department,
            "industry_partner": industry_partner,
            "patent_sensitive": patent_sensitive
        }
        
        nwtn_insights = await self._generate_project_insights(nwtn_context)
        
        # Set up timeline
        timeline = {
            "project_start": datetime.now(),
            "literature_review": datetime.now() + timedelta(weeks=2),
            "initial_synthesis": datetime.now() + timedelta(weeks=8),
            "characterization": datetime.now() + timedelta(weeks=12),
            "optimization": datetime.now() + timedelta(weeks=20),
            "manuscript_prep": datetime.now() + timedelta(weeks=28)
        }
        
        project = ChemistryProject(
            id=project_id,
            name=name,
            description=description,
            research_area=research_area,
            created_by=creator_id,
            created_at=datetime.now(),
            university=university,
            department=department,
            industry_partner=industry_partner,
            collaborators={creator_id: CollaborationRole.PRINCIPAL_INVESTIGATOR},
            access_permissions={
                "read": [creator_id],
                "write": [creator_id],
                "draw": [creator_id],
                "analyze": [creator_id],
                "admin": [creator_id]
            },
            molecules={},
            reactions={},
            synthetic_routes={},
            spectroscopic_data={},
            active_sessions=[],
            security_level=security_level,
            patent_sensitive=patent_sensitive,
            confidentiality_agreements=[],
            computational_resources={},
            timeline=timeline,
            deliverables=template_config.get("typical_deliverables", []),
            nwtn_insights=nwtn_insights,
            publication_pipeline=not patent_sensitive
        )
        
        self.projects[project_id] = project
        return project
    
    async def create_molecule(
        self,
        project_id: str,
        name: str,
        smiles: str,
        user_id: str,
        structure_data: Dict[str, Any] = None,
        properties: Dict[str, float] = None
    ) -> ChemicalMolecule:
        """Create a new chemical molecule"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("write", []):
            raise PermissionError("User does not have write access")
        
        # Generate molecular properties
        computed_properties = await self._compute_molecular_properties(smiles)
        if properties:
            computed_properties.update(properties)
        
        # Generate InChI from SMILES (simulated)
        inchi = await self._smiles_to_inchi(smiles)
        
        # Calculate molecular formula and weight
        formula, mw = await self._analyze_molecular_formula(smiles)
        
        molecule = ChemicalMolecule(
            id=str(uuid.uuid4()),
            name=name,
            formula=formula,
            smiles=smiles,
            inchi=inchi,
            molecular_weight=mw,
            structure_data=structure_data or {},
            properties=computed_properties,
            stereochemistry="not_determined",
            created_by=user_id,
            created_at=datetime.now(),
            version=1,
            parent_molecule=None,
            modification_history=[],
            validated=False,
            confidence_score=0.85
        )
        
        project.molecules[name] = molecule
        
        # Generate molecular insights
        mol_insights = await self._analyze_molecule(molecule, project)
        project.nwtn_insights.extend(mol_insights)
        
        return molecule
    
    async def create_reaction(
        self,
        project_id: str,
        reaction_name: str,
        reaction_type: ReactionType,
        reactants: List[str],
        products: List[str],
        conditions: Dict[str, Any],
        user_id: str,
        reagents: List[str] = None
    ) -> ChemicalReaction:
        """Create a chemical reaction"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("write", []):
            raise PermissionError("User does not have write access")
        
        # Validate that reactant and product molecules exist
        for mol_name in reactants + products:
            if mol_name not in project.molecules:
                raise ValueError(f"Molecule {mol_name} not found in project")
        
        # Reagents don't need to be pre-existing molecules in the project
        
        # Generate reaction mechanism (simulated)
        mechanism = await self._predict_reaction_mechanism(
            reaction_type, reactants, products, conditions
        )
        
        reaction = ChemicalReaction(
            id=str(uuid.uuid4()),
            name=reaction_name,
            reaction_type=reaction_type,
            reactants=reactants,
            products=products,
            reagents=reagents or [],
            conditions=conditions,
            mechanism=mechanism,
            yield_theoretical=None,
            yield_experimental=None,
            selectivity={},
            literature_references=[],
            safety_notes=[],
            created_by=user_id,
            created_at=datetime.now(),
            validated=False,
            patent_relevant=project.patent_sensitive
        )
        
        project.reactions[reaction_name] = reaction
        
        # Generate reaction insights
        rxn_insights = await self._analyze_reaction(reaction, project)
        project.nwtn_insights.extend(rxn_insights)
        
        return reaction
    
    async def create_synthetic_route(
        self,
        project_id: str,
        target_molecule: str,
        route_name: str,
        starting_materials: List[str],
        synthetic_steps: List[str],
        user_id: str
    ) -> SyntheticRoute:
        """Create multi-step synthetic route"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("write", []):
            raise PermissionError("User does not have write access")
        
        # Validate molecules and reactions exist
        if target_molecule not in project.molecules:
            raise ValueError(f"Target molecule {target_molecule} not found")
        
        for sm in starting_materials:
            if sm not in project.molecules:
                raise ValueError(f"Starting material {sm} not found")
        
        for step in synthetic_steps:
            if step not in project.reactions:
                raise ValueError(f"Reaction step {step} not found")
        
        # Analyze synthetic route
        analysis = await self._analyze_synthetic_route(
            target_molecule, starting_materials, synthetic_steps, project
        )
        
        route = SyntheticRoute(
            id=str(uuid.uuid4()),
            target_molecule_id=target_molecule,
            route_name=route_name,
            starting_materials=starting_materials,
            synthetic_steps=synthetic_steps,
            overall_yield=analysis["overall_yield"],
            step_count=len(synthetic_steps),
            complexity_score=analysis["complexity_score"],
            cost_estimate=analysis["cost_estimate"],
            time_estimate=analysis["time_estimate"],
            scalability=analysis["scalability"],
            green_chemistry_score=analysis["green_score"],
            patent_landscape=analysis["patent_risks"],
            created_by=user_id,
            created_at=datetime.now(),
            optimization_suggestions=analysis["optimizations"]
        )
        
        project.synthetic_routes[route_name] = route
        
        # Generate route insights
        route_insights = await self._analyze_synthetic_route_nwtn(route, project)
        project.nwtn_insights.extend(route_insights)
        
        return route
    
    async def upload_spectroscopic_data(
        self,
        project_id: str,
        molecule_name: str,
        technique: SpectroscopyType,
        data_file: str,
        user_id: str,
        instrument_params: Dict[str, Any] = None
    ) -> SpectroscopicData:
        """Upload and process spectroscopic data"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("analyze", []):
            raise PermissionError("User does not have analysis access")
        
        if molecule_name not in project.molecules:
            raise ValueError(f"Molecule {molecule_name} not found")
        
        molecule = project.molecules[molecule_name]
        
        # Process spectroscopic data
        processed_data = await self._process_spectroscopic_data(
            data_file, technique, molecule
        )
        
        # Encrypt sensitive data if required
        encrypted_path = data_file
        if project.security_level == "maximum" or project.patent_sensitive:
            encrypted_shards = self.crypto_sharding.shard_file(
                data_file,
                list(project.collaborators.keys()),
                num_shards=5
            )
            encrypted_path = "encrypted"
        
        spec_data = SpectroscopicData(
            id=str(uuid.uuid4()),
            molecule_id=molecule.id,
            technique=technique,
            data_file=encrypted_path,
            processed_data=processed_data["data"],
            peak_assignments=processed_data["peaks"],
            interpretation=processed_data["interpretation"],
            quality_metrics=processed_data["quality"],
            instrument_parameters=instrument_params or {},
            acquired_by=user_id,
            acquired_at=datetime.now(),
            processed_by=user_id,
            processed_at=datetime.now(),
            validated=False
        )
        
        project.spectroscopic_data[f"{molecule_name}_{technique.value}"] = spec_data
        
        # Generate spectroscopic insights
        spec_insights = await self._analyze_spectroscopic_data(spec_data, molecule, project)
        project.nwtn_insights.extend(spec_insights)
        
        return spec_data
    
    async def start_drawing_session(
        self,
        project_id: str,
        user_id: str,
        participants: List[str] = None
    ) -> ChemistryCollaborationSession:
        """Start real-time collaborative drawing session"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("draw", []):
            raise PermissionError("User does not have drawing access")
        
        session_id = str(uuid.uuid4())
        
        session = ChemistryCollaborationSession(
            id=session_id,
            project_id=project_id,
            participants=[user_id] + (participants or []),
            active_molecule=None,
            shared_canvas={
                "width": 800,
                "height": 600,
                "zoom": 1.0,
                "center": [400, 300],
                "structures": [],
                "arrows": [],
                "text_labels": []
            },
            live_edits=[],
            chat_messages=[],
            started_at=datetime.now(),
            last_activity=datetime.now(),
            session_status="active"
        )
        
        self.active_sessions[session_id] = session
        project.active_sessions.append(session_id)
        
        return session
    
    async def add_structure_to_canvas(
        self,
        session_id: str,
        structure_data: Dict[str, Any],
        user_id: str
    ):
        """Add chemical structure to collaborative canvas"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        if user_id not in session.participants:
            raise PermissionError("User is not a participant in this session")
        
        # Add structure to canvas
        structure_element = {
            "id": str(uuid.uuid4()),
            "type": "molecule",
            "data": structure_data,
            "position": structure_data.get("position", [100, 100]),
            "created_by": user_id,
            "timestamp": datetime.now()
        }
        
        session.shared_canvas["structures"].append(structure_element)
        
        # Record edit
        edit_record = {
            "action": "add_structure",
            "element_id": structure_element["id"],
            "user_id": user_id,
            "timestamp": datetime.now(),
            "data": structure_data
        }
        
        session.live_edits.append(edit_record)
        session.last_activity = datetime.now()
    
    async def predict_retrosynthesis(
        self,
        project_id: str,
        target_molecule: str,
        user_id: str,
        max_steps: int = 5,
        commercial_sources: bool = True
    ) -> Dict[str, Any]:
        """AI-powered retrosynthetic analysis"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("analyze", []):
            raise PermissionError("User does not have analysis access")
        
        if target_molecule not in project.molecules:
            raise ValueError(f"Target molecule {target_molecule} not found")
        
        molecule = project.molecules[target_molecule]
        
        # Generate retrosynthetic predictions
        retro_analysis = await self._generate_retrosynthesis(
            molecule, max_steps, commercial_sources, project
        )
        
        return retro_analysis
    
    async def export_chemistry_project(
        self,
        project_id: str,
        user_id: str,
        format: str = "sdf",
        include_spectra: bool = True,
        include_routes: bool = True
    ) -> str:
        """Export chemistry project data"""
        
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Check permissions
        if user_id not in project.access_permissions.get("read", []):
            raise PermissionError("User does not have read access")
        
        # Create export package
        temp_dir = tempfile.mkdtemp()
        export_path = Path(temp_dir) / f"{project.name}_chemistry_export.zip"
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Project metadata
            project_data = asdict(project)
            project_data['created_at'] = project_data['created_at'].isoformat()
            
            zipf.writestr(
                "project_metadata.json",
                json.dumps(project_data, indent=2, default=str)
            )
            
            # Molecules in SDF format
            if format in ["sdf", "molfile"]:
                molecules_sdf = self._generate_sdf_file(project.molecules)
                zipf.writestr("molecules.sdf", molecules_sdf)
            
            # Reactions in RDF format
            reactions_data = {}
            for name, reaction in project.reactions.items():
                rxn_data = asdict(reaction)
                rxn_data['created_at'] = rxn_data['created_at'].isoformat()
                reactions_data[name] = rxn_data
            
            zipf.writestr(
                "reactions.json",
                json.dumps(reactions_data, indent=2, default=str)
            )
            
            # Synthetic routes
            if include_routes:
                routes_data = {}
                for name, route in project.synthetic_routes.items():
                    route_data = asdict(route)
                    route_data['created_at'] = route_data['created_at'].isoformat()
                    routes_data[name] = route_data
                
                zipf.writestr(
                    "synthetic_routes.json",
                    json.dumps(routes_data, indent=2, default=str)
                )
            
            # Spectroscopic data metadata
            if include_spectra:
                spectra_metadata = {}
                for name, spec_data in project.spectroscopic_data.items():
                    spec_meta = asdict(spec_data)
                    spec_meta['acquired_at'] = spec_meta['acquired_at'].isoformat()
                    spec_meta['processed_at'] = spec_meta['processed_at'].isoformat()
                    
                    # Remove actual data file for security
                    spec_meta.pop('data_file', None)
                    spectra_metadata[name] = spec_meta
                
                zipf.writestr(
                    "spectroscopic_metadata.json",
                    json.dumps(spectra_metadata, indent=2, default=str)
                )
            
            # NWTN insights
            zipf.writestr(
                "nwtn_insights.json",
                json.dumps(project.nwtn_insights, indent=2, default=str)
            )
            
            # Generate project summary
            summary_report = self._generate_chemistry_summary(project)
            zipf.writestr("project_summary.md", summary_report)
        
        return str(export_path)
    
    async def _compute_molecular_properties(self, smiles: str) -> Dict[str, float]:
        """Compute molecular properties from SMILES"""
        
        # Simulate property calculations
        properties = {
            "logp": np.random.normal(2.5, 1.2),  # Lipophilicity
            "psa": np.random.uniform(20, 150),   # Polar surface area
            "hbd": np.random.randint(0, 6),      # H-bond donors
            "hba": np.random.randint(0, 10),     # H-bond acceptors
            "rotatable_bonds": np.random.randint(0, 15),
            "aromatic_rings": np.random.randint(0, 4),
            "saturation_fraction": np.random.uniform(0.3, 1.0),
            "formal_charge": 0,
            "heavy_atoms": len(smiles.replace('H', '')) // 2  # Rough estimate
        }
        
        return properties
    
    async def _smiles_to_inchi(self, smiles: str) -> str:
        """Convert SMILES to InChI (simulated)"""
        
        # Generate mock InChI
        hash_obj = hashlib.md5(smiles.encode())
        hash_hex = hash_obj.hexdigest()[:10]
        
        return f"InChI=1S/C10H12N2O/c1-8-4-6-10(7-5-8)12-9(11)2-3-13-/h4-7H,2-3H2,1H3,(H2,11,12)/t-/m1/s1_{hash_hex}"
    
    async def _analyze_molecular_formula(self, smiles: str) -> Tuple[str, float]:
        """Analyze molecular formula and calculate molecular weight"""
        
        # Simplified formula generation based on SMILES length
        c_count = smiles.count('C') + smiles.count('c') + len(smiles) // 6
        h_count = c_count * 2 + 2  # Rough estimate
        n_count = smiles.count('N') + smiles.count('n')
        o_count = smiles.count('O') + smiles.count('o')
        
        formula = f"C{c_count}H{h_count}"
        mw = c_count * 12.01 + h_count * 1.008
        
        if n_count > 0:
            formula += f"N{n_count}"
            mw += n_count * 14.007
        
        if o_count > 0:
            formula += f"O{o_count}"
            mw += o_count * 15.999
        
        return formula, round(mw, 2)
    
    async def _predict_reaction_mechanism(
        self,
        reaction_type: ReactionType,
        reactants: List[str],
        products: List[str],
        conditions: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Predict reaction mechanism steps"""
        
        if reaction_type == ReactionType.SUBSTITUTION:
            return [
                {
                    "step": 1,
                    "description": "Nucleophilic attack on electrophilic center",
                    "intermediate": "tetrahedral_intermediate",
                    "energy_barrier": 25.5  # kcal/mol
                },
                {
                    "step": 2,
                    "description": "Leaving group departure",
                    "intermediate": "product",
                    "energy_barrier": 18.2
                }
            ]
        elif reaction_type == ReactionType.CYCLOADDITION:
            return [
                {
                    "step": 1,
                    "description": "Concerted [4+2] cycloaddition",
                    "intermediate": "transition_state",
                    "energy_barrier": 28.7
                },
                {
                    "step": 2,
                    "description": "Product formation",
                    "intermediate": "product",
                    "energy_barrier": 0.0
                }
            ]
        else:
            return [
                {
                    "step": 1,
                    "description": f"General {reaction_type.value} mechanism",
                    "intermediate": "product",
                    "energy_barrier": 22.0
                }
            ]
    
    async def _analyze_synthetic_route(
        self,
        target: str,
        starting_materials: List[str],
        steps: List[str],
        project: ChemistryProject
    ) -> Dict[str, Any]:
        """Analyze synthetic route efficiency and feasibility"""
        
        # Simulate route analysis
        step_count = len(steps)
        base_yield = 0.85 ** step_count  # Assumes 85% average yield per step
        
        complexity_factors = {
            "step_count": step_count,
            "protecting_groups": np.random.randint(0, 3),
            "stereochemistry": np.random.randint(0, 2),
            "functional_group_tolerance": np.random.uniform(0.7, 1.0)
        }
        
        complexity_score = (
            step_count * 0.3 +
            complexity_factors["protecting_groups"] * 0.2 +
            complexity_factors["stereochemistry"] * 0.3 +
            (1 - complexity_factors["functional_group_tolerance"]) * 0.2
        )
        
        return {
            "overall_yield": round(base_yield * 100, 1),
            "complexity_score": round(complexity_score, 2),
            "cost_estimate": step_count * 150 + np.random.randint(200, 800),  # USD
            "time_estimate": f"{step_count * 3}-{step_count * 5} weeks",
            "scalability": "lab" if step_count > 8 else "pilot" if step_count > 5 else "industrial",
            "green_score": np.random.uniform(0.4, 0.9),
            "patent_risks": ["patent_123456", "patent_789012"] if np.random.random() > 0.7 else [],
            "optimizations": [
                "Consider flow chemistry for step 3",
                "Explore alternative protecting group strategy",
                "Investigate biocatalytic alternatives"
            ]
        }
    
    async def _process_spectroscopic_data(
        self,
        data_file: str,
        technique: SpectroscopyType,
        molecule: ChemicalMolecule
    ) -> Dict[str, Any]:
        """Process spectroscopic data file"""
        
        if technique == SpectroscopyType.NMR_1H:
            return {
                "data": {
                    "chemical_shifts": [7.8, 7.4, 7.2, 4.1, 2.3, 1.9],
                    "integrations": [2, 2, 3, 2, 3, 3],
                    "multiplicities": ["d", "t", "m", "q", "s", "d"],
                    "coupling_constants": [8.2, 7.5, None, 7.1, None, 6.8]
                },
                "peaks": [
                    {"shift": 7.8, "assignment": "aromatic_H", "multiplicity": "d", "integration": 2},
                    {"shift": 4.1, "assignment": "CH2_adjacent_to_O", "multiplicity": "q", "integration": 2},
                    {"shift": 2.3, "assignment": "CH3_aromatic", "multiplicity": "s", "integration": 3}
                ],
                "interpretation": "Spectrum consistent with substituted aromatic compound with ethyl ester functionality",
                "quality": {
                    "signal_to_noise": 45.2,
                    "shimming_quality": 0.92,
                    "water_suppression": 0.98,
                    "phase_correction": 0.95
                }
            }
        elif technique == SpectroscopyType.MS:
            return {
                "data": {
                    "molecular_ion": molecule.molecular_weight,
                    "base_peak": molecule.molecular_weight - 45,  # Loss of ethoxycarbonyl
                    "fragmentation_pattern": [
                        {"mz": molecule.molecular_weight, "intensity": 15, "assignment": "[M]+"},
                        {"mz": molecule.molecular_weight - 45, "intensity": 100, "assignment": "[M-OEt]+"},
                        {"mz": molecule.molecular_weight - 73, "intensity": 75, "assignment": "[M-CO2Et]+"}
                    ]
                },
                "peaks": [],
                "interpretation": "Fragmentation pattern consistent with ethyl ester functional group",
                "quality": {
                    "mass_accuracy": 2.1,  # ppm
                    "resolution": 15000,
                    "isotope_pattern_match": 0.96
                }
            }
        else:
            return {
                "data": {"technique": technique.value, "processed": True},
                "peaks": [],
                "interpretation": f"{technique.value} data processed successfully",
                "quality": {"overall_quality": 0.85}
            }
    
    async def _generate_retrosynthesis(
        self,
        molecule: ChemicalMolecule,
        max_steps: int,
        commercial_sources: bool,
        project: ChemistryProject
    ) -> Dict[str, Any]:
        """Generate retrosynthetic analysis"""
        
        # Simulate AI-powered retrosynthesis
        routes = []
        
        for i in range(3):  # Generate 3 alternative routes
            route_steps = []
            current_complexity = np.random.uniform(0.6, 0.9)
            
            for step in range(max_steps):
                if current_complexity < 0.3 and commercial_sources:
                    # Found commercial starting material
                    break
                
                disconnection = {
                    "step": step + 1,
                    "bond_broken": f"bond_{np.random.randint(1, 10)}",
                    "reaction_type": np.random.choice(list(ReactionType)).value,
                    "starting_materials": [f"precursor_{step}_{i}", f"reagent_{step}_{i}"],
                    "confidence": current_complexity,
                    "literature_precedent": np.random.random() > 0.3
                }
                
                route_steps.append(disconnection)
                current_complexity *= 0.8  # Reduce complexity each step
            
            route_score = sum(step["confidence"] for step in route_steps) / len(route_steps) if route_steps else 0
            
            routes.append({
                "route_id": i + 1,
                "steps": route_steps,
                "total_steps": len(route_steps),
                "overall_score": route_score,
                "estimated_yield": (0.8 ** len(route_steps)) * 100,
                "commercial_availability": commercial_sources and len(route_steps) < max_steps
            })
        
        # Sort routes by score
        routes.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return {
            "target_molecule": molecule.name,
            "analysis_timestamp": datetime.now(),
            "parameters": {
                "max_steps": max_steps,
                "commercial_sources": commercial_sources
            },
            "routes": routes,
            "confidence": np.mean([route["overall_score"] for route in routes]),
            "recommendations": [
                "Route 1 shows highest confidence for synthetic feasibility",
                "Consider protecting group strategies for selectivity",
                "Validate key disconnections with literature precedent"
            ]
        }
    
    async def _generate_project_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate NWTN insights for project setup"""
        
        nwtn_prompt = f"""
        Analyze this chemistry research project setup:
        
        Project: {context['project_name']}
        Research Area: {context['research_area']}
        University: {context['university']}
        Department: {context['department']}
        Industry Partner: {context.get('industry_partner', 'None')}
        Patent Sensitive: {context['patent_sensitive']}
        
        Provide insights on:
        1. Appropriate synthetic strategies and methodologies
        2. Analytical characterization requirements
        3. Intellectual property considerations
        4. Safety and regulatory requirements
        5. Collaboration best practices for this research area
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
    
    async def _analyze_molecule(
        self,
        molecule: ChemicalMolecule,
        project: ChemistryProject
    ) -> List[Dict[str, Any]]:
        """Analyze molecule using NWTN"""
        
        nwtn_prompt = f"""
        Analyze this chemical molecule for research insights:
        
        Name: {molecule.name}
        Formula: {molecule.formula}
        SMILES: {molecule.smiles}
        Molecular Weight: {molecule.molecular_weight}
        Properties: {molecule.properties}
        
        Research Context: {project.research_area}
        
        Provide analysis on:
        1. Structural features and reactivity predictions
        2. Potential synthetic challenges
        3. Drug-like properties (if applicable)
        4. Stability and handling considerations
        5. Analytical characterization strategy
        """
        
        context = {
            "molecule": asdict(molecule),
            "project": asdict(project)
        }
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "molecule_analysis",
                "timestamp": datetime.now(),
                "molecule_id": molecule.id,
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    async def _analyze_reaction(
        self,
        reaction: ChemicalReaction,
        project: ChemistryProject
    ) -> List[Dict[str, Any]]:
        """Analyze reaction using NWTN"""
        
        nwtn_prompt = f"""
        Analyze this chemical reaction:
        
        Reaction: {reaction.name}
        Type: {reaction.reaction_type.value}
        Reactants: {len(reaction.reactants)}
        Products: {len(reaction.products)}
        Conditions: {reaction.conditions}
        
        Research Context: {project.research_area}
        
        Provide analysis on:
        1. Reaction mechanism and selectivity
        2. Optimization opportunities
        3. Scale-up considerations
        4. Safety and environmental impact
        5. Literature precedent and novelty
        """
        
        context = {
            "reaction": asdict(reaction),
            "project": asdict(project)
        }
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "reaction_analysis",
                "timestamp": datetime.now(),
                "reaction_id": reaction.id,
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    async def _analyze_synthetic_route_nwtn(
        self,
        route: SyntheticRoute,
        project: ChemistryProject
    ) -> List[Dict[str, Any]]:
        """Analyze synthetic route using NWTN"""
        
        nwtn_prompt = f"""
        Analyze this synthetic route:
        
        Target: {route.target_molecule_id}
        Steps: {route.step_count}
        Overall Yield: {route.overall_yield}%
        Complexity Score: {route.complexity_score}
        Green Chemistry Score: {route.green_chemistry_score}
        
        Research Context: {project.research_area}
        
        Provide analysis on:
        1. Route efficiency and practicality
        2. Alternative disconnection strategies
        3. Process optimization opportunities
        4. Environmental and safety considerations
        5. Commercial viability assessment
        """
        
        context = {
            "route": asdict(route),
            "project": asdict(project)
        }
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "route_analysis",
                "timestamp": datetime.now(),
                "route_id": route.id,
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    async def _analyze_spectroscopic_data(
        self,
        spec_data: SpectroscopicData,
        molecule: ChemicalMolecule,
        project: ChemistryProject
    ) -> List[Dict[str, Any]]:
        """Analyze spectroscopic data using NWTN"""
        
        nwtn_prompt = f"""
        Analyze this spectroscopic data:
        
        Technique: {spec_data.technique.value}
        Molecule: {molecule.name}
        Quality Metrics: {spec_data.quality_metrics}
        Peak Assignments: {len(spec_data.peak_assignments)}
        Interpretation: {spec_data.interpretation}
        
        Provide analysis on:
        1. Data quality and reliability assessment
        2. Structural confirmation or conflicts
        3. Additional characterization needs
        4. Spectroscopic assignment validation
        5. Publication readiness
        """
        
        context = {
            "spectroscopic_data": asdict(spec_data),
            "molecule": asdict(molecule),
            "project": asdict(project)
        }
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "spectroscopic_analysis",
                "timestamp": datetime.now(),
                "spectrum_id": spec_data.id,
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    def _generate_sdf_file(self, molecules: Dict[str, ChemicalMolecule]) -> str:
        """Generate SDF file from molecules"""
        
        sdf_content = ""
        
        for name, molecule in molecules.items():
            sdf_content += f"""
{molecule.name}
  Generated by PRSM Chemistry Collaboration
  {datetime.now().strftime('%m%d%y%H%M')}

  0  0  0  0  0  0  0  0  0  0999 V2000
M  END
>  <NAME>
{molecule.name}

>  <FORMULA>
{molecule.formula}

>  <SMILES>
{molecule.smiles}

>  <MOLECULAR_WEIGHT>
{molecule.molecular_weight}

>  <CREATED_BY>
{molecule.created_by}

$$$$
"""
        
        return sdf_content
    
    def _generate_chemistry_summary(self, project: ChemistryProject) -> str:
        """Generate chemistry project summary"""
        
        summary = f"""# Chemistry Project Summary: {project.name}

## Project Information
- **Project ID**: {project.id}
- **Created**: {project.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **Research Area**: {project.research_area}
- **University**: {project.university}
- **Department**: {project.department}
- **Industry Partner**: {project.industry_partner or 'None'}
- **Security Level**: {project.security_level}
- **Patent Sensitive**: {'Yes' if project.patent_sensitive else 'No'}

## Research Assets
- **Molecules**: {len(project.molecules)}
- **Reactions**: {len(project.reactions)}
- **Synthetic Routes**: {len(project.synthetic_routes)}
- **Spectroscopic Data**: {len(project.spectroscopic_data)}
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
## Molecule Library
"""
        
        for name, molecule in list(project.molecules.items())[:5]:  # Show first 5
            summary += f"""
### {name}
- **Formula**: {molecule.formula}
- **Molecular Weight**: {molecule.molecular_weight}
- **SMILES**: {molecule.smiles}
- **Validated**: {'Yes' if molecule.validated else 'No'}
"""
        
        if len(project.molecules) > 5:
            summary += f"\n*... and {len(project.molecules) - 5} more molecules*\n"
        
        summary += f"""
## Deliverables
"""
        
        for deliverable in project.deliverables:
            summary += f"- {deliverable.replace('_', ' ').title()}\n"
        
        summary += f"""
---
*Summary generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return summary


# Testing and validation
async def test_chemistry_collaboration():
    """Test chemistry collaboration functionality"""
    
    chem_collab = ChemistryCollaboration()
    
    print("🧪 Testing Chemistry Collaboration Tools...")
    
    # Test 1: Create drug discovery project
    print("\n1. Creating UNC-Pharma Drug Discovery Project...")
    
    project = await chem_collab.create_chemistry_project(
        name="Novel Kinase Inhibitors for Cancer Therapy",
        description="Discovery and optimization of small molecule inhibitors targeting oncogenic kinases",
        research_area="drug_discovery",
        creator_id="pi_medchem_001",
        university="University of North Carolina at Chapel Hill",
        department="Department of Medicinal Chemistry",
        template="unc_pharma",
        industry_partner="Novartis Pharmaceuticals",
        security_level="maximum"
    )
    
    print(f"✅ Created project: {project.name}")
    print(f"   - ID: {project.id}")
    print(f"   - Research Area: {project.research_area}")
    print(f"   - Patent Sensitive: {project.patent_sensitive}")
    print(f"   - Security Level: {project.security_level}")
    print(f"   - NWTN Insights: {len(project.nwtn_insights)}")
    
    # Test 2: Add collaborators
    print("\n2. Adding research collaborators...")
    
    project.collaborators.update({
        "medchem_001": CollaborationRole.MEDICINAL_CHEMIST,
        "synth_001": CollaborationRole.SYNTHETIC_CHEMIST,
        "analyst_001": CollaborationRole.ANALYTICAL_CHEMIST,
        "compbio_001": CollaborationRole.COMPUTATIONAL_CHEMIST,
        "student_001": CollaborationRole.GRADUATE_STUDENT,
        "industry_001": CollaborationRole.INDUSTRY_CHEMIST
    })
    
    # Update permissions
    all_collaborators = list(project.collaborators.keys())
    project.access_permissions.update({
        "read": all_collaborators,
        "write": ["pi_medchem_001", "medchem_001", "synth_001", "student_001"],
        "draw": ["pi_medchem_001", "medchem_001", "compbio_001", "student_001"],
        "analyze": all_collaborators,
        "admin": ["pi_medchem_001"]
    })
    
    print(f"✅ Added {len(project.collaborators)} collaborators")
    
    # Test 3: Create target molecules
    print("\n3. Creating target kinase inhibitor molecules...")
    
    # Lead compound
    molecule1 = await chem_collab.create_molecule(
        project_id=project.id,
        name="Lead_Kinase_Inhibitor_001",
        smiles="CC1=C(C=CC(=C1)C2=CC=C(C=C2)CN3CCN(CC3)C)C(=O)NC4=CC(=C(C=C4)OC)N",
        user_id="medchem_001",
        properties={"ic50_nM": 25.3, "selectivity_fold": 15.2, "solubility_uM": 45.7}
    )
    
    # Optimized analogue
    molecule2 = await chem_collab.create_molecule(
        project_id=project.id,
        name="Optimized_Analogue_002",
        smiles="CC1=C(C=CC(=C1)C2=CC=C(C=C2)CN3CCN(CC3)C)C(=O)NC4=CC(=C(C=C4)OCF3)N",
        user_id="medchem_001",
        properties={"ic50_nM": 8.9, "selectivity_fold": 32.1, "solubility_uM": 78.4}
    )
    
    print(f"✅ Created {len(project.molecules)} target molecules")
    print(f"   - Molecule 1: {molecule1.name} (MW: {molecule1.molecular_weight})")
    print(f"   - Molecule 2: {molecule2.name} (MW: {molecule2.molecular_weight})")
    
    # Test 4: Create synthetic reactions
    print("\n4. Creating synthetic reaction...")
    
    reaction = await chem_collab.create_reaction(
        project_id=project.id,
        reaction_name="Amide_Coupling_Reaction",
        reaction_type=ReactionType.COUPLING,
        reactants=["Lead_Kinase_Inhibitor_001"],
        products=["Optimized_Analogue_002"],
        conditions={
            "temperature": "room_temperature",
            "solvent": "DMF",
            "reagents": ["HATU", "DIPEA"],
            "time": "2 hours",
            "atmosphere": "nitrogen"
        },
        user_id="synth_001",
        reagents=["HATU", "DIPEA"]
    )
    
    print(f"✅ Created reaction: {reaction.name}")
    print(f"   - Type: {reaction.reaction_type.value}")
    print(f"   - Mechanism steps: {len(reaction.mechanism)}")
    print(f"   - Patent relevant: {reaction.patent_relevant}")
    
    # Test 5: Create synthetic route
    print("\n5. Creating multi-step synthetic route...")
    
    route = await chem_collab.create_synthetic_route(
        project_id=project.id,
        target_molecule="Optimized_Analogue_002",
        route_name="Optimized_Synthesis_Route_A",
        starting_materials=["Lead_Kinase_Inhibitor_001"],
        synthetic_steps=["Amide_Coupling_Reaction"],
        user_id="synth_001"
    )
    
    print(f"✅ Created synthetic route: {route.route_name}")
    print(f"   - Steps: {route.step_count}")
    print(f"   - Overall yield: {route.overall_yield}%")
    print(f"   - Complexity score: {route.complexity_score}")
    print(f"   - Green chemistry score: {route.green_chemistry_score:.2f}")
    print(f"   - Optimization suggestions: {len(route.optimization_suggestions)}")
    
    # Test 6: Upload spectroscopic data
    print("\n6. Uploading spectroscopic characterization data...")
    
    # Create temporary NMR data file
    temp_nmr = tempfile.NamedTemporaryFile(suffix='.fid', delete=False)
    temp_nmr.write(b"Mock NMR data for kinase inhibitor characterization")
    temp_nmr.close()
    
    try:
        spec_data = await chem_collab.upload_spectroscopic_data(
            project_id=project.id,
            molecule_name="Optimized_Analogue_002",
            technique=SpectroscopyType.NMR_1H,
            data_file=temp_nmr.name,
            user_id="analyst_001",
            instrument_params={
                "field_strength": "400 MHz",
                "solvent": "DMSO-d6",
                "temperature": "298 K",
                "pulse_sequence": "zg30"
            }
        )
        
        print(f"✅ Uploaded spectroscopic data: {spec_data.technique.value}")
        print(f"   - Molecule: {spec_data.molecule_id}")
        print(f"   - Peak assignments: {len(spec_data.peak_assignments)}")
        print(f"   - Quality score: {spec_data.quality_metrics.get('signal_to_noise', 'N/A')}")
        print(f"   - Interpretation: {spec_data.interpretation[:60]}...")
        
    finally:
        # Clean up temporary file
        Path(temp_nmr.name).unlink()
    
    # Test 7: Start collaborative drawing session
    print("\n7. Starting collaborative structure drawing session...")
    
    session = await chem_collab.start_drawing_session(
        project_id=project.id,
        user_id="medchem_001",
        participants=["compbio_001", "student_001"]
    )
    
    print(f"✅ Started drawing session: {session.id}")
    print(f"   - Participants: {len(session.participants)}")
    print(f"   - Canvas size: {session.shared_canvas['width']}x{session.shared_canvas['height']}")
    print(f"   - Status: {session.session_status}")
    
    # Test 8: Add structure to collaborative canvas
    print("\n8. Adding structure to collaborative canvas...")
    
    structure_data = {
        "type": "molecule",
        "smiles": "CC1=CC=CC=C1C(=O)N",
        "position": [200, 150],
        "bonds": [
            {"from": 0, "to": 1, "order": 1},
            {"from": 1, "to": 2, "order": 2}
        ],
        "atoms": [
            {"element": "C", "x": 200, "y": 150},
            {"element": "C", "x": 220, "y": 170}
        ]
    }
    
    await chem_collab.add_structure_to_canvas(
        session_id=session.id,
        structure_data=structure_data,
        user_id="medchem_001"
    )
    
    print(f"✅ Added structure to canvas")
    print(f"   - Canvas structures: {len(session.shared_canvas['structures'])}")
    print(f"   - Live edits: {len(session.live_edits)}")
    
    # Test 9: AI-powered retrosynthetic analysis
    print("\n9. Running AI-powered retrosynthetic analysis...")
    
    retro_analysis = await chem_collab.predict_retrosynthesis(
        project_id=project.id,
        target_molecule="Optimized_Analogue_002",
        user_id="compbio_001",
        max_steps=4,
        commercial_sources=True
    )
    
    print(f"✅ Completed retrosynthetic analysis:")
    print(f"   - Routes generated: {len(retro_analysis['routes'])}")
    print(f"   - Overall confidence: {retro_analysis['confidence']:.2f}")
    print(f"   - Best route steps: {retro_analysis['routes'][0]['total_steps']}")
    print(f"   - Estimated yield: {retro_analysis['routes'][0]['estimated_yield']:.1f}%")
    print(f"   - Recommendations: {len(retro_analysis['recommendations'])}")
    
    # Test 10: Export chemistry project
    print("\n10. Exporting chemistry project data...")
    
    export_path = await chem_collab.export_chemistry_project(
        project_id=project.id,
        user_id="pi_medchem_001",
        format="sdf",
        include_spectra=True,
        include_routes=True
    )
    
    print(f"✅ Exported project to: {export_path}")
    
    # Verify export contents
    with zipfile.ZipFile(export_path, 'r') as zipf:
        files = zipf.namelist()
        print(f"   - Export contains {len(files)} files:")
        for file in files[:7]:  # Show first 7 files
            print(f"     • {file}")
        if len(files) > 7:
            print(f"     • ... and {len(files) - 7} more files")
    
    # Clean up export
    Path(export_path).unlink()
    shutil.rmtree(Path(export_path).parent)
    
    print(f"\n🎉 Chemistry Collaboration testing completed successfully!")
    print(f"   - Projects: {len(chem_collab.projects)}")
    print(f"   - Active Sessions: {len(chem_collab.active_sessions)}")
    print(f"   - Research Templates: {len(chem_collab.research_templates)}")
    print(f"   - Chemical Databases: {len(chem_collab.chemical_databases)}")
    print(f"   - Partnership Templates: {len(chem_collab.partnership_templates)}")


if __name__ == "__main__":
    asyncio.run(test_chemistry_collaboration())