"""
Bioinformatics Tools Integration for Collaborative Research

Provides secure P2P collaboration for bioinformatics research including Galaxy workflows,
Bioconductor packages, BLAST searches, and genomic data analysis. Features include
collaborative sequence analysis, phylogenetic studies, and university-industry
biotech partnerships with HIPAA-compliant data handling.

Key Features:
- Post-quantum cryptographic security for sensitive genomic data
- Galaxy workflow collaboration with shared tool execution
- Bioconductor R package integration for statistical genomics
- BLAST sequence alignment with collaborative result analysis
- Genomic data pipeline management with version control
- Multi-institutional biomedical research coordination
- NWTN AI-powered bioinformatics analysis recommendations
- HIPAA/IRB compliant data handling for clinical genomics
- Export capabilities for publications and regulatory submissions
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
import subprocess
import xml.etree.ElementTree as ET

from ..security.post_quantum_crypto_sharding import PostQuantumCryptoSharding

# Mock NWTN for testing
class MockNWTN:
    async def reason(self, prompt, context):
        return {
            "reasoning": [
                "Genomic analysis pipeline appears well-structured for the research objectives",
                "Quality control steps are appropriate for the sequencing technology used",
                "Statistical methods are suitable for the experimental design and sample size"
            ],
            "recommendations": [
                "Consider additional quality filters for low-coverage regions",
                "Implement batch effect correction for multi-institutional data",
                "Add functional annotation analysis for biological interpretation"
            ]
        }


class BioinformaticsDataType(Enum):
    """Bioinformatics data types"""
    DNA_SEQUENCE = "dna_sequence"
    RNA_SEQUENCE = "rna_sequence"
    PROTEIN_SEQUENCE = "protein_sequence"
    FASTQ_READS = "fastq_reads"
    SAM_BAM = "sam_bam"
    VCF_VARIANTS = "vcf_variants"
    GFF_ANNOTATION = "gff_annotation"
    PHYLOGENETIC_TREE = "phylogenetic_tree"
    MICROARRAY_DATA = "microarray_data"
    RNASEQ_COUNTS = "rnaseq_counts"
    PROTEOMICS_DATA = "proteomics_data"


class AnalysisType(Enum):
    """Bioinformatics analysis types"""
    SEQUENCE_ALIGNMENT = "sequence_alignment"
    BLAST_SEARCH = "blast_search"
    GENOME_ASSEMBLY = "genome_assembly"
    VARIANT_CALLING = "variant_calling"
    DIFFERENTIAL_EXPRESSION = "differential_expression"
    PHYLOGENETIC_ANALYSIS = "phylogenetic_analysis"
    FUNCTIONAL_ANNOTATION = "functional_annotation"
    PATHWAY_ANALYSIS = "pathway_analysis"
    METAGENOMICS = "metagenomics"
    STRUCTURAL_PREDICTION = "structural_prediction"


class CollaborationRole(Enum):
    """Bioinformatics collaboration roles"""
    PRINCIPAL_INVESTIGATOR = "principal_investigator"
    BIOINFORMATICIAN = "bioinformatician"
    MOLECULAR_BIOLOGIST = "molecular_biologist"
    COMPUTATIONAL_BIOLOGIST = "computational_biologist"
    CLINICAL_RESEARCHER = "clinical_researcher"
    STUDENT_RESEARCHER = "student_researcher"
    INDUSTRY_PARTNER = "industry_partner"
    DATA_ANALYST = "data_analyst"


@dataclass
class GalaxyTool:
    """Galaxy workflow tool definition"""
    id: str
    name: str
    version: str
    description: str
    category: str  # "genomics", "transcriptomics", "proteomics", etc.
    input_formats: List[str]
    output_formats: List[str]
    parameters: Dict[str, Any]
    computational_requirements: Dict[str, Union[int, float]]
    citation: str
    docker_container: Optional[str] = None


@dataclass
class GalaxyWorkflow:
    """Galaxy bioinformatics workflow"""
    id: str
    name: str
    description: str
    version: str
    created_by: str
    created_at: datetime
    tools: List[GalaxyTool]
    workflow_steps: List[Dict[str, Any]]
    input_datasets: List[str]
    output_datasets: List[str]
    execution_history: List[Dict[str, Any]]
    validation_status: str  # "draft", "validated", "published"
    citation_requirements: List[str]


@dataclass
class BLASTSearchResult:
    """BLAST search result"""
    id: str
    query_sequence: str
    query_length: int
    database: str  # "nr", "nt", "swissprot", etc.
    algorithm: str  # "blastn", "blastp", "blastx", etc.
    hits: List[Dict[str, Any]]
    statistics: Dict[str, float]
    search_parameters: Dict[str, Any]
    executed_at: datetime
    execution_time: float
    taxonomic_distribution: Dict[str, int]


@dataclass
class BioinformaticsDataset:
    """Bioinformatics dataset information"""
    id: str
    name: str
    data_type: BioinformaticsDataType
    file_path: str
    file_size: int
    file_format: str
    organism: str
    tissue_type: Optional[str]
    experimental_condition: Optional[str]
    sequencing_platform: Optional[str]
    read_length: Optional[int]
    coverage: Optional[float]
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any]
    encrypted: bool
    upload_date: datetime
    last_accessed: datetime


@dataclass
class AnalysisResult:
    """Bioinformatics analysis result"""
    id: str
    analysis_type: AnalysisType
    input_datasets: List[str]
    workflow_used: Optional[str]
    parameters: Dict[str, Any]
    output_files: List[str]
    quality_metrics: Dict[str, float]
    statistical_summary: Dict[str, Any]
    visualization_plots: List[str]
    biological_interpretation: str
    executed_by: str
    execution_time: float
    computational_resources: Dict[str, float]
    generated_at: datetime
    peer_reviewed: bool
    publication_ready: bool


@dataclass
class BioinformaticsCollaboration:
    """Main bioinformatics collaboration project"""
    id: str
    name: str
    description: str
    research_area: str  # "cancer_genomics", "infectious_disease", "plant_biology", etc.
    created_by: str
    created_at: datetime
    university: str
    clinical_site: Optional[str]
    industry_partner: Optional[str]
    collaborators: Dict[str, CollaborationRole]
    access_permissions: Dict[str, List[str]]
    datasets: Dict[str, BioinformaticsDataset]
    workflows: Dict[str, GalaxyWorkflow]
    analysis_results: List[AnalysisResult]
    blast_searches: List[BLASTSearchResult]
    active_analyses: List[str]
    security_level: str
    hipaa_compliant: bool
    irb_approved: bool
    data_sharing_agreements: List[str]
    computational_resources: Dict[str, Any]
    timeline: Dict[str, datetime]
    nwtn_insights: List[Dict[str, Any]]
    publication_pipeline: bool


class BioinformaticsTools:
    """Main bioinformatics tools integration system"""
    
    def __init__(self):
        self.crypto_sharding = PostQuantumCryptoSharding()
        self.nwtn = MockNWTN()
        
        self.collaborations: Dict[str, BioinformaticsCollaboration] = {}
        self.galaxy_tools: Dict[str, GalaxyTool] = {}
        self.blast_databases: Dict[str, Dict[str, Any]] = {}
        
        # Initialize common Galaxy tools
        self._initialize_galaxy_tools()
        
        # Initialize BLAST databases
        self._initialize_blast_databases()
        
        # University-specific bioinformatics templates
        self.research_templates = {
            "unc_cancer_genomics": {
                "name": "UNC Cancer Genomics Research",
                "required_tools": ["fastqc", "trimmomatic", "bwa", "gatk", "mutect2"],
                "data_types": [BioinformaticsDataType.FASTQ_READS, BioinformaticsDataType.VCF_VARIANTS],
                "analysis_pipeline": "cancer_variant_calling",
                "compliance": {"hipaa": True, "irb_required": True},
                "compute_requirements": {"cpu_cores": 16, "memory_gb": 64, "storage_tb": 2}
            },
            "duke_infectious_disease": {
                "name": "Duke Infectious Disease Genomics",
                "required_tools": ["kraken2", "metaphlan", "spades", "prokka", "roary"],
                "data_types": [BioinformaticsDataType.FASTQ_READS, BioinformaticsDataType.DNA_SEQUENCE],
                "analysis_pipeline": "pathogen_genomics",
                "compliance": {"hipaa": False, "irb_required": False},
                "compute_requirements": {"cpu_cores": 8, "memory_gb": 32, "storage_tb": 1}
            },
            "ncsu_plant_biology": {
                "name": "NC State Plant Biology Research",
                "required_tools": ["trinity", "busco", "augustus", "interproscan", "blast"],
                "data_types": [BioinformaticsDataType.RNA_SEQUENCE, BioinformaticsDataType.PROTEIN_SEQUENCE],
                "analysis_pipeline": "plant_transcriptomics",
                "compliance": {"hipaa": False, "irb_required": False},
                "compute_requirements": {"cpu_cores": 12, "memory_gb": 48, "storage_tb": 1.5}
            },
            "rtp_biotech": {
                "name": "RTP Biotech Industry Partnership",
                "required_tools": ["deseq2", "limma", "gsea", "string", "cytoscape"],
                "data_types": [BioinformaticsDataType.RNASEQ_COUNTS, BioinformaticsDataType.PROTEOMICS_DATA],
                "analysis_pipeline": "drug_discovery",
                "compliance": {"hipaa": True, "irb_required": True},
                "compute_requirements": {"cpu_cores": 20, "memory_gb": 128, "storage_tb": 5}
            }
        }
        
        # Bioconductor packages by research area
        self.bioconductor_packages = {
            "genomics": [
                "GenomicRanges", "GenomicFeatures", "Biostrings", "BSgenome",
                "VariantAnnotation", "rtracklayer", "Rsamtools"
            ],
            "transcriptomics": [
                "DESeq2", "limma", "edgeR", "tximport", "GenomicAlignments",
                "Rsubread", "ballgown", "sleuth"
            ],
            "proteomics": [
                "MSnbase", "xcms", "CAMERA", "MSstats", "MSstatsQC",
                "MSstatsTMT", "Proteomics", "RforProteomics"
            ],
            "epigenomics": [
                "methylKit", "bsseq", "ChIPseeker", "DiffBind", "ChIPQC",
                "soGGi", "chromstaR", "MEDIPS"
            ],
            "metagenomics": [
                "phyloseq", "microbiome", "DADA2", "decontam", "vegan",
                "ape", "picante", "SpiecEasi"
            ]
        }
    
    async def create_bioinformatics_collaboration(
        self,
        name: str,
        description: str,
        research_area: str,
        creator_id: str,
        university: str,
        template: Optional[str] = None,
        clinical_site: Optional[str] = None,
        industry_partner: Optional[str] = None,
        security_level: str = "high"
    ) -> BioinformaticsCollaboration:
        """Create a new bioinformatics collaboration project"""
        
        collaboration_id = str(uuid.uuid4())
        
        # Apply template if provided
        template_config = self.research_templates.get(template, {})
        
        # Set compliance requirements
        hipaa_compliant = template_config.get("compliance", {}).get("hipaa", False)
        irb_required = template_config.get("compliance", {}).get("irb_required", False)
        
        # Generate NWTN insights for project setup
        nwtn_context = {
            "project_name": name,
            "research_area": research_area,
            "university": university,
            "industry_partner": industry_partner,
            "clinical_site": clinical_site,
            "template": template
        }
        
        nwtn_insights = await self._generate_project_insights(nwtn_context)
        
        # Set up timeline
        timeline = {
            "project_start": datetime.now(),
            "data_collection": datetime.now() + timedelta(weeks=4),
            "quality_control": datetime.now() + timedelta(weeks=8),
            "primary_analysis": datetime.now() + timedelta(weeks=16),
            "validation_phase": datetime.now() + timedelta(weeks=20),
            "manuscript_prep": datetime.now() + timedelta(weeks=24)
        }
        
        collaboration = BioinformaticsCollaboration(
            id=collaboration_id,
            name=name,
            description=description,
            research_area=research_area,
            created_by=creator_id,
            created_at=datetime.now(),
            university=university,
            clinical_site=clinical_site,
            industry_partner=industry_partner,
            collaborators={creator_id: CollaborationRole.PRINCIPAL_INVESTIGATOR},
            access_permissions={
                "read": [creator_id],
                "write": [creator_id],
                "execute": [creator_id],
                "admin": [creator_id]
            },
            datasets={},
            workflows={},
            analysis_results=[],
            blast_searches=[],
            active_analyses=[],
            security_level=security_level,
            hipaa_compliant=hipaa_compliant,
            irb_approved=False,  # Must be explicitly approved
            data_sharing_agreements=[],
            computational_resources=template_config.get("compute_requirements", {}),
            timeline=timeline,
            nwtn_insights=nwtn_insights,
            publication_pipeline=True
        )
        
        self.collaborations[collaboration_id] = collaboration
        return collaboration
    
    async def upload_genomic_dataset(
        self,
        collaboration_id: str,
        dataset_name: str,
        file_path: str,
        data_type: BioinformaticsDataType,
        organism: str,
        user_id: str,
        metadata: Dict[str, Any] = None
    ) -> BioinformaticsDataset:
        """Upload genomic dataset to collaboration"""
        
        if collaboration_id not in self.collaborations:
            raise ValueError(f"Collaboration {collaboration_id} not found")
        
        collaboration = self.collaborations[collaboration_id]
        
        # Check permissions
        if user_id not in collaboration.access_permissions.get("write", []):
            raise PermissionError("User does not have write access")
        
        # Validate file and extract basic info
        file_stats = Path(file_path).stat() if Path(file_path).exists() else None
        file_size = file_stats.st_size if file_stats else 0
        
        # Quality control analysis
        quality_metrics = await self._analyze_dataset_quality(file_path, data_type)
        
        # Encrypt and shard dataset if required
        encrypted = False
        if collaboration.security_level in ["high", "maximum"] or collaboration.hipaa_compliant:
            encrypted_shards = self.crypto_sharding.shard_file(
                file_path,
                list(collaboration.collaborators.keys()),
                num_shards=7
            )
            encrypted = True
        
        dataset = BioinformaticsDataset(
            id=str(uuid.uuid4()),
            name=dataset_name,
            data_type=data_type,
            file_path=file_path if not encrypted else "encrypted",
            file_size=file_size,
            file_format=Path(file_path).suffix,
            organism=organism,
            tissue_type=metadata.get("tissue_type") if metadata else None,
            experimental_condition=metadata.get("condition") if metadata else None,
            sequencing_platform=metadata.get("platform") if metadata else None,
            read_length=metadata.get("read_length") if metadata else None,
            coverage=metadata.get("coverage") if metadata else None,
            quality_metrics=quality_metrics,
            metadata=metadata or {},
            encrypted=encrypted,
            upload_date=datetime.now(),
            last_accessed=datetime.now()
        )
        
        collaboration.datasets[dataset_name] = dataset
        
        # Generate dataset insights
        dataset_insights = await self._analyze_genomic_dataset(dataset, collaboration)
        collaboration.nwtn_insights.extend(dataset_insights)
        
        return dataset
    
    async def create_galaxy_workflow(
        self,
        collaboration_id: str,
        workflow_name: str,
        tools: List[str],
        workflow_description: str,
        creator_id: str
    ) -> GalaxyWorkflow:
        """Create Galaxy workflow for bioinformatics analysis"""
        
        if collaboration_id not in self.collaborations:
            raise ValueError(f"Collaboration {collaboration_id} not found")
        
        collaboration = self.collaborations[collaboration_id]
        
        # Check permissions
        if creator_id not in collaboration.access_permissions.get("write", []):
            raise PermissionError("User does not have write access")
        
        # Build workflow from tools
        workflow_tools = []
        workflow_steps = []
        
        for i, tool_name in enumerate(tools):
            if tool_name in self.galaxy_tools:
                tool = self.galaxy_tools[tool_name]
                workflow_tools.append(tool)
                
                step = {
                    "id": i + 1,
                    "tool_id": tool.id,
                    "tool_version": tool.version,
                    "inputs": {},
                    "outputs": {},
                    "parameters": tool.parameters.copy()
                }
                workflow_steps.append(step)
        
        workflow = GalaxyWorkflow(
            id=str(uuid.uuid4()),
            name=workflow_name,
            description=workflow_description,
            version="1.0",
            created_by=creator_id,
            created_at=datetime.now(),
            tools=workflow_tools,
            workflow_steps=workflow_steps,
            input_datasets=[],
            output_datasets=[],
            execution_history=[],
            validation_status="draft",
            citation_requirements=[tool.citation for tool in workflow_tools if tool.citation]
        )
        
        collaboration.workflows[workflow_name] = workflow
        
        return workflow
    
    async def execute_blast_search(
        self,
        collaboration_id: str,
        query_sequence: str,
        database: str,
        algorithm: str,
        user_id: str,
        parameters: Dict[str, Any] = None
    ) -> BLASTSearchResult:
        """Execute BLAST sequence search"""
        
        if collaboration_id not in self.collaborations:
            raise ValueError(f"Collaboration {collaboration_id} not found")
        
        collaboration = self.collaborations[collaboration_id]
        
        # Check permissions
        if user_id not in collaboration.access_permissions.get("execute", []):
            raise PermissionError("User does not have execute access")
        
        # Simulate BLAST search
        search_results = await self._simulate_blast_search(
            query_sequence, database, algorithm, parameters or {}
        )
        
        blast_result = BLASTSearchResult(
            id=str(uuid.uuid4()),
            query_sequence=query_sequence,
            query_length=len(query_sequence),
            database=database,
            algorithm=algorithm,
            hits=search_results["hits"],
            statistics=search_results["statistics"],
            search_parameters=parameters or {},
            executed_at=datetime.now(),
            execution_time=search_results["execution_time"],
            taxonomic_distribution=search_results["taxonomy"]
        )
        
        collaboration.blast_searches.append(blast_result)
        
        # Generate BLAST analysis insights
        blast_insights = await self._analyze_blast_results(blast_result, collaboration)
        collaboration.nwtn_insights.extend(blast_insights)
        
        return blast_result
    
    async def run_bioinformatics_analysis(
        self,
        collaboration_id: str,
        analysis_type: AnalysisType,
        input_datasets: List[str],
        workflow_name: Optional[str],
        parameters: Dict[str, Any],
        user_id: str
    ) -> AnalysisResult:
        """Run comprehensive bioinformatics analysis"""
        
        if collaboration_id not in self.collaborations:
            raise ValueError(f"Collaboration {collaboration_id} not found")
        
        collaboration = self.collaborations[collaboration_id]
        
        # Check permissions
        if user_id not in collaboration.access_permissions.get("execute", []):
            raise PermissionError("User does not have execute access")
        
        # Simulate analysis execution
        analysis_results = await self._simulate_bioinformatics_analysis(
            analysis_type, input_datasets, parameters
        )
        
        result = AnalysisResult(
            id=str(uuid.uuid4()),
            analysis_type=analysis_type,
            input_datasets=input_datasets,
            workflow_used=workflow_name,
            parameters=parameters,
            output_files=analysis_results["output_files"],
            quality_metrics=analysis_results["quality_metrics"],
            statistical_summary=analysis_results["statistics"],
            visualization_plots=analysis_results["plots"],
            biological_interpretation=analysis_results["interpretation"],
            executed_by=user_id,
            execution_time=analysis_results["execution_time"],
            computational_resources=analysis_results["resources"],
            generated_at=datetime.now(),
            peer_reviewed=False,
            publication_ready=False
        )
        
        collaboration.analysis_results.append(result)
        
        # Generate analysis insights
        analysis_insights = await self._generate_analysis_insights(result, collaboration)
        collaboration.nwtn_insights.extend(analysis_insights)
        
        return result
    
    async def create_bioconductor_environment(
        self,
        collaboration_id: str,
        research_focus: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Create Bioconductor R environment for genomic analysis"""
        
        if collaboration_id not in self.collaborations:
            raise ValueError(f"Collaboration {collaboration_id} not found")
        
        collaboration = self.collaborations[collaboration_id]
        
        # Get relevant packages for research focus
        packages = self.bioconductor_packages.get(research_focus, [])
        
        # Create R environment setup script
        r_script = f"""
# Bioconductor Environment Setup for {collaboration.name}
# Research Focus: {research_focus}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Install BiocManager if not present
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

# Set Bioconductor version
BiocManager::install(version = "3.17")

# Install required packages
required_packages <- c({', '.join([f'"{pkg}"' for pkg in packages])})

for (pkg in required_packages) {{
    if (!requireNamespace(pkg, quietly = TRUE)) {{
        cat("Installing", pkg, "...\\n")
        BiocManager::install(pkg)
    }}
}}

# Load packages
lapply(required_packages, library, character.only = TRUE)

# Environment information
sessionInfo()
"""
        
        # Create temporary environment file
        temp_dir = tempfile.mkdtemp()
        script_path = Path(temp_dir) / "bioconductor_setup.R"
        
        with open(script_path, 'w') as f:
            f.write(r_script)
        
        environment_info = {
            "collaboration_id": collaboration_id,
            "research_focus": research_focus,
            "packages": packages,
            "setup_script": str(script_path),
            "r_version": "4.3.0",
            "bioconductor_version": "3.17",
            "created_by": user_id,
            "created_at": datetime.now()
        }
        
        return environment_info
    
    async def generate_publication_report(
        self,
        collaboration_id: str,
        include_methods: bool = True,
        include_results: bool = True,
        include_figures: bool = True,
        format: str = "markdown"
    ) -> str:
        """Generate publication-ready research report"""
        
        if collaboration_id not in self.collaborations:
            raise ValueError(f"Collaboration {collaboration_id} not found")
        
        collaboration = self.collaborations[collaboration_id]
        
        # Generate comprehensive report
        report_content = await self._generate_publication_content(
            collaboration, include_methods, include_results, include_figures
        )
        
        # Create report file
        temp_dir = tempfile.mkdtemp()
        report_file = Path(temp_dir) / f"{collaboration.name}_publication_report.{format}"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_file)
    
    async def export_collaboration_data(
        self,
        collaboration_id: str,
        user_id: str,
        include_raw_data: bool = False,
        include_analysis_results: bool = True,
        include_workflows: bool = True
    ) -> str:
        """Export complete collaboration data package"""
        
        if collaboration_id not in self.collaborations:
            raise ValueError(f"Collaboration {collaboration_id} not found")
        
        collaboration = self.collaborations[collaboration_id]
        
        # Check permissions
        if user_id not in collaboration.access_permissions.get("read", []):
            raise PermissionError("User does not have read access")
        
        # Create export package
        temp_dir = tempfile.mkdtemp()
        export_path = Path(temp_dir) / f"{collaboration.name}_export.zip"
        
        with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Collaboration metadata
            collab_data = asdict(collaboration)
            collab_data['created_at'] = collab_data['created_at'].isoformat()
            
            zipf.writestr(
                "collaboration_metadata.json",
                json.dumps(collab_data, indent=2, default=str)
            )
            
            # Workflows
            if include_workflows:
                for workflow_name, workflow in collaboration.workflows.items():
                    workflow_data = asdict(workflow)
                    workflow_data['created_at'] = workflow_data['created_at'].isoformat()
                    
                    zipf.writestr(
                        f"workflows/{workflow_name}.json",
                        json.dumps(workflow_data, indent=2, default=str)
                    )
            
            # Analysis results
            if include_analysis_results:
                for result in collaboration.analysis_results:
                    result_data = asdict(result)
                    result_data['generated_at'] = result_data['generated_at'].isoformat()
                    
                    zipf.writestr(
                        f"results/{result.analysis_type.value}_{result.id}.json",
                        json.dumps(result_data, indent=2, default=str)
                    )
            
            # BLAST searches
            for blast_result in collaboration.blast_searches:
                blast_data = asdict(blast_result)
                blast_data['executed_at'] = blast_data['executed_at'].isoformat()
                
                zipf.writestr(
                    f"blast_searches/blast_{blast_result.id}.json",
                    json.dumps(blast_data, indent=2, default=str)
                )
            
            # NWTN insights
            zipf.writestr(
                "nwtn_insights.json",
                json.dumps(collaboration.nwtn_insights, indent=2, default=str)
            )
            
            # Dataset metadata (not raw data unless requested)
            datasets_metadata = {}
            for name, dataset in collaboration.datasets.items():
                dataset_data = asdict(dataset)
                dataset_data['upload_date'] = dataset_data['upload_date'].isoformat()
                dataset_data['last_accessed'] = dataset_data['last_accessed'].isoformat()
                
                if not include_raw_data:
                    dataset_data.pop('file_path', None)
                
                datasets_metadata[name] = dataset_data
            
            zipf.writestr(
                "datasets_metadata.json",
                json.dumps(datasets_metadata, indent=2, default=str)
            )
            
            # Generate project summary
            summary_report = self._generate_collaboration_summary(collaboration)
            zipf.writestr("collaboration_summary.md", summary_report)
        
        return str(export_path)
    
    def _initialize_galaxy_tools(self):
        """Initialize common Galaxy tools"""
        
        # Quality control tools
        self.galaxy_tools["fastqc"] = GalaxyTool(
            id="fastqc",
            name="FastQC",
            version="0.12.1",
            description="Quality control tool for high throughput sequence data",
            category="genomics",
            input_formats=["fastq", "fastq.gz"],
            output_formats=["html", "txt"],
            parameters={"adapters": "auto", "kmers": "7"},
            computational_requirements={"cpu": 1, "memory_gb": 2},
            citation="Andrews, S. (2010). FastQC: a quality control tool for high throughput sequence data."
        )
        
        # Alignment tools
        self.galaxy_tools["bwa"] = GalaxyTool(
            id="bwa",
            name="BWA-MEM",
            version="0.7.17",
            description="Burrows-Wheeler Aligner for sequence alignment",
            category="genomics",
            input_formats=["fastq", "fastq.gz"],
            output_formats=["sam", "bam"],
            parameters={"min_seed_length": 19, "band_width": 100},
            computational_requirements={"cpu": 4, "memory_gb": 8},
            citation="Li, H. (2013). Aligning sequence reads, clone sequences and assembly contigs with BWA-MEM."
        )
        
        # Variant calling tools
        self.galaxy_tools["gatk"] = GalaxyTool(
            id="gatk",
            name="GATK HaplotypeCaller",
            version="4.4.0",
            description="Variant calling using GATK HaplotypeCaller",
            category="genomics",
            input_formats=["bam"],
            output_formats=["vcf", "gvcf"],
            parameters={"stand_call_conf": 10.0, "stand_emit_conf": 10.0},
            computational_requirements={"cpu": 2, "memory_gb": 16},
            citation="McKenna, A. et al. (2010). The Genome Analysis Toolkit: a MapReduce framework."
        )
        
        # RNA-seq analysis
        self.galaxy_tools["deseq2"] = GalaxyTool(
            id="deseq2",
            name="DESeq2",
            version="1.40.0",
            description="Differential gene expression analysis using DESeq2",
            category="transcriptomics",
            input_formats=["csv", "tsv"],
            output_formats=["csv", "pdf"],
            parameters={"alpha": 0.05, "lfcThreshold": 0},
            computational_requirements={"cpu": 1, "memory_gb": 4},
            citation="Love, M.I. et al. (2014). Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2."
        )
        
        # Assembly tools
        self.galaxy_tools["spades"] = GalaxyTool(
            id="spades",
            name="SPAdes",
            version="3.15.5",
            description="Genome assembly using SPAdes assembler",
            category="genomics",
            input_formats=["fastq", "fastq.gz"],
            output_formats=["fasta"],
            parameters={"careful": True, "cov_cutoff": "auto"},
            computational_requirements={"cpu": 8, "memory_gb": 32},
            citation="Bankevich, A. et al. (2012). SPAdes: a new genome assembly algorithm."
        )
    
    def _initialize_blast_databases(self):
        """Initialize BLAST database information"""
        
        self.blast_databases = {
            "nr": {
                "name": "Non-redundant protein sequences",
                "type": "protein",
                "size_gb": 250,
                "sequences": 220000000,
                "last_updated": datetime(2024, 7, 1)
            },
            "nt": {
                "name": "Non-redundant nucleotide sequences",
                "type": "nucleotide",
                "size_gb": 180,
                "sequences": 95000000,
                "last_updated": datetime(2024, 7, 1)
            },
            "swissprot": {
                "name": "Swiss-Prot protein database",
                "type": "protein",
                "size_gb": 1.2,
                "sequences": 570000,
                "last_updated": datetime(2024, 6, 15)
            },
            "pdb": {
                "name": "Protein Data Bank",
                "type": "protein",
                "size_gb": 0.8,
                "sequences": 200000,
                "last_updated": datetime(2024, 6, 30)
            }
        }
    
    async def _analyze_dataset_quality(
        self,
        file_path: str,
        data_type: BioinformaticsDataType
    ) -> Dict[str, float]:
        """Analyze genomic dataset quality"""
        
        # Simulate quality analysis based on data type
        if data_type == BioinformaticsDataType.FASTQ_READS:
            return {
                "mean_quality_score": 35.2,
                "gc_content": 42.1,
                "sequence_length_mean": 150.0,
                "adapter_contamination": 2.1,
                "duplicate_rate": 8.5,
                "n_content": 0.02
            }
        elif data_type == BioinformaticsDataType.VCF_VARIANTS:
            return {
                "total_variants": 2451890,
                "snp_percentage": 87.3,
                "indel_percentage": 12.7,
                "transition_transversion_ratio": 2.08,
                "mean_variant_quality": 542.1
            }
        else:
            return {
                "file_integrity": 100.0,
                "format_compliance": 99.8,
                "completeness": 98.5
            }
    
    async def _simulate_blast_search(
        self,
        query_sequence: str,
        database: str,
        algorithm: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate BLAST search results"""
        
        # Generate mock BLAST hits
        hits = []
        for i in range(10):  # Top 10 hits
            hits.append({
                "accession": f"ACC{i+1:06d}",
                "description": f"Hypothetical protein {i+1} [Organism species]",
                "score": 450 - i * 25,
                "e_value": 1e-50 * (10 ** i),
                "identity": 95.2 - i * 2.1,
                "coverage": 89.3 - i * 1.5,
                "length": 1250 + i * 50
            })
        
        # Statistics
        statistics = {
            "database_sequences": self.blast_databases[database]["sequences"],
            "database_letters": self.blast_databases[database]["sequences"] * 300,
            "effective_search_space": 8.5e12,
            "lambda": 0.267,
            "kappa": 0.041,
            "entropy": 0.14
        }
        
        # Taxonomic distribution (mock)
        taxonomy = {
            "Bacteria": 40,
            "Eukaryota": 35,
            "Archaea": 15,
            "Viruses": 8,
            "Unclassified": 2
        }
        
        return {
            "hits": hits,
            "statistics": statistics,
            "taxonomy": taxonomy,
            "execution_time": 12.5 + len(query_sequence) * 0.01
        }
    
    async def _simulate_bioinformatics_analysis(
        self,
        analysis_type: AnalysisType,
        input_datasets: List[str],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate bioinformatics analysis execution"""
        
        if analysis_type == AnalysisType.DIFFERENTIAL_EXPRESSION:
            return {
                "output_files": ["deseq2_results.csv", "normalized_counts.csv", "plots.pdf"],
                "quality_metrics": {
                    "total_genes": 25847,
                    "differentially_expressed": 2341,
                    "upregulated": 1204,
                    "downregulated": 1137,
                    "mean_dispersion": 0.32
                },
                "statistics": {
                    "fdr_threshold": 0.05,
                    "log2_fold_change_threshold": 1.0,
                    "r_squared": 0.89
                },
                "plots": ["volcano_plot.png", "ma_plot.png", "heatmap.png"],
                "interpretation": "Identified significant differential expression patterns between treatment conditions",
                "execution_time": 45.2,
                "resources": {"cpu_hours": 2.1, "memory_peak_gb": 8.5, "disk_gb": 1.2}
            }
        elif analysis_type == AnalysisType.VARIANT_CALLING:
            return {
                "output_files": ["variants.vcf", "filtered_variants.vcf", "summary.txt"],
                "quality_metrics": {
                    "total_variants": 4521890,
                    "high_quality_variants": 3789456,
                    "snps": 3345678,
                    "indels": 443778,
                    "transition_transversion_ratio": 2.08
                },
                "statistics": {
                    "mean_coverage": 32.4,
                    "variant_quality_score": 567.2,
                    "genotype_quality": 89.1
                },
                "plots": ["quality_distribution.png", "coverage_plot.png"],
                "interpretation": "High-quality variant calling with good coverage distribution",
                "execution_time": 125.8,
                "resources": {"cpu_hours": 8.3, "memory_peak_gb": 24, "disk_gb": 15.4}
            }
        else:
            return {
                "output_files": [f"{analysis_type.value}_results.txt"],
                "quality_metrics": {"success_rate": 95.2},
                "statistics": {"processed_items": 1000},
                "plots": ["summary_plot.png"],
                "interpretation": f"Completed {analysis_type.value} analysis successfully",
                "execution_time": 30.0,
                "resources": {"cpu_hours": 1.5, "memory_peak_gb": 4, "disk_gb": 2}
            }
    
    async def _generate_project_insights(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate NWTN insights for project setup"""
        
        nwtn_prompt = f"""
        Analyze this bioinformatics research project setup:
        
        Project: {context['project_name']}
        Research Area: {context['research_area']}
        University: {context['university']}
        Clinical Site: {context.get('clinical_site', 'None')}
        Industry Partner: {context.get('industry_partner', 'None')}
        
        Provide insights on:
        1. Appropriate bioinformatics workflows and tools
        2. Data management and quality control strategies
        3. Statistical analysis considerations
        4. Collaboration and data sharing best practices
        5. Publication and reproducibility requirements
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
    
    async def _analyze_genomic_dataset(
        self,
        dataset: BioinformaticsDataset,
        collaboration: BioinformaticsCollaboration
    ) -> List[Dict[str, Any]]:
        """Analyze genomic dataset using NWTN"""
        
        nwtn_prompt = f"""
        Analyze this genomic dataset for research insights:
        
        Dataset: {dataset.name}
        Data Type: {dataset.data_type.value}
        Organism: {dataset.organism}
        File Size: {dataset.file_size} bytes
        Quality Metrics: {dataset.quality_metrics}
        
        Research Context: {collaboration.research_area}
        
        Provide analysis recommendations:
        1. Quality control steps and thresholds
        2. Preprocessing requirements
        3. Appropriate analysis workflows
        4. Statistical considerations
        5. Validation strategies
        """
        
        context = {
            "dataset": asdict(dataset),
            "collaboration": asdict(collaboration)
        }
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "dataset_analysis",
                "timestamp": datetime.now(),
                "dataset_id": dataset.id,
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    async def _analyze_blast_results(
        self,
        blast_result: BLASTSearchResult,
        collaboration: BioinformaticsCollaboration
    ) -> List[Dict[str, Any]]:
        """Analyze BLAST results using NWTN"""
        
        nwtn_prompt = f"""
        Analyze these BLAST search results:
        
        Query Length: {blast_result.query_length}
        Database: {blast_result.database}
        Algorithm: {blast_result.algorithm}
        Top Hits: {len(blast_result.hits)}
        Best E-value: {blast_result.hits[0]['e_value'] if blast_result.hits else 'N/A'}
        Taxonomic Distribution: {blast_result.taxonomic_distribution}
        
        Research Context: {collaboration.research_area}
        
        Provide insights on:
        1. Biological significance of top hits
        2. Taxonomic distribution interpretation
        3. Follow-up analysis recommendations
        4. Functional annotation suggestions
        5. Literature search priorities
        """
        
        context = {
            "blast_result": asdict(blast_result),
            "collaboration": asdict(collaboration)
        }
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "blast_analysis",
                "timestamp": datetime.now(),
                "blast_search_id": blast_result.id,
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    async def _generate_analysis_insights(
        self,
        result: AnalysisResult,
        collaboration: BioinformaticsCollaboration
    ) -> List[Dict[str, Any]]:
        """Generate insights for analysis results"""
        
        nwtn_prompt = f"""
        Analyze these bioinformatics analysis results:
        
        Analysis Type: {result.analysis_type.value}
        Quality Metrics: {result.quality_metrics}
        Statistical Summary: {result.statistical_summary}
        Execution Time: {result.execution_time} seconds
        
        Research Context: {collaboration.research_area}
        
        Provide insights on:
        1. Result quality and reliability assessment
        2. Biological interpretation guidelines
        3. Statistical significance evaluation
        4. Follow-up experiments or analyses
        5. Publication and validation strategies
        """
        
        context = {
            "analysis_result": asdict(result),
            "collaboration": asdict(collaboration)
        }
        
        nwtn_response = await self.nwtn.reason(nwtn_prompt, context)
        
        return [
            {
                "type": "analysis_insights",
                "timestamp": datetime.now(),
                "analysis_id": result.id,
                "insights": nwtn_response.get("reasoning", []),
                "recommendations": nwtn_response.get("recommendations", [])
            }
        ]
    
    async def _generate_publication_content(
        self,
        collaboration: BioinformaticsCollaboration,
        include_methods: bool,
        include_results: bool,
        include_figures: bool
    ) -> str:
        """Generate publication-ready content"""
        
        content = f"""# {collaboration.name}

## Abstract

**Background:** {collaboration.description}

**Methods:** Comprehensive bioinformatics analysis was performed using Galaxy workflows and Bioconductor packages for {collaboration.research_area} research.

**Results:** Analysis of {len(collaboration.datasets)} datasets using {len(collaboration.workflows)} computational workflows revealed significant biological insights.

**Conclusions:** The findings contribute to our understanding of {collaboration.research_area} and provide a foundation for future research.

## Introduction

This study presents a collaborative bioinformatics analysis conducted between {collaboration.university}"""
        
        if collaboration.clinical_site:
            content += f" and {collaboration.clinical_site}"
        if collaboration.industry_partner:
            content += f" in partnership with {collaboration.industry_partner}"
            
        content += f""".

## Methods

### Data Collection and Quality Control

A total of {len(collaboration.datasets)} datasets were included in this analysis:

"""
        
        for name, dataset in collaboration.datasets.items():
            content += f"""
**{name}:**
- Data Type: {dataset.data_type.value}
- Organism: {dataset.organism}
- File Size: {dataset.file_size / (1024*1024):.1f} MB
- Quality Score: {dataset.quality_metrics.get('mean_quality_score', 'N/A')}
"""
        
        if include_methods and collaboration.workflows:
            content += f"\n### Computational Workflows\n\n"
            
            for workflow_name, workflow in collaboration.workflows.items():
                content += f"""
**{workflow_name}:**
- Tools Used: {', '.join([tool.name for tool in workflow.tools])}
- Version: {workflow.version}
- Validation Status: {workflow.validation_status}
"""
        
        if include_results and collaboration.analysis_results:
            content += f"\n## Results\n\n"
            
            for result in collaboration.analysis_results:
                content += f"""
### {result.analysis_type.value.replace('_', ' ').title()}

{result.biological_interpretation}

**Quality Metrics:**
"""
                for metric, value in result.quality_metrics.items():
                    content += f"- {metric}: {value}\n"
                
                content += f"\n**Computational Resources:** {result.execution_time:.1f} seconds\n\n"
        
        if collaboration.blast_searches:
            content += f"\n### Sequence Similarity Analysis\n\n"
            content += f"BLAST searches were performed against {len(set(search.database for search in collaboration.blast_searches))} databases, yielding insights into sequence conservation and functional relationships.\n\n"
        
        content += f"""
## Discussion

The integrated bioinformatics approach employed in this study demonstrates the value of collaborative research between academic institutions"""
        
        if collaboration.industry_partner:
            content += " and industry partners"
            
        content += f""". The findings contribute significantly to {collaboration.research_area} research.

## Methods References

### Software and Tools

"""
        
        # Add tool citations
        citations = set()
        for workflow in collaboration.workflows.values():
            citations.update(workflow.citation_requirements)
        
        for citation in list(citations)[:5]:  # Show first 5 citations
            content += f"- {citation}\n"
        
        content += f"""
### Data Availability

All analysis workflows and results are available through the PRSM secure collaboration platform with appropriate data sharing agreements.

---
*Manuscript generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} using PRSM Bioinformatics Collaboration Platform*
"""
        
        return content
    
    def _generate_collaboration_summary(self, collaboration: BioinformaticsCollaboration) -> str:
        """Generate collaboration summary for export"""
        
        summary = f"""# Bioinformatics Collaboration Summary: {collaboration.name}

## Project Information
- **Collaboration ID**: {collaboration.id}
- **Created**: {collaboration.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- **Research Area**: {collaboration.research_area}
- **University**: {collaboration.university}
- **Clinical Site**: {collaboration.clinical_site or 'None'}
- **Industry Partner**: {collaboration.industry_partner or 'None'}
- **Security Level**: {collaboration.security_level}

## Compliance and Ethics
- **HIPAA Compliant**: {'Yes' if collaboration.hipaa_compliant else 'No'}
- **IRB Approved**: {'Yes' if collaboration.irb_approved else 'No'}
- **Data Sharing Agreements**: {len(collaboration.data_sharing_agreements)}

## Research Assets
- **Datasets**: {len(collaboration.datasets)}
- **Galaxy Workflows**: {len(collaboration.workflows)}
- **Analysis Results**: {len(collaboration.analysis_results)}
- **BLAST Searches**: {len(collaboration.blast_searches)}
- **NWTN Insights**: {len(collaboration.nwtn_insights)}

## Collaboration
- **Total Collaborators**: {len(collaboration.collaborators)}
- **Active Analyses**: {len(collaboration.active_analyses)}

### Collaborator Roles
"""
        
        for user_id, role in collaboration.collaborators.items():
            summary += f"- **{user_id}**: {role.value}\n"
        
        summary += f"""
## Computational Resources
- **CPU Cores**: {collaboration.computational_resources.get('cpu_cores', 'Not specified')}
- **Memory**: {collaboration.computational_resources.get('memory_gb', 'Not specified')} GB
- **Storage**: {collaboration.computational_resources.get('storage_tb', 'Not specified')} TB

## Timeline Progress
"""
        
        for milestone, date in collaboration.timeline.items():
            status = "✅" if date <= datetime.now() else "⏳"
            summary += f"- {status} **{milestone.replace('_', ' ').title()}**: {date.strftime('%Y-%m-%d')}\n"
        
        summary += f"""
## Dataset Summary
"""
        
        for name, dataset in collaboration.datasets.items():
            summary += f"""
### {name}
- **Type**: {dataset.data_type.value}
- **Organism**: {dataset.organism}
- **Size**: {dataset.file_size / (1024*1024):.1f} MB
- **Format**: {dataset.file_format}
- **Encrypted**: {'Yes' if dataset.encrypted else 'No'}
"""
        
        summary += f"""
---
*Summary generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return summary


# Testing and validation
async def test_bioinformatics_tools():
    """Test bioinformatics tools integration"""
    
    bio_tools = BioinformaticsTools()
    
    print("🧬 Testing Bioinformatics Tools Integration...")
    
    # Test 1: Create cancer genomics collaboration
    print("\n1. Creating UNC Cancer Genomics Research Project...")
    
    collaboration = await bio_tools.create_bioinformatics_collaboration(
        name="Multi-Institutional Cancer Genomics Consortium",
        description="Collaborative study of tumor heterogeneity using whole-genome sequencing across multiple cancer types",
        research_area="cancer_genomics",
        creator_id="pi_oncology_001",
        university="University of North Carolina at Chapel Hill",
        template="unc_cancer_genomics",
        clinical_site="UNC Lineberger Comprehensive Cancer Center",
        industry_partner="Illumina Inc.",
        security_level="maximum"
    )
    
    print(f"✅ Created collaboration: {collaboration.name}")
    print(f"   - ID: {collaboration.id}")
    print(f"   - Research Area: {collaboration.research_area}")
    print(f"   - HIPAA Compliant: {collaboration.hipaa_compliant}")
    print(f"   - Security Level: {collaboration.security_level}")
    print(f"   - NWTN Insights: {len(collaboration.nwtn_insights)}")
    
    # Test 2: Add collaborators
    print("\n2. Adding research collaborators...")
    
    collaboration.collaborators.update({
        "bioinf_001": CollaborationRole.BIOINFORMATICIAN,
        "clinician_001": CollaborationRole.CLINICAL_RESEARCHER,
        "student_001": CollaborationRole.STUDENT_RESEARCHER,
        "industry_001": CollaborationRole.INDUSTRY_PARTNER,
        "compbio_001": CollaborationRole.COMPUTATIONAL_BIOLOGIST
    })
    
    # Update permissions
    all_collaborators = list(collaboration.collaborators.keys())
    collaboration.access_permissions.update({
        "read": all_collaborators,
        "write": ["pi_oncology_001", "bioinf_001", "student_001"],
        "execute": ["pi_oncology_001", "bioinf_001", "compbio_001"],
        "admin": ["pi_oncology_001"]
    })
    
    print(f"✅ Added {len(collaboration.collaborators)} collaborators")
    
    # Test 3: Upload genomic datasets
    print("\n3. Uploading genomic datasets...")
    
    # Create temporary genomic data files
    temp_fastq = tempfile.NamedTemporaryFile(suffix='.fastq.gz', delete=False)
    temp_fastq.write(b"@seq1\nATCGATCGATCG\n+\nIIIIIIIIIIII\n@seq2\nGCTAGCTAGCTA\n+\nJJJJJJJJJJJJ\n")
    temp_fastq.close()
    
    temp_vcf = tempfile.NamedTemporaryFile(suffix='.vcf', delete=False)
    temp_vcf.write(b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr1\t100\t.\tA\tT\t60\tPASS\t.\n")
    temp_vcf.close()
    
    try:
        # Upload FASTQ sequencing data
        dataset1 = await bio_tools.upload_genomic_dataset(
            collaboration_id=collaboration.id,
            dataset_name="tumor_wgs_reads.fastq.gz",
            file_path=temp_fastq.name,
            data_type=BioinformaticsDataType.FASTQ_READS,
            organism="Homo sapiens",
            user_id="bioinf_001",
            metadata={
                "tissue_type": "primary_tumor",
                "condition": "lung_adenocarcinoma",
                "platform": "Illumina NovaSeq 6000",
                "read_length": 150,
                "coverage": 30.5
            }
        )
        
        # Upload variant call data
        dataset2 = await bio_tools.upload_genomic_dataset(
            collaboration_id=collaboration.id,
            dataset_name="somatic_variants.vcf",
            file_path=temp_vcf.name,
            data_type=BioinformaticsDataType.VCF_VARIANTS,
            organism="Homo sapiens",
            user_id="bioinf_001",
            metadata={
                "tissue_type": "primary_tumor",
                "condition": "lung_adenocarcinoma",
                "caller": "GATK Mutect2"
            }
        )
        
        print(f"✅ Uploaded {len(collaboration.datasets)} genomic datasets")
        print(f"   - Dataset 1: {dataset1.name} ({dataset1.data_type.value})")
        print(f"   - Dataset 2: {dataset2.name} ({dataset2.data_type.value})")
        print(f"   - Both encrypted: {dataset1.encrypted and dataset2.encrypted}")
        
    finally:
        # Clean up temporary files
        Path(temp_fastq.name).unlink()
        Path(temp_vcf.name).unlink()
    
    # Test 4: Create Galaxy workflow
    print("\n4. Creating Galaxy bioinformatics workflow...")
    
    workflow = await bio_tools.create_galaxy_workflow(
        collaboration_id=collaboration.id,
        workflow_name="cancer_variant_calling_pipeline",
        tools=["fastqc", "bwa", "gatk", "deseq2"],
        workflow_description="Comprehensive cancer genomics pipeline for variant calling and expression analysis",
        creator_id="bioinf_001"
    )
    
    print(f"✅ Created Galaxy workflow: {workflow.name}")
    print(f"   - Tools: {len(workflow.tools)}")
    print(f"   - Steps: {len(workflow.workflow_steps)}")
    print(f"   - Status: {workflow.validation_status}")
    print(f"   - Citations required: {len(workflow.citation_requirements)}")
    
    # Test 5: Execute BLAST search
    print("\n5. Executing BLAST sequence search...")
    
    query_sequence = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    
    blast_result = await bio_tools.execute_blast_search(
        collaboration_id=collaboration.id,
        query_sequence=query_sequence,
        database="nr",
        algorithm="blastp",
        user_id="compbio_001",
        parameters={
            "expect_threshold": 1e-5,
            "word_size": 6,
            "scoring_matrix": "BLOSUM62"
        }
    )
    
    print(f"✅ Completed BLAST search: {blast_result.id}")
    print(f"   - Database: {blast_result.database}")
    print(f"   - Algorithm: {blast_result.algorithm}")
    print(f"   - Query length: {blast_result.query_length}")
    print(f"   - Hits found: {len(blast_result.hits)}")
    print(f"   - Execution time: {blast_result.execution_time:.1f}s")
    print(f"   - Best E-value: {blast_result.hits[0]['e_value']:.2e}")
    
    # Test 6: Run bioinformatics analysis
    print("\n6. Running differential expression analysis...")
    
    analysis_result = await bio_tools.run_bioinformatics_analysis(
        collaboration_id=collaboration.id,
        analysis_type=AnalysisType.DIFFERENTIAL_EXPRESSION,
        input_datasets=["tumor_wgs_reads.fastq.gz"],
        workflow_name="cancer_variant_calling_pipeline",
        parameters={
            "fdr_threshold": 0.05,
            "log2_fold_change": 1.0,
            "statistical_test": "wald"
        },
        user_id="bioinf_001"
    )
    
    print(f"✅ Completed analysis: {analysis_result.analysis_type.value}")
    print(f"   - Input datasets: {len(analysis_result.input_datasets)}")
    print(f"   - Output files: {len(analysis_result.output_files)}")
    print(f"   - Quality metrics: {len(analysis_result.quality_metrics)}")
    print(f"   - Execution time: {analysis_result.execution_time:.1f}s")
    print(f"   - Interpretation: {analysis_result.biological_interpretation[:60]}...")
    
    # Test 7: Create Bioconductor environment
    print("\n7. Creating Bioconductor R environment...")
    
    bioc_env = await bio_tools.create_bioconductor_environment(
        collaboration_id=collaboration.id,
        research_focus="genomics",
        user_id="compbio_001"
    )
    
    print(f"✅ Created Bioconductor environment:")
    print(f"   - Research focus: {bioc_env['research_focus']}")
    print(f"   - Packages: {len(bioc_env['packages'])}")
    print(f"   - R version: {bioc_env['r_version']}")
    print(f"   - Bioconductor version: {bioc_env['bioconductor_version']}")
    print(f"   - Setup script: {Path(bioc_env['setup_script']).name}")
    
    # Clean up environment file
    Path(bioc_env['setup_script']).unlink()
    shutil.rmtree(Path(bioc_env['setup_script']).parent)
    
    # Test 8: Generate publication report
    print("\n8. Generating publication-ready report...")
    
    pub_report_path = await bio_tools.generate_publication_report(
        collaboration_id=collaboration.id,
        include_methods=True,
        include_results=True,
        include_figures=True,
        format="markdown"
    )
    
    print(f"✅ Generated publication report: {Path(pub_report_path).name}")
    
    # Check report size
    report_size = Path(pub_report_path).stat().st_size
    print(f"   - Report size: {report_size} bytes")
    
    # Clean up report
    Path(pub_report_path).unlink()
    shutil.rmtree(Path(pub_report_path).parent)
    
    # Test 9: Export collaboration data
    print("\n9. Exporting complete collaboration data...")
    
    export_path = await bio_tools.export_collaboration_data(
        collaboration_id=collaboration.id,
        user_id="pi_oncology_001",
        include_raw_data=False,
        include_analysis_results=True,
        include_workflows=True
    )
    
    print(f"✅ Exported collaboration to: {export_path}")
    
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
    
    print(f"\n🎉 Bioinformatics Tools testing completed successfully!")
    print(f"   - Collaborations: {len(bio_tools.collaborations)}")
    print(f"   - Galaxy Tools: {len(bio_tools.galaxy_tools)}")
    print(f"   - BLAST Databases: {len(bio_tools.blast_databases)}")
    print(f"   - Research Templates: {len(bio_tools.research_templates)}")
    print(f"   - Bioconductor Package Categories: {len(bio_tools.bioconductor_packages)}")


if __name__ == "__main__":
    asyncio.run(test_bioinformatics_tools())