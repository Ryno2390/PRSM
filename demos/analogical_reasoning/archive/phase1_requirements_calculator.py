#!/usr/bin/env python3
"""
Phase 1 Week 1 Requirements Calculator
Calculates exact storage and compute requirements for arXiv + PubMed ingestion

This analysis determines:
1. Exact data volumes for arXiv and PubMed Open Access
2. Storage requirements at each processing stage
3. Compute requirements and processing time
4. What can be done with current hardware vs cloud needs
"""

import math
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class StorageRequirement:
    """Storage requirement for a processing stage"""
    stage_name: str
    description: str
    size_gb: float
    temporary: bool  # Can be deleted after processing

@dataclass
class ComputeRequirement:
    """Compute requirement for processing"""
    task_name: str
    cpu_cores_needed: int
    ram_gb_needed: int
    processing_hours: float
    parallelizable: bool

@dataclass
class HardwareAssessment:
    """Assessment of hardware capabilities"""
    available_storage_gb: float
    required_storage_gb: float
    storage_shortfall_gb: float
    can_handle_locally: bool
    cloud_storage_needed_gb: float
    estimated_cost_usd: float

class Phase1RequirementsCalculator:
    """Calculates exact requirements for Phase 1 Week 1 ingestion"""
    
    def __init__(self):
        # User's current hardware
        self.laptop_storage_gb = 308.45
        self.external_drive_gb = 1000
        self.total_available_storage = self.laptop_storage_gb + self.external_drive_gb
        
        # Assume typical laptop specs
        self.laptop_cpu_cores = 8  # Modern laptop
        self.laptop_ram_gb = 16    # Typical for development
        
        # Phase 1 Week 1 targets
        self.arxiv_papers = 2_400_000
        self.pubmed_oa_papers = 8_500_000
        self.total_phase1_papers = self.arxiv_papers + self.pubmed_oa_papers
        
        # File size estimates (based on research)
        self.avg_paper_pdf_mb = 2.5    # Average scientific PDF
        self.avg_paper_text_kb = 50    # Extracted text
        self.avg_soc_data_kb = 5       # Structured SOC data per paper
        self.avg_patterns_kb = 2       # Pattern data per paper
        
        # Processing requirements
        self.api_calls_per_paper = 15  # SOC extraction + pattern analysis
        self.seconds_per_api_call = 2  # Average API response time
        self.papers_processed_parallel = 50  # Concurrent processing limit
    
    def calculate_storage_requirements(self) -> List[StorageRequirement]:
        """Calculate storage requirements for each processing stage"""
        
        requirements = []
        
        # Stage 1: Raw PDF Storage
        total_pdf_gb = (self.total_phase1_papers * self.avg_paper_pdf_mb) / 1024
        requirements.append(StorageRequirement(
            stage_name="Raw PDFs",
            description="Downloaded PDF files from arXiv and PubMed",
            size_gb=total_pdf_gb,
            temporary=True  # Can delete after text extraction
        ))
        
        # Stage 2: Extracted Text
        total_text_gb = (self.total_phase1_papers * self.avg_paper_text_kb) / (1024 * 1024)
        requirements.append(StorageRequirement(
            stage_name="Extracted Text",
            description="Plain text extracted from PDFs",
            size_gb=total_text_gb,
            temporary=True  # Can delete after SOC extraction
        ))
        
        # Stage 3: SOC Data
        total_soc_gb = (self.total_phase1_papers * self.avg_soc_data_kb) / (1024 * 1024)
        requirements.append(StorageRequirement(
            stage_name="SOC Data",
            description="Structured Subjects-Objects-Concepts data",
            size_gb=total_soc_gb,
            temporary=False  # Keep permanently
        ))
        
        # Stage 4: Pattern Data
        total_patterns_gb = (self.total_phase1_papers * self.avg_patterns_kb) / (1024 * 1024)
        requirements.append(StorageRequirement(
            stage_name="Pattern Catalog",
            description="Extracted patterns from SOCs",
            size_gb=total_patterns_gb,
            temporary=False  # Keep permanently
        ))
        
        # Stage 5: Database/Index
        database_gb = total_soc_gb * 0.3  # ~30% overhead for database indexes
        requirements.append(StorageRequirement(
            stage_name="Database/Indexes",
            description="Database storage and search indexes",
            size_gb=database_gb,
            temporary=False
        ))
        
        # Stage 6: Working Space
        working_space_gb = max(50, total_pdf_gb * 0.1)  # 10% of PDF storage for temp files
        requirements.append(StorageRequirement(
            stage_name="Working Space",
            description="Temporary processing files and buffers",
            size_gb=working_space_gb,
            temporary=True
        ))
        
        return requirements
    
    def calculate_compute_requirements(self) -> List[ComputeRequirement]:
        """Calculate compute requirements for processing"""
        
        requirements = []
        
        # Task 1: PDF Download
        download_hours = (self.total_phase1_papers * 0.5) / 3600  # 0.5 seconds per download
        requirements.append(ComputeRequirement(
            task_name="PDF Download",
            cpu_cores_needed=2,
            ram_gb_needed=4,
            processing_hours=download_hours / 10,  # 10 parallel downloads
            parallelizable=True
        ))
        
        # Task 2: Text Extraction
        extraction_hours = (self.total_phase1_papers * 3) / 3600  # 3 seconds per PDF
        requirements.append(ComputeRequirement(
            task_name="Text Extraction",
            cpu_cores_needed=4,
            ram_gb_needed=8,
            processing_hours=extraction_hours / 4,  # 4 parallel processes
            parallelizable=True
        ))
        
        # Task 3: SOC Extraction (API calls)
        total_api_calls = self.total_phase1_papers * self.api_calls_per_paper
        api_hours = (total_api_calls * self.seconds_per_api_call) / 3600
        requirements.append(ComputeRequirement(
            task_name="SOC Extraction",
            cpu_cores_needed=2,
            ram_gb_needed=4,
            processing_hours=api_hours / self.papers_processed_parallel,
            parallelizable=True
        ))
        
        # Task 4: Pattern Extraction
        pattern_hours = (self.total_phase1_papers * 1) / 3600  # 1 second per paper
        requirements.append(ComputeRequirement(
            task_name="Pattern Extraction",
            cpu_cores_needed=4,
            ram_gb_needed=8,
            processing_hours=pattern_hours / 4,
            parallelizable=True
        ))
        
        return requirements
    
    def assess_hardware_capability(self) -> HardwareAssessment:
        """Assess if current hardware can handle Phase 1"""
        
        storage_requirements = self.calculate_storage_requirements()
        
        # Calculate total storage needed
        max_concurrent_storage = 0
        permanent_storage = 0
        
        # Find maximum concurrent storage (worst case scenario)
        temp_storage = sum(req.size_gb for req in storage_requirements if req.temporary)
        permanent_storage = sum(req.size_gb for req in storage_requirements if not req.temporary)
        max_concurrent_storage = temp_storage + permanent_storage
        
        # Calculate what can be optimized
        optimized_storage = permanent_storage + max(
            req.size_gb for req in storage_requirements if req.temporary
        )  # Process one temporary stage at a time
        
        # Storage assessment
        storage_shortfall = max(0, optimized_storage - self.total_available_storage)
        can_handle_locally = storage_shortfall == 0
        
        # Cost estimation for cloud storage if needed
        cloud_storage_cost = 0
        if storage_shortfall > 0:
            # AWS S3 pricing: ~$0.023/GB/month
            cloud_storage_cost = storage_shortfall * 0.023 * 3  # 3 months
        
        return HardwareAssessment(
            available_storage_gb=self.total_available_storage,
            required_storage_gb=optimized_storage,
            storage_shortfall_gb=storage_shortfall,
            can_handle_locally=can_handle_locally,
            cloud_storage_needed_gb=storage_shortfall,
            estimated_cost_usd=cloud_storage_cost
        )
    
    def generate_implementation_plan(self):
        """Generate implementation plan based on hardware assessment"""
        
        print("💻 PHASE 1 WEEK 1 HARDWARE REQUIREMENTS ANALYSIS")
        print("=" * 70)
        print(f"Target: arXiv ({self.arxiv_papers:,}) + PubMed OA ({self.pubmed_oa_papers:,}) = {self.total_phase1_papers:,} papers")
        print()
        
        # Storage Analysis
        print("📊 STORAGE REQUIREMENTS ANALYSIS")
        print("-" * 50)
        
        storage_requirements = self.calculate_storage_requirements()
        
        print(f"{'Stage':<20} {'Size (GB)':<12} {'Temporary':<10} {'Description'}")
        print("-" * 70)
        
        total_temp = 0
        total_permanent = 0
        
        for req in storage_requirements:
            temp_status = "Yes" if req.temporary else "No"
            print(f"{req.stage_name:<20} {req.size_gb:<12.1f} {temp_status:<10} {req.description}")
            
            if req.temporary:
                total_temp += req.size_gb
            else:
                total_permanent += req.size_gb
        
        print("-" * 70)
        print(f"{'Permanent Storage':<20} {total_permanent:<12.1f}")
        print(f"{'Peak Temp Storage':<20} {max(req.size_gb for req in storage_requirements if req.temporary):<12.1f}")
        print(f"{'Optimized Total':<20} {total_permanent + max(req.size_gb for req in storage_requirements if req.temporary):<12.1f}")
        
        # Hardware Assessment
        print(f"\n💻 CURRENT HARDWARE ASSESSMENT")
        print("-" * 50)
        
        hardware = self.assess_hardware_capability()
        
        print(f"Available Storage:")
        print(f"   Laptop: {self.laptop_storage_gb:.1f} GB")
        print(f"   External Drive: {self.external_drive_gb:.1f} GB")
        print(f"   Total: {hardware.available_storage_gb:.1f} GB")
        
        print(f"\nRequired Storage: {hardware.required_storage_gb:.1f} GB")
        print(f"Shortfall: {hardware.storage_shortfall_gb:.1f} GB")
        
        if hardware.can_handle_locally:
            print("✅ CAN HANDLE LOCALLY!")
        else:
            print(f"❌ NEED ADDITIONAL STORAGE: {hardware.storage_shortfall_gb:.1f} GB")
            print(f"   Estimated cloud cost: ${hardware.estimated_cost_usd:.2f}")
        
        # Compute Assessment
        print(f"\n⚡ COMPUTE REQUIREMENTS")
        print("-" * 50)
        
        compute_requirements = self.calculate_compute_requirements()
        
        print(f"{'Task':<20} {'CPU Cores':<10} {'RAM (GB)':<10} {'Hours':<10}")
        print("-" * 55)
        
        total_hours = 0
        max_cores = 0
        max_ram = 0
        
        for req in compute_requirements:
            print(f"{req.task_name:<20} {req.cpu_cores_needed:<10} {req.ram_gb_needed:<10} {req.processing_hours:<10.1f}")
            total_hours += req.processing_hours
            max_cores = max(max_cores, req.cpu_cores_needed)
            max_ram = max(max_ram, req.ram_gb_needed)
        
        print("-" * 55)
        print(f"Peak Requirements: {max_cores} cores, {max_ram} GB RAM")
        print(f"Total Processing: {total_hours:.1f} hours ({total_hours/24:.1f} days)")
        
        print(f"\nLaptop Capability:")
        print(f"   Available: {self.laptop_cpu_cores} cores, {self.laptop_ram_gb} GB RAM")
        
        if max_cores <= self.laptop_cpu_cores and max_ram <= self.laptop_ram_gb:
            print("✅ LAPTOP CAN HANDLE COMPUTE REQUIREMENTS!")
        else:
            print("❌ LAPTOP INSUFFICIENT FOR COMPUTE")
            if max_cores > self.laptop_cpu_cores:
                print(f"   Need {max_cores - self.laptop_cpu_cores} more CPU cores")
            if max_ram > self.laptop_ram_gb:
                print(f"   Need {max_ram - self.laptop_ram_gb} more GB RAM")
        
        return hardware
    
    def calculate_cost_breakdown(self, hardware: HardwareAssessment):
        """Calculate detailed cost breakdown"""
        
        print(f"\n💰 COST BREAKDOWN")
        print("-" * 50)
        
        # API costs (main expense)
        total_api_calls = self.total_phase1_papers * self.api_calls_per_paper
        api_cost = total_api_calls * 0.002  # $0.002 per API call
        
        print(f"API Processing:")
        print(f"   Total API calls: {total_api_calls:,}")
        print(f"   Cost per call: $0.002")
        print(f"   Total API cost: ${api_cost:,.2f}")
        
        # Storage costs
        storage_cost = hardware.estimated_cost_usd
        print(f"\nCloud Storage (if needed): ${storage_cost:.2f}")
        
        # Compute costs (if cloud needed)
        if not hardware.can_handle_locally:
            # AWS EC2 pricing for processing
            compute_requirements = self.calculate_compute_requirements()
            total_hours = sum(req.processing_hours for req in compute_requirements)
            cloud_compute_cost = total_hours * 0.50  # ~$0.50/hour for suitable instance
            print(f"Cloud Compute: ${cloud_compute_cost:.2f}")
        else:
            cloud_compute_cost = 0
            print(f"Cloud Compute: $0.00 (local processing)")
        
        total_cost = api_cost + storage_cost + cloud_compute_cost
        
        print(f"\n🎯 TOTAL PHASE 1 COST: ${total_cost:,.2f}")
        print(f"   API Processing: ${api_cost:,.2f} ({api_cost/total_cost*100:.1f}%)")
        print(f"   Storage: ${storage_cost:.2f} ({storage_cost/total_cost*100:.1f}%)")
        print(f"   Compute: ${cloud_compute_cost:.2f} ({cloud_compute_cost/total_cost*100:.1f}%)")
        
        return total_cost
    
    def generate_recommendations(self, hardware: HardwareAssessment):
        """Generate specific recommendations"""
        
        print(f"\n🎯 IMPLEMENTATION RECOMMENDATIONS")
        print("=" * 70)
        
        if hardware.can_handle_locally:
            print("✅ LOCAL PROCESSING RECOMMENDED")
            print("\nImplementation Plan:")
            print("1. Use external drive for temporary PDF storage")
            print("2. Process papers in batches (delete PDFs after text extraction)")
            print("3. Keep only SOCs and patterns permanently")
            print("4. Run processing overnight (estimated 3-5 days)")
            print("5. Monitor disk space and clean temp files regularly")
            
        else:
            print("🌩️  HYBRID LOCAL + CLOUD RECOMMENDED")
            print(f"\nStorage Gap: {hardware.storage_shortfall_gb:.1f} GB")
            print("\nOption A - Additional External Storage:")
            additional_storage_gb = math.ceil(hardware.storage_shortfall_gb / 1000) * 1000
            external_drive_cost = additional_storage_gb * 0.05  # ~$0.05/GB for external drives
            print(f"   Buy {additional_storage_gb} GB external drive: ~${external_drive_cost:.0f}")
            print(f"   One-time cost, reusable for future phases")
            
            print(f"\nOption B - Cloud Storage:")
            print(f"   AWS S3: ${hardware.estimated_cost_usd:.2f} for 3 months")
            print(f"   No upfront cost, pay as you go")
            
            print(f"\nOption C - Optimize Processing:")
            print("   Process in smaller batches (1M papers at a time)")
            print("   Delete intermediate files aggressively")
            print("   May extend timeline but reduce storage needs")
        
        print(f"\n📅 TIMELINE ESTIMATE:")
        compute_requirements = self.calculate_compute_requirements()
        total_processing_days = sum(req.processing_hours for req in compute_requirements) / 24
        
        if hardware.can_handle_locally:
            print(f"   Total processing: {total_processing_days:.1f} days")
            print("   Recommended: Start this weekend!")
        else:
            print(f"   With optimized batching: {total_processing_days * 1.5:.1f} days")
            print("   With cloud acceleration: {total_processing_days * 0.5:.1f} days")

def main():
    calculator = Phase1RequirementsCalculator()
    
    # Generate analysis
    hardware = calculator.generate_implementation_plan()
    total_cost = calculator.calculate_cost_breakdown(hardware)
    calculator.generate_recommendations(hardware)
    
    print(f"\n🚀 EXECUTIVE SUMMARY")
    print("=" * 70)
    print(f"Phase 1 Target: {calculator.total_phase1_papers:,} papers")
    print(f"Storage Required: {hardware.required_storage_gb:.1f} GB")
    print(f"Current Available: {hardware.available_storage_gb:.1f} GB")
    print(f"Can Handle Locally: {'YES' if hardware.can_handle_locally else 'NO'}")
    print(f"Total Cost: ${total_cost:,.2f}")
    print(f"Recommendation: {'Start immediately with local processing' if hardware.can_handle_locally else 'Consider hybrid approach or additional storage'}")

if __name__ == "__main__":
    main()