"""
PRSM Evidence Generation Framework
=================================

🎯 PURPOSE:
Comprehensive evidence generation framework for PRSM system validation,
investment readiness, and compliance documentation.

🚀 KEY CAPABILITIES:
- Automated evidence collection across all PRSM subsystems
- Investment-ready documentation and financial projections
- Compliance and audit trail generation
- Multi-format output (JSON, HTML, PDF, Markdown)
- Secure evidence packaging and validation
- Executive summaries and stakeholder reports

📦 COMPONENTS:
- EvidenceGenerator: Core evidence generation engine
- EvidenceType: Available evidence types
- EvidenceFormat: Supported output formats
- CLI tools for automated evidence generation
- Demo scripts and comprehensive documentation

🔧 USAGE:
    from evidence import generate_investment_package, generate_system_health_report
    
    # Generate investment package
    package_id = await generate_investment_package()
    
    # Generate system health report
    health_id = await generate_system_health_report()
    
    # Custom evidence generation
    generator = EvidenceGenerator()
    custom_id = await generator.generate_comprehensive_evidence_package(
        evidence_types=[EvidenceType.SECURITY_ASSESSMENT],
        formats=[EvidenceFormat.JSON, EvidenceFormat.PDF]
    )
"""

from .evidence_framework import (
    EvidenceGenerator,
    EvidenceType,
    EvidenceFormat,
    EvidenceMetadata,
    EvidenceContent,
    generate_investment_package,
    generate_system_health_report
)

__all__ = [
    "EvidenceGenerator",
    "EvidenceType",
    "EvidenceFormat", 
    "EvidenceMetadata",
    "EvidenceContent",
    "generate_investment_package",
    "generate_system_health_report"
]

__version__ = "1.0.0"