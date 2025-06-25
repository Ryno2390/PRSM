#!/usr/bin/env python3
"""
PRSM Evidence Generation CLI
===========================

🎯 PURPOSE:
Command-line interface for generating evidence packages for investors,
stakeholders, auditors, and regulatory compliance.

🚀 USAGE:
    python evidence/evidence_cli.py investment-package
    python evidence/evidence_cli.py system-health
    python evidence/evidence_cli.py --type security --format pdf
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.text import Text

from evidence_framework import (
    EvidenceGenerator, EvidenceType, EvidenceFormat,
    generate_investment_package, generate_system_health_report
)

console = Console()


class EvidenceCLI:
    """Command-line interface for evidence generation"""
    
    def __init__(self):
        self.generator = EvidenceGenerator()
    
    async def generate_evidence(self,
                              evidence_types: List[str],
                              formats: List[str],
                              classification: str = "internal",
                              output_dir: Optional[str] = None) -> str:
        """Generate evidence package with specified parameters"""
        
        # Convert string types to enums
        try:
            type_enums = [EvidenceType(t) for t in evidence_types]
        except ValueError as e:
            console.print(f"❌ Invalid evidence type: {e}")
            return ""
        
        try:
            format_enums = [EvidenceFormat(f) for f in formats]
        except ValueError as e:
            console.print(f"❌ Invalid format: {e}")
            return ""
        
        # Set output directory if provided
        if output_dir:
            self.generator.output_dir = Path(output_dir)
            self.generator.output_dir.mkdir(exist_ok=True)
        
        console.print(Panel(
            f"🎯 Generating PRSM Evidence Package\n"
            f"Types: {', '.join(evidence_types)}\n"
            f"Formats: {', '.join(formats)}\n"
            f"Classification: {classification}",
            title="Evidence Generation",
            border_style="blue"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Generating evidence package...", total=None)
            
            try:
                package_id = await self.generator.generate_comprehensive_evidence_package(
                    evidence_types=type_enums,
                    formats=format_enums,
                    classification=classification
                )
                
                progress.update(task, description="✅ Evidence package generated successfully")
                
                # Display results
                await self._display_package_info(package_id)
                
                return package_id
                
            except Exception as e:
                progress.update(task, description=f"❌ Failed: {str(e)}")
                console.print(f"Error generating evidence: {str(e)}")
                return ""
    
    async def _display_package_info(self, package_id: str):
        """Display information about generated package"""
        
        evidence = await self.generator.get_evidence_package(package_id)
        if not evidence:
            console.print("❌ Package not found")
            return
        
        # Create info table
        table = Table(title=f"Evidence Package: {package_id}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Evidence ID", evidence.metadata.evidence_id)
        table.add_row("Type", evidence.metadata.evidence_type.value)
        table.add_row("Generated", evidence.metadata.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
        table.add_row("Classification", evidence.metadata.classification)
        table.add_row("Validation Hash", evidence.metadata.validation_hash[:16] + "...")
        
        if evidence.metadata.tags:
            table.add_row("Tags", ", ".join(evidence.metadata.tags))
        
        console.print(table)
        
        # Display executive summary if available
        if evidence.executive_summary:
            exec_summary = evidence.executive_summary.get("executive_overview", {})
            if exec_summary:
                console.print("\n📊 Executive Summary:")
                console.print(f"   System Status: {exec_summary.get('system_status', 'N/A')}")
                console.print(f"   Health Score: {exec_summary.get('overall_health_score', 0):.1%}")
                console.print(f"   Performance Grade: {exec_summary.get('performance_grade', 'N/A')}")
                console.print(f"   Readiness Level: {exec_summary.get('readiness_level', 'N/A')}")
        
        # Display output files
        console.print(f"\n📁 Output Directory: {self.generator.output_dir}")
        
        # List generated files
        output_files = list(self.generator.output_dir.glob(f"{evidence.metadata.evidence_id}*"))
        if output_files:
            console.print("\n📄 Generated Files:")
            for file_path in output_files:
                file_size = file_path.stat().st_size / 1024  # KB
                console.print(f"   • {file_path.name} ({file_size:.1f} KB)")
    
    async def list_packages(self):
        """List all generated evidence packages"""
        
        packages = await self.generator.list_evidence_packages()
        
        if not packages:
            console.print("No evidence packages found.")
            return
        
        table = Table(title="Evidence Packages")
        table.add_column("Evidence ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Generated", style="yellow")
        table.add_column("Classification", style="red")
        
        for package in packages:
            generated_time = datetime.fromisoformat(package["generated_at"].replace('Z', '+00:00'))
            table.add_row(
                package["evidence_id"][:16] + "...",
                package["evidence_type"],
                generated_time.strftime("%Y-%m-%d %H:%M"),
                package["classification"]
            )
        
        console.print(table)
    
    async def validate_package(self, package_id: str):
        """Validate evidence package integrity"""
        
        console.print(f"🔍 Validating package: {package_id}")
        
        is_valid = await self.generator.validate_evidence_integrity(package_id)
        
        if is_valid:
            console.print("✅ Package integrity validated successfully")
        else:
            console.print("❌ Package integrity validation failed")
        
        return is_valid
    
    async def quick_investment_package(self):
        """Generate quick investment package"""
        
        console.print("🚀 Generating Investment Package...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Creating comprehensive investment package...", total=None)
            
            try:
                package_id = await generate_investment_package()
                progress.update(task, description="✅ Investment package created")
                
                await self._display_package_info(package_id)
                return package_id
                
            except Exception as e:
                progress.update(task, description=f"❌ Failed: {str(e)}")
                console.print(f"Error: {str(e)}")
                return ""
    
    async def quick_health_report(self):
        """Generate quick system health report"""
        
        console.print("🔧 Generating System Health Report...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Running system health assessment...", total=None)
            
            try:
                package_id = await generate_system_health_report()
                progress.update(task, description="✅ Health report generated")
                
                await self._display_package_info(package_id)
                return package_id
                
            except Exception as e:
                progress.update(task, description=f"❌ Failed: {str(e)}")
                console.print(f"Error: {str(e)}")
                return ""


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    
    parser = argparse.ArgumentParser(
        description="PRSM Evidence Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s investment-package                    # Generate investment package
  %(prog)s system-health                         # Generate system health report
  %(prog)s --type system_health --format json   # Custom evidence generation
  %(prog)s list                                  # List generated packages
  %(prog)s validate PACKAGE_ID                  # Validate package integrity

Evidence Types:
  system_health, performance_metrics, integration_validation,
  security_assessment, financial_projections, technical_architecture,
  compliance_audit, investment_package, executive_summary

Formats:
  json, html, markdown, zip
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Investment package command
    investment_parser = subparsers.add_parser(
        "investment-package",
        help="Generate comprehensive investment package"
    )
    
    # System health command
    health_parser = subparsers.add_parser(
        "system-health", 
        help="Generate system health report"
    )
    
    # Custom evidence command
    custom_parser = subparsers.add_parser(
        "generate",
        help="Generate custom evidence package"
    )
    custom_parser.add_argument(
        "--type", 
        nargs="+",
        choices=[t.value for t in EvidenceType],
        default=["system_health"],
        help="Evidence types to generate"
    )
    custom_parser.add_argument(
        "--format",
        nargs="+", 
        choices=[f.value for f in EvidenceFormat],
        default=["json", "html"],
        help="Output formats"
    )
    custom_parser.add_argument(
        "--classification",
        choices=["internal", "confidential", "public"],
        default="internal",
        help="Security classification"
    )
    custom_parser.add_argument(
        "--output",
        help="Output directory"
    )
    
    # List packages command
    list_parser = subparsers.add_parser(
        "list",
        help="List generated evidence packages"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate evidence package integrity"
    )
    validate_parser.add_argument(
        "package_id",
        help="Package ID to validate"
    )
    
    return parser


async def main():
    """Main CLI entry point"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    cli = EvidenceCLI()
    
    try:
        if args.command == "investment-package":
            package_id = await cli.quick_investment_package()
            if package_id:
                console.print(f"\n🎉 Investment package generated: {package_id}")
        
        elif args.command == "system-health":
            package_id = await cli.quick_health_report()
            if package_id:
                console.print(f"\n🎉 Health report generated: {package_id}")
        
        elif args.command == "generate":
            package_id = await cli.generate_evidence(
                evidence_types=args.type,
                formats=args.format,
                classification=args.classification,
                output_dir=args.output
            )
            if package_id:
                console.print(f"\n🎉 Evidence package generated: {package_id}")
        
        elif args.command == "list":
            await cli.list_packages()
        
        elif args.command == "validate":
            is_valid = await cli.validate_package(args.package_id)
            if not is_valid:
                sys.exit(1)
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        console.print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())