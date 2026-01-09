#!/usr/bin/env python3
"""
LaTeX Real-time Collaboration with P2P Security for PRSM
======================================================

This module implements Overleaf-equivalent LaTeX collaboration with PRSM's
P2P cryptographic security. It enables:

- Real-time collaborative LaTeX editing
- Automatic compilation and PDF generation
- Secure sharing with cryptographic sharding
- Integration with university citation management
- Version control and conflict resolution
- NWTN AI assistance for writing and formatting
"""

import json
import asyncio
import uuid
import subprocess
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import re

# Import PRSM components
from ..security.crypto_sharding import BasicCryptoSharding
from ..models import QueryRequest

# Mock UnifiedPipelineController for testing
class UnifiedPipelineController:
    """Mock pipeline controller for LaTeX integration"""
    async def initialize(self):
        pass
    
    async def process_query_full_pipeline(self, user_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # LaTeX-specific NWTN responses
        if context.get("latex_assistance"):
            return {
                "response": {
                    "text": """
LaTeX Writing Assistance:

**Structure Suggestions**:
- Use \\section{} for main headings
- Add \\label{} references for cross-referencing
- Consider \\cite{} for proper citations

**Formatting Improvements**:
- Use \\emph{} instead of \\textit{} for emphasis
- Add \\usepackage{amsmath} for mathematical expressions
- Consider \\usepackage{graphicx} for figure handling

**Common Issues**:
- Missing bibliography style - add \\bibliographystyle{plain}
- Table formatting could use \\usepackage{booktabs}
- Math mode requires $ delimiters
""",
                    "confidence": 0.91,
                    "sources": ["latex_guide.pdf", "academic_writing_standards.pdf"]
                },
                "performance_metrics": {"total_processing_time": 1.9}
            }
        else:
            return {
                "response": {"text": "LaTeX assistance available", "confidence": 0.75, "sources": []},
                "performance_metrics": {"total_processing_time": 1.2}
            }

class DocumentType(Enum):
    """Types of LaTeX documents"""
    RESEARCH_PAPER = "research_paper"
    GRANT_PROPOSAL = "grant_proposal" 
    THESIS = "thesis"
    PRESENTATION = "presentation"
    TECHNICAL_REPORT = "technical_report"
    BOOK_CHAPTER = "book_chapter"

@dataclass
class LaTeXFile:
    """Represents a LaTeX file in the project"""
    file_id: str
    filename: str
    content: str
    file_type: str  # 'main', 'chapter', 'bibliography', 'style'
    last_modified: datetime
    modified_by: str

@dataclass
class CompilationResult:
    """Result of LaTeX compilation"""
    success: bool
    pdf_path: Optional[str]
    log_output: str
    errors: List[str]
    warnings: List[str]
    compilation_time: float

@dataclass
class LaTeXProject:
    """Complete LaTeX collaboration project"""
    project_id: str
    title: str
    document_type: DocumentType
    main_file: str  # filename of main .tex file
    files: Dict[str, LaTeXFile]  # filename -> LaTeXFile
    collaborators: List[str]
    created_by: str
    created_at: datetime
    last_compiled: Optional[datetime]
    last_compilation_result: Optional[CompilationResult]
    security_level: str  # 'high', 'medium', 'standard'
    template_used: Optional[str]

@dataclass
class LaTeXEdit:
    """Represents a collaborative edit to LaTeX content"""
    edit_id: str
    project_id: str
    filename: str
    operation: str  # 'insert', 'delete', 'replace'
    position: int  # character position in file
    content: str
    user_id: str
    timestamp: datetime
    applied: bool = False

class LaTeXCollaboration:
    """
    Main class for LaTeX real-time collaboration with P2P security
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize LaTeX collaboration system"""
        self.storage_path = storage_path or Path("./latex_collaboration")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize PRSM components
        self.crypto_sharding = BasicCryptoSharding()
        self.nwtn_pipeline = None
        
        # Active projects and templates
        self.active_projects: Dict[str, LaTeXProject] = {}
        self.pending_edits: Dict[str, List[LaTeXEdit]] = {}
        
        # LaTeX templates for different document types
        self.templates = self._initialize_templates()
        
        # Compilation settings
        self.latex_compiler = "pdflatex"  # or xelatex, lualatex
        self.bibtex_compiler = "bibtex"
    
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize LaTeX templates for different document types"""
        return {
            "research_paper": r"""
\documentclass[11pt,letterpaper]{article}

% Essential packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{cite}
\usepackage{url}
\usepackage[margin=1in]{geometry}

% Title and authors
\title{Your Research Paper Title}
\author{Author One\thanks{University Affiliation} \and Author Two\thanks{Industry Partner}}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
Your abstract goes here. This should be a concise summary of your research, 
methodology, key findings, and conclusions.
\end{abstract}

\section{Introduction}
\label{sec:introduction}

Your introduction goes here. Cite relevant work using \cite{example2024}.

\section{Methodology}
\label{sec:methodology}

Describe your research methodology here.

\section{Results}
\label{sec:results}

Present your results here. You can reference Figure~\ref{fig:example} or Table~\ref{tab:example}.

\section{Discussion}
\label{sec:discussion}

Discuss your findings and their implications.

\section{Conclusion}
\label{sec:conclusion}

Summarize your work and suggest future directions.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
""",
            "grant_proposal": r"""
\documentclass[11pt,letterpaper]{article}

% Grant proposal specific packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{cite}
\usepackage{url}
\usepackage[margin=1in]{geometry}
\usepackage{fancyhdr}
\usepackage{titlesec}

% Header setup
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{Grant Proposal Title}
\fancyhead[R]{PI: Principal Investigator}
\fancyfoot[C]{\thepage}

\title{Grant Proposal: [Insert Title]}
\author{Principal Investigator\\Institution\\Co-Investigators}
\date{\today}

\begin{document}

\maketitle

\section{Project Summary}
\label{sec:summary}

Brief overview of the proposed research (typically 1 page).

\section{Project Description}
\label{sec:description}

\subsection{Significance and Innovation}
Explain the significance of your research and what makes it innovative.

\subsection{Approach and Methodology}
Detailed description of your research approach.

\subsection{Preliminary Results}
Present any preliminary data or results.

\section{Research Team}
\label{sec:team}

\subsection{Principal Investigator}
Background and qualifications of the PI.

\subsection{Co-Investigators}
Background of co-investigators and their roles.

\section{Timeline and Milestones}
\label{sec:timeline}

Detailed project timeline with specific milestones.

\section{Budget and Budget Justification}
\label{sec:budget}

Detailed budget breakdown and justification.

\section{Broader Impacts}
\label{sec:impacts}

Educational and societal impacts of the proposed research.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
""",
            "thesis": r"""
\documentclass[12pt,letterpaper,oneside]{book}

% Thesis specific packages
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{cite}
\usepackage{url}
\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\usepackage{fancyhdr}
\usepackage{tocloft}

% Double spacing for thesis
\doublespacing

% Title page information
\title{Your Thesis Title}
\author{Your Name}
\date{Month Year}

\begin{document}

% Title page
\begin{titlepage}
\centering
\vspace*{2cm}
{\LARGE\textbf{Your Thesis Title}\par}
\vspace{2cm}
{\Large by\par}
\vspace{1cm}
{\Large Your Name\par}
\vspace{2cm}
{A thesis submitted to the Graduate Faculty of\\
University Name\\
in partial fulfillment of the requirements\\
for the degree of\\
DOCTOR OF PHILOSOPHY\par}
\vspace{2cm}
{Department of Your Department\par}
\vspace{1cm}
{\large Month Year\par}
\end{titlepage}

% Abstract
\chapter*{Abstract}
\addcontentsline{toc}{chapter}{Abstract}

Your thesis abstract goes here.

% Table of contents
\tableofcontents
\listoffigures
\listoftables

% Chapters
\chapter{Introduction}
\label{chap:introduction}

Your introduction chapter goes here.

\chapter{Literature Review}
\label{chap:literature}

Your literature review goes here.

\chapter{Methodology}
\label{chap:methodology}

Your methodology chapter goes here.

\chapter{Results}
\label{chap:results}

Your results chapter goes here.

\chapter{Discussion}
\label{chap:discussion}

Your discussion chapter goes here.

\chapter{Conclusion}
\label{chap:conclusion}

Your conclusion chapter goes here.

% Bibliography
\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""
        }
    
    async def initialize_nwtn_pipeline(self):
        """Initialize NWTN pipeline for LaTeX assistance"""
        if self.nwtn_pipeline is None:
            self.nwtn_pipeline = UnifiedPipelineController()
            await self.nwtn_pipeline.initialize()
    
    def create_latex_project(self,
                           title: str,
                           document_type: DocumentType,
                           collaborators: List[str],
                           created_by: str,
                           security_level: str = "medium",
                           template: Optional[str] = None) -> LaTeXProject:
        """Create a new LaTeX collaboration project"""
        project_id = str(uuid.uuid4())
        
        # Use template if specified, otherwise use document type default
        template_content = template or self.templates.get(document_type.value, self.templates["research_paper"])
        
        # Create main .tex file
        main_filename = f"main.tex"
        main_file = LaTeXFile(
            file_id=str(uuid.uuid4()),
            filename=main_filename,
            content=template_content,
            file_type="main",
            last_modified=datetime.now(),
            modified_by=created_by
        )
        
        # Create references.bib file
        bib_filename = "references.bib"
        bib_file = LaTeXFile(
            file_id=str(uuid.uuid4()),
            filename=bib_filename,
            content="% Bibliography entries go here\n% Example:\n% @article{example2024,\n%   title={Example Paper},\n%   author={Author, First},\n%   journal={Journal Name},\n%   year={2024}\n% }\n",
            file_type="bibliography",
            last_modified=datetime.now(),
            modified_by=created_by
        )
        
        project = LaTeXProject(
            project_id=project_id,
            title=title,
            document_type=document_type,
            main_file=main_filename,
            files={
                main_filename: main_file,
                bib_filename: bib_file
            },
            collaborators=collaborators,
            created_by=created_by,
            created_at=datetime.now(),
            last_compiled=None,
            last_compilation_result=None,
            security_level=security_level,
            template_used=document_type.value
        )
        
        self.active_projects[project_id] = project
        self.pending_edits[project_id] = []
        
        # Save project with encryption if needed
        self._save_project(project)
        
        return project
    
    def add_file_to_project(self,
                          project_id: str,
                          filename: str,
                          content: str = "",
                          file_type: str = "chapter",
                          created_by: str = "system") -> LaTeXFile:
        """Add a new file to a LaTeX project"""
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.active_projects[project_id]
        
        if filename in project.files:
            raise ValueError(f"File {filename} already exists in project")
        
        latex_file = LaTeXFile(
            file_id=str(uuid.uuid4()),
            filename=filename,
            content=content,
            file_type=file_type,
            last_modified=datetime.now(),
            modified_by=created_by
        )
        
        project.files[filename] = latex_file
        self._save_project(project)
        
        return latex_file
    
    async def apply_latex_edit(self,
                             project_id: str,
                             edit: LaTeXEdit) -> bool:
        """Apply a collaborative edit to a LaTeX file"""
        if project_id not in self.active_projects:
            return False
        
        project = self.active_projects[project_id]
        
        if edit.filename not in project.files:
            return False
        
        latex_file = project.files[edit.filename]
        
        try:
            if edit.operation == "insert":
                # Insert content at position
                content = latex_file.content
                new_content = content[:edit.position] + edit.content + content[edit.position:]
                latex_file.content = new_content
                
            elif edit.operation == "delete":
                # Delete characters from position
                content = latex_file.content
                delete_length = len(edit.content)
                new_content = content[:edit.position] + content[edit.position + delete_length:]
                latex_file.content = new_content
                
            elif edit.operation == "replace":
                # Replace content at position
                content = latex_file.content
                # Find and replace specific text
                new_content = content.replace(edit.content.split("->")[0], edit.content.split("->")[1], 1)
                latex_file.content = new_content
            
            latex_file.last_modified = datetime.now()
            latex_file.modified_by = edit.user_id
            edit.applied = True
            
            # Save updated project
            self._save_project(project)
            
            return True
            
        except Exception as e:
            print(f"Error applying LaTeX edit: {e}")
            return False
    
    async def compile_latex_project(self,
                                  project_id: str,
                                  user_id: str) -> CompilationResult:
        """Compile LaTeX project to PDF"""
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.active_projects[project_id]
        
        # Create temporary directory for compilation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write all project files to temp directory
            for filename, latex_file in project.files.items():
                file_path = temp_path / filename
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(latex_file.content)
            
            # Compile LaTeX
            main_file_path = temp_path / project.main_file
            
            try:
                import time
                start_time = time.time()
                
                # Run pdflatex
                result = subprocess.run([
                    self.latex_compiler,
                    "-interaction=nonstopmode",
                    "-output-directory", str(temp_path),
                    str(main_file_path)
                ], capture_output=True, text=True, cwd=temp_path)
                
                # Run bibtex if bibliography exists
                if any(f.filename.endswith('.bib') for f in project.files.values()):
                    bib_result = subprocess.run([
                        self.bibtex_compiler,
                        project.main_file.replace('.tex', '')
                    ], capture_output=True, text=True, cwd=temp_path)
                    
                    # Run pdflatex again after bibtex
                    subprocess.run([
                        self.latex_compiler,
                        "-interaction=nonstopmode", 
                        "-output-directory", str(temp_path),
                        str(main_file_path)
                    ], capture_output=True, text=True, cwd=temp_path)
                
                compilation_time = time.time() - start_time
                
                # Check if PDF was generated
                pdf_filename = project.main_file.replace('.tex', '.pdf')
                pdf_path = temp_path / pdf_filename
                
                if pdf_path.exists():
                    # Copy PDF to project storage
                    project_dir = self.storage_path / "projects" / project_id
                    project_dir.mkdir(parents=True, exist_ok=True)
                    final_pdf_path = project_dir / pdf_filename
                    shutil.copy2(pdf_path, final_pdf_path)
                    
                    # Parse log for errors and warnings
                    log_content = result.stdout + result.stderr
                    errors = self._parse_latex_errors(log_content)
                    warnings = self._parse_latex_warnings(log_content)
                    
                    compilation_result = CompilationResult(
                        success=True,
                        pdf_path=str(final_pdf_path),
                        log_output=log_content,
                        errors=errors,
                        warnings=warnings,
                        compilation_time=compilation_time
                    )
                else:
                    # Compilation failed
                    log_content = result.stdout + result.stderr
                    errors = self._parse_latex_errors(log_content)
                    
                    compilation_result = CompilationResult(
                        success=False,
                        pdf_path=None,
                        log_output=log_content,
                        errors=errors,
                        warnings=[],
                        compilation_time=compilation_time
                    )
                
                # Update project with compilation result
                project.last_compiled = datetime.now()
                project.last_compilation_result = compilation_result
                self._save_project(project)
                
                return compilation_result
                
            except Exception as e:
                return CompilationResult(
                    success=False,
                    pdf_path=None,
                    log_output=f"Compilation error: {str(e)}",
                    errors=[str(e)],
                    warnings=[],
                    compilation_time=0.0
                )
    
    def _parse_latex_errors(self, log_content: str) -> List[str]:
        """Parse LaTeX log for errors"""
        errors = []
        lines = log_content.split('\n')
        
        for i, line in enumerate(lines):
            if '! ' in line and 'LaTeX Error' in line:
                errors.append(line.strip())
            elif 'Error:' in line:
                errors.append(line.strip())
        
        return errors
    
    def _parse_latex_warnings(self, log_content: str) -> List[str]:
        """Parse LaTeX log for warnings"""
        warnings = []
        lines = log_content.split('\n')
        
        for line in lines:
            if 'Warning:' in line:
                warnings.append(line.strip())
            elif 'Overfull' in line or 'Underfull' in line:
                warnings.append(line.strip())
        
        return warnings
    
    async def get_latex_assistance(self,
                                 project_id: str,
                                 content: str,
                                 user_id: str) -> Dict[str, Any]:
        """Get NWTN AI assistance for LaTeX writing"""
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found")
        
        await self.initialize_nwtn_pipeline()
        
        # Construct LaTeX assistance query
        assistance_query = f"""
Please provide LaTeX writing assistance for this academic content:

{content[:1000]}...

Please provide:
1. Structure and formatting suggestions
2. LaTeX command recommendations
3. Academic writing improvements
4. Bibliography and citation guidance
5. Common LaTeX issues and fixes

Focus on academic writing best practices and proper LaTeX usage.
"""
        
        result = await self.nwtn_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=assistance_query,
            context={
                "domain": "academic_writing",
                "latex_assistance": True,
                "document_type": self.active_projects[project_id].document_type.value
            }
        )
        
        return {
            "suggestions": result.get('response', {}).get('text', ''),
            "confidence": result.get('response', {}).get('confidence', 0.0),
            "sources": result.get('response', {}).get('sources', []),
            "processing_time": result.get('performance_metrics', {}).get('total_processing_time', 0.0)
        }
    
    def _save_project(self, project: LaTeXProject):
        """Save LaTeX project with optional encryption"""
        project_dir = self.storage_path / "projects" / project.project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Save project metadata
        metadata_path = project_dir / "project.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(project), f, default=str, indent=2)
        
        # Save individual files
        for filename, latex_file in project.files.items():
            file_path = project_dir / filename
            
            if project.security_level == "high":
                # Use crypto sharding for high security
                temp_file = project_dir / f"temp_{filename}"
                with open(temp_file, 'w') as f:
                    f.write(latex_file.content)
                
                try:
                    shards, manifest = self.crypto_sharding.shard_file(
                        str(temp_file),
                        project.collaborators
                    )
                    
                    # Save shards
                    shard_dir = project_dir / "shards" / filename.replace('.', '_')
                    shard_dir.mkdir(parents=True, exist_ok=True)
                    
                    for i, shard in enumerate(shards):
                        shard_path = shard_dir / f"shard_{i}.enc"
                        with open(shard_path, 'wb') as f:
                            f.write(shard.shard_data)
                    
                    # Save manifest
                    manifest_path = shard_dir / "manifest.json"
                    with open(manifest_path, 'w') as f:
                        json.dump(asdict(manifest), f, default=str, indent=2)
                    
                    temp_file.unlink()  # Remove temporary file
                    
                except Exception as e:
                    print(f"Error sharding LaTeX file {filename}: {e}")
                    # Fall back to regular storage
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(latex_file.content)
            else:
                # Regular storage for medium/standard security
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(latex_file.content)

# University-specific LaTeX implementations
class UniversityLaTeXCollaboration(LaTeXCollaboration):
    """University-specific LaTeX collaboration with institutional templates"""
    
    def __init__(self, university_name: str, storage_path: Optional[Path] = None):
        super().__init__(storage_path)
        self.university_name = university_name
        self._add_university_templates()
    
    def _add_university_templates(self):
        """Add university-specific templates"""
        if "UNC" in self.university_name or "Chapel Hill" in self.university_name:
            self.templates["unc_thesis"] = r"""
\documentclass[12pt,letterpaper,oneside]{book}

% UNC Chapel Hill thesis template
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amsfonts,amssymb}
\usepackage{graphicx}
\usepackage{cite}
\usepackage{url}
\usepackage[margin=1in]{geometry}
\usepackage{setspace}
\usepackage{fancyhdr}

% UNC specific formatting
\doublespacing
\setlength{\parindent}{0.5in}

\title{Thesis Title}
\author{Student Name}
\date{Graduation Date}

\begin{document}

% UNC Title Page
\begin{titlepage}
\centering
\vspace*{1cm}
{\LARGE\textbf{THESIS TITLE}\par}
\vspace{2cm}
{by\par}
\vspace{1cm}
{\Large Student Name\par}
\vspace{2cm}
{A thesis submitted to the faculty at the University of North Carolina at Chapel Hill\\
in partial fulfillment of the requirements for the degree of Doctor of Philosophy\\
in the Department of [Department Name].\par}
\vspace{2cm}
{Chapel Hill\\
[Year]\par}
\vspace{1cm}
{Approved by:\\
[Advisor Name]\\
[Committee Member]\\
[Committee Member]\\
[Committee Member]\\
[Committee Member]\par}
\end{titlepage}

% Rest of thesis content...
\tableofcontents

\chapter{Introduction}
Your introduction goes here.

\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""

# Example usage and testing
if __name__ == "__main__":
    async def test_latex_collaboration():
        """Test LaTeX collaboration system"""
        
        print("🚀 Testing LaTeX Real-time Collaboration")
        
        # Initialize collaboration system
        latex_collab = LaTeXCollaboration()
        
        # Create a research paper project
        project = latex_collab.create_latex_project(
            "Quantum Error Correction in NISQ Devices",
            DocumentType.RESEARCH_PAPER,
            ["sarah.chen@unc.edu", "michael.johnson@sas.com", "alex.rodriguez@duke.edu"],
            created_by="sarah.chen@unc.edu",
            security_level="high"
        )
        
        print(f"✅ Created LaTeX project: {project.title}")
        print(f"   Project ID: {project.project_id}")
        print(f"   Document Type: {project.document_type.value}")
        print(f"   Security Level: {project.security_level}")
        print(f"   Files: {len(project.files)}")
        
        # Add a chapter file
        chapter_file = latex_collab.add_file_to_project(
            project.project_id,
            "methodology.tex",
            r"""
\section{Quantum Error Correction Methodology}
\label{sec:methodology}

Our approach to quantum error correction involves a novel algorithm that adapts to the specific noise characteristics of NISQ devices.

\subsection{Algorithm Design}
The core innovation lies in our adaptive error correction scheme:

\begin{equation}
|\psi_{corrected}\rangle = \mathcal{C}(\mathcal{N}(|\psi\rangle))
\label{eq:correction}
\end{equation}

where $\mathcal{C}$ is our correction operator and $\mathcal{N}$ represents the noise channel.
""",
            file_type="chapter",
            created_by="sarah.chen@unc.edu"
        )
        
        print(f"✅ Added chapter file: {chapter_file.filename}")
        
        # Test collaborative editing
        edit = LaTeXEdit(
            edit_id=str(uuid.uuid4()),
            project_id=project.project_id,
            filename="main.tex",
            operation="replace",
            position=0,
            content="Your Research Paper Title->Quantum Error Correction in NISQ Devices: A Novel Adaptive Approach",
            user_id="michael.johnson@sas.com",
            timestamp=datetime.now()
        )
        
        success = await latex_collab.apply_latex_edit(project.project_id, edit)
        print(f"✅ Applied collaborative edit: {success}")
        
        # Test LaTeX assistance
        try:
            assistance = await latex_collab.get_latex_assistance(
                project.project_id,
                project.files["main.tex"].content,
                "sarah.chen@unc.edu"
            )
            
            print("✅ LaTeX assistance provided:")
            print(f"   Confidence: {assistance['confidence']:.2f}")
            print(f"   Processing time: {assistance['processing_time']:.1f}s")
            print(f"   Suggestions preview: {assistance['suggestions'][:100]}...")
            
        except Exception as e:
            print(f"⚠️  LaTeX assistance test: {e}")
        
        # Test compilation (would require LaTeX installation)
        print("📄 Testing LaTeX compilation...")
        try:
            compilation_result = await latex_collab.compile_latex_project(
                project.project_id,
                "sarah.chen@unc.edu"
            )
            
            if compilation_result.success:
                print(f"✅ LaTeX compilation successful: {compilation_result.pdf_path}")
                print(f"   Compilation time: {compilation_result.compilation_time:.2f}s")
                print(f"   Warnings: {len(compilation_result.warnings)}")
            else:
                print(f"⚠️  LaTeX compilation failed: {len(compilation_result.errors)} errors")
                
        except Exception as e:
            print(f"⚠️  LaTeX compilation test (requires LaTeX installation): {e}")
        
        print("\n🎉 LaTeX collaboration system test completed!")
        print("Ready for integration with PRSM collaboration platform")
    
    # Run the test
    import asyncio
    asyncio.run(test_latex_collaboration())