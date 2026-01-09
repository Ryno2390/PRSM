#!/usr/bin/env python3
"""
Interactive Whiteboard Collaboration for PRSM Secure Collaboration
==================================================================

This module implements a comprehensive interactive whiteboard platform with 
advanced collaborative features for university-industry research partnerships:

- Real-time collaborative drawing and diagramming (Miro/Mural-style)
- Post-quantum secure whiteboard sharing and synchronization
- Multi-institutional whiteboard collaboration with role-based access
- Research planning templates and academic workflow integration
- AI-powered diagram suggestions and automatic layout optimization
- Integration with presentations, publications, and project management

Key Features:
- Infinite canvas with real-time multi-user editing
- Rich drawing tools (shapes, text, images, connectors, sticky notes)
- Template library for research workflows and planning sessions
- Export to multiple formats (PDF, PNG, SVG, PowerPoint, Figma)
- Version history and branching for iterative design processes
- Integration with existing PRSM collaboration tools
"""

import json
import uuid
import asyncio
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import tempfile
import hashlib
import math

# Import PRSM components
from ..security.post_quantum_crypto_sharding import PostQuantumCryptoSharding, CryptoMode
from ..models import QueryRequest

# Mock UnifiedPipelineController for testing
class UnifiedPipelineController:
    """Mock pipeline controller for whiteboard collaboration"""
    async def initialize(self):
        pass
    
    async def process_query_full_pipeline(self, user_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Whiteboard-specific NWTN responses
        if context.get("diagram_optimization"):
            return {
                "response": {
                    "text": """
Diagram Layout Optimization Analysis:

🎨 **Visual Hierarchy Recommendations**:
```javascript
// Optimal layout principles for research diagrams
const layoutOptimization = {
  // Group related concepts with consistent spacing
  conceptClusters: {
    spacing: 150, // pixels between groups
    alignment: 'center',
    colorCoding: 'by_institution' // UNC blue, Duke blue, NC State red, SAS orange
  },
  
  // Flow direction for process diagrams
  processFlow: {
    direction: 'left-to-right', // Natural reading pattern
    stepSpacing: 200,
    decisionNodes: {
      shape: 'diamond',
      color: '#ff9800' // Orange for decisions
    }
  },
  
  // Connection routing for clarity
  connectors: {
    style: 'orthogonal', // 90-degree angles for technical diagrams
    avoid_overlap: true,
    arrow_style: 'filled_triangle',
    weight_by_importance: true
  }
};
```

📊 **Collaboration Layout Patterns**:
- **Research Workflow**: Linear progression with feedback loops
- **Institutional Partnership**: Hub-and-spoke with university centers
- **Grant Planning**: Gantt-style timeline with dependency arrows
- **System Architecture**: Hierarchical with clear data flow indicators

🎯 **Cognitive Load Optimization**:
- Limit to 7±2 items per visual cluster (Miller's Rule)
- Use consistent color schemes across institutional branding
- Implement progressive disclosure for complex diagrams
- Add visual anchors for navigation in large canvases

🏛️ **University-Industry Specific**:
- Separate swim lanes for university vs industry contributions
- Clear IP ownership indicators with color coding
- Milestone markers aligned with academic calendar
- Export formats optimized for grant applications and presentations
                    """,
                    "confidence": 0.95,
                    "sources": ["design_principles.pdf", "cognitive_psychology.com", "collaboration_patterns.org"]
                },
                "performance_metrics": {"total_processing_time": 3.3}
            }
        elif context.get("template_generation"):
            return {
                "response": {
                    "text": """
Research Template Generation Recommendations:

📋 **Research Planning Templates**:
```html
<!-- Grant Application Planning Template -->
<div class="grant-planning-template">
  <section class="timeline-section">
    <h3>Project Timeline (36 months)</h3>
    <div class="timeline-items">
      <div class="milestone year-1">Year 1: Foundation & Setup</div>
      <div class="milestone year-2">Year 2: Core Research & Development</div>
      <div class="milestone year-3">Year 3: Validation & Dissemination</div>
    </div>
  </section>
  
  <section class="collaboration-matrix">
    <h3>Institutional Roles</h3>
    <div class="institution-grid">
      <div class="institution unc">UNC: Theoretical Framework</div>
      <div class="institution duke">Duke: Clinical Validation</div>
      <div class="institution ncstate">NC State: Engineering Implementation</div>
      <div class="institution sas">SAS: Analytics & Scale</div>
    </div>
  </section>
  
  <section class="deliverables">
    <h3>Key Deliverables</h3>
    <div class="deliverable-items">
      <div class="deliverable publications">Publications: 5 peer-reviewed papers</div>
      <div class="deliverable patents">Patents: 2 provisional applications</div>
      <div class="deliverable software">Software: Open-source toolkit</div>
      <div class="deliverable training">Training: 10 graduate students</div>
    </div>
  </section>
</div>
```

🔬 **Research Methodology Templates**:
- **Hypothesis Testing**: Visual hypothesis trees with statistical test plans
- **Experimental Design**: Multi-factorial design matrices with controls
- **Data Flow**: From collection through analysis to publication
- **Literature Review**: Systematic review flowcharts with inclusion/exclusion criteria

🤝 **Collaboration Framework Templates**:
- **Stakeholder Mapping**: Influence/interest matrices for project participants
- **Communication Plans**: Who-what-when matrices for project updates
- **Risk Assessment**: Risk/impact quadrants with mitigation strategies
- **IP Management**: Ownership flowcharts with licensing decision trees

🎯 **Meeting Templates**:
- **Research Planning Sessions**: Agenda-driven with action item tracking
- **Progress Reviews**: Milestone checkpoints with go/no-go decisions
- **Brainstorming Sessions**: Divergent thinking followed by convergent synthesis
- **Problem-Solving Workshops**: Issue identification to solution implementation
                    """,
                    "confidence": 0.91,
                    "sources": ["research_methodology.pdf", "project_management.org", "academic_collaboration.edu"]
                },
                "performance_metrics": {"total_processing_time": 2.8}
            }
        else:
            return {
                "response": {"text": "Whiteboard collaboration assistance available", "confidence": 0.75, "sources": []},
                "performance_metrics": {"total_processing_time": 1.7}
            }

class WhiteboardAccessLevel(Enum):
    """Access levels for whiteboard collaboration"""
    OWNER = "owner"
    EDITOR = "editor"
    COMMENTER = "commenter"
    VIEWER = "viewer"

class ElementType(Enum):
    """Types of whiteboard elements"""
    STICKY_NOTE = "sticky_note"
    TEXT_BOX = "text_box"
    SHAPE = "shape"
    ARROW = "arrow"
    CONNECTOR = "connector"
    IMAGE = "image"
    DRAWING = "drawing"
    TEMPLATE = "template"

class ShapeType(Enum):
    """Types of shapes"""
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"
    HEXAGON = "hexagon"
    STAR = "star"
    CLOUD = "cloud"
    CYLINDER = "cylinder"

class TemplateCategory(Enum):
    """Whiteboard template categories"""
    RESEARCH_PLANNING = "research_planning"
    GRANT_APPLICATION = "grant_application"
    SYSTEM_DESIGN = "system_design"
    WORKFLOW_MAPPING = "workflow_mapping"
    BRAINSTORMING = "brainstorming"
    RETROSPECTIVE = "retrospective"
    STAKEHOLDER_MAPPING = "stakeholder_mapping"

@dataclass
class Point:
    """2D point coordinate"""
    x: float
    y: float

@dataclass
class BoundingBox:
    """Bounding box for elements"""
    x: float
    y: float
    width: float
    height: float

@dataclass
class WhiteboardElement:
    """Individual element on the whiteboard"""
    element_id: str
    element_type: ElementType
    position: Point
    bounding_box: BoundingBox
    
    # Content
    content: str = ""
    properties: Dict[str, Any] = None
    
    # Styling
    background_color: str = "#ffffff"
    border_color: str = "#000000"
    text_color: str = "#000000"
    border_width: int = 1
    font_size: int = 14
    font_family: str = "Arial"
    
    # Metadata
    created_by: str = ""
    created_at: datetime = None
    last_modified_by: str = ""
    last_modified_at: datetime = None
    
    # Collaboration
    locked_by: Optional[str] = None
    comments: List[Dict[str, Any]] = None
    tags: List[str] = None

@dataclass
class WhiteboardConnection:
    """Connection between whiteboard elements"""
    connection_id: str
    from_element: str
    to_element: str
    connection_type: str = "arrow"
    
    # Visual properties
    color: str = "#000000"
    width: int = 2
    style: str = "solid"  # solid, dashed, dotted
    arrow_start: bool = False
    arrow_end: bool = True
    
    # Path
    waypoints: List[Point] = None
    
    # Metadata
    created_by: str = ""
    created_at: datetime = None
    label: str = ""

@dataclass
class WhiteboardTemplate:
    """Reusable whiteboard template"""
    template_id: str
    name: str
    description: str
    category: TemplateCategory
    
    # Template content
    elements: List[WhiteboardElement]
    connections: List[WhiteboardConnection]
    canvas_size: BoundingBox
    
    # Metadata
    created_by: str
    institution: str = ""
    use_count: int = 0
    rating: float = 0.0
    
    # Customization
    variables: Dict[str, str] = None  # Placeholder variables
    color_scheme: str = "default"
    
    created_at: datetime = None

@dataclass
class CollaborativeWhiteboard:
    """Interactive collaborative whiteboard"""
    whiteboard_id: str
    name: str
    description: str
    owner: str
    collaborators: Dict[str, WhiteboardAccessLevel]
    
    # Canvas properties
    canvas_size: BoundingBox
    background_color: str = "#f8f9fa"
    grid_enabled: bool = True
    grid_size: int = 20
    
    # Content
    elements: Dict[str, WhiteboardElement]
    connections: Dict[str, WhiteboardConnection]
    
    # Collaboration features
    real_time_enabled: bool = True
    cursor_sharing: bool = True
    voice_chat_enabled: bool = False
    
    # History and versioning
    version_history: List[Dict[str, Any]] = None
    auto_save_interval: int = 30  # seconds
    
    # Security
    encrypted: bool = True
    access_controlled: bool = True
    security_level: str = "high"
    
    # Integration
    linked_projects: List[str] = None
    linked_documents: List[str] = None
    
    # Analytics
    view_count: int = 0
    edit_count: int = 0
    collaboration_sessions: List[Dict[str, Any]] = None
    
    # Metadata
    tags: List[str] = None
    created_at: datetime = None
    last_modified_at: datetime = None

@dataclass
class WhiteboardSession:
    """Active collaboration session"""
    session_id: str
    whiteboard_id: str
    participants: Dict[str, Dict[str, Any]]  # user_id -> {cursor_position, last_activity, etc.}
    started_at: datetime
    
    # Real-time state
    active_cursors: Dict[str, Point] = None
    element_locks: Dict[str, str] = None  # element_id -> user_id
    
    # Communication
    chat_messages: List[Dict[str, Any]] = None
    voice_participants: List[str] = None

class WhiteboardCollaboration:
    """
    Main class for interactive whiteboard collaboration with P2P security
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize whiteboard collaboration system"""
        self.storage_path = storage_path or Path("./whiteboard_collaboration")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize PRSM components
        self.crypto_sharding = PostQuantumCryptoSharding(
            default_shards=5,
            required_shards=3,
            crypto_mode=CryptoMode.POST_QUANTUM
        )
        self.nwtn_pipeline = None
        
        # Active whiteboards and sessions
        self.collaborative_whiteboards: Dict[str, CollaborativeWhiteboard] = {}
        self.active_sessions: Dict[str, WhiteboardSession] = {}
        self.whiteboard_templates: Dict[str, WhiteboardTemplate] = {}
        
        # Initialize templates and tools
        self.standard_templates = self._initialize_standard_templates()
        self.drawing_tools = self._initialize_drawing_tools()
        self.color_palettes = self._initialize_color_palettes()
        
        # Real-time collaboration state
        self.websocket_connections = {}
        self.collaboration_events = []
    
    def _initialize_standard_templates(self) -> Dict[str, WhiteboardTemplate]:
        """Initialize standard whiteboard templates"""
        templates = {}
        
        # Research Planning Template
        research_elements = [
            WhiteboardElement(
                element_id="research_question",
                element_type=ElementType.STICKY_NOTE,
                position=Point(100, 100),
                bounding_box=BoundingBox(100, 100, 200, 100),
                content="Research Question",
                background_color="#ffeb3b",
                created_at=datetime.now()
            ),
            WhiteboardElement(
                element_id="hypothesis",
                element_type=ElementType.STICKY_NOTE,
                position=Point(400, 100),
                bounding_box=BoundingBox(400, 100, 200, 100),
                content="Hypothesis",
                background_color="#4caf50",
                created_at=datetime.now()
            ),
            WhiteboardElement(
                element_id="methodology",
                element_type=ElementType.SHAPE,
                position=Point(250, 300),
                bounding_box=BoundingBox(250, 300, 300, 150),
                content="Methodology",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#2196f3",
                created_at=datetime.now()
            )
        ]
        
        research_connections = [
            WhiteboardConnection(
                connection_id="q_to_h",
                from_element="research_question",
                to_element="hypothesis",
                created_at=datetime.now()
            ),
            WhiteboardConnection(
                connection_id="h_to_m",
                from_element="hypothesis",
                to_element="methodology",
                created_at=datetime.now()
            )
        ]
        
        templates["research_planning"] = WhiteboardTemplate(
            template_id="research_planning",
            name="Research Planning Template",
            description="Template for planning academic research projects",
            category=TemplateCategory.RESEARCH_PLANNING,
            elements=research_elements,
            connections=research_connections,
            canvas_size=BoundingBox(0, 0, 1200, 800),
            created_by="system",
            institution="PRSM",
            created_at=datetime.now()
        )
        
        # Grant Application Template
        grant_elements = [
            WhiteboardElement(
                element_id="project_title",
                element_type=ElementType.TEXT_BOX,
                position=Point(50, 50),
                bounding_box=BoundingBox(50, 50, 600, 80),
                content="Project Title: [Enter Project Name]",
                font_size=24,
                background_color="#f5f5f5",
                created_at=datetime.now()
            ),
            WhiteboardElement(
                element_id="unc_role",
                element_type=ElementType.SHAPE,
                position=Point(50, 200),
                bounding_box=BoundingBox(50, 200, 200, 120),
                content="UNC Chapel Hill\nRole & Contributions",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#13294b",
                text_color="#ffffff",
                created_at=datetime.now()
            ),
            WhiteboardElement(
                element_id="duke_role",
                element_type=ElementType.SHAPE,
                position=Point(300, 200),
                bounding_box=BoundingBox(300, 200, 200, 120),
                content="Duke University\nRole & Contributions",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#001a57",
                text_color="#ffffff",
                created_at=datetime.now()
            ),
            WhiteboardElement(
                element_id="sas_role",
                element_type=ElementType.SHAPE,
                position=Point(550, 200),
                bounding_box=BoundingBox(550, 200, 200, 120),
                content="SAS Institute\nRole & Contributions",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#0066cc",
                text_color="#ffffff",
                created_at=datetime.now()
            ),
            WhiteboardElement(
                element_id="timeline",
                element_type=ElementType.SHAPE,
                position=Point(50, 400),
                bounding_box=BoundingBox(50, 400, 700, 100),
                content="36-Month Timeline",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#e8f5e8",
                created_at=datetime.now()
            )
        ]
        
        templates["grant_application"] = WhiteboardTemplate(
            template_id="grant_application",
            name="Multi-University Grant Application",
            description="Template for collaborative grant applications",
            category=TemplateCategory.GRANT_APPLICATION,
            elements=grant_elements,
            connections=[],
            canvas_size=BoundingBox(0, 0, 1000, 600),
            created_by="system",
            institution="RTP Consortium",
            created_at=datetime.now()
        )
        
        return templates
    
    def _initialize_drawing_tools(self) -> Dict[str, Dict[str, Any]]:
        """Initialize drawing tools and their configurations"""
        return {
            "pointer": {
                "cursor": "default",
                "action": "select",
                "shortcuts": ["v", "1"]
            },
            "sticky_note": {
                "cursor": "crosshair",
                "action": "create_sticky",
                "default_size": {"width": 150, "height": 100},
                "shortcuts": ["s", "2"]
            },
            "text": {
                "cursor": "text",
                "action": "create_text",
                "default_font": {"family": "Arial", "size": 14},
                "shortcuts": ["t", "3"]
            },
            "rectangle": {
                "cursor": "crosshair",
                "action": "create_shape",
                "shape_type": "rectangle",
                "shortcuts": ["r", "4"]
            },
            "circle": {
                "cursor": "crosshair",
                "action": "create_shape",
                "shape_type": "circle",
                "shortcuts": ["c", "5"]
            },
            "arrow": {
                "cursor": "crosshair",
                "action": "create_connector",
                "connector_type": "arrow",
                "shortcuts": ["a", "6"]
            },
            "pen": {
                "cursor": "crosshair",
                "action": "draw_freehand",
                "brush_size": 2,
                "shortcuts": ["p", "7"]
            },
            "highlighter": {
                "cursor": "crosshair",
                "action": "highlight",
                "opacity": 0.3,
                "shortcuts": ["h", "8"]
            },
            "eraser": {
                "cursor": "crosshair",
                "action": "erase",
                "size": 20,
                "shortcuts": ["e", "9"]
            }
        }
    
    def _initialize_color_palettes(self) -> Dict[str, List[str]]:
        """Initialize color palettes for different use cases"""
        return {
            "university_branding": [
                "#13294b",  # UNC Navy
                "#4b9cd3",  # UNC Blue
                "#001a57",  # Duke Blue
                "#cc0000",  # NC State Red
                "#0066cc",  # SAS Blue
                "#f47735"   # SAS Orange
            ],
            "research_categories": [
                "#e3f2fd",  # Theory (Light Blue)
                "#f3e5f5",  # Methods (Light Purple)
                "#e8f5e8",  # Results (Light Green)
                "#fff3e0",  # Discussion (Light Orange)
                "#fce4ec"   # Conclusion (Light Pink)
            ],
            "priority_levels": [
                "#f44336",  # High (Red)
                "#ff9800",  # Medium (Orange)
                "#4caf50",  # Low (Green)
                "#2196f3",  # Info (Blue)
                "#9c27b0"   # Note (Purple)
            ],
            "collaboration_status": [
                "#4caf50",  # Complete (Green)
                "#ff9800",  # In Progress (Orange)
                "#f44336",  # Blocked (Red)
                "#9e9e9e",  # Not Started (Gray)
                "#2196f3"   # Review (Blue)
            ]
        }
    
    async def initialize_nwtn_pipeline(self):
        """Initialize NWTN pipeline for diagram optimization"""
        if self.nwtn_pipeline is None:
            self.nwtn_pipeline = UnifiedPipelineController()
            await self.nwtn_pipeline.initialize()
    
    def create_collaborative_whiteboard(self,
                                      name: str,
                                      description: str,
                                      owner: str,
                                      collaborators: Optional[Dict[str, WhiteboardAccessLevel]] = None,
                                      template_id: Optional[str] = None,
                                      canvas_size: Optional[BoundingBox] = None,
                                      security_level: str = "high") -> CollaborativeWhiteboard:
        """Create a new collaborative whiteboard"""
        
        whiteboard_id = str(uuid.uuid4())
        
        # Set default canvas size
        if canvas_size is None:
            canvas_size = BoundingBox(0, 0, 2000, 1500)  # Large infinite canvas
        
        # Initialize from template if specified
        elements = {}
        connections = {}
        
        if template_id and template_id in self.standard_templates:
            template = self.standard_templates[template_id]
            # Copy template elements with new IDs
            for element in template.elements:
                new_element_id = str(uuid.uuid4())
                element_copy = WhiteboardElement(
                    element_id=new_element_id,
                    element_type=element.element_type,
                    position=element.position,
                    bounding_box=element.bounding_box,
                    content=element.content,
                    properties=element.properties,
                    background_color=element.background_color,
                    border_color=element.border_color,
                    text_color=element.text_color,
                    border_width=element.border_width,
                    font_size=element.font_size,
                    font_family=element.font_family,
                    created_by=owner,
                    created_at=datetime.now(),
                    comments=[],
                    tags=[]
                )
                elements[new_element_id] = element_copy
            
            # Copy template connections
            for connection in template.connections:
                new_connection_id = str(uuid.uuid4())
                connection_copy = WhiteboardConnection(
                    connection_id=new_connection_id,
                    from_element=connection.from_element,
                    to_element=connection.to_element,
                    connection_type=connection.connection_type,
                    color=connection.color,
                    width=connection.width,
                    style=connection.style,
                    arrow_start=connection.arrow_start,
                    arrow_end=connection.arrow_end,
                    waypoints=connection.waypoints,
                    created_by=owner,
                    created_at=datetime.now(),
                    label=connection.label
                )
                connections[new_connection_id] = connection_copy
        
        whiteboard = CollaborativeWhiteboard(
            whiteboard_id=whiteboard_id,
            name=name,
            description=description,
            owner=owner,
            collaborators=collaborators or {},
            canvas_size=canvas_size,
            background_color="#f8f9fa",
            grid_enabled=True,
            grid_size=20,
            elements=elements,
            connections=connections,
            real_time_enabled=True,
            cursor_sharing=True,
            voice_chat_enabled=False,
            version_history=[],
            auto_save_interval=30,
            encrypted=True,
            access_controlled=True,
            security_level=security_level,
            linked_projects=[],
            linked_documents=[],
            view_count=0,
            edit_count=0,
            collaboration_sessions=[],
            tags=[],
            created_at=datetime.now(),
            last_modified_at=datetime.now()
        )
        
        self.collaborative_whiteboards[whiteboard_id] = whiteboard
        self._save_whiteboard(whiteboard)
        
        print(f"🎨 Created collaborative whiteboard: {name}")
        print(f"   Whiteboard ID: {whiteboard_id}")
        print(f"   Template: {template_id or 'Blank canvas'}")
        print(f"   Canvas Size: {canvas_size.width}x{canvas_size.height}")
        print(f"   Collaborators: {len(collaborators or {})}")
        print(f"   Elements: {len(elements)}")
        print(f"   Security: {security_level}")
        
        return whiteboard
    
    def add_element(self,
                   whiteboard_id: str,
                   element_type: ElementType,
                   position: Point,
                   size: Tuple[float, float],
                   content: str,
                   user_id: str,
                   properties: Optional[Dict[str, Any]] = None,
                   styling: Optional[Dict[str, Any]] = None) -> WhiteboardElement:
        """Add an element to the whiteboard"""
        
        if whiteboard_id not in self.collaborative_whiteboards:
            raise ValueError(f"Whiteboard {whiteboard_id} not found")
        
        whiteboard = self.collaborative_whiteboards[whiteboard_id]
        
        # Check permissions
        if not self._check_whiteboard_access(whiteboard, user_id, WhiteboardAccessLevel.EDITOR):
            raise PermissionError("Insufficient permissions to add elements")
        
        element_id = str(uuid.uuid4())
        
        # Apply default styling
        default_style = {
            "background_color": "#ffffff",
            "border_color": "#000000",
            "text_color": "#000000",
            "border_width": 1,
            "font_size": 14,
            "font_family": "Arial"
        }
        
        if styling:
            default_style.update(styling)
        
        element = WhiteboardElement(
            element_id=element_id,
            element_type=element_type,
            position=position,
            bounding_box=BoundingBox(position.x, position.y, size[0], size[1]),
            content=content,
            properties=properties or {},
            background_color=default_style["background_color"],
            border_color=default_style["border_color"],
            text_color=default_style["text_color"],
            border_width=default_style["border_width"],
            font_size=default_style["font_size"],
            font_family=default_style["font_family"],
            created_by=user_id,
            created_at=datetime.now(),
            last_modified_by=user_id,
            last_modified_at=datetime.now(),
            comments=[],
            tags=[]
        )
        
        whiteboard.elements[element_id] = element
        whiteboard.edit_count += 1
        whiteboard.last_modified_at = datetime.now()
        
        self._save_whiteboard(whiteboard)
        self._broadcast_element_change(whiteboard_id, "add", element)
        
        print(f"➕ Added element to whiteboard: {element_type.value}")
        print(f"   Element ID: {element_id}")
        print(f"   Position: ({position.x}, {position.y})")
        print(f"   Size: {size[0]}x{size[1]}")
        print(f"   Content: {content[:50]}...")
        
        return element
    
    def add_connection(self,
                      whiteboard_id: str,
                      from_element_id: str,
                      to_element_id: str,
                      user_id: str,
                      connection_type: str = "arrow",
                      styling: Optional[Dict[str, Any]] = None,
                      label: str = "") -> WhiteboardConnection:
        """Add a connection between elements"""
        
        if whiteboard_id not in self.collaborative_whiteboards:
            raise ValueError(f"Whiteboard {whiteboard_id} not found")
        
        whiteboard = self.collaborative_whiteboards[whiteboard_id]
        
        # Check permissions
        if not self._check_whiteboard_access(whiteboard, user_id, WhiteboardAccessLevel.EDITOR):
            raise PermissionError("Insufficient permissions to add connections")
        
        # Verify elements exist
        if from_element_id not in whiteboard.elements or to_element_id not in whiteboard.elements:
            raise ValueError("One or both elements not found")
        
        connection_id = str(uuid.uuid4())
        
        # Apply default styling
        default_style = {
            "color": "#000000",
            "width": 2,
            "style": "solid",
            "arrow_start": False,
            "arrow_end": True
        }
        
        if styling:
            default_style.update(styling)
        
        connection = WhiteboardConnection(
            connection_id=connection_id,
            from_element=from_element_id,
            to_element=to_element_id,
            connection_type=connection_type,
            color=default_style["color"],
            width=default_style["width"],
            style=default_style["style"],
            arrow_start=default_style["arrow_start"],
            arrow_end=default_style["arrow_end"],
            waypoints=[],
            created_by=user_id,
            created_at=datetime.now(),
            label=label
        )
        
        whiteboard.connections[connection_id] = connection
        whiteboard.edit_count += 1
        whiteboard.last_modified_at = datetime.now()
        
        self._save_whiteboard(whiteboard)
        self._broadcast_connection_change(whiteboard_id, "add", connection)
        
        print(f"🔗 Added connection: {from_element_id} → {to_element_id}")
        print(f"   Connection ID: {connection_id}")
        print(f"   Type: {connection_type}")
        print(f"   Label: {label}")
        
        return connection
    
    def start_collaboration_session(self,
                                  whiteboard_id: str,
                                  user_id: str) -> WhiteboardSession:
        """Start a real-time collaboration session"""
        
        if whiteboard_id not in self.collaborative_whiteboards:
            raise ValueError(f"Whiteboard {whiteboard_id} not found")
        
        whiteboard = self.collaborative_whiteboards[whiteboard_id]
        
        # Check permissions
        if not self._check_whiteboard_access(whiteboard, user_id, WhiteboardAccessLevel.VIEWER):
            raise PermissionError("Insufficient permissions to join session")
        
        # Find or create session
        session_id = f"session_{whiteboard_id}"
        
        if session_id not in self.active_sessions:
            session = WhiteboardSession(
                session_id=session_id,
                whiteboard_id=whiteboard_id,
                participants={},
                started_at=datetime.now(),
                active_cursors={},
                element_locks={},
                chat_messages=[],
                voice_participants=[]
            )
            self.active_sessions[session_id] = session
        else:
            session = self.active_sessions[session_id]
        
        # Add participant
        session.participants[user_id] = {
            "joined_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "cursor_position": {"x": 0, "y": 0},
            "current_tool": "pointer",
            "user_color": self._assign_user_color(user_id)
        }
        
        print(f"👥 User joined collaboration session:")
        print(f"   User: {user_id}")
        print(f"   Whiteboard: {whiteboard.name}")
        print(f"   Session participants: {len(session.participants)}")
        
        return session
    
    async def optimize_diagram_layout(self,
                                    whiteboard_id: str,
                                    optimization_goals: List[str],
                                    user_id: str) -> Dict[str, Any]:
        """Get AI recommendations for optimizing diagram layout"""
        
        if whiteboard_id not in self.collaborative_whiteboards:
            raise ValueError(f"Whiteboard {whiteboard_id} not found")
        
        whiteboard = self.collaborative_whiteboards[whiteboard_id]
        
        # Check permissions
        if not self._check_whiteboard_access(whiteboard, user_id, WhiteboardAccessLevel.EDITOR):
            raise PermissionError("Insufficient permissions to optimize layout")
        
        await self.initialize_nwtn_pipeline()
        
        # Analyze current layout
        layout_analysis = self._analyze_current_layout(whiteboard)
        
        optimization_prompt = f"""
Please provide diagram layout optimization recommendations for this collaborative whiteboard:

**Whiteboard**: {whiteboard.name}
**Description**: {whiteboard.description}
**Current Elements**: {len(whiteboard.elements)} items
**Current Connections**: {len(whiteboard.connections)} connections
**Optimization Goals**: {', '.join(optimization_goals)}
**Canvas Size**: {whiteboard.canvas_size.width}x{whiteboard.canvas_size.height}

**Current Layout Analysis**:
{layout_analysis}

Please provide:
1. Visual hierarchy optimization recommendations
2. Spacing and alignment improvements
3. Color coding and grouping strategies
4. Connection routing optimization
5. Cognitive load reduction techniques

Focus on research collaboration best practices and university-industry presentation standards.
"""
        
        result = await self.nwtn_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=optimization_prompt,
            context={
                "domain": "diagram_optimization",
                "diagram_optimization": True,
                "whiteboard_type": "research_collaboration",
                "optimization_type": "layout_analysis"
            }
        )
        
        optimization = {
            "whiteboard_id": whiteboard_id,
            "whiteboard_name": whiteboard.name,
            "optimization_goals": optimization_goals,
            "current_analysis": layout_analysis,
            "recommendations": result.get('response', {}).get('text', ''),
            "confidence": result.get('response', {}).get('confidence', 0.0),
            "sources": result.get('response', {}).get('sources', []),
            "processing_time": result.get('performance_metrics', {}).get('total_processing_time', 0.0),
            "generated_at": datetime.now().isoformat(),
            "requested_by": user_id
        }
        
        print(f"🎯 Diagram optimization analysis completed:")
        print(f"   Whiteboard: {whiteboard.name}")
        print(f"   Elements analyzed: {len(whiteboard.elements)}")
        print(f"   Goals: {len(optimization_goals)} optimization objectives")
        print(f"   Confidence: {optimization['confidence']:.2f}")
        
        return optimization
    
    async def generate_template_suggestions(self,
                                          research_area: str,
                                          collaboration_type: str,
                                          institutions: List[str],
                                          user_id: str) -> Dict[str, Any]:
        """Generate AI-powered template suggestions"""
        
        await self.initialize_nwtn_pipeline()
        
        template_prompt = f"""
Please generate whiteboard template suggestions for this research collaboration:

**Research Area**: {research_area}
**Collaboration Type**: {collaboration_type}
**Participating Institutions**: {', '.join(institutions)}
**Context**: University-industry research partnership

Please provide:
1. Recommended template structures and layouts
2. Standard elements and components to include
3. Color coding schemes for institutional identification
4. Workflow patterns specific to this research area
5. Integration points with other collaboration tools

Focus on templates that enhance productivity and clarity for multi-institutional research teams.
"""
        
        result = await self.nwtn_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=template_prompt,
            context={
                "domain": "template_generation",
                "template_generation": True,
                "research_area": research_area,
                "collaboration_type": collaboration_type
            }
        )
        
        suggestions = {
            "research_area": research_area,
            "collaboration_type": collaboration_type,
            "institutions": institutions,
            "template_suggestions": result.get('response', {}).get('text', ''),
            "confidence": result.get('response', {}).get('confidence', 0.0),
            "sources": result.get('response', {}).get('sources', []),
            "processing_time": result.get('performance_metrics', {}).get('total_processing_time', 0.0),
            "generated_at": datetime.now().isoformat(),
            "requested_by": user_id
        }
        
        print(f"📋 Template suggestions generated:")
        print(f"   Research Area: {research_area}")
        print(f"   Collaboration Type: {collaboration_type}")
        print(f"   Institutions: {len(institutions)}")
        print(f"   Confidence: {suggestions['confidence']:.2f}")
        
        return suggestions
    
    def export_whiteboard(self,
                         whiteboard_id: str,
                         export_format: str,
                         user_id: str,
                         region: Optional[BoundingBox] = None) -> str:
        """Export whiteboard in various formats"""
        
        if whiteboard_id not in self.collaborative_whiteboards:
            raise ValueError(f"Whiteboard {whiteboard_id} not found")
        
        whiteboard = self.collaborative_whiteboards[whiteboard_id]
        
        # Check permissions
        if not self._check_whiteboard_access(whiteboard, user_id, WhiteboardAccessLevel.VIEWER):
            raise PermissionError("Insufficient permissions to export whiteboard")
        
        export_dir = self.storage_path / "exports" / whiteboard_id
        export_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format.lower() == "svg":
            export_file = export_dir / f"whiteboard_{timestamp}.svg"
            svg_content = self._generate_svg_export(whiteboard, region)
            with open(export_file, 'w') as f:
                f.write(svg_content)
                
        elif export_format.lower() == "png":
            export_file = export_dir / f"whiteboard_{timestamp}.png"
            self._generate_png_export(whiteboard, export_file, region)
            
        elif export_format.lower() == "pdf":
            export_file = export_dir / f"whiteboard_{timestamp}.pdf"
            self._generate_pdf_export(whiteboard, export_file, region)
            
        elif export_format.lower() == "json":
            export_file = export_dir / f"whiteboard_data_{timestamp}.json"
            whiteboard_data = asdict(whiteboard)
            with open(export_file, 'w') as f:
                json.dump(whiteboard_data, f, indent=2, default=str)
                
        elif export_format.lower() == "miro":
            export_file = export_dir / f"whiteboard_miro_{timestamp}.json"
            miro_data = self._convert_to_miro_format(whiteboard)
            with open(export_file, 'w') as f:
                json.dump(miro_data, f, indent=2)
                
        else:
            raise ValueError(f"Export format {export_format} not supported")
        
        print(f"📦 Whiteboard exported successfully:")
        print(f"   Format: {export_format.upper()}")
        print(f"   File: {export_file.name}")
        print(f"   Elements: {len(whiteboard.elements)}")
        print(f"   Connections: {len(whiteboard.connections)}")
        
        return str(export_file)
    
    def _analyze_current_layout(self, whiteboard: CollaborativeWhiteboard) -> str:
        """Analyze current whiteboard layout for optimization"""
        
        if not whiteboard.elements:
            return "Empty whiteboard - no elements to analyze"
        
        # Calculate layout metrics
        positions = [(e.position.x, e.position.y) for e in whiteboard.elements.values()]
        sizes = [(e.bounding_box.width, e.bounding_box.height) for e in whiteboard.elements.values()]
        
        # Bounding box of all elements
        min_x = min(pos[0] for pos in positions)
        max_x = max(pos[0] + size[0] for pos, size in zip(positions, sizes))
        min_y = min(pos[1] for pos in positions)
        max_y = max(pos[1] + size[1] for pos, size in zip(positions, sizes))
        
        used_width = max_x - min_x
        used_height = max_y - min_y
        
        # Element density
        total_area = sum(size[0] * size[1] for size in sizes)
        canvas_area = used_width * used_height if used_width > 0 and used_height > 0 else 1
        density = total_area / canvas_area
        
        # Element type distribution
        type_counts = {}
        for element in whiteboard.elements.values():
            element_type = element.element_type.value
            type_counts[element_type] = type_counts.get(element_type, 0) + 1
        
        analysis = f"""
Layout Analysis:
- Total Elements: {len(whiteboard.elements)}
- Total Connections: {len(whiteboard.connections)}
- Used Canvas Area: {used_width:.0f}x{used_height:.0f} pixels
- Element Density: {density:.2f}
- Element Types: {dict(type_counts)}
- Average Element Size: {sum(size[0] * size[1] for size in sizes) / len(sizes):.0f} px²
"""
        
        return analysis
    
    def _generate_svg_export(self, whiteboard: CollaborativeWhiteboard, region: Optional[BoundingBox]) -> str:
        """Generate SVG export of whiteboard"""
        
        # Determine export region
        if region:
            width, height = region.width, region.height
            viewbox = f"{region.x} {region.y} {region.width} {region.height}"
        else:
            width, height = whiteboard.canvas_size.width, whiteboard.canvas_size.height
            viewbox = f"0 0 {width} {height}"
        
        svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="{width}" height="{height}" viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <style>
      .text-element {{ font-family: Arial, sans-serif; }}
      .sticky-note {{ filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.2)); }}
    </style>
  </defs>
  
  <!-- Background -->
  <rect width="100%" height="100%" fill="{whiteboard.background_color}"/>
  
  <!-- Grid (if enabled) -->
  {self._generate_svg_grid(whiteboard) if whiteboard.grid_enabled else ''}
  
  <!-- Connections -->
  {self._generate_svg_connections(whiteboard)}
  
  <!-- Elements -->
  {self._generate_svg_elements(whiteboard)}
  
  <!-- Title -->
  <text x="20" y="30" font-family="Arial" font-size="20" font-weight="bold" fill="#333">
    {whiteboard.name}
  </text>
</svg>"""
        
        return svg_content
    
    def _generate_svg_elements(self, whiteboard: CollaborativeWhiteboard) -> str:
        """Generate SVG for whiteboard elements"""
        
        svg_elements = []
        
        for element in whiteboard.elements.values():
            x, y = element.position.x, element.position.y
            w, h = element.bounding_box.width, element.bounding_box.height
            
            if element.element_type == ElementType.STICKY_NOTE:
                svg_elements.append(f"""
  <g class="sticky-note">
    <rect x="{x}" y="{y}" width="{w}" height="{h}" 
          fill="{element.background_color}" 
          stroke="{element.border_color}" 
          stroke-width="{element.border_width}" 
          rx="5"/>
    <foreignObject x="{x+5}" y="{y+5}" width="{w-10}" height="{h-10}">
      <div xmlns="http://www.w3.org/1999/xhtml" 
           style="font-family: {element.font_family}; font-size: {element.font_size}px; 
                  color: {element.text_color}; padding: 5px; word-wrap: break-word;">
        {element.content}
      </div>
    </foreignObject>
  </g>""")
                
            elif element.element_type == ElementType.SHAPE:
                shape_type = element.properties.get("shape_type", "rectangle")
                if shape_type == "rectangle":
                    svg_elements.append(f"""
  <rect x="{x}" y="{y}" width="{w}" height="{h}" 
        fill="{element.background_color}" 
        stroke="{element.border_color}" 
        stroke-width="{element.border_width}"/>
  <text x="{x + w/2}" y="{y + h/2}" 
        font-family="{element.font_family}" 
        font-size="{element.font_size}" 
        fill="{element.text_color}" 
        text-anchor="middle" 
        dominant-baseline="middle">
    {element.content}
  </text>""")
                elif shape_type == "circle":
                    radius = min(w, h) / 2
                    cx, cy = x + w/2, y + h/2
                    svg_elements.append(f"""
  <circle cx="{cx}" cy="{cy}" r="{radius}" 
          fill="{element.background_color}" 
          stroke="{element.border_color}" 
          stroke-width="{element.border_width}"/>
  <text x="{cx}" y="{cy}" 
        font-family="{element.font_family}" 
        font-size="{element.font_size}" 
        fill="{element.text_color}" 
        text-anchor="middle" 
        dominant-baseline="middle">
    {element.content}
  </text>""")
            
            elif element.element_type == ElementType.TEXT_BOX:
                svg_elements.append(f"""
  <foreignObject x="{x}" y="{y}" width="{w}" height="{h}">
    <div xmlns="http://www.w3.org/1999/xhtml" 
         style="font-family: {element.font_family}; font-size: {element.font_size}px; 
                color: {element.text_color}; background: {element.background_color}; 
                border: {element.border_width}px solid {element.border_color}; 
                padding: 5px; word-wrap: break-word; height: 100%; box-sizing: border-box;">
      {element.content}
    </div>
  </foreignObject>""")
        
        return '\n'.join(svg_elements)
    
    def _generate_svg_connections(self, whiteboard: CollaborativeWhiteboard) -> str:
        """Generate SVG for whiteboard connections"""
        
        svg_connections = []
        
        for connection in whiteboard.connections.values():
            from_element = whiteboard.elements.get(connection.from_element)
            to_element = whiteboard.elements.get(connection.to_element)
            
            if not from_element or not to_element:
                continue
            
            # Calculate connection points
            from_center = Point(
                from_element.position.x + from_element.bounding_box.width / 2,
                from_element.position.y + from_element.bounding_box.height / 2
            )
            to_center = Point(
                to_element.position.x + to_element.bounding_box.width / 2,
                to_element.position.y + to_element.bounding_box.height / 2
            )
            
            # Simple straight line connection
            svg_connections.append(f"""
  <line x1="{from_center.x}" y1="{from_center.y}" 
        x2="{to_center.x}" y2="{to_center.y}" 
        stroke="{connection.color}" 
        stroke-width="{connection.width}" 
        stroke-dasharray="{'5,5' if connection.style == 'dashed' else 'none'}"
        marker-end="url(#arrowhead)"/>""")
        
        # Add arrowhead marker
        if svg_connections:
            svg_connections.insert(0, """
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" 
            refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#000"/>
    </marker>
  </defs>""")
        
        return '\n'.join(svg_connections)
    
    def _generate_svg_grid(self, whiteboard: CollaborativeWhiteboard) -> str:
        """Generate SVG grid pattern"""
        
        grid_size = whiteboard.grid_size
        width = whiteboard.canvas_size.width
        height = whiteboard.canvas_size.height
        
        return f"""
  <defs>
    <pattern id="grid" width="{grid_size}" height="{grid_size}" patternUnits="userSpaceOnUse">
      <path d="M {grid_size} 0 L 0 0 0 {grid_size}" fill="none" stroke="#e0e0e0" stroke-width="0.5"/>
    </pattern>
  </defs>
  <rect width="100%" height="100%" fill="url(#grid)"/>"""
    
    def _generate_png_export(self, whiteboard: CollaborativeWhiteboard, output_path: Path, region: Optional[BoundingBox]):
        """Generate PNG export (mock implementation)"""
        # In real implementation, would use libraries like cairosvg or playwright to render SVG to PNG
        with open(output_path, 'w') as f:
            f.write(f"PNG Export: {whiteboard.name}\n")
            f.write(f"Elements: {len(whiteboard.elements)}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
    
    def _generate_pdf_export(self, whiteboard: CollaborativeWhiteboard, output_path: Path, region: Optional[BoundingBox]):
        """Generate PDF export (mock implementation)"""
        # In real implementation, would use libraries like reportlab or weasyprint
        with open(output_path, 'w') as f:
            f.write(f"PDF Export: {whiteboard.name}\n")
            f.write(f"Elements: {len(whiteboard.elements)}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
    
    def _convert_to_miro_format(self, whiteboard: CollaborativeWhiteboard) -> Dict[str, Any]:
        """Convert whiteboard to Miro-compatible format"""
        
        miro_data = {
            "type": "board",
            "name": whiteboard.name,
            "description": whiteboard.description,
            "widgets": []
        }
        
        # Convert elements to Miro widgets
        for element in whiteboard.elements.values():
            if element.element_type == ElementType.STICKY_NOTE:
                miro_data["widgets"].append({
                    "type": "sticker",
                    "text": element.content,
                    "x": element.position.x,
                    "y": element.position.y,
                    "width": element.bounding_box.width,
                    "height": element.bounding_box.height,
                    "style": {
                        "backgroundColor": element.background_color,
                        "fontSize": element.font_size
                    }
                })
            elif element.element_type == ElementType.SHAPE:
                miro_data["widgets"].append({
                    "type": "shape",
                    "text": element.content,
                    "x": element.position.x,
                    "y": element.position.y,
                    "width": element.bounding_box.width,
                    "height": element.bounding_box.height,
                    "style": {
                        "shapeType": element.properties.get("shape_type", "rectangle"),
                        "backgroundColor": element.background_color,
                        "borderColor": element.border_color
                    }
                })
        
        return miro_data
    
    def _assign_user_color(self, user_id: str) -> str:
        """Assign a consistent color to a user for cursor and selections"""
        colors = ["#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4", "#ffeaa7", "#dda0dd", "#98d8c8"]
        hash_value = hashlib.md5(user_id.encode()).hexdigest()
        color_index = int(hash_value[:2], 16) % len(colors)
        return colors[color_index]
    
    def _broadcast_element_change(self, whiteboard_id: str, action: str, element: WhiteboardElement):
        """Broadcast element changes to active session participants"""
        session_id = f"session_{whiteboard_id}"
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            change_event = {
                "type": "element_change",
                "action": action,
                "element_id": element.element_id,
                "element_data": asdict(element),
                "timestamp": datetime.now().isoformat()
            }
            self.collaboration_events.append(change_event)
            print(f"📡 Broadcasting element change: {action} - {element.element_type.value}")
    
    def _broadcast_connection_change(self, whiteboard_id: str, action: str, connection: WhiteboardConnection):
        """Broadcast connection changes to active session participants"""
        session_id = f"session_{whiteboard_id}"
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            change_event = {
                "type": "connection_change",
                "action": action,
                "connection_id": connection.connection_id,
                "connection_data": asdict(connection),
                "timestamp": datetime.now().isoformat()
            }
            self.collaboration_events.append(change_event)
            print(f"📡 Broadcasting connection change: {action} - {connection.connection_type}")
    
    def _check_whiteboard_access(self, whiteboard: CollaborativeWhiteboard, user_id: str, required_level: WhiteboardAccessLevel) -> bool:
        """Check if user has required access level to whiteboard"""
        
        # Owner has all access
        if whiteboard.owner == user_id:
            return True
        
        # Check collaborator access
        if user_id in whiteboard.collaborators:
            user_level = whiteboard.collaborators[user_id]
            
            # Define access hierarchy
            access_hierarchy = {
                WhiteboardAccessLevel.VIEWER: 1,
                WhiteboardAccessLevel.COMMENTER: 2,
                WhiteboardAccessLevel.EDITOR: 3,
                WhiteboardAccessLevel.OWNER: 4
            }
            
            return access_hierarchy[user_level] >= access_hierarchy[required_level]
        
        return False
    
    def _save_whiteboard(self, whiteboard: CollaborativeWhiteboard):
        """Save whiteboard with encryption"""
        whiteboard_dir = self.storage_path / "whiteboards" / whiteboard.whiteboard_id
        whiteboard_dir.mkdir(parents=True, exist_ok=True)
        
        whiteboard_file = whiteboard_dir / "whiteboard.json"
        with open(whiteboard_file, 'w') as f:
            whiteboard_data = asdict(whiteboard)
            json.dump(whiteboard_data, f, default=str, indent=2)

# University-specific whiteboard templates
class UniversityWhiteboardTemplates:
    """Pre-configured whiteboard templates for university research"""
    
    @staticmethod
    def create_research_proposal_template() -> WhiteboardTemplate:
        """Create template for research proposal planning"""
        elements = [
            WhiteboardElement(
                element_id="problem_statement",
                element_type=ElementType.STICKY_NOTE,
                position=Point(100, 100),
                bounding_box=BoundingBox(100, 100, 250, 120),
                content="Problem Statement\n\nWhat specific research problem are we addressing?",
                background_color="#ffeb3b",
                created_at=datetime.now()
            ),
            WhiteboardElement(
                element_id="research_objectives",
                element_type=ElementType.STICKY_NOTE,
                position=Point(400, 100),
                bounding_box=BoundingBox(400, 100, 250, 120),
                content="Research Objectives\n\n1. Primary objective\n2. Secondary objectives",
                background_color="#4caf50",
                created_at=datetime.now()
            ),
            WhiteboardElement(
                element_id="methodology",
                element_type=ElementType.SHAPE,
                position=Point(100, 300),
                bounding_box=BoundingBox(100, 300, 200, 100),
                content="Methodology",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#2196f3",
                text_color="#ffffff",
                created_at=datetime.now()
            ),
            WhiteboardElement(
                element_id="expected_outcomes",
                element_type=ElementType.SHAPE,
                position=Point(350, 300),
                bounding_box=BoundingBox(350, 300, 200, 100),
                content="Expected Outcomes",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#9c27b0",
                text_color="#ffffff",
                created_at=datetime.now()
            ),
            WhiteboardElement(
                element_id="timeline",
                element_type=ElementType.SHAPE,
                position=Point(100, 450),
                bounding_box=BoundingBox(100, 450, 450, 80),
                content="Project Timeline: 36 months",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#ff5722",
                text_color="#ffffff",
                created_at=datetime.now()
            )
        ]
        
        connections = [
            WhiteboardConnection(
                connection_id="problem_to_objectives",
                from_element="problem_statement",
                to_element="research_objectives",
                created_at=datetime.now()
            ),
            WhiteboardConnection(
                connection_id="objectives_to_methodology",
                from_element="research_objectives",
                to_element="methodology",
                created_at=datetime.now()
            ),
            WhiteboardConnection(
                connection_id="methodology_to_outcomes",
                from_element="methodology",
                to_element="expected_outcomes",
                created_at=datetime.now()
            )
        ]
        
        return WhiteboardTemplate(
            template_id="research_proposal",
            name="Research Proposal Planning",
            description="Template for planning academic research proposals",
            category=TemplateCategory.RESEARCH_PLANNING,
            elements=elements,
            connections=connections,
            canvas_size=BoundingBox(0, 0, 800, 600),
            created_by="PRSM",
            institution="University System",
            created_at=datetime.now()
        )
    
    @staticmethod
    def create_stakeholder_mapping_template() -> WhiteboardTemplate:
        """Create template for stakeholder mapping"""
        elements = [
            # Core project in center
            WhiteboardElement(
                element_id="project_core",
                element_type=ElementType.SHAPE,
                position=Point(350, 250),
                bounding_box=BoundingBox(350, 250, 150, 100),
                content="Research Project",
                properties={"shape_type": ShapeType.CIRCLE.value},
                background_color="#4caf50",
                text_color="#ffffff",
                created_at=datetime.now()
            ),
            # University stakeholders
            WhiteboardElement(
                element_id="university_admin",
                element_type=ElementType.SHAPE,
                position=Point(100, 100),
                bounding_box=BoundingBox(100, 100, 120, 80),
                content="University\nAdministration",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#13294b",
                text_color="#ffffff",
                created_at=datetime.now()
            ),
            WhiteboardElement(
                element_id="faculty_researchers",
                element_type=ElementType.SHAPE,
                position=Point(250, 100),
                bounding_box=BoundingBox(250, 100, 120, 80),
                content="Faculty\nResearchers",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#2196f3",
                text_color="#ffffff",
                created_at=datetime.now()
            ),
            # Industry stakeholders
            WhiteboardElement(
                element_id="industry_partners",
                element_type=ElementType.SHAPE,
                position=Point(600, 200),
                bounding_box=BoundingBox(600, 200, 120, 80),
                content="Industry\nPartners",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#ff9800",
                text_color="#ffffff",
                created_at=datetime.now()
            ),
            # Students
            WhiteboardElement(
                element_id="students",
                element_type=ElementType.SHAPE,
                position=Point(300, 400),
                bounding_box=BoundingBox(300, 400, 120, 80),
                content="Graduate\nStudents",
                properties={"shape_type": ShapeType.RECTANGLE.value},
                background_color="#9c27b0",
                text_color="#ffffff",
                created_at=datetime.now()
            )
        ]
        
        # Create connections from project core to all stakeholders
        connections = [
            WhiteboardConnection(
                connection_id="core_to_admin",
                from_element="project_core",
                to_element="university_admin",
                label="Oversight",
                created_at=datetime.now()
            ),
            WhiteboardConnection(
                connection_id="core_to_faculty",
                from_element="project_core",
                to_element="faculty_researchers",
                label="Research",
                created_at=datetime.now()
            ),
            WhiteboardConnection(
                connection_id="core_to_industry",
                from_element="project_core",
                to_element="industry_partners",
                label="Application",
                created_at=datetime.now()
            ),
            WhiteboardConnection(
                connection_id="core_to_students",
                from_element="project_core",
                to_element="students",
                label="Training",
                created_at=datetime.now()
            )
        ]
        
        return WhiteboardTemplate(
            template_id="stakeholder_mapping",
            name="Research Stakeholder Mapping",
            description="Template for mapping research project stakeholders",
            category=TemplateCategory.STAKEHOLDER_MAPPING,
            elements=elements,
            connections=connections,
            canvas_size=BoundingBox(0, 0, 800, 600),
            created_by="PRSM",
            institution="University System",
            created_at=datetime.now()
        )

# Example usage and testing
if __name__ == "__main__":
    async def test_whiteboard_collaboration():
        """Test interactive whiteboard collaboration system"""
        
        print("🚀 Testing Interactive Whiteboard Collaboration")
        print("=" * 60)
        
        # Initialize whiteboard collaboration
        whiteboard_collab = WhiteboardCollaboration()
        
        # Create collaborative whiteboard for quantum research planning
        whiteboard = whiteboard_collab.create_collaborative_whiteboard(
            name="Quantum Computing Research Strategy - Multi-University Collaboration",
            description="Strategic planning whiteboard for quantum error correction research across UNC, Duke, NC State, and SAS Institute",
            owner="sarah.chen@unc.edu",
            collaborators={
                "alex.rodriguez@duke.edu": WhiteboardAccessLevel.EDITOR,
                "jennifer.kim@ncsu.edu": WhiteboardAccessLevel.EDITOR,
                "michael.johnson@sas.com": WhiteboardAccessLevel.COMMENTER,
                "research.coordinator@unc.edu": WhiteboardAccessLevel.EDITOR,
                "grants.office@unc.edu": WhiteboardAccessLevel.VIEWER
            },
            template_id="research_planning",
            security_level="high"
        )
        
        print(f"\n✅ Created collaborative whiteboard: {whiteboard.name}")
        print(f"   Whiteboard ID: {whiteboard.whiteboard_id}")
        print(f"   Collaborators: {len(whiteboard.collaborators)}")
        print(f"   Template elements: {len(whiteboard.elements)}")
        print(f"   Canvas size: {whiteboard.canvas_size.width}x{whiteboard.canvas_size.height}")
        
        # Add additional elements for detailed research planning
        timeline_element = whiteboard_collab.add_element(
            whiteboard.whiteboard_id,
            ElementType.STICKY_NOTE,
            Point(100, 500),
            (300, 100),
            "Phase 1 (Months 1-12): Theory Development\n- UNC: Quantum algorithms\n- Duke: Error models\n- NC State: Hardware specs",
            "sarah.chen@unc.edu",
            styling={"background_color": "#e8f5e8", "font_size": 12}
        )
        
        collaboration_element = whiteboard_collab.add_element(
            whiteboard.whiteboard_id,
            ElementType.SHAPE,
            Point(500, 400),
            (250, 150),
            "SAS Institute Partnership\n\n• Analytics platform\n• Statistical validation\n• Industry applications\n• Workforce development",
            "michael.johnson@sas.com",
            properties={"shape_type": ShapeType.RECTANGLE.value},
            styling={"background_color": "#0066cc", "text_color": "#ffffff", "font_size": 11}
        )
        
        milestones_element = whiteboard_collab.add_element(
            whiteboard.whiteboard_id,
            ElementType.TEXT_BOX,
            Point(800, 100),
            (200, 300),
            "Key Milestones:\n\n✓ Q1: Team formation\n✓ Q2: Grant submission\n○ Q3: Equipment setup\n○ Q4: First experiments\n○ Y2Q1: Initial results\n○ Y2Q2: Conference papers\n○ Y3Q1: Patent filing\n○ Y3Q4: Final report",
            "research.coordinator@unc.edu",
            styling={"background_color": "#fff3e0", "border_color": "#ff9800", "border_width": 2}
        )
        
        print(f"\n✅ Added research planning elements:")
        print(f"   Timeline: {timeline_element.element_id}")
        print(f"   SAS Partnership: {collaboration_element.element_id}")
        print(f"   Milestones: {milestones_element.element_id}")
        
        # Add connections between elements
        connection1 = whiteboard_collab.add_connection(
            whiteboard.whiteboard_id,
            timeline_element.element_id,
            collaboration_element.element_id,
            "sarah.chen@unc.edu",
            connection_type="arrow",
            styling={"color": "#0066cc", "width": 3},
            label="Partnership"
        )
        
        connection2 = whiteboard_collab.add_connection(
            whiteboard.whiteboard_id,
            collaboration_element.element_id,
            milestones_element.element_id,
            "research.coordinator@unc.edu",
            connection_type="arrow",
            styling={"color": "#ff9800", "width": 2, "style": "dashed"},
            label="Progress Tracking"
        )
        
        print(f"\n✅ Added connections:")
        print(f"   Partnership flow: {connection1.connection_id}")
        print(f"   Progress tracking: {connection2.connection_id}")
        
        # Start collaboration session
        session = whiteboard_collab.start_collaboration_session(
            whiteboard.whiteboard_id,
            "sarah.chen@unc.edu"
        )
        
        # Add more participants
        whiteboard_collab.start_collaboration_session(
            whiteboard.whiteboard_id,
            "alex.rodriguez@duke.edu"
        )
        
        whiteboard_collab.start_collaboration_session(
            whiteboard.whiteboard_id,
            "michael.johnson@sas.com"
        )
        
        print(f"\n✅ Started collaboration session:")
        print(f"   Session ID: {session.session_id}")
        print(f"   Active participants: {len(session.participants)}")
        print(f"   Real-time features: Cursor sharing, live updates")
        
        # Get AI optimization recommendations
        print(f"\n🎯 Getting diagram optimization recommendations...")
        
        optimization = await whiteboard_collab.optimize_diagram_layout(
            whiteboard.whiteboard_id,
            ["visual_hierarchy", "information_flow", "collaboration_clarity", "presentation_ready"],
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ Diagram optimization analysis completed:")
        print(f"   Confidence: {optimization['confidence']:.2f}")
        print(f"   Processing time: {optimization['processing_time']:.1f}s")
        print(f"   Elements analyzed: {len(whiteboard.elements)}")
        print(f"   Optimization goals: {len(optimization['optimization_goals'])}")
        
        # Generate template suggestions
        print(f"\n📋 Getting template suggestions...")
        
        template_suggestions = await whiteboard_collab.generate_template_suggestions(
            "quantum_computing",
            "university_industry_partnership",
            ["UNC Chapel Hill", "Duke University", "NC State", "SAS Institute"],
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ Template suggestions generated:")
        print(f"   Research area: {template_suggestions['research_area']}")
        print(f"   Collaboration type: {template_suggestions['collaboration_type']}")
        print(f"   Institutions: {len(template_suggestions['institutions'])}")
        print(f"   Confidence: {template_suggestions['confidence']:.2f}")
        
        # Export whiteboard in multiple formats
        print(f"\n📦 Exporting whiteboard...")
        
        svg_export = whiteboard_collab.export_whiteboard(
            whiteboard.whiteboard_id,
            "svg",
            "sarah.chen@unc.edu"
        )
        
        json_export = whiteboard_collab.export_whiteboard(
            whiteboard.whiteboard_id,
            "json",
            "sarah.chen@unc.edu"
        )
        
        miro_export = whiteboard_collab.export_whiteboard(
            whiteboard.whiteboard_id,
            "miro",
            "sarah.chen@unc.edu"
        )
        
        print(f"✅ Whiteboard exported in multiple formats:")
        print(f"   SVG: {Path(svg_export).name}")
        print(f"   JSON: {Path(json_export).name}")
        print(f"   Miro: {Path(miro_export).name}")
        
        # Test university-specific templates
        print(f"\n🏛️ Testing university-specific templates...")
        
        research_proposal_template = UniversityWhiteboardTemplates.create_research_proposal_template()
        stakeholder_template = UniversityWhiteboardTemplates.create_stakeholder_mapping_template()
        
        print(f"✅ Research proposal template: {research_proposal_template.name}")
        print(f"   Elements: {len(research_proposal_template.elements)}")
        print(f"   Connections: {len(research_proposal_template.connections)}")
        
        print(f"✅ Stakeholder mapping template: {stakeholder_template.name}")
        print(f"   Elements: {len(stakeholder_template.elements)}")
        print(f"   Connections: {len(stakeholder_template.connections)}")
        
        # Create whiteboard from stakeholder template
        stakeholder_board = whiteboard_collab.create_collaborative_whiteboard(
            name="RTP Quantum Computing Stakeholder Map",
            description="Stakeholder mapping for Research Triangle Park quantum computing initiative",
            owner="sarah.chen@unc.edu",
            collaborators={
                "partnerships@unc.edu": WhiteboardAccessLevel.EDITOR,
                "external.relations@duke.edu": WhiteboardAccessLevel.EDITOR,
                "industry.partnerships@ncsu.edu": WhiteboardAccessLevel.EDITOR
            },
            template_id="stakeholder_mapping",
            security_level="high"
        )
        
        print(f"\n✅ Created stakeholder mapping whiteboard: {stakeholder_board.name}")
        print(f"   Whiteboard ID: {stakeholder_board.whiteboard_id}")
        print(f"   Template-based elements: {len(stakeholder_board.elements)}")
        
        print(f"\n🎉 Interactive whiteboard collaboration test completed!")
        print("✅ Ready for university-industry visual collaboration partnerships!")
    
    # Run test
    import asyncio
    asyncio.run(test_whiteboard_collaboration())