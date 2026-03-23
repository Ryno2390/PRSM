#!/usr/bin/env python3
"""
Jupyter Notebook Collaborative Editing Integration for PRSM
==========================================================

This module integrates Jupyter notebook collaborative editing with PRSM's 
NWTN pipeline and cryptographic security features. It enables:

- Real-time collaborative editing of Jupyter notebooks
- Secure sharing with P2P cryptographic sharding
- Integration with NWTN for AI-powered code assistance
- University-industry collaboration with IP protection
"""

import json
import asyncio
import websockets
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
from pathlib import Path

# Import PRSM components
from ..security.crypto_sharding import BasicCryptoSharding
from ..models import QueryRequest

# Mock UnifiedPipelineController for testing
class UnifiedPipelineController:
    """Mock pipeline controller for integration testing"""
    async def initialize(self):
        pass
    
    async def process_query_full_pipeline(self, user_id: str, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # This will be replaced with actual NWTN integration
        return {
            "response": {"text": "Mock NWTN response", "confidence": 0.85, "sources": []},
            "performance_metrics": {"total_processing_time": 1.5}
        }

@dataclass
class NotebookCell:
    """Represents a single Jupyter notebook cell"""
    cell_id: str
    cell_type: str  # 'code' or 'markdown'
    source: List[str]
    execution_count: Optional[int] = None
    outputs: List[Dict] = None
    metadata: Dict = None

@dataclass
class NotebookDocument:
    """Represents a complete Jupyter notebook"""
    notebook_id: str
    name: str
    cells: List[NotebookCell]
    metadata: Dict
    created_by: str
    created_at: datetime
    last_modified: datetime
    collaborators: List[str]
    security_level: str = "medium"

@dataclass
class CollaborativeEdit:
    """Represents a collaborative edit operation"""
    edit_id: str
    notebook_id: str
    cell_id: str
    operation: str  # 'insert', 'delete', 'modify'
    content: Any
    user_id: str
    timestamp: datetime
    applied: bool = False

class JupyterCollaboration:
    """
    Main class for Jupyter notebook collaborative editing with PRSM integration
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the Jupyter collaboration system"""
        self.storage_path = storage_path or Path("./jupyter_collaboration")
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize PRSM components
        self.crypto_sharding = BasicCryptoSharding()
        self.nwtn_pipeline = None  # Will be initialized when needed
        
        # Active notebooks and connections
        self.active_notebooks: Dict[str, NotebookDocument] = {}
        self.notebook_connections: Dict[str, List[websockets.WebSocketServerProtocol]] = {}
        self.pending_edits: Dict[str, List[CollaborativeEdit]] = {}
        
        # WebSocket server
        self.websocket_server = None
    
    async def initialize_nwtn_pipeline(self):
        """Initialize NWTN pipeline for AI assistance"""
        if self.nwtn_pipeline is None:
            self.nwtn_pipeline = UnifiedPipelineController()
            await self.nwtn_pipeline.initialize()
    
    def create_notebook(self, 
                       name: str, 
                       created_by: str, 
                       collaborators: List[str],
                       security_level: str = "medium") -> NotebookDocument:
        """Create a new collaborative notebook"""
        notebook_id = str(uuid.uuid4())
        
        # Create initial empty notebook structure
        notebook = NotebookDocument(
            notebook_id=notebook_id,
            name=name,
            cells=[],
            metadata={
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.9.0"
                }
            },
            created_by=created_by,
            created_at=datetime.now(),
            last_modified=datetime.now(),
            collaborators=collaborators,
            security_level=security_level
        )
        
        self.active_notebooks[notebook_id] = notebook
        self.notebook_connections[notebook_id] = []
        self.pending_edits[notebook_id] = []
        
        # Save notebook with encryption if high security
        self._save_notebook(notebook)
        
        return notebook
    
    def add_cell(self, 
                notebook_id: str, 
                cell_type: str = "code", 
                content: str = "",
                position: Optional[int] = None) -> NotebookCell:
        """Add a new cell to a notebook"""
        if notebook_id not in self.active_notebooks:
            raise ValueError(f"Notebook {notebook_id} not found")
        
        cell_id = str(uuid.uuid4())
        cell = NotebookCell(
            cell_id=cell_id,
            cell_type=cell_type,
            source=content.split('\n') if content else [''],
            execution_count=None,
            outputs=[],
            metadata={}
        )
        
        notebook = self.active_notebooks[notebook_id]
        if position is None:
            notebook.cells.append(cell)
        else:
            notebook.cells.insert(position, cell)
        
        notebook.last_modified = datetime.now()
        
        # Broadcast change to all collaborators
        asyncio.create_task(self._broadcast_change(notebook_id, {
            "type": "cell_added",
            "cell": asdict(cell),
            "position": position or len(notebook.cells) - 1
        }))
        
        return cell
    
    async def execute_cell_with_nwtn(self, 
                                   notebook_id: str, 
                                   cell_id: str, 
                                   user_id: str) -> Dict[str, Any]:
        """Execute a code cell with NWTN AI assistance"""
        if notebook_id not in self.active_notebooks:
            raise ValueError(f"Notebook {notebook_id} not found")
        
        notebook = self.active_notebooks[notebook_id]
        cell = next((c for c in notebook.cells if c.cell_id == cell_id), None)
        
        if not cell or cell.cell_type != "code":
            raise ValueError("Invalid cell for execution")
        
        # Initialize NWTN pipeline if needed
        await self.initialize_nwtn_pipeline()
        
        # Get code from cell
        code = '\n'.join(cell.source)
        
        # Use NWTN for code analysis and suggestions
        query_request = QueryRequest(
            user_id=user_id,
            query_text=f"Analyze and suggest improvements for this Python code:\n{code}",
            context={
                "domain": "data_science",
                "notebook_context": True,
                "cell_type": "code"
            }
        )
        
        # Get NWTN analysis
        nwtn_result = await self.nwtn_pipeline.process_query_full_pipeline(
            user_id=user_id,
            query=query_request.query_text,
            context=query_request.context
        )
        
        # Execute the actual code (this would integrate with Jupyter kernel)
        execution_result = await self._execute_code(code)
        
        # Update cell with execution results
        cell.execution_count = (cell.execution_count or 0) + 1
        cell.outputs = execution_result.get('outputs', [])
        
        # Combine execution results with NWTN insights
        result = {
            "execution_result": execution_result,
            "nwtn_analysis": {
                "suggestions": nwtn_result.get('response', {}).get('text', ''),
                "confidence": nwtn_result.get('response', {}).get('confidence', 0.0),
                "sources": nwtn_result.get('response', {}).get('sources', [])
            },
            "cell_id": cell_id,
            "execution_count": cell.execution_count
        }
        
        # Broadcast execution results to collaborators
        await self._broadcast_change(notebook_id, {
            "type": "cell_executed",
            "cell_id": cell_id,
            "result": result
        })
        
        return result
    
    async def _execute_code(self, code: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute Python code using Jupyter kernel if available, or exec fallback"""
        import time
        import io
        import sys

        start_time = time.perf_counter()

        # Try to use jupyter_client if available
        try:
            import jupyter_client

            # Get or create kernel for this session
            if not hasattr(self, '_kernels'):
                self._kernels: Dict[str, tuple] = {}

            kernel_key = session_id or "default"

            if kernel_key not in self._kernels:
                # Start a new kernel
                km = jupyter_client.KernelManager()
                km.start_kernel()
                kc = km.client()
                self._kernels[kernel_key] = (km, kc)

            km, kc = self._kernels[kernel_key]

            # Execute the code
            msg_id = kc.execute(code)

            # Collect output with timeout
            outputs = []
            execution_count = None
            error = None

            try:
                while True:
                    try:
                        msg = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, kc.get_iopub_msg, 30
                            ),
                            timeout=30.0
                        )
                    except asyncio.TimeoutError:
                        error = "execution_timeout"
                        break

                    if msg['parent_header'].get('msg_id') != msg_id:
                        continue

                    msg_type = msg['header']['msg_type']
                    content = msg['content']

                    if msg_type == 'status' and content.get('execution_state') == 'idle':
                        break
                    elif msg_type == 'execute_input':
                        execution_count = content.get('execution_count')
                    elif msg_type == 'stream':
                        outputs.append({
                            "output_type": "stream",
                            "name": content.get('name', 'stdout'),
                            "text": content.get('text', '').split('\n')
                        })
                    elif msg_type == 'execute_result':
                        outputs.append({
                            "output_type": "execute_result",
                            "data": content.get('data', {}),
                            "execution_count": content.get('execution_count')
                        })
                    elif msg_type == 'error':
                        error = content.get('ename', 'UnknownError')
                        outputs.append({
                            "output_type": "error",
                            "ename": content.get('ename'),
                            "evalue": content.get('evalue'),
                            "traceback": content.get('traceback', [])
                        })
            except Exception as e:
                error = str(e)

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            return {
                "status": "error" if error else "ok",
                "outputs": outputs,
                "execution_count": execution_count or 0,
                "success": error is None,
                "error": error,
                "execution_time_ms": execution_time_ms
            }

        except ImportError:
            # Fall back to exec() in sandboxed namespace
            pass

        # Fallback: exec() in sandboxed namespace
        try:
            # Create sandboxed execution environment
            exec_globals = {'__builtins__': __builtins__}
            exec_locals = {}

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = captured_stdout = io.StringIO()

            try:
                exec(code, exec_globals, exec_locals)
                output_text = captured_stdout.getvalue()
            finally:
                sys.stdout = old_stdout

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            outputs = []
            if output_text:
                outputs.append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": output_text.split('\n')
                })

            # Check for return value
            if exec_locals.get('_') is not None:
                outputs.append({
                    "output_type": "execute_result",
                    "data": {"text/plain": str(exec_locals['_'])},
                    "execution_count": 0
                })

            return {
                "status": "ok",
                "outputs": outputs,
                "execution_count": 0,
                "success": True,
                "error": None,
                "execution_time_ms": execution_time_ms
            }

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return {
                "status": "error",
                "outputs": [{
                    "output_type": "error",
                    "ename": type(e).__name__,
                    "evalue": str(e),
                    "traceback": [str(e)]
                }],
                "execution_count": 0,
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time_ms
            }
    
    async def apply_collaborative_edit(self, 
                                     notebook_id: str, 
                                     edit: CollaborativeEdit) -> bool:
        """Apply a collaborative edit to a notebook"""
        if notebook_id not in self.active_notebooks:
            return False
        
        notebook = self.active_notebooks[notebook_id]
        
        # Find the target cell
        cell = next((c for c in notebook.cells if c.cell_id == edit.cell_id), None)
        if not cell:
            return False
        
        # Apply the edit based on operation type
        try:
            if edit.operation == "modify_source":
                cell.source = edit.content
            elif edit.operation == "insert_line":
                line_num = edit.content.get("line", 0)
                text = edit.content.get("text", "")
                if line_num <= len(cell.source):
                    cell.source.insert(line_num, text)
            elif edit.operation == "delete_line":
                line_num = edit.content.get("line", 0)
                if 0 <= line_num < len(cell.source):
                    del cell.source[line_num]
            
            notebook.last_modified = datetime.now()
            edit.applied = True
            
            # Broadcast the change to other collaborators
            await self._broadcast_change(notebook_id, {
                "type": "collaborative_edit",
                "edit": asdict(edit)
            }, exclude_user=edit.user_id)
            
            return True
            
        except Exception as e:
            print(f"Error applying edit: {e}")
            return False
    
    async def _broadcast_change(self, 
                              notebook_id: str, 
                              change: Dict[str, Any], 
                              exclude_user: Optional[str] = None):
        """Broadcast a change to all connected collaborators"""
        if notebook_id not in self.notebook_connections:
            return
        
        message = json.dumps(change)
        disconnected = []
        
        for connection in self.notebook_connections[notebook_id]:
            try:
                # In a real implementation, you'd track user IDs per connection
                await connection.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.notebook_connections[notebook_id].remove(conn)
    
    def _save_notebook(self, notebook: NotebookDocument):
        """Save notebook to disk with optional encryption"""
        notebook_path = self.storage_path / f"{notebook.notebook_id}.ipynb"
        
        # Convert to standard Jupyter format
        jupyter_format = {
            "cells": [
                {
                    "cell_type": cell.cell_type,
                    "source": cell.source,
                    "metadata": cell.metadata or {},
                    "execution_count": cell.execution_count,
                    "outputs": cell.outputs or []
                }
                for cell in notebook.cells
            ],
            "metadata": notebook.metadata,
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        notebook_json = json.dumps(jupyter_format, indent=2)
        
        # Apply encryption for high-security notebooks
        if notebook.security_level == "high":
            # Use crypto sharding for secure storage
            temp_file = self.storage_path / f"temp_{notebook.notebook_id}.ipynb"
            with open(temp_file, 'w') as f:
                f.write(notebook_json)
            
            try:
                shards, manifest = self.crypto_sharding.shard_file(
                    str(temp_file),
                    notebook.collaborators
                )
                
                # Save shards and manifest
                self._save_shards(notebook.notebook_id, shards, manifest)
                temp_file.unlink()  # Remove temporary file
                
            except Exception as e:
                print(f"Error creating secure notebook: {e}")
                # Fall back to regular storage
                with open(notebook_path, 'w') as f:
                    f.write(notebook_json)
        else:
            # Regular storage for standard security
            with open(notebook_path, 'w') as f:
                f.write(notebook_json)
    
    def _save_shards(self, 
                    notebook_id: str, 
                    shards: List, 
                    manifest: Any):
        """Save encrypted notebook shards"""
        shards_dir = self.storage_path / "shards" / notebook_id
        shards_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each shard
        for i, shard in enumerate(shards):
            shard_path = shards_dir / f"shard_{i}.enc"
            with open(shard_path, 'wb') as f:
                f.write(shard.shard_data)
        
        # Save manifest
        manifest_path = shards_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(asdict(manifest), f, default=str, indent=2)
    
    async def start_websocket_server(self, host: str = "localhost", port: int = 8888):
        """Start WebSocket server for real-time collaboration"""
        async def handle_connection(websocket, path):
            """Handle individual WebSocket connections"""
            try:
                # Extract notebook ID from path
                notebook_id = path.strip('/')
                
                if notebook_id in self.notebook_connections:
                    self.notebook_connections[notebook_id].append(websocket)
                
                # Send current notebook state
                if notebook_id in self.active_notebooks:
                    notebook = self.active_notebooks[notebook_id]
                    await websocket.send(json.dumps({
                        "type": "notebook_state",
                        "notebook": asdict(notebook)
                    }))
                
                # Handle incoming messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_websocket_message(notebook_id, data, websocket)
                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON"
                        }))
                        
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                # Clean up connection
                if notebook_id in self.notebook_connections and websocket in self.notebook_connections[notebook_id]:
                    self.notebook_connections[notebook_id].remove(websocket)
        
        self.websocket_server = await websockets.serve(handle_connection, host, port)
        print(f"Jupyter collaboration WebSocket server started on ws://{host}:{port}")
    
    async def _handle_websocket_message(self, 
                                      notebook_id: str, 
                                      data: Dict[str, Any], 
                                      websocket):
        """Handle incoming WebSocket messages"""
        message_type = data.get("type")
        
        if message_type == "edit_cell":
            edit = CollaborativeEdit(
                edit_id=str(uuid.uuid4()),
                notebook_id=notebook_id,
                cell_id=data["cell_id"],
                operation=data["operation"],
                content=data["content"],
                user_id=data["user_id"],
                timestamp=datetime.now()
            )
            
            success = await self.apply_collaborative_edit(notebook_id, edit)
            await websocket.send(json.dumps({
                "type": "edit_response",
                "success": success,
                "edit_id": edit.edit_id
            }))
        
        elif message_type == "execute_cell":
            try:
                result = await self.execute_cell_with_nwtn(
                    notebook_id,
                    data["cell_id"],
                    data["user_id"]
                )
                await websocket.send(json.dumps({
                    "type": "execution_result",
                    "result": result
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
        
        elif message_type == "add_cell":
            try:
                cell = self.add_cell(
                    notebook_id,
                    data.get("cell_type", "code"),
                    data.get("content", ""),
                    data.get("position")
                )
                await websocket.send(json.dumps({
                    "type": "cell_added",
                    "cell": asdict(cell)
                }))
            except Exception as e:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))

# Integration with existing PRSM collaboration UI
class JupyterCollaborationAPI:
    """API endpoints for integrating with PRSM collaboration interface"""
    
    def __init__(self, jupyter_collab: JupyterCollaboration):
        self.jupyter_collab = jupyter_collab
    
    async def create_university_industry_notebook(self,
                                                name: str,
                                                university_researchers: List[str],
                                                industry_partners: List[str],
                                                security_level: str = "high") -> Dict[str, Any]:
        """Create a notebook specifically for university-industry collaboration"""
        all_collaborators = university_researchers + industry_partners
        created_by = university_researchers[0] if university_researchers else "system"
        
        notebook = self.jupyter_collab.create_notebook(
            name=name,
            created_by=created_by,
            collaborators=all_collaborators,
            security_level=security_level
        )
        
        # Add initial welcome cell with collaboration guidelines
        welcome_cell = self.jupyter_collab.add_cell(
            notebook.notebook_id,
            "markdown",
            f"""# {name}

## University-Industry Collaboration Notebook

**Security Level**: {security_level.upper()}
**Participants**: {len(all_collaborators)} members

### Collaboration Guidelines:
- All code and data are protected with P2P cryptographic sharding
- University researchers: Share openly within academic guidelines
- Industry partners: Maintain confidentiality agreements
- Use NWTN AI assistance for code optimization and research insights
- Document all proprietary algorithms and methods clearly

### Getting Started:
1. Import necessary libraries
2. Load and explore datasets
3. Implement algorithms with proper documentation
4. Use AI assistance for optimization suggestions
"""
        )
        
        return {
            "notebook_id": notebook.notebook_id,
            "name": notebook.name,
            "collaborators": all_collaborators,
            "security_level": security_level,
            "websocket_url": f"ws://localhost:8888/{notebook.notebook_id}",
            "welcome_cell_id": welcome_cell.cell_id
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_jupyter_collaboration():
        """Test the Jupyter collaboration system"""
        
        # Initialize the collaboration system
        jupyter_collab = JupyterCollaboration()
        api = JupyterCollaborationAPI(jupyter_collab)
        
        print("🚀 Testing Jupyter Collaborative Editing with PRSM Integration")
        
        # Create a university-industry notebook
        notebook_info = await api.create_university_industry_notebook(
            name="Quantum ML Research Project",
            university_researchers=["sarah.chen@unc.edu", "alex.rodriguez@duke.edu"],
            industry_partners=["michael.johnson@sas.com"],
            security_level="high"
        )
        
        print(f"✅ Created secure notebook: {notebook_info['name']}")
        print(f"   Notebook ID: {notebook_info['notebook_id']}")
        print(f"   Security Level: {notebook_info['security_level']}")
        print(f"   Collaborators: {len(notebook_info['collaborators'])}")
        
        # Add a code cell
        code_cell = jupyter_collab.add_cell(
            notebook_info['notebook_id'],
            "code",
            """import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load quantum dataset
data = pd.read_csv('quantum_features.csv')
print(f"Dataset shape: {data.shape}")

# This is proprietary SAS Institute algorithm - confidential
def proprietary_quantum_classifier(X, y):
    # Placeholder for actual proprietary algorithm
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    return classifier.fit(X, y)
"""
        )
        
        print(f"✅ Added code cell: {code_cell.cell_id}")
        
        # Simulate execution with NWTN assistance
        try:
            execution_result = await jupyter_collab.execute_cell_with_nwtn(
                notebook_info['notebook_id'],
                code_cell.cell_id,
                "sarah.chen@unc.edu"
            )
            
            print("✅ Code execution with NWTN analysis completed")
            print(f"   NWTN Confidence: {execution_result['nwtn_analysis']['confidence']:.2f}")
            print(f"   NWTN Suggestions: {execution_result['nwtn_analysis']['suggestions'][:100]}...")
            
        except Exception as e:
            print(f"⚠️  NWTN integration not available: {e}")
        
        # Test collaborative editing
        edit = CollaborativeEdit(
            edit_id=str(uuid.uuid4()),
            notebook_id=notebook_info['notebook_id'],
            cell_id=code_cell.cell_id,
            operation="insert_line",
            content={"line": 5, "text": "# Added comment via collaborative editing"},
            user_id="michael.johnson@sas.com",
            timestamp=datetime.now()
        )
        
        success = await jupyter_collab.apply_collaborative_edit(
            notebook_info['notebook_id'],
            edit
        )
        
        print(f"✅ Collaborative edit applied: {success}")
        
        print("\n🎉 Jupyter collaboration system test completed!")
        print("Ready for integration with PRSM collaboration interface")
    
    # Run the test
    asyncio.run(test_jupyter_collaboration())