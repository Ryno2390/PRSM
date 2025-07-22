#!/usr/bin/env python3
"""
NWTN Background Execution Manager
================================

This module provides background execution capabilities for long-running NWTN 
deep reasoning tasks with real-time progress monitoring, checkpointing, and
FTNS payment tracking.

Features:
- Background process execution (2-4 hour DEEP reasoning runs)
- Real-time progress monitoring with live updates
- Resumable checkpointing system
- FTNS payment tracking throughout execution
- System stability monitoring
- Automatic result persistence
"""

import asyncio
import json
import time
import os
import sys
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import signal
import atexit
from uuid import uuid4

from prsm.nwtn.meta_reasoning_engine import MetaReasoningEngine, ThinkingMode
from prsm.nwtn.external_storage_config import get_external_storage_manager

logger = structlog.get_logger(__name__)

class ExecutionStatus(Enum):
    """Status of background execution"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ExecutionProgress:
    """Real-time execution progress tracking"""
    execution_id: str
    status: ExecutionStatus
    started_at: datetime
    updated_at: datetime
    query: str
    thinking_mode: str
    
    # Progress metrics
    total_permutations: int = 5040
    completed_permutations: int = 0
    current_sequence: str = ""
    estimated_completion: Optional[datetime] = None
    
    # Performance metrics
    permutations_per_minute: float = 0.0
    elapsed_hours: float = 0.0
    estimated_hours_remaining: float = 0.0
    
    # System metrics
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # FTNS metrics
    ftns_tokens_distributed: float = 0.0
    ftns_payments_count: int = 0
    
    # Results
    confidence_score: float = 0.0
    papers_analyzed: int = 0
    final_answer_length: int = 0
    citations_count: int = 0
    
    # Error tracking
    error_count: int = 0
    last_error: str = ""
    recovery_count: int = 0

class BackgroundExecutionManager:
    """Manages background execution of NWTN deep reasoning tasks"""
    
    def __init__(self, progress_dir: str = "nwtn_progress"):
        self.progress_dir = Path(progress_dir)
        self.progress_dir.mkdir(exist_ok=True)
        
        self.active_executions: Dict[str, ExecutionProgress] = {}
        self.progress_callbacks: Dict[str, Callable] = {}
        
        # Background task tracking
        self.background_tasks: Dict[str, asyncio.Task] = {}
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        logger.info("Background Execution Manager initialized",
                   progress_dir=str(self.progress_dir))
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal", signal=signum)
            asyncio.create_task(self.shutdown_all())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_background_execution(self, 
                                 query: str,
                                 thinking_mode: ThinkingMode = ThinkingMode.DEEP,
                                 context: Dict[str, Any] = None,
                                 progress_callback: Optional[Callable] = None) -> str:
        """
        Start a background NWTN deep reasoning execution
        
        Args:
            query: The reasoning query
            thinking_mode: DEEP for full 5,040 permutations
            context: Execution context including FTNS budget
            progress_callback: Optional callback for progress updates
            
        Returns:
            execution_id: Unique ID for tracking this execution
        """
        execution_id = str(uuid4())
        
        # Initialize progress tracking
        progress = ExecutionProgress(
            execution_id=execution_id,
            status=ExecutionStatus.INITIALIZING,
            started_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            query=query,
            thinking_mode=thinking_mode.value,
            total_permutations=5040 if thinking_mode == ThinkingMode.DEEP else 
                              720 if thinking_mode == ThinkingMode.INTERMEDIATE else 7
        )
        
        self.active_executions[execution_id] = progress
        
        if progress_callback:
            self.progress_callbacks[execution_id] = progress_callback
        
        # Start background task in thread pool if no event loop
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(
                self._execute_background_reasoning(execution_id, query, thinking_mode, context or {})
            )
            self.background_tasks[execution_id] = task
        except RuntimeError:
            # No event loop running, start in thread
            import threading
            thread = threading.Thread(
                target=self._run_in_thread,
                args=(execution_id, query, thinking_mode, context or {})
            )
            thread.daemon = True
            thread.start()
            # Store thread reference
            self.background_tasks[execution_id] = thread
        
        # Save initial progress
        self._save_progress(execution_id)
        
        logger.info("Started background execution",
                   execution_id=execution_id,
                   query=query[:100] + "..." if len(query) > 100 else query,
                   thinking_mode=thinking_mode.value)
        
        return execution_id
    
    def _run_in_thread(self, execution_id: str, query: str, thinking_mode: ThinkingMode, context: Dict[str, Any]):
        """Run async reasoning in a separate thread with its own event loop"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Ensure we have the progress tracking in the thread
            logger.info("Starting thread execution", execution_id=execution_id)
            
            # Run the async reasoning
            loop.run_until_complete(
                self._execute_background_reasoning(execution_id, query, thinking_mode, context)
            )
            
            logger.info("Thread execution completed", execution_id=execution_id)
            
        except Exception as e:
            logger.error("Thread execution failed", execution_id=execution_id, error=str(e))
            
            # Update progress with failure
            if execution_id in self.active_executions:
                progress = self.active_executions[execution_id]
                progress.status = ExecutionStatus.FAILED
                progress.last_error = str(e)
                progress.error_count += 1
                progress.updated_at = datetime.now(timezone.utc)
                self._save_progress(execution_id)
        finally:
            # Clean up loop
            try:
                loop.close()
            except:
                pass
    
    async def _execute_background_reasoning(self, 
                                          execution_id: str,
                                          query: str,
                                          thinking_mode: ThinkingMode,
                                          context: Dict[str, Any]):
        """Execute the reasoning in background with progress tracking"""
        
        progress = self.active_executions[execution_id]
        
        try:
            # Update status
            progress.status = ExecutionStatus.RUNNING
            progress.updated_at = datetime.now(timezone.utc)
            self._update_progress(execution_id)
            
            # Initialize NWTN
            logger.info("Initializing NWTN for background execution",
                       execution_id=execution_id)
            
            meta_engine = MetaReasoningEngine()
            await meta_engine.initialize_external_knowledge_base()
            
            # Setup progress monitoring
            original_update_method = None
            if hasattr(meta_engine, '_update_progress'):
                original_update_method = meta_engine._update_progress
                meta_engine._update_progress = lambda seq, total, current: self._reasoning_progress_callback(
                    execution_id, seq, total, current
                )
            
            # Start reasoning with progress tracking
            start_time = time.time()
            
            reasoning_result = await meta_engine.meta_reason(
                query=query,
                context=context,
                thinking_mode=thinking_mode,
                include_world_model=True
            )
            
            end_time = time.time()
            
            # Update final progress
            progress.status = ExecutionStatus.COMPLETED
            progress.completed_permutations = progress.total_permutations
            progress.elapsed_hours = (end_time - start_time) / 3600
            progress.confidence_score = getattr(reasoning_result, 'confidence_score', 0.0)
            progress.ftns_tokens_distributed = getattr(reasoning_result, 'ftns_rewards_distributed', 0.0)
            progress.papers_analyzed = len(getattr(reasoning_result, 'retrieved_papers', []))
            
            if hasattr(reasoning_result, 'final_answer'):
                progress.final_answer_length = len(reasoning_result.final_answer)
                progress.citations_count = reasoning_result.final_answer.count('[')
            
            progress.updated_at = datetime.now(timezone.utc)
            self._update_progress(execution_id)
            
            # Save final result
            result_file = self.progress_dir / f"{execution_id}_result.json"
            with open(result_file, 'w') as f:
                json.dump({
                    'execution_id': execution_id,
                    'query': query,
                    'thinking_mode': thinking_mode.value,
                    'completed_at': datetime.now(timezone.utc).isoformat(),
                    'duration_hours': progress.elapsed_hours,
                    'confidence_score': progress.confidence_score,
                    'ftns_tokens_distributed': progress.ftns_tokens_distributed,
                    'papers_analyzed': progress.papers_analyzed,
                    'final_answer': getattr(reasoning_result, 'final_answer', ''),
                    'citations_count': progress.citations_count
                }, f, indent=2)
            
            logger.info("Background execution completed successfully",
                       execution_id=execution_id,
                       duration_hours=progress.elapsed_hours,
                       permutations=progress.completed_permutations)
            
        except Exception as e:
            progress.status = ExecutionStatus.FAILED
            progress.last_error = str(e)
            progress.error_count += 1
            progress.updated_at = datetime.now(timezone.utc)
            self._update_progress(execution_id)
            
            logger.error("Background execution failed",
                        execution_id=execution_id,
                        error=str(e),
                        exc_info=True)
            raise
    
    def _reasoning_progress_callback(self, execution_id: str, sequence: int, total: int, current_desc: str):
        """Callback for reasoning progress updates"""
        if execution_id not in self.active_executions:
            return
        
        progress = self.active_executions[execution_id]
        progress.completed_permutations = sequence
        progress.current_sequence = current_desc
        progress.updated_at = datetime.now(timezone.utc)
        
        # Calculate performance metrics
        elapsed_seconds = (progress.updated_at - progress.started_at).total_seconds()
        if elapsed_seconds > 0:
            progress.elapsed_hours = elapsed_seconds / 3600
            progress.permutations_per_minute = (sequence / elapsed_seconds) * 60
            
            if progress.permutations_per_minute > 0:
                remaining_permutations = total - sequence
                remaining_minutes = remaining_permutations / progress.permutations_per_minute
                progress.estimated_hours_remaining = remaining_minutes / 60
                progress.estimated_completion = progress.updated_at + \
                    timedelta(minutes=remaining_minutes)
        
        # Update system metrics
        try:
            import psutil
            process = psutil.Process()
            progress.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            progress.cpu_usage_percent = process.cpu_percent()
        except:
            pass
        
        # Save progress every 100 sequences
        if sequence % 100 == 0:
            self._save_progress(execution_id)
        
        # Call user callback
        if execution_id in self.progress_callbacks:
            try:
                self.progress_callbacks[execution_id](progress)
            except Exception as e:
                logger.warning("Progress callback failed", error=str(e))
    
    def _update_progress(self, execution_id: str):
        """Update progress and notify callbacks"""
        if execution_id not in self.active_executions:
            return
        
        progress = self.active_executions[execution_id]
        
        # Save to file
        self._save_progress(execution_id)
        
        # Call callback
        if execution_id in self.progress_callbacks:
            try:
                self.progress_callbacks[execution_id](progress)
            except Exception as e:
                logger.warning("Progress callback failed", error=str(e))
    
    def _save_progress(self, execution_id: str):
        """Save progress to file"""
        if execution_id not in self.active_executions:
            return
        
        progress = self.active_executions[execution_id]
        progress_file = self.progress_dir / f"{execution_id}_progress.json"
        
        # Convert to dict and handle datetime serialization
        progress_dict = asdict(progress)
        progress_dict['started_at'] = progress.started_at.isoformat()
        progress_dict['updated_at'] = progress.updated_at.isoformat()
        progress_dict['status'] = progress.status.value
        
        if progress.estimated_completion:
            progress_dict['estimated_completion'] = progress.estimated_completion.isoformat()
        
        with open(progress_file, 'w') as f:
            json.dump(progress_dict, f, indent=2)
    
    def get_execution_status(self, execution_id: str) -> Optional[ExecutionProgress]:
        """Get current status of an execution"""
        return self.active_executions.get(execution_id)
    
    def list_active_executions(self) -> Dict[str, ExecutionProgress]:
        """List all active executions"""
        return self.active_executions.copy()
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        if execution_id not in self.background_tasks:
            return False
        
        task = self.background_tasks[execution_id]
        task.cancel()
        
        if execution_id in self.active_executions:
            progress = self.active_executions[execution_id]
            progress.status = ExecutionStatus.CANCELLED
            progress.updated_at = datetime.now(timezone.utc)
            self._save_progress(execution_id)
        
        logger.info("Cancelled execution", execution_id=execution_id)
        return True
    
    async def shutdown_all(self):
        """Gracefully shutdown all executions"""
        logger.info("Shutting down all background executions")
        
        for execution_id in list(self.background_tasks.keys()):
            self.cancel_execution(execution_id)
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks.values(), return_exceptions=True)

# Global instance
_background_manager = None

def get_background_manager() -> BackgroundExecutionManager:
    """Get the global background execution manager"""
    global _background_manager
    if _background_manager is None:
        _background_manager = BackgroundExecutionManager()
    return _background_manager