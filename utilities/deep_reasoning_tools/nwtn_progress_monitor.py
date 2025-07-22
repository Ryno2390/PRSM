#!/usr/bin/env python3
"""
NWTN Progress Monitor
====================

Real-time monitoring utility for NWTN deep reasoning background executions.
Provides live progress updates, system metrics, and FTNS payment tracking.

Usage:
    python nwtn_progress_monitor.py [execution_id]
    python nwtn_progress_monitor.py --list
    python nwtn_progress_monitor.py --start "query" [--mode DEEP]
"""

import asyncio
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add PRSM to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from prsm.nwtn.background_execution_manager import get_background_manager, ExecutionStatus, ThinkingMode
from prsm.nwtn.meta_reasoning_engine import ThinkingMode

class ProgressMonitor:
    """Real-time progress monitor for NWTN executions"""
    
    def __init__(self):
        self.manager = get_background_manager()
        self.monitoring = False
    
    async def start_execution_async(self, query: str, mode: ThinkingMode = ThinkingMode.DEEP):
        """Start a new background execution (async version)"""
        print(f"🚀 Starting NWTN {mode.value.upper()} reasoning...")
        print(f"📝 Query: {query}")
        print("=" * 70)
        
        execution_id = self.manager.start_background_execution(
            query=query,
            thinking_mode=mode,
            context={'user_id': 'progress_monitor', 'budget_ftns': 1000.0},
            progress_callback=self._progress_callback
        )
        
        print(f"✅ Execution started with ID: {execution_id}")
        print(f"📊 Monitor with: python {__file__} {execution_id}")
        print("🔄 Starting real-time monitoring...")
        print("=" * 70)
        
        return execution_id
    
    def start_execution(self, query: str, mode: ThinkingMode = ThinkingMode.DEEP):
        """Start a new background execution (sync wrapper)"""
        return asyncio.run(self.start_execution_async(query, mode))
    
    def monitor_execution(self, execution_id: str, refresh_interval: float = 10.0):
        """Monitor an execution in real-time"""
        self.monitoring = True
        
        print(f"📊 Monitoring NWTN execution: {execution_id}")
        print("Press Ctrl+C to stop monitoring (execution continues)")
        print("=" * 70)
        
        try:
            while self.monitoring:
                progress = self.manager.get_execution_status(execution_id)
                
                if not progress:
                    print(f"❌ Execution {execution_id} not found")
                    break
                
                self._display_progress(progress)
                
                if progress.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
                    break
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print("\n👋 Stopped monitoring (execution continues in background)")
            self.monitoring = False
    
    def _display_progress(self, progress):
        """Display formatted progress information"""
        # Clear screen and reset cursor
        print("\033[H\033[J", end="")
        
        # Header
        print("🧠 NWTN DEEP REASONING - REAL-TIME PROGRESS")
        print("=" * 70)
        print(f"🆔 Execution ID: {progress.execution_id}")
        print(f"📝 Query: {progress.query[:60]}{'...' if len(progress.query) > 60 else ''}")
        print(f"⏰ Started: {progress.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"🔄 Status: {progress.status.value.upper()}")
        
        # Progress metrics
        print("\n📊 REASONING PROGRESS")
        print("-" * 30)
        completion_pct = (progress.completed_permutations / progress.total_permutations) * 100
        print(f"🎯 Permutations: {progress.completed_permutations:,} / {progress.total_permutations:,} ({completion_pct:.1f}%)")
        
        # Progress bar
        bar_width = 50
        filled = int(bar_width * completion_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"📈 Progress: [{bar}] {completion_pct:.1f}%")
        
        if progress.current_sequence:
            print(f"🔍 Current: {progress.current_sequence}")
        
        # Performance metrics
        print("\n⚡ PERFORMANCE METRICS")
        print("-" * 30)
        print(f"⏱️  Elapsed: {progress.elapsed_hours:.2f} hours")
        
        if progress.permutations_per_minute > 0:
            print(f"🏃 Speed: {progress.permutations_per_minute:.1f} permutations/min")
            print(f"⏳ Remaining: {progress.estimated_hours_remaining:.2f} hours")
            
            if progress.estimated_completion:
                eta = progress.estimated_completion.strftime('%H:%M:%S UTC')
                print(f"🎯 ETA: {eta}")
        
        # System metrics
        if progress.memory_usage_mb > 0:
            print(f"💾 Memory: {progress.memory_usage_mb:.0f} MB")
        if progress.cpu_usage_percent > 0:
            print(f"🖥️  CPU: {progress.cpu_usage_percent:.1f}%")
        
        # FTNS metrics
        print("\n💰 FTNS PAYMENT TRACKING")
        print("-" * 30)
        print(f"🪙 Tokens Distributed: {progress.ftns_tokens_distributed:.2f}")
        print(f"💳 Payment Events: {progress.ftns_payments_count}")
        
        # Results (if available)
        if progress.status == ExecutionStatus.COMPLETED:
            print("\n🎉 COMPLETION RESULTS")
            print("-" * 30)
            print(f"🧠 Confidence: {progress.confidence_score:.3f}")
            print(f"📄 Papers Analyzed: {progress.papers_analyzed}")
            print(f"📝 Answer Length: {progress.final_answer_length:,} chars")
            print(f"📚 Citations: {progress.citations_count}")
        
        # Error tracking
        if progress.error_count > 0:
            print(f"\n⚠️  Errors: {progress.error_count} (recovered: {progress.recovery_count})")
            if progress.last_error:
                print(f"❌ Last Error: {progress.last_error}")
        
        print(f"\n🔄 Last Updated: {progress.updated_at.strftime('%H:%M:%S UTC')}")
        print("=" * 70)
    
    def _progress_callback(self, progress):
        """Callback for progress updates (used during execution start)"""
        if progress.completed_permutations % 500 == 0:  # Update every 500 sequences
            completion = (progress.completed_permutations / progress.total_permutations) * 100
            print(f"🔄 Progress: {progress.completed_permutations:,}/5,040 ({completion:.1f}%) - {progress.current_sequence}")
    
    def list_executions(self):
        """List all active executions"""
        executions = self.manager.list_active_executions()
        
        if not executions:
            print("No active NWTN executions found")
            return
        
        print("🧠 ACTIVE NWTN EXECUTIONS")
        print("=" * 70)
        
        for exec_id, progress in executions.items():
            status_icon = {
                ExecutionStatus.INITIALIZING: "🔄",
                ExecutionStatus.RUNNING: "⚡",
                ExecutionStatus.COMPLETED: "✅",
                ExecutionStatus.FAILED: "❌",
                ExecutionStatus.CANCELLED: "🛑"
            }.get(progress.status, "❓")
            
            completion = (progress.completed_permutations / progress.total_permutations) * 100
            
            print(f"{status_icon} {exec_id}")
            print(f"   Query: {progress.query[:50]}...")
            print(f"   Status: {progress.status.value} ({completion:.1f}% complete)")
            print(f"   Started: {progress.started_at.strftime('%Y-%m-%d %H:%M UTC')}")
            print(f"   Duration: {progress.elapsed_hours:.2f} hours")
            print()

def main():
    parser = argparse.ArgumentParser(description="NWTN Progress Monitor")
    parser.add_argument("execution_id", nargs="?", help="Execution ID to monitor")
    parser.add_argument("--list", action="store_true", help="List all active executions")
    parser.add_argument("--start", help="Start new execution with this query")
    parser.add_argument("--mode", choices=["QUICK", "INTERMEDIATE", "DEEP"], 
                       default="DEEP", help="Thinking mode for new execution")
    parser.add_argument("--refresh", type=float, default=10.0, 
                       help="Refresh interval in seconds (default: 10)")
    
    args = parser.parse_args()
    
    monitor = ProgressMonitor()
    
    if args.list:
        monitor.list_executions()
    elif args.start:
        mode = ThinkingMode(args.mode.lower())
        execution_id = monitor.start_execution(args.start, mode)
        # Start monitoring the new execution
        monitor.monitor_execution(execution_id, args.refresh)
    elif args.execution_id:
        monitor.monitor_execution(args.execution_id, args.refresh)
    else:
        print("Usage:")
        print("  Monitor execution:  python nwtn_progress_monitor.py <execution_id>")
        print("  List executions:    python nwtn_progress_monitor.py --list")
        print('  Start & monitor:    python nwtn_progress_monitor.py --start "your query"')

if __name__ == "__main__":
    main()