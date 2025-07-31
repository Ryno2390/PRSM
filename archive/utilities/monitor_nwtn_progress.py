#!/usr/bin/env python3
"""
NWTN Deep Reasoning Progress Monitor
=====================================

Monitors the progress of NWTN deep reasoning pipeline execution in real-time.
Shows current phase, step progress, and estimated completion time.
"""

import time
import os
import re
from datetime import datetime

def monitor_nwtn_progress():
    """Monitor NWTN pipeline progress from logs with detailed phase tracking"""
    
    print("🔍 NWTN Deep Reasoning Progress Monitor")
    print("=" * 50)
    print(f"Started monitoring at: {datetime.now().strftime('%H:%M:%S')}")
    print()
    
    # Find the most recent log file
    log_files = ["nwtn_progress_demo.log", "nwtn_deep_reasoning_complete.log", "nwtn_breakthrough_final.log"]
    current_log = None
    for log_file in log_files:
        if os.path.exists(log_file):
            current_log = log_file
            break
    
    if not current_log:
        print("❌ No NWTN log files found")
        return
        
    print(f"📋 Monitoring log file: {current_log}")
    print()
    
    # Progress tracking with detailed phases
    phases_completed = {
        "orchestrator_init": False,
        "session_created": False,
        "phase_1_started": False, 
        "semantic_retrieval": False,
        "content_analysis": False,
        "candidate_generation": False,
        "phase_2_validation": False,
        "claude_synthesis": False,
        "pipeline_complete": False
    }
    
    start_time = time.time()
    last_position = 0
    
    # Monitoring loop with comprehensive progress tracking
    try:
        while True:
            # Check if process is still running
            result = os.popen("ps aux | grep 'python.*test_revolutionary' | grep -v grep").read()
            
            elapsed = time.time() - start_time
            elapsed_mins = int(elapsed // 60)
            elapsed_secs = int(elapsed % 60)
            
            # Read new log entries
            try:
                with open(current_log, 'r') as f:
                    f.seek(last_position)
                    new_lines = f.readlines()
                    last_position = f.tell()
                    
                    # Process new log lines for progress indicators
                    for line in new_lines:
                        if "Enhanced NWTN Orchestrator initialized" in line:
                            phases_completed["orchestrator_init"] = True
                            print("✅ NWTN Orchestrator initialized")
                            
                        elif "Session created" in line:
                            phases_completed["session_created"] = True  
                            print("✅ Session created with budget allocation")
                            
                        elif "Phase 1: System 1 Creative Generation" in line:
                            phases_completed["phase_1_started"] = True
                            print("🧠 Phase 1: System 1 Creative Generation STARTED")
                            
                        elif "Step 1.1: Semantic retrieval from 100K arXiv papers" in line:
                            print("🔍 Step 1.1: Semantic retrieval from 100K arXiv papers...")
                            
                        elif "Semantic retrieval completed" in line:
                            phases_completed["semantic_retrieval"] = True
                            # Extract papers found
                            if "papers_found=" in line:
                                papers = line.split("papers_found=")[1].split()[0]
                                print(f"✅ Semantic retrieval completed - {papers} papers found")
                            else:
                                print("✅ Semantic retrieval completed")
                                
                        elif "Step 1.2: Content analysis of retrieved papers" in line:
                            print("📊 Step 1.2: Content analysis of retrieved papers...")
                            
                        elif "Content analysis completed" in line:
                            phases_completed["content_analysis"] = True
                            # Extract concepts
                            if "concepts_extracted=" in line:
                                concepts = line.split("concepts_extracted=")[1].split()[0]
                                print(f"✅ Content analysis completed - {concepts} concepts extracted")
                            else:
                                print("✅ Content analysis completed")
                                
                        elif "Step 1.3: Candidate Generation" in line:
                            print("🎯 Step 1.3: Candidate generation using 7 reasoning engines...")
                            
                        elif "Executing agent" in line and "executor" in line:
                            if not phases_completed["claude_synthesis"]:
                                print("🤖 Claude API synthesis in progress...")
                                
                        elif "Model execution completed successfully" in line:
                            phases_completed["claude_synthesis"] = True
                            print("✅ Claude API synthesis completed")
                            
                        elif "REVOLUTIONARY BREAKTHROUGH ANALYSIS COMPLETE" in line:
                            phases_completed["pipeline_complete"] = True
                            print("🎉 NWTN Deep Reasoning Pipeline COMPLETE!")
            
            except FileNotFoundError:
                pass
            
            # Status display
            print(f"\r⏱️  Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d} | ", end="")
            
            if result.strip():
                # Extract CPU and memory usage
                process_line = result.strip().split('\n')[0]
                parts = process_line.split()
                if len(parts) >= 4:
                    cpu = parts[2]
                    memory = parts[3]
                    print(f"🔄 RUNNING (CPU: {cpu}%, MEM: {memory}%) | ", end="")
                else:
                    print("🔄 RUNNING | ", end="")
                    
                # Show current phase
                if phases_completed["pipeline_complete"]:
                    print("✅ COMPLETE", end="")
                elif phases_completed["claude_synthesis"]:
                    print("🤖 Final synthesis", end="")
                elif phases_completed["content_analysis"]:
                    print("🎯 Candidate generation", end="")
                elif phases_completed["semantic_retrieval"]:
                    print("📊 Content analysis", end="")
                elif phases_completed["phase_1_started"]:
                    print("🔍 Semantic retrieval", end="")
                elif phases_completed["session_created"]:
                    print("🧠 Starting Phase 1", end="")
                elif phases_completed["orchestrator_init"]:
                    print("⚙️  Session setup", end="")
                else:
                    print("🚀 Initializing", end="")
            else:
                if phases_completed["pipeline_complete"]:
                    print("✅ COMPLETED SUCCESSFULLY", end="")
                    break
                else:
                    print("❌ PROCESS STOPPED", end="")
                    break
                    
            print(" " * 10, end="")  # Clear rest of line
            time.sleep(2)  # Check every 2 seconds for more responsive monitoring
            
    except KeyboardInterrupt:
        print(f"\n\n🛑 Monitoring stopped at {datetime.now().strftime('%H:%M:%S')}")
        print(f"Total monitoring time: {elapsed_mins:02d}:{elapsed_secs:02d}")
        
        # Show final progress summary
        print("\n📊 Final Progress Summary:")
        for phase, completed in phases_completed.items():
            status = "✅" if completed else "⏸️"
            print(f"   {status} {phase.replace('_', ' ').title()}")
            
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")
        import traceback
        traceback.print_exc()

def check_current_status():
    """Check current NWTN pipeline status"""
    print("📊 Current NWTN Pipeline Status:")
    print("-" * 30)
    
    # Check for running processes
    result = os.popen("ps aux | grep 'python.*test_revolutionary' | grep -v grep").read()
    
    if result.strip():
        print("✅ NWTN Deep Reasoning Pipeline: RUNNING")
        
        # Try to get process details
        lines = result.strip().split('\n')
        for line in lines:
            parts = line.split()
            if len(parts) >= 11:
                cpu_usage = parts[2]
                memory_usage = parts[3]
                start_time = ' '.join(parts[8:10])
                print(f"   CPU: {cpu_usage}% | Memory: {memory_usage}% | Started: {start_time}")
    else:
        print("❌ NWTN Deep Reasoning Pipeline: NOT RUNNING")
        print("\nTo start the deep reasoning test:")
        print("   python test_revolutionary_breakthrough.py &")
    
    print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        check_current_status()
    else:
        monitor_nwtn_progress()