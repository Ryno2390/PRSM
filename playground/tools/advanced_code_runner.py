#!/usr/bin/env python3
"""
Advanced PRSM Code Runner with Execution-Guided Generation
Integrates EG-CFG methodology for enhanced code generation and execution.

This tool provides an interactive environment for testing the cutting-edge
Execution-Guided Classifier-Free Guidance approach in PRSM.
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from prsm.agents.executors.execution_guided_code_runner import (
    EGCFGAgent, 
    GenerationConfig, 
    ExecutionGuidedCodeRunner,
    ExecutionStatus
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedCodeRunnerTool:
    """Advanced code runner tool with EG-CFG integration"""
    
    def __init__(self):
        self.sessions = {}
        self.current_session_id = None
        self.session_counter = 0
        
        print("🚀 Advanced PRSM Code Runner with EG-CFG")
        print("Based on: Execution Guided Line-by-Line Code Generation (arXiv:2506.10948v1)")
        print("=" * 80)
    
    async def start_interactive_session(self):
        """Start interactive code generation session"""
        print("\n🎮 Interactive Mode - Advanced Code Generation")
        print("Commands:")
        print("  generate <prompt>           - Generate code with EG-CFG")
        print("  config <setting>=<value>    - Adjust generation settings")
        print("  analyze <code>              - Analyze existing code")
        print("  compare <prompt>            - Compare EG-CFG vs traditional")
        print("  benchmark <task>            - Run performance benchmarks")
        print("  session <new|load|save>     - Manage sessions")
        print("  stats                       - Show execution statistics")
        print("  help                        - Show this help")
        print("  quit                        - Exit")
        print("-" * 80)
        
        # Create default session
        await self._create_new_session()
        
        while True:
            try:
                command = input(f"\n🔧 runner[{self.current_session_id}]> ").strip()
                
                if not command:
                    continue
                
                parts = command.split(' ', 1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                if cmd == "quit" or cmd == "exit":
                    print("👋 Thanks for using the Advanced Code Runner!")
                    break
                elif cmd == "help":
                    await self._show_help()
                elif cmd == "generate":
                    await self._handle_generate(args)
                elif cmd == "config":
                    await self._handle_config(args)
                elif cmd == "analyze":
                    await self._handle_analyze(args)
                elif cmd == "compare":
                    await self._handle_compare(args)
                elif cmd == "benchmark":
                    await self._handle_benchmark(args)
                elif cmd == "session":
                    await self._handle_session(args)
                elif cmd == "stats":
                    await self._handle_stats()
                else:
                    print(f"❌ Unknown command: {cmd}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n🛑 Session interrupted. Type 'quit' to exit.")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                logger.error(f"Interactive session error: {str(e)}")
    
    async def _create_new_session(self, session_name: str = None):
        """Create a new code generation session"""
        self.session_counter += 1
        session_id = session_name or f"session_{self.session_counter}"
        
        config = GenerationConfig(
            candidates_per_line=3,
            temperatures=[0.7, 0.9, 1.2],
            enable_parallel_execution=True,
            beam_search_width=5
        )
        
        agent = EGCFGAgent(config)
        
        self.sessions[session_id] = {
            "agent": agent,
            "config": config,
            "history": [],
            "created_at": time.time()
        }
        
        self.current_session_id = session_id
        print(f"✅ Created new session: {session_id}")
    
    async def _handle_generate(self, prompt: str):
        """Handle code generation command"""
        if not prompt:
            print("❌ Please provide a generation prompt")
            return
        
        print(f"🔄 Generating code with EG-CFG methodology...")
        print(f"📝 Prompt: {prompt}")
        
        session = self.sessions[self.current_session_id]
        agent = session["agent"]
        
        # Prepare task
        task = {
            "prompt": prompt,
            "context": {
                "max_lines": 15,
                "enable_validation": True,
                "require_completion": True
            }
        }
        
        try:
            start_time = time.time()
            result = await agent.process_coding_task(task)
            execution_time = time.time() - start_time
            
            # Display results
            print(f"\n📊 Generation Results:")
            print(f"⏱️  Total Time: {execution_time:.2f}s")
            print(f"✅ Success: {result['success']}")
            print(f"🎯 Final Score: {result.get('final_score', 0):.2f}")
            print(f"🔄 Candidates Explored: {result.get('candidate_count', 0)}")
            
            print(f"\n📝 Generated Code:")
            print("-" * 40)
            print(result['generated_code'])
            print("-" * 40)
            
            # Show execution traces
            if result.get('execution_traces'):
                print(f"\n🔍 Execution Trace:")
                for i, trace in enumerate(result['execution_traces'][:5]):  # Show first 5
                    status_emoji = self._get_status_emoji(trace['status'])
                    print(f"  {i+1}. {status_emoji} Line {trace['line_number']}: {trace['status']}")
                    if trace.get('output'):
                        print(f"     💬 Output: {trace['output'][:50]}...")
                    if trace.get('error_message'):
                        print(f"     ❌ Error: {trace['error_message'][:50]}...")
            
            # Save to session history
            session["history"].append({
                "prompt": prompt,
                "result": result,
                "timestamp": time.time()
            })
            
        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")
            logger.error(f"Code generation error: {str(e)}")
    
    async def _handle_config(self, config_str: str):
        """Handle configuration changes"""
        if not config_str or '=' not in config_str:
            # Show current config
            session = self.sessions[self.current_session_id]
            config = session["config"]
            
            print(f"\n⚙️ Current Configuration:")
            print(f"  Candidates per line: {config.candidates_per_line}")
            print(f"  Temperatures: {config.temperatures}")
            print(f"  Completion horizons: {config.completion_horizons}")
            print(f"  CFG strengths: {config.cfg_strengths}")
            print(f"  Max execution time: {config.max_execution_time}s")
            print(f"  Parallel execution: {config.enable_parallel_execution}")
            print(f"  AST validation: {config.enable_ast_validation}")
            print(f"  Beam search width: {config.beam_search_width}")
            return
        
        try:
            setting, value = config_str.split('=', 1)
            setting = setting.strip()
            value = value.strip()
            
            session = self.sessions[self.current_session_id]
            config = session["config"]
            
            # Parse and apply configuration
            if setting == "candidates_per_line":
                config.candidates_per_line = int(value)
            elif setting == "temperatures":
                config.temperatures = [float(x.strip()) for x in value.split(',')]
            elif setting == "max_execution_time":
                config.max_execution_time = float(value)
            elif setting == "parallel_execution":
                config.enable_parallel_execution = value.lower() in ['true', '1', 'yes']
            elif setting == "beam_search_width":
                config.beam_search_width = int(value)
            else:
                print(f"❌ Unknown setting: {setting}")
                return
            
            print(f"✅ Updated {setting} = {value}")
            
            # Recreate agent with new config
            session["agent"] = EGCFGAgent(config)
            
        except Exception as e:
            print(f"❌ Configuration error: {str(e)}")
    
    async def _handle_analyze(self, code: str):
        """Handle code analysis"""
        if not code:
            print("❌ Please provide code to analyze")
            return
        
        print(f"🔍 Analyzing code with EG-CFG insights...")
        
        session = self.sessions[self.current_session_id]
        runner = session["agent"].code_runner
        
        # Create candidate for analysis
        from prsm.agents.executors.execution_guided_code_runner import CodeCandidate
        
        candidate = CodeCandidate(
            candidate_id="analysis",
            code_lines=code.split('\n'),
            temperature=1.0,
            execution_traces=[]
        )
        
        # Execute and analyze
        analyzed = runner._execute_single_candidate(candidate)
        
        print(f"\n📊 Analysis Results:")
        print(f"✅ Syntax Valid: {analyzed.syntax_valid}")
        print(f"🎯 Quality Score: {analyzed.total_score:.2f}")
        print(f"📏 Lines of Code: {len(analyzed.code_lines)}")
        
        # Show execution analysis
        success_count = sum(1 for t in analyzed.execution_traces if t.status == ExecutionStatus.SUCCESS)
        error_count = sum(1 for t in analyzed.execution_traces if t.status == ExecutionStatus.ERROR)
        
        print(f"✅ Successful Lines: {success_count}")
        print(f"❌ Error Lines: {error_count}")
        
        if analyzed.execution_traces:
            avg_exec_time = sum(t.execution_time for t in analyzed.execution_traces) / len(analyzed.execution_traces)
            print(f"⏱️  Average Execution Time: {avg_exec_time:.3f}s")
    
    async def _handle_compare(self, prompt: str):
        """Compare EG-CFG vs traditional approaches"""
        if not prompt:
            print("❌ Please provide a prompt for comparison")
            return
        
        print(f"⚖️ Comparing generation approaches...")
        print(f"📝 Prompt: {prompt}")
        
        session = self.sessions[self.current_session_id]
        
        # EG-CFG approach
        print(f"\n🔬 EG-CFG Approach:")
        egcfg_start = time.time()
        egcfg_task = {
            "prompt": prompt,
            "context": {"max_lines": 10, "methodology": "EG-CFG"}
        }
        egcfg_result = await session["agent"].process_coding_task(egcfg_task)
        egcfg_time = time.time() - egcfg_start
        
        print(f"  ⏱️ Time: {egcfg_time:.2f}s")
        print(f"  🎯 Score: {egcfg_result.get('final_score', 0):.2f}")
        print(f"  🔄 Candidates: {egcfg_result.get('candidate_count', 0)}")
        
        # Mock traditional approach
        print(f"\n📝 Traditional Approach (simulated):")
        traditional_start = time.time()
        await asyncio.sleep(0.5)  # Simulate generation time
        traditional_time = time.time() - traditional_start
        
        print(f"  ⏱️ Time: {traditional_time:.2f}s")
        print(f"  🎯 Score: 2.5 (estimated)")
        print(f"  🔄 Candidates: 1")
        
        # Comparison
        print(f"\n📊 Comparison:")
        speedup = traditional_time / egcfg_time if egcfg_time > 0 else 1
        quality_improvement = (egcfg_result.get('final_score', 0) - 2.5) / 2.5 * 100
        
        print(f"  ⚡ Speed: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
        print(f"  📈 Quality: {quality_improvement:+.1f}% vs traditional")
        print(f"  🎯 Methodology: Execution-guided with real-time feedback")
    
    async def _handle_benchmark(self, task_type: str):
        """Run performance benchmarks"""
        print(f"🏁 Running benchmark: {task_type or 'general'}")
        
        benchmarks = {
            "simple": "Create a function that adds two numbers",
            "algorithm": "Implement binary search algorithm",
            "data_structure": "Create a stack data structure",
            "complex": "Build a simple calculator with error handling"
        }
        
        task_type = task_type or "simple"
        if task_type not in benchmarks:
            print(f"❌ Unknown benchmark: {task_type}")
            print(f"Available: {', '.join(benchmarks.keys())}")
            return
        
        prompt = benchmarks[task_type]
        session = self.sessions[self.current_session_id]
        
        print(f"📝 Benchmark Task: {prompt}")
        print(f"🔄 Running with EG-CFG...")
        
        start_time = time.time()
        result = await session["agent"].process_coding_task({
            "prompt": prompt,
            "context": {"max_lines": 20, "benchmark": True}
        })
        total_time = time.time() - start_time
        
        print(f"\n📊 Benchmark Results:")
        print(f"  ⏱️ Execution Time: {total_time:.2f}s")
        print(f"  ✅ Success: {result['success']}")
        print(f"  🎯 Quality Score: {result.get('final_score', 0):.2f}")
        print(f"  📏 Code Lines: {len(result['generated_code'].split())}")
        print(f"  🔄 Exploration: {result.get('candidate_count', 0)} candidates")
        
        # Performance rating
        if result['success'] and result.get('final_score', 0) > 3.0:
            rating = "🌟 Excellent"
        elif result['success'] and result.get('final_score', 0) > 2.0:
            rating = "✅ Good"
        elif result['success']:
            rating = "⚠️ Fair"
        else:
            rating = "❌ Poor"
        
        print(f"  📈 Rating: {rating}")
    
    async def _handle_session(self, action: str):
        """Handle session management"""
        if action == "new":
            session_name = input("Session name (optional): ").strip() or None
            await self._create_new_session(session_name)
        elif action == "list":
            print(f"\n📋 Available Sessions:")
            for sid, session in self.sessions.items():
                active = "🟢" if sid == self.current_session_id else "⚪"
                created = time.strftime("%H:%M:%S", time.localtime(session["created_at"]))
                history_count = len(session["history"])
                print(f"  {active} {sid} (created: {created}, history: {history_count})")
        elif action.startswith("switch"):
            parts = action.split(' ', 1)
            if len(parts) > 1:
                session_id = parts[1]
                if session_id in self.sessions:
                    self.current_session_id = session_id
                    print(f"✅ Switched to session: {session_id}")
                else:
                    print(f"❌ Session not found: {session_id}")
        else:
            print("Session commands: new, list, switch <name>")
    
    async def _handle_stats(self):
        """Show execution statistics"""
        session = self.sessions[self.current_session_id]
        agent = session["agent"]
        
        stats = agent.get_agent_status()
        
        print(f"\n📊 Session Statistics:")
        print(f"  Agent ID: {stats['agent_id']}")
        print(f"  Status: {stats['status']}")
        print(f"  Methodology: {stats['methodology']}")
        print(f"  Research Base: {stats['research_base']}")
        
        exec_stats = stats.get('execution_stats', {})
        if 'total_candidates' in exec_stats:
            print(f"\n🔄 Execution Statistics:")
            print(f"  Total Candidates: {exec_stats['total_candidates']}")
            print(f"  Successful: {exec_stats['successful_candidates']}")
            print(f"  Success Rate: {exec_stats['success_rate']*100:.1f}%")
            print(f"  Average Score: {exec_stats['average_score']:.2f}")
            print(f"  Average Time: {exec_stats['average_execution_time']:.3f}s")
        
        # Session history
        history = session["history"]
        if history:
            print(f"\n📚 Session History ({len(history)} generations):")
            for i, entry in enumerate(history[-3:], 1):  # Show last 3
                success = "✅" if entry["result"]["success"] else "❌"
                prompt_preview = entry["prompt"][:40] + "..." if len(entry["prompt"]) > 40 else entry["prompt"]
                print(f"  {i}. {success} {prompt_preview}")
    
    async def _show_help(self):
        """Show detailed help information"""
        help_text = """
🚀 Advanced PRSM Code Runner - Help

METHODOLOGY:
This tool implements Execution-Guided Classifier-Free Guidance (EG-CFG)
based on research paper "Execution Guided Line-by-Line Code Generation"
(arXiv:2506.10948v1)

KEY FEATURES:
• Line-by-line code execution during generation
• Real-time execution feedback integration
• Parallel candidate exploration
• Classifier-Free Guidance for optimal selection
• Advanced AST validation and error recovery

COMMANDS:

generate <prompt>
  Generate code using EG-CFG methodology
  Example: generate "Create a fibonacci function"

config [setting=value]
  View or change generation settings
  Settings: candidates_per_line, temperatures, max_execution_time,
           parallel_execution, beam_search_width
  Example: config candidates_per_line=5

analyze <code>
  Analyze existing code with EG-CFG insights
  Example: analyze "def hello(): print('world')"

compare <prompt>
  Compare EG-CFG vs traditional generation
  Example: compare "Sort a list of numbers"

benchmark <task>
  Run performance benchmarks
  Tasks: simple, algorithm, data_structure, complex
  Example: benchmark algorithm

session <action>
  Manage sessions (new, list, switch <name>)
  Example: session new

stats
  Show execution statistics and session history

RESEARCH BACKGROUND:
EG-CFG represents a breakthrough in code generation by incorporating
execution feedback directly into the generation process, achieving
SOTA results on coding benchmarks like MBPP (96.6%) and HumanEval-ET (87.19%).
        """
        print(help_text)
    
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for execution status"""
        status_map = {
            "success": "✅",
            "error": "❌", 
            "timeout": "⏱️",
            "syntax_error": "🔧",
            "incomplete": "⚠️"
        }
        return status_map.get(status, "❓")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Advanced PRSM Code Runner with EG-CFG")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    parser.add_argument("--prompt", help="Generate code for specific prompt")
    parser.add_argument("--benchmark", help="Run specific benchmark")
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    tool = AdvancedCodeRunnerTool()
    
    if args.interactive:
        await tool.start_interactive_session()
    elif args.prompt:
        await tool._create_new_session()
        await tool._handle_generate(args.prompt)
        if tool.sessions:
            await tool._handle_stats()
    elif args.benchmark:
        await tool._create_new_session()
        await tool._handle_benchmark(args.benchmark)
    else:
        print("🚀 Advanced PRSM Code Runner")
        print("Use --interactive for interactive mode")
        print("Use --prompt 'your prompt' for single generation")
        print("Use --benchmark <task> for benchmarking")
        print("Use --help for full options")

if __name__ == "__main__":
    asyncio.run(main())