#!/usr/bin/env python3
"""
PRSM Developer Playground Launcher
Interactive environment for exploring PRSM capabilities with examples,
tutorials, and development tools.
"""

import os
import sys
import asyncio
import argparse
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import webbrowser
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExampleMetadata:
    """Metadata for playground examples"""
    name: str
    description: str
    category: str
    difficulty: str  # "beginner", "intermediate", "advanced"
    duration: int    # estimated minutes
    prerequisites: List[str]
    tags: List[str]
    file_path: str

@dataclass
class TutorialMetadata:
    """Metadata for tutorials"""
    name: str
    description: str
    learning_objectives: List[str]
    duration: int
    prerequisites: List[str]
    steps: List[str]
    directory: str

class PlaygroundLauncher:
    """Main playground launcher and interface"""
    
    def __init__(self):
        self.playground_dir = Path(__file__).parent
        self.examples_dir = self.playground_dir / "examples"
        self.tutorials_dir = self.playground_dir / "tutorials"
        self.tools_dir = self.playground_dir / "tools"
        self.templates_dir = self.playground_dir / "templates"
        
        self.examples: Dict[str, ExampleMetadata] = {}
        self.tutorials: Dict[str, TutorialMetadata] = {}
        
        self._load_examples()
        self._load_tutorials()
    
    def _load_examples(self):
        """Load example metadata"""
        # Basic examples
        self.examples.update({
            "basic/hello_prsm": ExampleMetadata(
                name="Hello PRSM",
                description="Your first PRSM program demonstrating basic setup and API usage",
                category="basic",
                difficulty="beginner",
                duration=5,
                prerequisites=[],
                tags=["getting-started", "api", "basic"],
                file_path="examples/basic/hello_prsm.py"
            ),
            "basic/simple_agent": ExampleMetadata(
                name="Simple AI Agent",
                description="Create and run a basic AI agent with PRSM",
                category="basic",
                difficulty="beginner",
                duration=10,
                prerequisites=["basic/hello_prsm"],
                tags=["agent", "ai", "basic"],
                file_path="examples/basic/simple_agent.py"
            ),
            "basic/api_integration": ExampleMetadata(
                name="API Integration",
                description="Integrate with PRSM APIs for various operations",
                category="basic",
                difficulty="beginner",
                duration=15,
                prerequisites=["basic/hello_prsm"],
                tags=["api", "integration", "rest"],
                file_path="examples/basic/api_integration.py"
            ),
            "ai_models/model_loading": ExampleMetadata(
                name="AI Model Loading",
                description="Load and manage different types of AI models",
                category="ai_models",
                difficulty="intermediate",
                duration=20,
                prerequisites=["basic/simple_agent"],
                tags=["ai", "models", "pytorch", "tensorflow"],
                file_path="examples/ai_models/model_loading.py"
            ),
            "ai_models/distributed_inference": ExampleMetadata(
                name="Distributed AI Inference",
                description="Run AI model inference across distributed P2P networks",
                category="ai_models",
                difficulty="advanced",
                duration=30,
                prerequisites=["ai_models/model_loading", "p2p_network/basic_network"],
                tags=["ai", "distributed", "p2p", "inference"],
                file_path="examples/ai_models/distributed_inference.py"
            ),
            "p2p_network/basic_network": ExampleMetadata(
                name="Basic P2P Network",
                description="Set up and run a basic peer-to-peer network",
                category="p2p_network",
                difficulty="intermediate",
                duration=25,
                prerequisites=["basic/simple_agent"],
                tags=["p2p", "network", "distributed"],
                file_path="examples/p2p_network/basic_network.py"
            ),
            "orchestration/multi_agent": ExampleMetadata(
                name="Multi-Agent Orchestration",
                description="Coordinate multiple AI agents for complex tasks",
                category="orchestration",
                difficulty="advanced",
                duration=45,
                prerequisites=["ai_models/model_loading", "p2p_network/basic_network"],
                tags=["orchestration", "multi-agent", "coordination"],
                file_path="examples/orchestration/multi_agent.py"
            ),
            "monitoring/dashboard_integration": ExampleMetadata(
                name="Monitoring Dashboard",
                description="Integrate with PRSM's real-time monitoring dashboard",
                category="monitoring",
                difficulty="intermediate",
                duration=20,
                prerequisites=["basic/simple_agent"],
                tags=["monitoring", "dashboard", "metrics"],
                file_path="examples/monitoring/dashboard_integration.py"
            ),
            "enterprise/security_example": ExampleMetadata(
                name="Enterprise Security",
                description="Implement enterprise-grade security features",
                category="enterprise",
                difficulty="advanced",
                duration=35,
                prerequisites=["basic/api_integration"],
                tags=["security", "enterprise", "authentication"],
                file_path="examples/enterprise/security_example.py"
            )
        })
    
    def _load_tutorials(self):
        """Load tutorial metadata"""
        self.tutorials.update({
            "getting-started": TutorialMetadata(
                name="Getting Started with PRSM",
                description="Complete introduction to PRSM concepts and basic usage",
                learning_objectives=[
                    "Understand PRSM architecture and core concepts",
                    "Set up development environment",
                    "Create your first AI agent",
                    "Understand P2P network basics"
                ],
                duration=30,
                prerequisites=[],
                steps=[
                    "Environment setup and installation",
                    "PRSM concepts overview",
                    "Creating your first agent",
                    "Basic API usage",
                    "Next steps and resources"
                ],
                directory="tutorials/01-getting-started"
            ),
            "first-agent": TutorialMetadata(
                name="Building Your First AI Agent",
                description="Step-by-step guide to creating and deploying an AI agent",
                learning_objectives=[
                    "Design agent architecture",
                    "Implement agent logic",
                    "Test and debug agents",
                    "Deploy to P2P network"
                ],
                duration=45,
                prerequisites=["getting-started"],
                steps=[
                    "Agent design principles",
                    "Implementation walkthrough",
                    "Testing and validation",
                    "Deployment and monitoring",
                    "Scaling and optimization"
                ],
                directory="tutorials/02-first-agent"
            ),
            "distributed-ai": TutorialMetadata(
                name="Distributed AI with P2P Networks",
                description="Learn to build distributed AI systems using PRSM",
                learning_objectives=[
                    "Understand distributed AI concepts",
                    "Set up P2P networks",
                    "Implement distributed inference",
                    "Handle network failures"
                ],
                duration=60,
                prerequisites=["first-agent"],
                steps=[
                    "P2P network fundamentals",
                    "Distributed model serving",
                    "Consensus and coordination",
                    "Fault tolerance",
                    "Performance optimization"
                ],
                directory="tutorials/03-distributed-ai"
            ),
            "orchestration": TutorialMetadata(
                name="Advanced Agent Orchestration",
                description="Master complex multi-agent workflows and coordination",
                learning_objectives=[
                    "Design multi-agent systems",
                    "Implement coordination protocols",
                    "Handle complex workflows",
                    "Monitor system performance"
                ],
                duration=75,
                prerequisites=["distributed-ai"],
                steps=[
                    "Multi-agent architecture",
                    "Coordination patterns",
                    "Workflow management",
                    "Performance monitoring",
                    "Production deployment"
                ],
                directory="tutorials/04-orchestration"
            ),
            "production": TutorialMetadata(
                name="Production Deployment",
                description="Deploy PRSM applications to production environments",
                learning_objectives=[
                    "Production architecture design",
                    "Security implementation",
                    "Monitoring and alerting",
                    "Scaling strategies"
                ],
                duration=90,
                prerequisites=["orchestration"],
                steps=[
                    "Production planning",
                    "Security hardening",
                    "Deployment automation",
                    "Monitoring setup",
                    "Maintenance procedures"
                ],
                directory="tutorials/05-production"
            )
        })
    
    def list_examples(self, category: Optional[str] = None, difficulty: Optional[str] = None):
        """List available examples with filtering"""
        print("🎮 Available Examples")
        print("=" * 50)
        
        filtered_examples = self.examples.items()
        
        if category:
            filtered_examples = [(k, v) for k, v in filtered_examples if v.category == category]
        
        if difficulty:
            filtered_examples = [(k, v) for k, v in filtered_examples if v.difficulty == difficulty]
        
        # Group by category
        by_category = {}
        for key, example in filtered_examples:
            if example.category not in by_category:
                by_category[example.category] = []
            by_category[example.category].append((key, example))
        
        for cat, examples in sorted(by_category.items()):
            print(f"\n📂 {cat.replace('_', ' ').title()}")
            print("-" * 30)
            
            for key, example in examples:
                duration_str = f"{example.duration}min"
                difficulty_emoji = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}
                diff_emoji = difficulty_emoji.get(example.difficulty, "⚪")
                
                print(f"{diff_emoji} {key}")
                print(f"   {example.description}")
                print(f"   Duration: {duration_str} | Prerequisites: {', '.join(example.prerequisites) or 'None'}")
                print(f"   Tags: {', '.join(example.tags)}")
                print()
    
    def list_tutorials(self):
        """List available tutorials"""
        print("📚 Available Tutorials")
        print("=" * 50)
        
        for key, tutorial in self.tutorials.items():
            duration_str = f"{tutorial.duration}min"
            print(f"🎓 {key}")
            print(f"   {tutorial.description}")
            print(f"   Duration: {duration_str} | Prerequisites: {', '.join(tutorial.prerequisites) or 'None'}")
            print(f"   Learning Objectives:")
            for obj in tutorial.learning_objectives:
                print(f"     • {obj}")
            print()
    
    async def run_example(self, example_key: str, debug: bool = False):
        """Run a specific example"""
        if example_key not in self.examples:
            print(f"❌ Example '{example_key}' not found")
            self.list_examples()
            return False
        
        example = self.examples[example_key]
        example_path = self.playground_dir / example.file_path
        
        if not example_path.exists():
            print(f"❌ Example file not found: {example_path}")
            return False
        
        print(f"🚀 Running Example: {example.name}")
        print(f"📝 Description: {example.description}")
        print(f"⏱️  Estimated Duration: {example.duration} minutes")
        print(f"🎯 Difficulty: {example.difficulty}")
        print("=" * 60)
        
        # Check prerequisites
        if example.prerequisites:
            print(f"📋 Prerequisites: {', '.join(example.prerequisites)}")
            response = input("Have you completed the prerequisites? (y/N): ")
            if response.lower() != 'y':
                print("💡 Please complete prerequisites first")
                return False
        
        try:
            # Run the example
            env = os.environ.copy()
            env['PYTHONPATH'] = str(project_root)
            
            cmd = [sys.executable, str(example_path)]
            if debug:
                cmd.extend(['--debug'])
            
            print(f"🏃 Executing: {' '.join(cmd)}")
            print("-" * 60)
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            # Stream output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                print(line.decode().rstrip())
            
            await process.wait()
            
            print("-" * 60)
            if process.returncode == 0:
                print("✅ Example completed successfully!")
                print(f"💡 Next steps: Try related examples or tutorials")
                self._suggest_next_steps(example_key)
            else:
                print(f"❌ Example failed with return code: {process.returncode}")
            
            return process.returncode == 0
            
        except Exception as e:
            print(f"❌ Error running example: {str(e)}")
            return False
    
    def _suggest_next_steps(self, completed_example: str):
        """Suggest next examples or tutorials"""
        print("\n🎯 Suggested Next Steps:")
        
        # Find examples that have this as a prerequisite
        next_examples = []
        for key, example in self.examples.items():
            if completed_example in example.prerequisites:
                next_examples.append((key, example))
        
        if next_examples:
            print("📚 Examples you can now try:")
            for key, example in next_examples[:3]:  # Show max 3
                print(f"   • {key}: {example.description}")
        
        # Suggest tutorials
        completed_category = self.examples[completed_example].category
        relevant_tutorials = []
        
        for key, tutorial in self.tutorials.items():
            if completed_example in tutorial.prerequisites or completed_category in tutorial.name.lower():
                relevant_tutorials.append((key, tutorial))
        
        if relevant_tutorials:
            print("🎓 Relevant tutorials:")
            for key, tutorial in relevant_tutorials[:2]:  # Show max 2
                print(f"   • {key}: {tutorial.description}")
    
    async def run_tutorial(self, tutorial_key: str):
        """Run an interactive tutorial"""
        if tutorial_key not in self.tutorials:
            print(f"❌ Tutorial '{tutorial_key}' not found")
            self.list_tutorials()
            return False
        
        tutorial = self.tutorials[tutorial_key]
        tutorial_path = self.playground_dir / tutorial.directory
        
        print(f"🎓 Starting Tutorial: {tutorial.name}")
        print(f"📝 Description: {tutorial.description}")
        print(f"⏱️  Estimated Duration: {tutorial.duration} minutes")
        print("=" * 60)
        
        print("🎯 Learning Objectives:")
        for obj in tutorial.learning_objectives:
            print(f"   • {obj}")
        
        # Check prerequisites
        if tutorial.prerequisites:
            print(f"\n📋 Prerequisites: {', '.join(tutorial.prerequisites)}")
            response = input("Have you completed the prerequisites? (y/N): ")
            if response.lower() != 'y':
                print("💡 Please complete prerequisites first")
                return False
        
        print(f"\n📚 Tutorial Steps:")
        for i, step in enumerate(tutorial.steps, 1):
            print(f"   {i}. {step}")
        
        input("\nPress Enter to begin tutorial...")
        
        # Check if tutorial files exist
        if tutorial_path.exists():
            readme_path = tutorial_path / "README.md"
            if readme_path.exists():
                print(f"\n📖 Opening tutorial content...")
                # In a real implementation, this could open an interactive tutorial
                print(f"Tutorial content available at: {readme_path}")
                
                # Simulate interactive tutorial
                for i, step in enumerate(tutorial.steps, 1):
                    print(f"\n📝 Step {i}: {step}")
                    response = input("Complete this step and press Enter to continue (or 'q' to quit): ")
                    if response.lower() == 'q':
                        print("Tutorial paused. You can resume anytime.")
                        return False
                
                print("\n🎉 Tutorial completed successfully!")
                print("💡 Consider trying related examples or the next tutorial")
                return True
            else:
                print(f"❌ Tutorial content not found at: {readme_path}")
                return False
        else:
            print(f"❌ Tutorial directory not found: {tutorial_path}")
            return False
    
    def generate_template(self, template_name: str, output_dir: str):
        """Generate a project template"""
        templates = {
            "basic_agent": "Basic AI Agent Template",
            "p2p_network": "P2P Network Application Template",
            "enterprise_app": "Enterprise Application Template",
            "research_project": "Research Project Template"
        }
        
        if template_name not in templates:
            print(f"❌ Template '{template_name}' not found")
            print("Available templates:")
            for name, desc in templates.items():
                print(f"   • {name}: {desc}")
            return False
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🏗️  Generating template: {templates[template_name]}")
        print(f"📂 Output directory: {output_path.absolute()}")
        
        # Generate template files based on template type
        if template_name == "basic_agent":
            self._generate_basic_agent_template(output_path)
        elif template_name == "p2p_network":
            self._generate_p2p_template(output_path)
        elif template_name == "enterprise_app":
            self._generate_enterprise_template(output_path)
        elif template_name == "research_project":
            self._generate_research_template(output_path)
        
        print("✅ Template generated successfully!")
        print(f"💡 Next steps:")
        print(f"   1. cd {output_path}")
        print(f"   2. pip install -r requirements.txt")
        print(f"   3. python main.py")
        
        return True
    
    def _generate_basic_agent_template(self, output_path: Path):
        """Generate basic agent template files"""
        # Main application file
        main_py = '''#!/usr/bin/env python3
"""
Basic PRSM AI Agent Template
Generated by PRSM Developer Playground
"""

import asyncio
import sys
from pathlib import Path

# Add PRSM to path (adjust as needed)
prsm_path = Path(__file__).parent.parent
sys.path.append(str(prsm_path))

from prsm.agents.base import BaseAgent
from prsm.core.config import PRSMConfig

class MyAgent(BaseAgent):
    """Simple AI agent implementation"""
    
    def __init__(self, config: PRSMConfig):
        super().__init__(config)
        self.name = "MyAgent"
    
    async def initialize(self):
        """Initialize agent resources"""
        print(f"🤖 Initializing {self.name}...")
        # Add initialization logic here
        
    async def process_task(self, task):
        """Process a task"""
        print(f"📝 Processing task: {task}")
        # Add task processing logic here
        return {"status": "completed", "result": f"Processed: {task}"}
    
    async def shutdown(self):
        """Cleanup agent resources"""
        print(f"🛑 Shutting down {self.name}...")
        # Add cleanup logic here

async def main():
    """Main application entry point"""
    print("🚀 Starting Basic PRSM Agent")
    
    # Create configuration
    config = PRSMConfig()
    
    # Create and initialize agent
    agent = MyAgent(config)
    await agent.initialize()
    
    try:
        # Example tasks
        tasks = ["analyze data", "generate report", "optimize performance"]
        
        for task in tasks:
            result = await agent.process_task(task)
            print(f"✅ Task result: {result}")
        
        print("🎉 All tasks completed successfully!")
        
    except KeyboardInterrupt:
        print("\\n🛑 Stopping agent...")
    finally:
        await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        # Requirements file
        requirements_txt = '''# PRSM Basic Agent Requirements
prsm-sdk
asyncio
'''
        
        # README file
        readme_md = '''# Basic PRSM Agent

This is a basic AI agent template created with PRSM Developer Playground.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the agent:
   ```bash
   python main.py
   ```

## Customization

- Modify `MyAgent` class in `main.py`
- Add your business logic to `process_task()` method
- Customize initialization and shutdown procedures

## Next Steps

- Explore PRSM documentation
- Try more advanced examples
- Join the PRSM community
'''
        
        # Write files
        (output_path / "main.py").write_text(main_py)
        (output_path / "requirements.txt").write_text(requirements_txt)
        (output_path / "README.md").write_text(readme_md)
    
    def _generate_p2p_template(self, output_path: Path):
        """Generate P2P network template"""
        # Simplified P2P template
        main_py = '''#!/usr/bin/env python3
"""
P2P Network Application Template
Generated by PRSM Developer Playground
"""

import asyncio
from prsm.federation.p2p_network import P2PNetwork
from prsm.core.config import PRSMConfig

async def main():
    print("🌐 Starting P2P Network Application")
    
    config = PRSMConfig()
    network = P2PNetwork(config)
    
    await network.start()
    print("✅ P2P Network started successfully!")
    
    try:
        await asyncio.sleep(60)  # Run for 1 minute
    except KeyboardInterrupt:
        print("\\n🛑 Stopping network...")
    finally:
        await network.stop()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        (output_path / "main.py").write_text(main_py)
        (output_path / "requirements.txt").write_text("prsm-sdk\\naiohttp\\n")
        (output_path / "README.md").write_text("# P2P Network Application\\n\\nA P2P network application template.\\n")
    
    def _generate_enterprise_template(self, output_path: Path):
        """Generate enterprise application template"""
        # Simplified enterprise template
        main_py = '''#!/usr/bin/env python3
"""
Enterprise PRSM Application Template
Generated by PRSM Developer Playground
"""

import asyncio
from prsm.enterprise.security import EnterpriseSecurityManager
from prsm.monitoring.dashboard import MonitoringDashboard
from prsm.core.config import PRSMConfig

async def main():
    print("🏢 Starting Enterprise PRSM Application")
    
    config = PRSMConfig()
    config.security_enabled = True
    config.monitoring_enabled = True
    
    # Initialize enterprise components
    security = EnterpriseSecurityManager(config)
    dashboard = MonitoringDashboard(config)
    
    await security.initialize()
    await dashboard.start()
    
    print("✅ Enterprise application started!")
    
    try:
        await asyncio.sleep(3600)  # Run for 1 hour
    except KeyboardInterrupt:
        print("\\n🛑 Stopping application...")
    finally:
        await security.shutdown()
        await dashboard.stop()

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        (output_path / "main.py").write_text(main_py)
        (output_path / "requirements.txt").write_text("prsm-sdk\\nflask\\nprometheus-client\\n")
        (output_path / "README.md").write_text("# Enterprise PRSM Application\\n\\nAn enterprise-ready PRSM application template.\\n")
    
    def _generate_research_template(self, output_path: Path):
        """Generate research project template"""
        # Simplified research template
        main_py = '''#!/usr/bin/env python3
"""
Research Project Template
Generated by PRSM Developer Playground
"""

import asyncio
import numpy as np
from prsm.research.experiment import ExperimentFramework
from prsm.core.config import PRSMConfig

async def main():
    print("🔬 Starting Research Project")
    
    config = PRSMConfig()
    experiment = ExperimentFramework(config)
    
    # Run experiment
    results = await experiment.run_experiment({
        "name": "distributed_ai_research",
        "parameters": {"nodes": 5, "models": 3},
        "metrics": ["accuracy", "latency", "throughput"]
    })
    
    print(f"📊 Results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        (output_path / "main.py").write_text(main_py)
        (output_path / "requirements.txt").write_text("prsm-sdk\\nnumpy\\nmatplotlib\\n")
        (output_path / "README.md").write_text("# Research Project\\n\\nA research project template for PRSM experiments.\\n")
    
    def interactive_mode(self):
        """Start interactive playground mode"""
        print("🎮 PRSM Developer Playground - Interactive Mode")
        print("=" * 60)
        print("Commands:")
        print("  examples [category] [difficulty] - List examples")
        print("  tutorials                        - List tutorials")
        print("  run <example>                    - Run an example")
        print("  tutorial <name>                  - Start a tutorial")
        print("  template <name> <output_dir>     - Generate template")
        print("  help                             - Show this help")
        print("  quit                             - Exit playground")
        print("=" * 60)
        
        while True:
            try:
                command = input("\n🎮 playground> ").strip().split()
                
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == "quit" or cmd == "exit":
                    print("👋 Thanks for using PRSM Developer Playground!")
                    break
                elif cmd == "help":
                    print("Available commands: examples, tutorials, run, tutorial, template, help, quit")
                elif cmd == "examples":
                    category = command[1] if len(command) > 1 else None
                    difficulty = command[2] if len(command) > 2 else None
                    self.list_examples(category, difficulty)
                elif cmd == "tutorials":
                    self.list_tutorials()
                elif cmd == "run":
                    if len(command) < 2:
                        print("Usage: run <example_key>")
                    else:
                        asyncio.run(self.run_example(command[1]))
                elif cmd == "tutorial":
                    if len(command) < 2:
                        print("Usage: tutorial <tutorial_key>")
                    else:
                        asyncio.run(self.run_tutorial(command[1]))
                elif cmd == "template":
                    if len(command) < 3:
                        print("Usage: template <template_name> <output_dir>")
                    else:
                        self.generate_template(command[1], command[2])
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Thanks for using PRSM Developer Playground!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="PRSM Developer Playground")
    parser.add_argument("--example", help="Run specific example")
    parser.add_argument("--tutorial", help="Run specific tutorial")
    parser.add_argument("--template", help="Generate project template")
    parser.add_argument("--output", help="Output directory for template")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive mode")
    parser.add_argument("--list-examples", action="store_true", help="List all examples")
    parser.add_argument("--list-tutorials", action="store_true", help="List all tutorials")
    parser.add_argument("--category", help="Filter examples by category")
    parser.add_argument("--difficulty", help="Filter examples by difficulty")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", type=int, default=8888, help="Port for web interface")
    
    args = parser.parse_args()
    
    launcher = PlaygroundLauncher()
    
    if args.interactive:
        launcher.interactive_mode()
    elif args.example:
        asyncio.run(launcher.run_example(args.example, args.debug))
    elif args.tutorial:
        asyncio.run(launcher.run_tutorial(args.tutorial))
    elif args.template:
        if not args.output:
            print("❌ --output directory required for template generation")
            sys.exit(1)
        launcher.generate_template(args.template, args.output)
    elif args.list_examples:
        launcher.list_examples(args.category, args.difficulty)
    elif args.list_tutorials:
        launcher.list_tutorials()
    else:
        # Default: show welcome screen and enter interactive mode
        print("🚀 Welcome to PRSM Developer Playground!")
        print("🎯 Your gateway to exploring PRSM's capabilities")
        print()
        launcher.list_examples()
        print()
        print("💡 Use --interactive for interactive mode, or --help for options")

if __name__ == "__main__":
    main()