#!/usr/bin/env python3
"""
Simple AI Agent Example
This example demonstrates creating and running a basic AI agent with PRSM.
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

class SimpleAgent:
    """A simple AI agent for demonstration purposes"""
    
    def __init__(self, name: str = "SimpleAgent"):
        self.name = name
        self.tasks_completed = 0
        self.start_time = time.time()
        self.capabilities = [
            "text_processing",
            "data_analysis", 
            "task_planning",
            "report_generation"
        ]
        
        print(f"🤖 Initialized {self.name}")
        print(f"   Capabilities: {', '.join(self.capabilities)}")
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return results"""
        print(f"📝 Processing task: {task['type']}")
        
        # Simulate processing time
        processing_time = task.get('complexity', 1) * 0.5
        await asyncio.sleep(processing_time)
        
        # Generate result based on task type
        if task['type'] == 'text_processing':
            result = await self._process_text(task['data'])
        elif task['type'] == 'data_analysis':
            result = await self._analyze_data(task['data'])
        elif task['type'] == 'task_planning':
            result = await self._plan_tasks(task['data'])
        elif task['type'] == 'report_generation':
            result = await self._generate_report(task['data'])
        else:
            result = {"error": f"Unknown task type: {task['type']}"}
        
        self.tasks_completed += 1
        
        return {
            "task_id": task.get('id', 'unknown'),
            "agent": self.name,
            "status": "completed",
            "processing_time": processing_time,
            "result": result,
            "timestamp": time.time()
        }
    
    async def _process_text(self, text: str) -> Dict[str, Any]:
        """Process text data"""
        word_count = len(text.split())
        char_count = len(text)
        
        return {
            "type": "text_analysis",
            "word_count": word_count,
            "character_count": char_count,
            "average_word_length": char_count / max(word_count, 1),
            "summary": f"Processed text with {word_count} words"
        }
    
    async def _analyze_data(self, data: List[float]) -> Dict[str, Any]:
        """Analyze numerical data"""
        if not data:
            return {"error": "No data provided"}
        
        total = sum(data)
        count = len(data)
        average = total / count
        minimum = min(data)
        maximum = max(data)
        
        return {
            "type": "data_analysis",
            "total": total,
            "count": count,
            "average": average,
            "min": minimum,
            "max": maximum,
            "summary": f"Analyzed {count} data points, average: {average:.2f}"
        }
    
    async def _plan_tasks(self, objectives: List[str]) -> Dict[str, Any]:
        """Create a task plan"""
        plan = []
        
        for i, objective in enumerate(objectives, 1):
            plan.append({
                "step": i,
                "objective": objective,
                "estimated_duration": "30 minutes",
                "priority": "medium" if i <= len(objectives) // 2 else "low"
            })
        
        return {
            "type": "task_plan",
            "total_objectives": len(objectives),
            "plan": plan,
            "estimated_total_time": f"{len(objectives) * 30} minutes",
            "summary": f"Created plan for {len(objectives)} objectives"
        }
    
    async def _generate_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary report"""
        report_sections = []
        
        if 'metrics' in data:
            report_sections.append("Performance Metrics Analysis")
        if 'tasks' in data:
            report_sections.append("Task Execution Summary")
        if 'timeline' in data:
            report_sections.append("Timeline and Milestones")
        
        return {
            "type": "report",
            "sections": report_sections,
            "total_sections": len(report_sections),
            "generated_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "summary": f"Generated report with {len(report_sections)} sections"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        uptime = time.time() - self.start_time
        
        return {
            "name": self.name,
            "status": "active",
            "uptime_seconds": uptime,
            "tasks_completed": self.tasks_completed,
            "capabilities": self.capabilities,
            "performance": {
                "tasks_per_minute": (self.tasks_completed / max(uptime / 60, 1)),
                "average_processing_time": 1.5  # Simulated
            }
        }

async def run_agent_demo():
    """Run a demonstration of the simple agent"""
    print("🚀 Simple AI Agent Demo")
    print("=" * 50)
    
    # Create agent
    agent = SimpleAgent("DemoAgent")
    
    # Define sample tasks
    tasks = [
        {
            "id": "task_1",
            "type": "text_processing",
            "data": "The Protocol for Recursive Scientific Modeling (PRSM) enables distributed AI research through peer-to-peer networks.",
            "complexity": 1
        },
        {
            "id": "task_2", 
            "type": "data_analysis",
            "data": [85.2, 92.1, 78.5, 95.3, 88.7, 91.2, 84.9],
            "complexity": 2
        },
        {
            "id": "task_3",
            "type": "task_planning",
            "data": [
                "Initialize P2P network",
                "Load AI models",
                "Distribute computation tasks",
                "Aggregate results",
                "Generate final report"
            ],
            "complexity": 1
        },
        {
            "id": "task_4",
            "type": "report_generation",
            "data": {
                "metrics": {"accuracy": 0.95, "latency": 120},
                "tasks": ["model_training", "validation", "deployment"],
                "timeline": "2024-Q1"
            },
            "complexity": 2
        }
    ]
    
    print(f"\n📋 Processing {len(tasks)} tasks...")
    
    # Process tasks
    results = []
    for task in tasks:
        print(f"\n🔄 Starting {task['type']} (ID: {task['id']})")
        result = await agent.process_task(task)
        results.append(result)
        
        print(f"✅ Completed in {result['processing_time']:.1f}s")
        print(f"   Result: {result['result']['summary']}")
    
    # Show final status
    print(f"\n📊 Final Agent Status:")
    status = agent.get_status()
    print(f"   Agent: {status['name']}")
    print(f"   Tasks Completed: {status['tasks_completed']}")
    print(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
    print(f"   Performance: {status['performance']['tasks_per_minute']:.1f} tasks/minute")
    
    # Show detailed results
    print(f"\n📈 Task Results Summary:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['task_id']}: {result['status']} ({result['processing_time']:.1f}s)")
    
    print(f"\n🎉 Agent demo completed successfully!")
    print(f"💡 Next steps:")
    print(f"   • Try: python playground_launcher.py --example ai_models/model_loading")
    print(f"   • Try: python playground_launcher.py --example p2p_network/basic_network")
    
    return results

async def main():
    """Main function"""
    try:
        results = await run_agent_demo()
        
        # Save results for reference
        output_file = Path(__file__).parent / "agent_demo_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    
    if success:
        print("\n✅ Simple Agent example completed successfully!")
    else:
        print("\n❌ Example failed. Check the logs for details.")
        sys.exit(1)