# PRSM Core Concepts

Understanding the three pillars of PRSM: NWTN, FTNS, and Teachers.

## ⏱️ Time: 10 minutes

## 🎯 Learning Goals

- Understand PRSM's distributed AI architecture
- Learn how NWTN orchestrates AI queries
- Master FTNS economic incentives
- Explore Teacher model specialization

## 🧠 The Three Pillars of PRSM

### 1. NWTN (Newton) - AI Orchestrator

NWTN is PRSM's central intelligence that coordinates AI models to solve complex problems.

**Key Concepts:**
- **Query Decomposition**: Breaks complex problems into manageable tasks
- **Model Selection**: Chooses optimal AI models for each task
- **Context Management**: Maintains conversation history and reasoning
- **Quality Assurance**: Validates and combines responses

**Simple Example:**
```python
# NWTN coordinates this complex query:
query = "Design a sustainable water purification system for rural communities"

# NWTN breaks it down:
# 1. Research existing purification methods
# 2. Analyze rural infrastructure constraints  
# 3. Evaluate sustainability criteria
# 4. Design system architecture
# 5. Calculate implementation costs
```

### 2. FTNS (Fungible Tokens for Node Support) - Economic Engine

FTNS tokens provide economic incentives for quality AI responses and network participation.

**Key Concepts:**
- **Token Economics**: Pay for AI processing with FTNS
- **Quality Incentives**: Better responses earn more tokens
- **Resource Allocation**: Distribute computing fairly
- **Network Effects**: More participants = better performance

**Practical Example:**
```python
# You start with FTNS tokens
user_budget = 100  # FTNS tokens

# Each query costs tokens based on complexity
simple_query_cost = 5    # "What's the weather?"
complex_query_cost = 50  # "Analyze climate change impacts"

# Better models cost more but give better results
gpt4_cost = 25          # High-quality, expensive
gpt3_cost = 10          # Good quality, moderate cost
local_model_cost = 3    # Basic quality, very cheap
```

### 3. Teachers - Specialized AI Models

Teachers are AI models fine-tuned for specific domains and tasks.

**Key Concepts:**
- **Domain Specialization**: Expert knowledge in specific fields
- **Fine-Tuning**: Optimized for particular problem types  
- **Knowledge Distillation**: Compress large models efficiently
- **Continuous Learning**: Improve through feedback

**Teacher Types:**
```python
# Science Teachers
biology_teacher = "Specialized in biological processes"
chemistry_teacher = "Expert in molecular interactions"
physics_teacher = "Focused on physical phenomena"

# Creative Teachers  
writing_teacher = "Optimized for content creation"
art_teacher = "Specialized in visual concepts"
music_teacher = "Expert in musical theory"

# Technical Teachers
coding_teacher = "Focused on software development"
math_teacher = "Specialized in mathematical problem-solving"
data_teacher = "Expert in data analysis"
```

## 🔄 How They Work Together

Let's see a complete example of PRSM's architecture in action:

```python
#!/usr/bin/env python3
\"\"\"
PRSM Architecture Demo
Shows how NWTN, FTNS, and Teachers collaborate
\"\"\"

import asyncio
from prsm.nwtn.orchestrator import NWTNOrchestrator
from prsm.core.models import UserInput

async def architecture_demo():
    print("🏗️  PRSM Architecture in Action\\n")
    
    # Initialize NWTN orchestrator
    orchestrator = NWTNOrchestrator()
    
    # Complex scientific query
    query = UserInput(
        user_id="demo_user",
        prompt=\\\"\\\"\\\"
        I'm designing a small greenhouse for my backyard. 
        I want to grow tomatoes year-round in Minnesota. 
        What should I consider for temperature control, 
        lighting, and water management?
        \\\"\\\"\\\",
        context_allocation=75  # Allocate 75 FTNS tokens
    )
    
    print("📝 User Query:")
    print(f"   {query.prompt}")
    print(f"💰 FTNS Budget: {query.context_allocation} tokens\\n")
    
    print("🧠 NWTN Processing Steps:")
    print("   1. Analyzing query complexity...")
    print("   2. Identifying required expertise areas...")
    print("   3. Selecting appropriate teacher models...")
    print("   4. Decomposing into sub-queries...")
    print("   5. Coordinating teacher responses...")
    print("   6. Synthesizing final answer...\\n")
    
    # Process through PRSM
    response = await orchestrator.process_query(query)
    
    print("✅ PRSM Response:")
    print("=" * 60)
    print(response.final_answer)
    print("=" * 60)
    
    print(f"\\n📊 Processing Details:")
    print(f"   💰 FTNS Used: {response.ftns_charged}/{query.context_allocation}")
    print(f"   ⏱️  Processing Time: {response.processing_time:.2f}s")
    print(f"   🎯 Quality Score: {response.quality_score}/100")
    
    if response.reasoning_trace:
        print(f"\\n🧭 NWTN Reasoning Process:")
        for i, step in enumerate(response.reasoning_trace, 1):
            print(f"   {i}. {step}")
    
    print("\\n🏫 Teachers Involved:")
    if hasattr(response, 'teachers_used'):
        for teacher in response.teachers_used:
            print(f"   🎓 {teacher}")
    else:
        print("   🎓 General Purpose Teacher (GPT-4)")
        print("   🎓 Environmental Systems Teacher")
        print("   🎓 Agriculture Specialist Teacher")

if __name__ == "__main__":
    asyncio.run(architecture_demo())
```

## 💡 Key Insights

### NWTN as Conductor
Think of NWTN like a symphony conductor:
- **Reads the Music**: Understands your complex query
- **Coordinates Musicians**: Assigns tasks to different AI models
- **Maintains Tempo**: Manages processing flow and timing
- **Ensures Harmony**: Combines responses coherently

### FTNS as Currency
FTNS tokens work like credits in an arcade:
- **Pay to Play**: Use tokens for AI processing
- **Better Games Cost More**: Advanced models require more tokens
- **Earn Rewards**: Quality contributions earn tokens back
- **Shared Economy**: Everyone benefits from network effects

### Teachers as Specialists
Teachers are like university professors:
- **Domain Expertise**: Deep knowledge in specific fields
- **Specialized Training**: Optimized for particular problem types
- **Continuous Learning**: Improve through experience
- **Collaboration**: Work together on complex problems

## 🔬 Interactive Concept Test

Create `concept_test.py` to experiment with these ideas:

```python
#!/usr/bin/env python3
\"\"\"
Interactive PRSM Concept Explorer
Experiment with different query types and see how PRSM responds
\"\"\"

import asyncio
from prsm.nwtn.orchestrator import NWTNOrchestrator
from prsm.core.models import UserInput

# Test queries showing different PRSM capabilities
TEST_QUERIES = {
    "simple": {
        "prompt": "What is photosynthesis?",
        "tokens": 15,
        "expected": "Single teacher, direct answer"
    },
    "complex": {
        "prompt": "Design a sustainable urban transportation system that reduces emissions by 50% while maintaining accessibility for disabled users",
        "tokens": 100,
        "expected": "Multiple teachers, decomposed reasoning"
    },
    "creative": {
        "prompt": "Write a haiku about machine learning that also explains gradient descent",
        "tokens": 30,
        "expected": "Creative + technical teachers"
    },
    "scientific": {
        "prompt": "Explain how CRISPR gene editing could be used to develop drought-resistant crops, including ethical considerations",
        "tokens": 80,
        "expected": "Biology + ethics teachers"
    }
}

async def test_concept(query_type, query_data):
    print(f"\\n🧪 Testing: {query_type.upper()} Query")
    print(f"📝 Prompt: {query_data['prompt']}")
    print(f"💰 FTNS Budget: {query_data['tokens']}")
    print(f"🎯 Expected: {query_data['expected']}")
    print("-" * 50)
    
    orchestrator = NWTNOrchestrator()
    
    query = UserInput(
        user_id="concept_test",
        prompt=query_data['prompt'],
        context_allocation=query_data['tokens']
    )
    
    try:
        response = await orchestrator.process_query(query)
        
        print("✅ Response received!")
        print(f"💰 FTNS Used: {response.ftns_charged}")
        print(f"⏱️  Time: {response.processing_time:.2f}s")
        
        # Show reasoning for complex queries
        if query_type == "complex" and response.reasoning_trace:
            print("\\n🧭 NWTN Reasoning:")
            for step in response.reasoning_trace:
                print(f"   • {step}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    print("🎓 PRSM Concept Explorer")
    print("Testing different query types to understand PRSM architecture")
    print("=" * 60)
    
    for query_type, query_data in TEST_QUERIES.items():
        success = await test_concept(query_type, query_data)
        if not success:
            print("⚠️  Skipping remaining tests due to error")
            break
        
        # Pause between tests
        await asyncio.sleep(1)
    
    print("\\n🎉 Concept exploration complete!")
    print("💡 Notice how PRSM adapts its approach based on query complexity")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🎯 Key Takeaways

1. **NWTN Orchestrates**: Coordinates multiple AI models intelligently
2. **FTNS Incentivizes**: Provides economic framework for quality
3. **Teachers Specialize**: Domain-specific models for better results
4. **Together They Scale**: Handle problems too complex for single models

## 🤔 Reflection Questions

Before moving on, consider:

1. How would PRSM handle a query in your field of expertise?
2. What kinds of problems benefit most from multi-model coordination?
3. How might FTNS economics encourage better AI responses?
4. What teacher specializations would be most valuable?

## 🚀 What's Next?

Now that you understand PRSM's architecture:

- **Try the Examples**: Run the code above to see concepts in action
- **Learn the API**: [API Fundamentals](./api-fundamentals.md)
- **Configure Your Environment**: [Configuration Deep Dive](./configuration.md)

---

**Architecture Understood!** 🏗️

**Next Tutorial** → [API Fundamentals](./api-fundamentals.md)