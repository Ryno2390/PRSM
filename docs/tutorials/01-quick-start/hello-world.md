# Hello World Tutorial

Welcome to PRSM! This is your first tutorial - let's get you running a query in under 2 minutes.

## ⏱️ Time: 2 minutes

## 🎯 Goal
Run your first PRSM query and see the magic happen.

## 🚀 Let's Go!

### Step 1: Quick Setup Check
```bash
# Verify PRSM is ready
prsm-dev status
```

You should see all green checkmarks ✅. If not, run:
```bash
prsm-dev setup
```

### Step 2: Your First Query

Create a file called `hello_prsm.py`:

```python
#!/usr/bin/env python3
\"\"\"
PRSM Hello World - Your First AI Query
This example shows the simplest possible PRSM interaction.
\"\"\"

import asyncio
from prsm.nwtn.orchestrator import NWTNOrchestrator
from prsm.core.models import UserInput, PRSMSession

async def hello_world():
    print("🧠 Initializing PRSM...")
    
    # Create the NWTN orchestrator (PRSM's AI coordinator)
    orchestrator = NWTNOrchestrator()
    
    # Create a user session with FTNS budget
    session = PRSMSession(
        user_id="hello_world_user",
        nwtn_context_allocation=50  # 50 FTNS tokens for this query
    )
    
    # Your first AI query
    query = UserInput(
        user_id="hello_world_user",
        prompt="Hello PRSM! Please introduce yourself and explain what you do in simple terms.",
        context_allocation=25  # Use 25 FTNS for this specific query
    )
    
    print("🚀 Sending your first query to PRSM...")
    print(f"📝 Query: {query.prompt}")
    print(f"💰 FTNS Budget: {query.context_allocation} tokens")
    print("⏳ Processing...\n")
    
    try:
        # Process the query through PRSM
        response = await orchestrator.process_query(query)
        
        # Display the results
        print("✅ SUCCESS! PRSM Response:")
        print("=" * 50)
        print(f"🤖 Answer: {response.final_answer}")
        print(f"💰 FTNS Used: {response.ftns_charged}")
        print(f"⏱️  Processing Time: {response.processing_time:.2f} seconds")
        
        if response.reasoning_trace:
            print(f"\\n🧭 AI Reasoning Steps:")
            for i, step in enumerate(response.reasoning_trace, 1):
                print(f"   {i}. {step}")
        
        print("\\n🎉 Congratulations! You just ran your first PRSM query!")
        print("💡 Next: Try the Foundation tutorials to learn more.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\\n🔧 Troubleshooting:")
        print("1. Check API keys: config/api_keys.env")
        print("2. Verify services: prsm-dev status")
        print("3. Run setup: prsm-dev setup")
        return False
    
    return True

if __name__ == "__main__":
    print("🎯 PRSM Hello World Tutorial")
    print("This is your first PRSM query!")
    print("-" * 40)
    
    # Run the hello world example
    success = asyncio.run(hello_world())
    
    if success:
        print("\\n✨ Tutorial Complete!")
        print("🎓 You're ready for more advanced tutorials.")
    else:
        print("\\n❌ Tutorial failed. Check the troubleshooting steps above.")
```

### Step 3: Run It!

```bash
python hello_prsm.py
```

## 🎉 Expected Output

You should see something like:

```
🎯 PRSM Hello World Tutorial
This is your first PRSM query!
----------------------------------------
🧠 Initializing PRSM...
🚀 Sending your first query to PRSM...
📝 Query: Hello PRSM! Please introduce yourself and explain what you do in simple terms.
💰 FTNS Budget: 25 tokens
⏳ Processing...

✅ SUCCESS! PRSM Response:
==================================================
🤖 Answer: Hello! I'm PRSM (Protocol for Recursive Scientific Modeling), 
a decentralized AI framework designed to advance scientific discovery. 
I coordinate multiple AI models to solve complex problems by breaking 
them into smaller tasks and using economic incentives (FTNS tokens) 
to ensure quality responses. Think of me as an AI orchestrator that 
can tackle scientific challenges too complex for any single AI model.

💰 FTNS Used: 23.4
⏱️  Processing Time: 2.84 seconds

🧭 AI Reasoning Steps:
   1. Analyzed user query for introduction request
   2. Selected appropriate response tone and complexity
   3. Structured explanation of PRSM's core functionality
   4. Validated response quality and completeness

🎉 Congratulations! You just ran your first PRSM query!
💡 Next: Try the Foundation tutorials to learn more.

✨ Tutorial Complete!
🎓 You're ready for more advanced tutorials.
```

## 🔍 What Just Happened?

1. **NWTN Orchestrator**: PRSM's AI coordinator that manages your queries
2. **FTNS Tokens**: PRSM's internal currency for AI processing (like credits)
3. **Reasoning Trace**: Shows how PRSM "thinks" through problems
4. **Processing Time**: Real-time performance metrics

## 🐛 Troubleshooting

### ❌ "Module not found" error
```bash
# Install PRSM in development mode
prsm-dev setup
```

### ❌ "API key not configured" error
```bash
# Edit your API keys
nano config/api_keys.env
```

### ❌ "Connection refused" error
```bash
# Start required services
prsm-dev start
```

### ❌ Still having issues?
```bash
# Run comprehensive diagnostics
prsm-dev diagnose
```

## 🎯 What's Next?

Now that you've successfully run your first PRSM query:

1. **Try Different Queries**: Change the prompt and see how PRSM responds
2. **Learn the Basics**: Continue to [Foundation Tutorials](../02-foundation/)
3. **Join the Community**: Share your first success story!

### Quick Experiments

Try changing the prompt in `hello_prsm.py` to:

```python
# Scientific query
prompt="Explain photosynthesis in terms a 12-year-old would understand"

# Creative query  
prompt="Write a haiku about artificial intelligence"

# Problem-solving query
prompt="How would you approach reducing plastic waste in oceans?"
```

## 📚 Related

- [Setup Verification](./setup-verification.md) - Comprehensive system check
- [Foundation Tutorials](../02-foundation/) - Learn PRSM concepts
- [API Reference](../../API_REFERENCE.md) - Complete API documentation

---

**Congratulations!** 🎉 You've completed your first PRSM tutorial.

**Next Tutorial** → [Setup Verification](./setup-verification.md)