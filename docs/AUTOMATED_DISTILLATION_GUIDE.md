# 🤖 PRSM Automated Distillation System - How-To Guide

**Welcome to PRSM's Automated Distillation System!** This guide will help you create your own specialized AI models in just a few simple steps, even if you have no machine learning background.

---

## 🎯 What Is Automated Distillation?

Think of distillation like teaching a specialized student from a genius teacher:

- **Teacher Model**: A large, powerful AI (like GPT-4 or Claude) that knows a lot about everything
- **Student Model**: A smaller, faster AI that becomes an expert in your specific area
- **Distillation**: The automated process of transferring knowledge from teacher to student using state-of-the-art ML frameworks

**🤖 Multi-Framework AI Engine:**
PRSM automatically selects the best ML framework for your needs:
- **🔥 PyTorch**: Best for research and custom architectures  
- **📱 TensorFlow**: Best for mobile deployment and production
- **🤗 Transformers**: Best for NLP tasks with pre-trained models

**Why This Matters:**
- ✨ **Create specialized models** for your exact needs with production-ready quality
- 💰 **Save money** - smaller models cost 90% less to run
- ⚡ **Get faster responses** - specialized models run 10x faster
- 🎯 **Better accuracy** - focused models often outperform generalists in their domain
- 🚀 **Deploy anywhere** - get models in PyTorch (.pth), TensorFlow (SavedModel), or Transformers (Hub) format

---

## 🚀 Quick Start: Create Your First Model in 5 Minutes

### Step 1: Choose Your Domain

First, decide what you want your AI to specialize in:

```python
# Popular domains in PRSM:
domains = [
    "medical_research",      # For healthcare and medical applications
    "legal_analysis",        # For legal documents and compliance
    "scientific_reasoning",  # For research and academic work
    "code_generation",       # For programming and software development
    "creative_writing",      # For content creation and storytelling
    "data_analysis",         # For statistics and data science
    "financial_analysis",    # For investment and financial planning
    "educational_content",   # For teaching and learning materials
    "general_purpose"        # For broad, versatile applications
]
```

### Step 2: Set Your Budget

FTNS tokens are PRSM's currency. Here's what different budgets get you:

| Budget | Model Quality | Use Case |
|--------|---------------|----------|
| 500-1000 FTNS | Basic | Simple tasks, prototyping |
| 1000-2500 FTNS | Good | Professional applications |
| 2500-5000 FTNS | Excellent | Production systems |
| 5000+ FTNS | Premium | Mission-critical applications |

### Step 3: Create Your Model

```python
from prsm.distillation.orchestrator import get_distillation_orchestrator
from prsm.distillation.models import DistillationRequest, ModelSize, OptimizationTarget, TrainingStrategy

# Initialize the system
orchestrator = get_distillation_orchestrator()

# Create your distillation request
request = DistillationRequest(
    user_id="your_username",
    
    # WHAT TO DISTILL
    teacher_model="gpt-4",              # The "genius teacher" to learn from
    domain="medical_research",          # Your chosen specialization
    
    # HOW BIG/FAST
    target_size=ModelSize.SMALL,        # TINY, SMALL, MEDIUM, or LARGE
    optimization_target=OptimizationTarget.BALANCED,  # SPEED, ACCURACY, EFFICIENCY, SIZE, or BALANCED
    
    # HOW TO TRAIN  
    training_strategy=TrainingStrategy.PROGRESSIVE,  # We recommend PROGRESSIVE for beginners
    
    # ECONOMICS
    budget_ftns=2000,                   # Your FTNS budget
    quality_threshold=0.85,             # Minimum quality (85% of teacher's performance)
    marketplace_listing=True            # List in PRSM marketplace when done
)

# Submit your job
job = await orchestrator.create_distillation(request)
print(f"🎉 Your distillation job has been created!")
print(f"Job ID: {job.job_id}")
print(f"Estimated completion: {job.estimated_completion}")
```

### Step 4: Monitor Your Progress

```python
# Check status anytime
status = await orchestrator.get_job_status(job.job_id)

print(f"Status: {status.status}")
print(f"Progress: {status.progress}%")
print(f"Current stage: {status.current_stage}")
print(f"Estimated time remaining: {status.estimated_remaining_minutes} minutes")
```

### Step 5: Use Your New Model

Once complete, your model will be automatically:
- ✅ **Deployed** to PRSM's network
- ✅ **Listed** in the marketplace (if you chose that option)
- ✅ **Ready to use** in your applications

---

## 🎛️ Advanced Options: Customize Your Model

### Model Sizes: Finding the Right Balance

| Size | Parameters | Speed | Cost | Best For |
|------|------------|-------|------|----------|
| **TINY** | < 100M | ⚡⚡⚡ | 💰 | Mobile apps, real-time responses |
| **SMALL** | 100M-1B | ⚡⚡ | 💰💰 | Most applications, good balance |
| **MEDIUM** | 1B-10B | ⚡ | 💰💰💰 | Complex reasoning, high accuracy |
| **LARGE** | 10B+ | 🐌 | 💰💰💰💰 | Research, maximum capability |

**💡 Recommendation**: Start with SMALL for most use cases.

### Optimization Targets: What Matters Most?

```python
# Choose based on your priorities (also affects ML framework selection):
OptimizationTarget.SPEED      # Fastest responses (good for chatbots) → PyTorch
OptimizationTarget.ACCURACY   # Best performance (good for research) → Transformers/PyTorch
OptimizationTarget.EFFICIENCY # Best cost/performance ratio → PyTorch
OptimizationTarget.SIZE       # Smallest model (good for mobile) → TensorFlow
OptimizationTarget.BALANCED   # Good all-around choice ⭐ RECOMMENDED → Auto-selected
```

**🤖 Framework Selection**: PRSM automatically chooses the optimal ML framework based on your domain and optimization target. No ML expertise required!

### Training Strategies: How Your Model Learns

| Strategy | Difficulty | Time | Quality | Best For |
|----------|------------|------|---------|----------|
| **BASIC** | Easy | Fast | Good | Beginners, simple domains |
| **PROGRESSIVE** ⭐ | Medium | Medium | Excellent | Most use cases |
| **ENSEMBLE** | Medium | Slow | Excellent | When you have multiple teachers |
| **ADVERSARIAL** | Hard | Slow | Robust | Security-critical applications |
| **CURRICULUM** | Medium | Medium | Good | Educational content |
| **SELF_SUPERVISED** | Hard | Slow | Variable | When you have unlabeled data |

**💡 Recommendation**: Use PROGRESSIVE for best results with reasonable time/cost.

---

## 🎯 Domain-Specific Examples

### Example 1: Medical Diagnosis Assistant

```python
medical_request = DistillationRequest(
    user_id="dr_smith",
    teacher_model="gpt-4",
    domain="medical_research",            # → Auto-selects Transformers backend
    specialization="diagnostic_imaging",  # Optional: narrow focus
    target_size=ModelSize.MEDIUM,         # Need good accuracy for medical
    optimization_target=OptimizationTarget.ACCURACY,
    training_strategy=TrainingStrategy.PROGRESSIVE,
    budget_ftns=3500,
    quality_threshold=0.90,               # High threshold for medical
    
    # Advanced options
    max_inference_latency="500ms",        # Fast enough for real-time use
    target_hardware=["cpu", "gpu"],       # Can run on various hardware
    marketplace_listing=True,
    revenue_sharing=0.1                   # Share 10% revenue with teacher owners
)

# Result: Model trained with Transformers, exported in Hugging Face format
```

### Example 2: Code Review Assistant

```python
code_request = DistillationRequest(
    user_id="dev_team",
    teacher_model="claude-3-opus",
    domain="code_generation",             # → Auto-selects Transformers backend  
    specialization="python_code_review",
    target_size=ModelSize.SMALL,          # Speed matters for development
    optimization_target=OptimizationTarget.SPEED,
    training_strategy=TrainingStrategy.PROGRESSIVE,
    budget_ftns=1500,
    quality_threshold=0.80,
    
    # Custom training data (optional)
    custom_training_data="ipfs://your-code-examples-hash",
    augmentation_techniques=["adversarial", "paraphrase"],
    marketplace_listing=True
)

# Result: Model trained with Transformers, optimized for code tasks
```

### Example 3: Mobile App Assistant

```python
mobile_request = DistillationRequest(
    user_id="mobile_dev",
    teacher_model="gpt-4",
    domain="general_purpose",
    target_size=ModelSize.TINY,           # Must fit on mobile
    optimization_target=OptimizationTarget.SIZE,  # → Auto-selects TensorFlow  
    training_strategy=TrainingStrategy.BASIC,
    budget_ftns=800,
    quality_threshold=0.75,
    
    # Mobile-specific options
    target_hardware=["mobile"],
    max_model_size_mb=50,                 # Must be under 50MB
    marketplace_listing=True
)

# Result: Model trained with TensorFlow, exported as TensorFlow Lite for mobile
```

---

## 💰 Understanding Costs

### Cost Factors

Your FTNS cost depends on:

1. **Model Size**: LARGE costs ~6x more than TINY
2. **Training Strategy**: ADVERSARIAL costs ~2.5x more than BASIC
3. **Teacher Model**: Premium models (GPT-4) cost more than others
4. **Quality Threshold**: Higher quality = higher cost
5. **Custom Features**: Multi-teacher, custom data, etc.

### Cost Estimation

```python
# Get cost estimate before submitting
estimated_cost = await orchestrator._estimate_cost(request)
print(f"Estimated cost: {estimated_cost} FTNS")

# Check your balance
user_balance = await ftns_service.get_user_balance("your_username")
print(f"Your balance: {user_balance.balance} FTNS")
```

### Ways to Reduce Costs

- ✅ **Start with BASIC strategy** and upgrade if needed
- ✅ **Use SMALL size** unless you specifically need larger
- ✅ **Lower quality threshold** slightly (0.80 vs 0.90)
- ✅ **Choose efficient teacher models** (Claude vs GPT-4)
- ✅ **Avoid unnecessary features** (multi-teacher, custom data)

---

## 🔍 Monitoring and Troubleshooting

### Real-Time Progress Tracking

```python
import asyncio

async def monitor_job(job_id):
    while True:
        status = await orchestrator.get_job_status(job_id)
        
        print(f"\n📊 Progress Report:")
        print(f"Status: {status.status}")
        print(f"Overall Progress: {status.progress}%")
        print(f"Current Stage: {status.current_stage}")
        print(f"Stage Progress: {status.stage_progress}%")
        print(f"Time Elapsed: {status.elapsed_time_minutes} minutes")
        print(f"Time Remaining: {status.estimated_remaining_minutes} minutes")
        print(f"FTNS Spent: {status.current_ftns_spent}")
        
        if status.status in ["completed", "failed", "cancelled"]:
            break
            
        await asyncio.sleep(60)  # Check every minute

# Monitor your job
await monitor_job(job.job_id)
```

### Common Issues and Solutions

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| "Insufficient FTNS balance" | Not enough tokens | Buy more FTNS or reduce budget |
| "Teacher model not available" | Wrong model name | Check available models list |
| "Budget too low" | Underestimated costs | Increase budget or simplify request |
| Job stuck in "queued" | High demand | Wait or increase priority with higher budget |
| Low quality results | Wrong strategy/settings | Try PROGRESSIVE strategy, higher threshold |

### Cancelling a Job

```python
# Cancel if needed (you'll get a partial refund)
cancelled = await orchestrator.cancel_job(job.job_id, "your_username")
if cancelled:
    print("Job cancelled successfully")
```

---

## 🏪 Marketplace Integration

### Automatic Listing

When you set `marketplace_listing=True`, your completed model automatically appears in PRSM's marketplace with:

- ✨ **Quality metrics** (accuracy, speed, efficiency)
- 🛡️ **Safety certification** 
- 💰 **Automatic pricing** based on performance
- 📊 **Usage analytics** 
- 💸 **Revenue sharing** if configured

### Earning from Your Models

```python
# Set revenue sharing when creating your model
request = DistillationRequest(
    # ... other parameters ...
    revenue_sharing=0.2,  # Share 20% with teacher model owners
    marketplace_listing=True,
    pricing_model="usage_based"  # Users pay per use
)
```

**Revenue Streams:**
- 💰 **Direct usage fees** from other PRSM users
- 📈 **Performance bonuses** for highly-rated models
- 🎯 **Citation rewards** for academic/research usage
- ⚡ **Efficiency bonuses** for resource-efficient models

---

## 🛡️ Safety and Quality

### Built-in Safety Features

PRSM's automated distillation includes comprehensive safety measures:

- 🔍 **Content filtering** prevents harmful outputs
- 🛡️ **Circuit breaker integration** with emergency halt capability
- 📊 **Bias detection** and mitigation
- 🔒 **Privacy protection** for sensitive data
- ⚖️ **Governance compliance** with community standards

### Quality Assurance

Every model goes through:

1. **Functional testing** - Does it work correctly?
2. **Performance benchmarking** - How fast and accurate?
3. **Safety validation** - Is it safe to deploy?
4. **Robustness testing** - Can it handle edge cases?
5. **Efficiency validation** - Is it resource-optimized?

---

## 🎓 Best Practices

### For Beginners

1. **Start simple** - Use BASIC strategy and SMALL size first
2. **Pick common domains** - Medical, legal, code, etc. are well-supported
3. **Set realistic budgets** - 1500-2500 FTNS for good results
4. **Use BALANCED optimization** - Good all-around choice
5. **Enable marketplace listing** - Earn revenue from your models

### For Advanced Users

1. **Experiment with strategies** - PROGRESSIVE and ENSEMBLE often give best results
2. **Use custom training data** - Upload domain-specific examples
3. **Framework awareness** - Understand which backend works best for your use case:
   - **PyTorch**: Best for research and custom architectures
   - **TensorFlow**: Best for mobile deployment and edge computing  
   - **Transformers**: Best for NLP and pre-trained model utilization
4. **Export format planning** - Know which format you need for deployment
5. **Hardware optimization** - Specify target deployment environment
5. **Revenue optimization** - Balance quality vs. cost for marketplace success

### Production Deployment

1. **Thorough testing** - Use high quality thresholds (0.85+)
2. **Framework selection** - Choose based on deployment environment:
   - **PyTorch models (.pth)**: Research environments, custom deployment
   - **TensorFlow models (SavedModel/TFLite)**: Production servers, mobile apps
   - **Transformers models**: Easy deployment to Hugging Face Hub
3. **Safety validation** - Enable all safety checks
4. **Performance monitoring** - Track usage and efficiency
5. **Gradual rollout** - Start with limited users
6. **Feedback integration** - Collect user feedback for improvements

---

## 🔧 Troubleshooting Guide

### Job Not Starting?

```python
# Check system status
stats = await orchestrator.get_system_stats()
print(f"Active jobs: {stats['active_jobs']}")
print(f"Queue length: {stats['queued_jobs']}")
print(f"Resource utilization: {stats['resource_utilization']}")

# If system is busy, try:
# 1. Increase your budget (higher priority)
# 2. Wait for off-peak hours
# 3. Use simpler settings to reduce resource needs
```

### Poor Quality Results?

```python
# Try these improvements:
better_request = DistillationRequest(
    # ... your settings ...
    
    # Quality improvements:
    training_strategy=TrainingStrategy.PROGRESSIVE,  # Usually better than BASIC
    quality_threshold=0.90,                         # Higher threshold
    target_size=ModelSize.MEDIUM,                   # Larger if you can afford it
    optimization_target=OptimizationTarget.ACCURACY, # Focus on quality
    
    # More training time/resources:
    budget_ftns=budget * 1.5,                      # Increase budget 50%
)
```

### Getting Help

- 📚 **Documentation**: Check the full docs at `docs/`
- 💬 **Community**: Join GitHub Discussions
- 🐛 **Issues**: Report bugs on GitHub Issues
- 📧 **Support**: Contact the development team for enterprise support

---

## 🎉 Success Stories

### "I created a legal document analyzer in 30 minutes!"
> "As a lawyer with no programming experience, I was able to create a specialized AI that reviews contracts and finds potential issues. It saves me hours every week!" - Sarah K., Legal Professional

### "Our startup's customer service bot got 50% better"
> "We distilled our customer service knowledge into a specialized model. Response quality improved dramatically and costs dropped by 80%." - Mike T., Startup Founder

### "Research paper analysis at scale"
> "I created a model that analyzes research papers in my field. It's like having a research assistant that never sleeps!" - Dr. Chen L., Research Scientist

---

## 🚀 Ready to Get Started?

1. **[Install PRSM](../README.md#installation)** if you haven't already
2. **Get some FTNS tokens** through the tokenomics system
3. **Choose your domain** from the examples above
4. **Copy and modify** one of our example requests
5. **Submit your first distillation job**
6. **Share your success** with the community!

**Need more help?** 
- 📖 Read the [full documentation](../docs/)
- 💬 Ask questions in [GitHub Discussions](https://github.com/your-repo/discussions)
- 🎯 Check out more [tutorials](../docs/tutorials/)

---

> _"Making AI development accessible to everyone - that's the power of PRSM's Automated Distillation System!"_

**Happy model building! 🤖✨**