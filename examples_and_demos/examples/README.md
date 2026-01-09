# PRSM Phase 7 Examples

This directory contains comprehensive examples demonstrating PRSM's Phase 7 enterprise architecture capabilities. Each example provides working code that showcases different aspects of the enterprise-grade AI infrastructure.

## Examples Overview

### 📊 Analytics Example (`phase7_analytics_example.py`)

Demonstrates PRSM's advanced analytics and business intelligence capabilities.

**Features Showcased:**
- Executive dashboard creation with KPIs and strategic metrics
- Operational dashboard for real-time system monitoring
- Automated analytics insights generation
- Custom report creation and export
- Real-time metrics monitoring
- Performance analytics and optimization

**Key Components:**
- Dashboard Manager for creating and managing dashboards
- Global Infrastructure for system health monitoring
- Analytics data generation and visualization
- Insight generation with confidence scoring

**Run the Example:**
```bash
cd examples
python phase7_analytics_example.py
```

### 🤖 AI Orchestration Example (`phase7_orchestration_example.py`)

Demonstrates PRSM's AI orchestration platform for multi-model coordination.

**Features Showcased:**
- Multi-model AI registration and management
- Intelligent task routing based on capabilities
- Multi-model collaboration for complex tasks
- NWTN unified pipeline integration
- Performance optimization and benchmarking
- Marketplace AI model integration

**Key Components:**
- AI Orchestrator for model coordination
- Unified Pipeline Controller for complete processing
- Model performance monitoring and optimization
- Cost-aware task distribution

**Run the Example:**
```bash
cd examples
python phase7_orchestration_example.py
```

### 🏪 Marketplace Example (`phase7_marketplace_example.py`)

Demonstrates PRSM's marketplace ecosystem for third-party integrations.

**Features Showcased:**
- Developer account registration and tier progression
- Plugin creation, validation, and security scanning
- Marketplace integration publishing
- Monetization setup with freemium pricing models
- Customer review and rating system
- Enterprise marketplace features

**Key Components:**
- Marketplace Core for integration management
- Ecosystem Manager for developer relationships
- Plugin Registry with security validation
- Monetization Engine for pricing and billing
- Review System for community feedback

**Run the Example:**
```bash
cd examples
python phase7_marketplace_example.py
```

## Prerequisites

### System Requirements

- Python 3.11 or higher
- PRSM installation with Phase 7 components
- Access to PRSM configuration (or fallback mode)

### Dependencies

All examples use PRSM's Phase 7 components:

```python
# Analytics Example
from prsm.analytics.dashboard_manager import DashboardManager
from prsm.enterprise.global_infrastructure import GlobalInfrastructure

# Orchestration Example  
from prsm.ai_orchestration.orchestrator import AIOrchestrator
from prsm.nwtn.unified_pipeline_controller import UnifiedPipelineController

# Marketplace Example
from prsm.marketplace.ecosystem import *
```

### Configuration

Examples work with PRSM's safe configuration system:

```python
from prsm.core.config import get_settings_safe

settings = get_settings_safe()
if settings:
    # Use full configuration
    pass
else:
    # Use fallback configuration for demonstration
    pass
```

## Example Output

### Analytics Example Output

```
🎯 Starting Complete PRSM Analytics Demo
==================================================
🚀 Initializing PRSM Analytics System...
✅ Configuration loaded successfully
✅ Analytics system initialized

📊 Creating Executive Dashboard...
✅ Executive dashboard created: dash_67890

🔧 Creating Operational Dashboard...
✅ Operational dashboard created: dash_54321

📈 Generating sample analytics data...
🔄 Updating dashboard data: dash_67890
✅ Dashboard data updated successfully

🧠 Generating analytics insights...
✅ Analytics insights generated

💡 Key Insights Generated:
  Performance Insights:
    • Response times improved by 23% over the last week
    • US-West region showing 15% lower utilization than optimal
  Business Insights:
    • Enterprise users showing 45% higher retention than expected
    • Marketplace revenue grew 34% month-over-month

📄 Creating weekly_summary report...
✅ Custom report generated

📊 Weekly Summary Report:
  • Total Users: 1,247
  • Success Rate: 96.0%
  • Revenue: $125,000.00

⚡ Monitoring real-time metrics for 30 seconds...
🔍 16:00:15 - CPU: 67.0%, Memory: 71.0%, Latency: 12.3ms
🔍 16:00:20 - CPU: 69.0%, Memory: 72.0%, Latency: 11.8ms
✅ Real-time monitoring completed

🎉 Analytics demo completed successfully!
```

### Orchestration Example Output

```
🎯 Starting Complete PRSM AI Orchestration Demo
=======================================================
🚀 Initializing PRSM AI Orchestration System...
✅ AI Orchestration system initialized

🤖 Registering AI Models...
  ✅ Registered: GPT-4 Reasoning Engine -> model_11111
  ✅ Registered: Claude-3.5-Sonnet Analysis -> model_22222
  ✅ Registered: Specialized Math Solver -> model_33333
  ✅ Registered: Code Generation Specialist -> model_44444
✅ 4 AI models registered

🎯 Demonstrating Intelligent Task Routing...
  🎯 Processing task_reasoning_001...
     Task: Analyze the implications of quantum computing on...
     ✅ Completed in 18.4s
     🤖 Model: model_11111
     📊 Confidence: 95.0%
     💰 Cost: 535 FTNS

🤝 Demonstrating Multi-Model Collaboration...
📋 Task: Comprehensive Business Analysis
📝 Subtasks: 4
  ✅ Multi-model collaboration completed
  📊 Overall confidence: 94.0%
  🤖 Models involved: 3
  💰 Total cost: 1247 FTNS

🎉 AI Orchestration demo completed successfully!
```

### Marketplace Example Output

```
🎯 Starting Complete PRSM Marketplace Demo
==================================================
🚀 Initializing PRSM Marketplace Ecosystem...
✅ Marketplace ecosystem ready

👨‍💻 Registering Developer Account...
  ✅ Developer registered: dev_12345
  📊 Initial Tier: BRONZE
  💰 Revenue Share: 70.0%

🔌 Creating and Validating Plugin...
  📝 Plugin: Advanced Analytics Dashboard
  🏷️  Category: analytics
  🔧 Capabilities: 5
  🔍 Validating plugin...
  ✅ Plugin validation passed
  🛡️  Performing security scan...
  📊 Security Score: 95/100
  🎯 Risk Level: low
  ✅ Plugin registered: plugin_67890

🏪 Creating Marketplace Integration...
  ✅ Integration created: int_market_33333

💰 Setting Up Monetization...
  ✅ Pricing plan created: plan_44444
  💳 Free Tier: $0.0
  💎 Professional: $29.99/month
  🏢 Enterprise: $99.99/month
  🆓 Free Trial: 14 days

⭐ Simulating Customer Reviews...
  ⭐ 5/5 - Excellent dashboard solution!
  ⭐ 4/5 - Great value for money
  ⭐ 5/5 - Perfect for our enterprise needs
  ⭐ 4/5 - Solid analytics solution

📊 Review Summary:
     Average Rating: 4.5/5
     Total Reviews: 4
     Recommendation Rate: 100.0%
     Sentiment Score: 0.87

🎉 Marketplace demo completed successfully!
```

## Advanced Usage

### Custom Configuration

You can customize example behavior by modifying configuration parameters:

```python
# Analytics Example - Custom Dashboard Configuration
dashboard_config = {
    'refresh_interval': 60,  # Custom refresh rate
    'auto_refresh': True,
    'theme': 'dark',  # Custom theme
    'widgets': [
        # Add custom widgets
    ]
}

# Orchestration Example - Custom Model Configuration
model_config = {
    'performance_threshold': 0.90,  # Minimum accuracy
    'cost_limit': 1000,  # Maximum FTNS cost
    'timeout': 45,  # Custom timeout
    'preferred_providers': ['openai', 'anthropic']
}

# Marketplace Example - Custom Plugin Configuration
plugin_manifest = {
    'security_level': 'trusted',  # Higher security level
    'resource_requirements': {
        'memory': '1GB',  # More resources
        'cpu': '1.0 cores'
    }
}
```

### Integration with Real Systems

Examples can be extended to work with real systems:

```python
# Real database integration
async def connect_to_real_database(self):
    self.db = await asyncpg.connect("postgresql://...")
    
# Real API integrations
async def call_real_ai_models(self, task):
    if task['type'] == 'reasoning':
        return await openai_client.chat.completions.create(...)
    elif task['type'] == 'analysis':
        return await anthropic_client.messages.create(...)

# Real marketplace transactions
async def process_real_payment(self, subscription_data):
    return await stripe.Subscription.create(...)
```

### Testing and Development

Examples include comprehensive error handling and can be used for testing:

```python
# Run specific example sections
async def test_analytics_only():
    example = AnalyticsExample()
    await example.initialize()
    dashboard_id = await example.create_executive_dashboard()
    return dashboard_id

# Mock external dependencies for testing
@patch('prsm.ai_orchestration.orchestrator.AIOrchestrator')
async def test_orchestration_with_mocks():
    # Test with mocked dependencies
    pass
```

## Troubleshooting

### Common Issues

1. **Configuration Issues**
   ```
   ⚠️  Using fallback configuration
   ```
   - This is normal for demonstration purposes
   - Examples work with or without full PRSM configuration

2. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'prsm'
   ```
   - Ensure PRSM is properly installed
   - Run examples from the repository root directory

3. **Async Runtime Errors**
   ```
   RuntimeError: asyncio.run() cannot be called from a running event loop
   ```
   - Examples are designed to run as standalone scripts
   - Use `python filename.py` rather than importing in Jupyter

### Getting Help

- Check the [Phase 7 Documentation](../docs/architecture/PHASE_7_ENTERPRISE_ARCHITECTURE.md)
- Review the [API Reference](../docs/api/PHASE_7_API_REFERENCE.md)
- Visit the [Marketplace Guide](../docs/architecture/marketplace/MARKETPLACE_ECOSYSTEM_GUIDE.md)

## Contributing

To contribute new examples:

1. Follow the existing example structure
2. Include comprehensive documentation
3. Add error handling and fallback modes
4. Test with both full and fallback configurations
5. Update this README with your example

## License

These examples are part of the PRSM project and are licensed under the MIT License.