# PRSM Tutorial Examples

This directory contains runnable examples for all PRSM tutorials.

## 📚 Available Examples

### Quick Start Examples
- `hello_world_complete.py` - Complete Hello World demonstration with all features

### Foundation Examples (Coming Soon)
- `concepts_explorer.py` - Interactive PRSM architecture exploration
- `api_examples.py` - REST API and SDK usage examples
- `configuration_demo.py` - Environment and deployment configuration

### Development Examples (Coming Soon)
- `custom_teacher.py` - Build and deploy custom teacher models
- `tool_integration.py` - Add external tools and capabilities
- `local_development.py` - Local development and debugging

## 🚀 Quick Start

1. **Setup PRSM Environment**:
   ```bash
   prsm-dev setup
   ```

2. **Run Hello World Example**:
   ```bash
   cd examples/tutorials
   python hello_world_complete.py
   ```

3. **Verify Everything Works**:
   ```bash
   prsm-dev status
   ```

## 🎯 Learning Path

Follow the examples in this order:

1. **hello_world_complete.py** - Your first PRSM experience
2. **concepts_explorer.py** - Understand PRSM architecture  
3. **api_examples.py** - Learn API integration
4. **custom_teacher.py** - Build specialized AI models

## 🔧 Prerequisites

All examples require:
- PRSM development environment setup
- API keys configured in `config/api_keys.env`
- Docker services running (Redis, IPFS)

## 🆘 Troubleshooting

If examples fail:

1. **Check Setup**: `prsm-dev diagnose`
2. **Verify Services**: `prsm-dev status`
3. **Review Logs**: Check console output for specific errors
4. **Get Help**: See [Troubleshooting Guide](../../docs/TROUBLESHOOTING_GUIDE.md)

---

**Ready to learn?** Start with `hello_world_complete.py`!