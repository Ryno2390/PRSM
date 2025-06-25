# PRSM Developer Tools

This directory contains developer tools and utilities to enhance your PRSM development experience.

## Available Tools

### 🏃 Code Runner (`code_runner.py`)
Interactive code execution environment for testing PRSM code snippets.

**Features:**
- Syntax highlighting and error checking
- Real-time execution with output capture
- Integration with PRSM components
- Save and share code snippets

**Usage:**
```bash
python tools/code_runner.py
```

### ⚙️ Configuration Generator (`config_generator.py`)
Generate PRSM configuration files for different deployment scenarios.

**Features:**
- Templates for development, staging, production
- Security configuration wizards
- Performance optimization settings
- Validation and testing tools

**Usage:**
```bash
python tools/config_generator.py --template production --output config.yaml
```

### 🔧 Performance Tester (`performance_tester.py`)
Benchmark and test PRSM applications for performance optimization.

**Features:**
- Load testing for P2P networks
- AI model performance benchmarking
- Resource usage monitoring
- Comparative analysis tools

**Usage:**
```bash
python tools/performance_tester.py --test network --nodes 5 --duration 60
```

### 📝 Template Generator (`template_generator.py`)
Generate project templates and boilerplate code for PRSM applications.

**Features:**
- Multiple project templates
- Custom template creation
- Dependency management
- Best practices integration

**Usage:**
```bash
python tools/template_generator.py --template enterprise_app --output my_app
```

## Tool Development Guidelines

### Adding New Tools

1. **Create Tool File**: Add your tool in this directory
2. **Follow Naming Convention**: Use snake_case for file names
3. **Include Documentation**: Add comprehensive docstrings and README updates
4. **Implement Help**: Support `--help` command-line argument
5. **Error Handling**: Implement robust error handling and user feedback
6. **Testing**: Include test cases and validation

### Tool Template

```python
#!/usr/bin/env python3
"""
Tool Name - Brief Description
Detailed description of what this tool does and how to use it.
"""

import argparse
import logging
from pathlib import Path

def main():
    """Main tool function"""
    parser = argparse.ArgumentParser(description="Tool description")
    parser.add_argument("--input", help="Input parameter")
    parser.add_argument("--output", help="Output parameter")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Tool implementation here
    print("Tool executed successfully!")

if __name__ == "__main__":
    main()
```

## Integration with Playground

All tools are integrated with the main playground launcher:

```bash
# Access tools through launcher
python playground_launcher.py --tool performance_tester --args "--test network"

# Or run directly
python tools/performance_tester.py --test network
```

## Contributing

We welcome contributions to the PRSM developer tools! Please see the main contributing guidelines for details on how to submit improvements and new tools.