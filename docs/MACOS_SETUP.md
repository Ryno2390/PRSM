# macOS Setup Guide for PRSM

## Prerequisites

### 1. Install Homebrew (if not already installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Install Required System Dependencies
```bash
# Install LLVM and OpenMP support
brew install llvm libomp

# Install BLAS/LAPACK libraries
brew install openblas

# Install Python build tools
brew install python@3.11 cmake pkg-config
```

## Environment Setup

### Option 1: Use Pre-compiled Wheels (Recommended)
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip and install wheel support
pip install --upgrade pip wheel setuptools

# Install scipy with pre-compiled wheels (avoids compilation issues)
pip install --only-binary=scipy,scikit-learn,numpy scipy scikit-learn numpy

# Install remaining dependencies
pip install -r requirements.txt
```

### Option 2: Build from Source (Advanced)
If you need to build from source, set these environment variables:

```bash
# Add to ~/.zshrc or ~/.bash_profile
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
export LDFLAGS="-L$(brew --prefix openblas)/lib -L$(brew --prefix libomp)/lib"
export CPPFLAGS="-I$(brew --prefix openblas)/include -I$(brew --prefix libomp)/include"
export PKG_CONFIG_PATH="$(brew --prefix openblas)/lib/pkgconfig"
export PATH="$(brew --prefix llvm)/bin:$PATH"

# Reload shell configuration
source ~/.zshrc  # or ~/.bash_profile

# Create clean virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install with verbose output to debug issues
pip install --upgrade pip wheel setuptools
pip install numpy  # Install numpy first
pip install scipy --verbose  # Install scipy with build details
pip install -r requirements.txt
```

## Alternative: Use Conda (Most Reliable)
```bash
# Install miniconda
brew install --cask miniconda

# Create conda environment
conda create -n prsm python=3.11
conda activate prsm

# Install scientific packages via conda (pre-compiled)
conda install numpy scipy scikit-learn pandas

# Install remaining packages via pip
pip install -r requirements.txt
```

## Troubleshooting

### If you still get OpenMP errors:
```bash
# Force install OpenMP-compatible scipy
pip uninstall scipy
conda install scipy  # Use conda version

# Or use Intel MKL optimized version
pip install scipy-mkl
```

### For Apple Silicon (M1/M2) Macs:
```bash
# Use ARM64 optimized packages
pip install --upgrade pip
pip install --only-binary=:all: scipy scikit-learn numpy
```

### Verify Installation
```bash
python -c "import scipy; print(scipy.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
python -c "import numpy; print(numpy.__version__)"
```

## Quick Start (Recommended)
```bash
# Clone repository
git clone https://github.com/Ryno2390/PRSM.git
cd PRSM

# Use conda for easiest setup
conda create -n prsm python=3.11
conda activate prsm
conda install numpy scipy scikit-learn pandas
pip install -r requirements.txt

# Verify
python -c "import prsm; print('PRSM setup successful!')"
```

This approach avoids compilation issues by using pre-built scientific computing libraries.