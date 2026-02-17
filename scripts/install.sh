#!/bin/bash
# PRSM Installer
# Sets up a local PRSM development environment

set -e

echo "PRSM: Protocol for Recursive Scientific Modeling"
echo "Setting up local development environment..."
echo ""

# 1. OS Detection
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

echo "Detected OS: ${MACHINE}"

# 2. Check for Python 3.11+
if ! [ -x "$(command -v python3)" ]; then
  echo "Error: Python 3 is not installed. Please install Python 3.11 or higher." >&2
  exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: ${PYTHON_VERSION}"

# 3. Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 4. Upgrade pip and install PRSM
echo "Installing PRSM..."
pip install --upgrade pip
pip install -e .
echo "PRSM installed successfully."

# 5. Set up configuration
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit it with your settings."
  fi
fi

echo ""
echo "--------------------------------------------------------"
echo "Installation complete!"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the PRSM API server:"
echo "  prsm serve"
echo ""
echo "To verify it's running:"
echo "  curl http://localhost:8000/health"
echo "--------------------------------------------------------"
