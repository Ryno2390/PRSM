"""
PRSM Python SDK Setup
Official Python client for the Protocol for Recursive Scientific Modeling
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from version file safely
import importlib.util
version_file = os.path.join(os.path.dirname(__file__), "prsm_sdk", "__version__.py")
spec = importlib.util.spec_from_file_location("version", version_file)
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)
__version__ = version_module.__version__

setup(
    name="prsm-python-sdk",
    version=__version__,  # noqa: F821
    author="PRSM Development Team",
    author_email="dev@prsm.ai",
    description="Official Python SDK for PRSM (Protocol for Recursive Scientific Modeling)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PRSM-AI/PRSM",
    project_urls={
        "Bug Tracker": "https://github.com/PRSM-AI/PRSM/issues",
        "Documentation": "https://docs.prsm.ai/python-sdk",
        "Source Code": "https://github.com/PRSM-AI/PRSM/tree/main/sdks/python",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "pydantic>=2.0.0",
        "structlog>=23.0.0",
        "websockets>=11.0.0",
        "cryptography>=41.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
    },
    keywords=[
        "prsm", "ai", "machine-learning", "distributed-computing", 
        "blockchain", "scientific-modeling", "p2p", "ftns"
    ],
    entry_points={
        "console_scripts": [
            "prsm=prsm_sdk.cli:main",
        ],
    },
)