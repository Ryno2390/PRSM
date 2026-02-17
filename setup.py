from setuptools import setup, find_packages

setup(
    name="prsm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "email-validator>=2.0.0",
        "sqlalchemy>=2.0.0",
        "redis>=5.0.0",
        "aiohttp>=3.9.0",
        "structlog>=23.0.0",
        "rich>=13.7.0",
        "click>=8.1.0",
        "python-dotenv>=1.0.0",
        "httpx>=0.25.0",
        "cryptography>=41.0.0",
        "pyjwt>=2.8.0",
        "bcrypt>=4.1.0",
        "prometheus-client>=0.19.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.11",
    description="PRSM - Protocol for Recursive Scientific Modeling",
    author="PRSM Team",
    entry_points={
        "console_scripts": [
            "prsm=prsm.cli:main",
        ],
    },
    package_data={
        "prsm": ["*.py", "*/*.py", "*/*/*.py"]
    },
    include_package_data=True,
)