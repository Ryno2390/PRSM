from setuptools import setup, find_packages

setup(
    name="prsm",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "asyncpg>=0.29.0",
        "redis>=5.0.0",
        "aiohttp>=3.9.0",
        "structlog>=23.0.0",
        "psutil>=5.9.0",
        "bleach>=6.0.0",
        "html5lib>=1.1",
        "python-multipart>=0.0.6",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "pydantic-settings>=2.0.0"
    ],
    python_requires=">=3.9",
    description="PRSM - Production-Ready Semantic Marketplace",
    author="PRSM Development Team",
    package_data={
        "prsm": ["*.py", "*/*.py", "*/*/*.py"]
    },
    include_package_data=True,
)