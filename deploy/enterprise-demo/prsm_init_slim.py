"""Sprint 318a — slim package init for the enterprise
demo container.

The repo-canonical prsm/__init__.py imports prsm.core
which transitively pulls in DB, FTNS config, governance,
etc. — exactly the deps the slim demo image avoids.

This file is COPIED OVER prsm/__init__.py inside the
Docker build (see Dockerfile). The runtime image only
needs prsm.enterprise.* + prsm.compute.* importable.
"""
__version__ = "1.7.0-enterprise-demo"
__description__ = (
    "PRSM Enterprise Demo — §7 Confidentiality Mode + "
    "Federated Inference (slim demo image)"
)
