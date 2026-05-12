"""Sprint 318a — slim prsm.compute.inference init for the
demo container.

The repo-canonical __init__.py eagerly imports executor /
multi_stage_attestation / receipt / content_tier_gate /
etc., which transitively pull prsm.node, prsm.economy,
prsm.storage — exactly the deps the slim demo image
avoids.

This file is COPIED OVER prsm/compute/inference/__init__.py
inside the Docker build (see Dockerfile). The demo image
only needs the sprint-312/313/314/315/316/316a pipeline
modules to be importable on demand — not eagerly re-exported
at package level.
"""
