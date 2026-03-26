"""
prsm.compute.nwtn.synthesis — Nightly Synthesis & Project Ledger
=================================================================

Sub-phase 10.4: transforms the messy, ad-hoc Active Whiteboard into a
coherent, tamper-evident, accumulating Project Ledger.

Components
----------
NarrativeSynthesizer / SynthesisResult
    Re-constructor Agent.  Reads a ``WhiteboardSnapshot`` and synthesises
    it into a structured Markdown narrative (LLM-driven or template
    fallback).  Solves the "disconnected bullet points" problem — agents
    coming online in a fresh session read one coherent document, not a
    dump of raw BSC-filtered fragments.

ProjectLedger / LedgerEntry
    Append-only, tamper-evident ledger backed by a Markdown file (humans)
    and a JSON sidecar (machine verification).  Each entry is Ed25519-
    signed and SHA-256 hash-chained — modifying any historical entry
    breaks all subsequent signatures.

LedgerSigner / EntrySignature / VerificationResult
    Ed25519 signing and chain verification, built on PRSM's existing
    ``dag_signatures`` infrastructure.  A ``LedgerSigner`` manages one
    keypair per project; the public key travels with every entry so any
    verifier can check without a PKI server.

DAGAnchor / AnchorReceipt
    Optional PRSM DAG anchoring for major milestone entries.  Gracefully
    no-ops when the PRSM network is unavailable.

Typical end-of-session flow
---------------------------
.. code-block:: python

    from prsm.compute.nwtn.synthesis import (
        NarrativeSynthesizer, ProjectLedger, LedgerSigner, DAGAnchor,
    )
    from prsm.compute.nwtn.whiteboard import WhiteboardQuery

    # 1. Get the session snapshot from the whiteboard
    query = WhiteboardQuery(store)
    snapshot = await query.snapshot("sess-001")

    # 2. Synthesise
    synthesizer = NarrativeSynthesizer(backend_registry=backend)
    synthesis = await synthesizer.synthesise(snapshot, meta_plan=meta_plan)

    # 3. Sign and append to ledger
    signer = LedgerSigner(keyfile_path=Path(".prsm/ledger.key"))
    ledger = ProjectLedger(ledger_dir=Path(".prsm/ledger"), project_title="PRSM")
    ledger.load()
    entry = ledger.append(synthesis, signer)

    # 4. Optionally anchor to PRSM DAG (major milestones only)
    anchor = DAGAnchor(dag_ledger=dag_ledger, wallet_id="my-wallet")
    receipt = await anchor.anchor(entry.entry_index, entry.chain_hash, "sess-001")
    if receipt.success:
        ledger.update_dag_anchor(entry.entry_index, receipt.dag_tx_id)

    # 5. Verify the chain is intact
    result = ledger.verify()
    print(result)  # "Chain OK (3 entries verified)"

    # 6. New agents onboard from the ledger
    context = ledger.to_onboarding_context()
"""

from .dag_anchor import AnchorReceipt, DAGAnchor
from .ledger import LedgerEntry, ProjectLedger
from .reconstructor import NarrativeSynthesizer, SynthesisResult
from .signer import EntrySignature, LedgerSigner, VerificationResult, hash_content, GENESIS_HASH

__all__ = [
    # Synthesiser
    "NarrativeSynthesizer",
    "SynthesisResult",
    # Ledger
    "ProjectLedger",
    "LedgerEntry",
    # Signer
    "LedgerSigner",
    "EntrySignature",
    "VerificationResult",
    "hash_content",
    "GENESIS_HASH",
    # DAG anchor
    "DAGAnchor",
    "AnchorReceipt",
]
