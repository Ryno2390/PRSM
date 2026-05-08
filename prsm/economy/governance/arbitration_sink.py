"""PRSM-PROV-1 T6.5.gov.next — TokenWeightedVoting arbitration sink.

Production ``ArbitrationProposalSink`` implementation that wraps
``TokenWeightedVoting.create_proposal`` so the disputed-band
arbitration queue (``prsm/data/dedup/arbitration.py``) can surface
records as governance proposals of category
``ProposalCategory.ARBITRATION_DISPUTE``.

Lives under ``prsm/economy/governance/`` rather than the dedup
module so the dedup tier doesn't pull governance imports — keeps
``prsm/data/dedup/`` deployable on stripped-down nodes that don't
have an FTNS ledger or a voting backend.

Operator guidance:
  - ``proposer_id`` should be a Foundation-controlled address with
    sufficient FTNS to cover ``PROPOSAL_SUBMISSION_FEE`` per call.
    The uploader cannot be the proposer (would create the dispute
    they're a party to); the candidate parent's creator cannot be
    either (same conflict of interest). System-level proposer is
    the right pattern.
  - Sink failures (insufficient FTNS, eligibility rejection,
    backend unavailable) MUST surface as ``None`` per the
    ``ArbitrationProposalSink`` Protocol contract — the upload's
    arbitration record is still queued; the linkage to a council
    vote is what's lost. Operators should monitor logs for repeated
    failures (suggests Foundation address out of FTNS) and refill.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from prsm.core.models import GovernanceProposal as CoreGovernanceProposal
from prsm.data.dedup.arbitration import (
    DisputedAttributionRecord,
    render_arbitration_body,
)
from prsm.economy.governance.voting import ProposalCategory

logger = logging.getLogger(__name__)


# Truncate CIDs in the proposal title to 12 chars so council-UI list
# views stay readable. The full CIDs are in the description body.
_TITLE_CID_PREFIX_LEN = 12


class TokenWeightedVotingProposalSink:
    """ArbitrationProposalSink wrapping TokenWeightedVoting.

    Constructor:
        voting: A ``TokenWeightedVoting`` instance (or any object
            exposing ``async create_proposal(proposer_id, proposal)``
            returning a ``UUID``).
        proposer_id: Address of the system proposer that submits the
            arbitration proposal on behalf of the network. Must hold
            sufficient FTNS balance to cover the proposal submission
            fee. Recommended: the Foundation Safe address or a
            delegate.
    """

    def __init__(
        self,
        *,
        voting: Any,
        proposer_id: str,
    ) -> None:
        if voting is None:
            raise ValueError("voting backend must not be None")
        if not isinstance(proposer_id, str) or not proposer_id:
            raise ValueError(
                "proposer_id must be a non-empty string"
            )
        self._voting = voting
        self._proposer_id = proposer_id

    async def create_arbitration_proposal(
        self,
        record: DisputedAttributionRecord,
        record_id: str,
    ) -> Optional[str]:
        """Build a ``GovernanceProposal`` of category
        ``ARBITRATION_DISPUTE`` and submit it to the voting backend.

        Returns the str-formatted proposal UUID on success, or
        ``None`` on any backend failure. Per the
        ``ArbitrationProposalSink`` contract, this method MUST NOT
        raise.
        """
        title = (
            f"Disputed attribution: "
            f"{record.new_cid[:_TITLE_CID_PREFIX_LEN]}… vs "
            f"{record.candidate_parent_cid[:_TITLE_CID_PREFIX_LEN]}…"
        )
        proposal = CoreGovernanceProposal(
            proposer_id=self._proposer_id,
            title=title,
            description=render_arbitration_body(record),
            proposal_type=ProposalCategory.ARBITRATION_DISPUTE.value,
            metadata={
                "arbitration_record_id": record_id,
                "fingerprint_kind": record.fingerprint_kind,
                "similarity": record.similarity,
                "new_cid": record.new_cid,
                "candidate_parent_cid": record.candidate_parent_cid,
            },
        )
        try:
            proposal_uuid = await self._voting.create_proposal(
                self._proposer_id, proposal,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TokenWeightedVotingProposalSink: backend rejected "
                "arbitration proposal (record_id=%s): %s",
                record_id,
                exc,
            )
            return None
        if proposal_uuid is None:
            return None
        return str(proposal_uuid)
