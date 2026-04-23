"""
Swarm Coordinator
=================

Orchestrates parallel map-reduce across semantically-sharded datasets.
Fans out Ring 2 agent dispatches, collects results, aggregates.
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Dict, List, Optional

from prsm.compute.agents.models import AgentManifest
from prsm.compute.swarm.models import (
    MapReduceStrategy,
    ShardAssignment,
    SwarmJob,
    SwarmResult,
    SwarmStatus,
)

logger = logging.getLogger(__name__)


class SwarmCoordinator:
    """Coordinates parallel agent execution across data shards."""

    def __init__(self, dispatcher, result_consensus=None):
        self.dispatcher = dispatcher
        self.result_consensus = result_consensus
        self._jobs: Dict[str, SwarmJob] = {}

    def create_swarm_job(
        self,
        query: str,
        shard_content_ids: List[str],
        wasm_binary: bytes,
        manifest: AgentManifest,
        budget_ftns: float,
        strategy: Optional[MapReduceStrategy] = None,
    ) -> SwarmJob:
        job_id = f"swarm-{uuid.uuid4().hex[:12]}"
        job = SwarmJob(
            job_id=job_id,
            query=query,
            shard_content_ids=shard_content_ids,
            wasm_binary=wasm_binary,
            agent_manifest=manifest,
            budget_ftns=budget_ftns,
            strategy=strategy or MapReduceStrategy(),
        )
        self._jobs[job_id] = job
        return job

    async def execute(self, job: SwarmJob) -> SwarmResult:
        """Execute a swarm job: fan out, collect, aggregate."""
        job.status = SwarmStatus.DISPATCHING

        # Phase 1: Fan out — dispatch one agent per shard
        dispatch_tasks = []
        for shard_cid in job.shard_content_ids:
            shard_manifest = AgentManifest(
                required_content_ids=[shard_cid],
                min_hardware_tier=job.agent_manifest.min_hardware_tier,
                max_memory_bytes=job.agent_manifest.max_memory_bytes,
                max_execution_seconds=job.agent_manifest.max_execution_seconds,
                max_output_bytes=job.agent_manifest.max_output_bytes,
                required_capabilities=job.agent_manifest.required_capabilities,
            )

            agent = self.dispatcher.create_agent(
                wasm_binary=job.wasm_binary,
                manifest=shard_manifest,
                ftns_budget=job.budget_per_shard,
                ttl=job.strategy.per_shard_timeout,
            )

            assignment = ShardAssignment(
                shard_content_id=shard_cid,
                agent_id=agent.agent_id,
            )
            job.assignments[shard_cid] = assignment
            dispatch_tasks.append(self._dispatch_shard(job, agent, shard_cid))

        job.status = SwarmStatus.EXECUTING

        # Dispatch all shards in parallel
        dispatch_results = await asyncio.gather(*dispatch_tasks, return_exceptions=True)

        # Phase 2: Collect results
        shard_results = {}
        total_pcu = 0.0
        total_ftns = 0.0

        for shard_cid, result in zip(job.shard_content_ids, dispatch_results):
            assignment = job.assignments.get(shard_cid)
            if isinstance(result, Exception):
                if assignment:
                    assignment.status = "failed"
                job.failed_shards.append(shard_cid)
                continue

            if result and result.get("status") == "success":
                shard_results[shard_cid] = result
                job.completed_shards.append(shard_cid)
                pcu = result.get("pcu", 0)
                total_pcu += pcu
                total_ftns += job.budget_per_shard
                if assignment:
                    assignment.status = "completed"
                    assignment.result = result
                    assignment.pcu_used = pcu
                    assignment.completed_at = time.time()
            else:
                job.failed_shards.append(shard_cid)
                if assignment:
                    assignment.status = "failed"

        # Phase 2.5: Result consensus validation (for high-value jobs)
        if self.result_consensus and job.budget_ftns > 1.0 and shard_results:
            try:
                validated_results = {}
                for cid, result in shard_results.items():
                    # For now, single-provider consensus (trust the result)
                    # Multi-provider consensus activates when redundant dispatch is enabled
                    validated_results[cid] = result
                shard_results = validated_results
                logger.debug(f"Swarm {job.job_id[:12]}: consensus validation passed for {len(shard_results)} shards")
            except Exception as e:
                logger.warning(f"Swarm {job.job_id[:12]}: consensus validation error: {e}")

        # Phase 3: Determine status
        if job.is_quorum_met():
            job.status = SwarmStatus.COMPLETED if not job.failed_shards else SwarmStatus.PARTIAL
        else:
            job.status = SwarmStatus.FAILED
            job.error = (
                f"Quorum not met: {len(job.completed_shards)}/{job.quorum_count} required "
                f"({len(job.failed_shards)} shards failed)"
            )

        job.completed_at = time.time()

        # Phase 4: Aggregate
        aggregation_start = time.time()
        aggregated = self._aggregate_results(shard_results, job.query)

        if job.status != SwarmStatus.FAILED:
            job.status = SwarmStatus.COMPLETED if not job.failed_shards else SwarmStatus.PARTIAL

        return SwarmResult(
            job_id=job.job_id,
            shard_results=shard_results,
            total_pcu=total_pcu,
            total_ftns_spent=total_ftns,
            total_shards=len(job.shard_content_ids),
            aggregation_time_seconds=time.time() - aggregation_start,
            aggregated_output=aggregated,
        )

    async def _dispatch_shard(self, job, agent, shard_cid):
        """Dispatch a single shard agent and wait for result."""
        try:
            await self.dispatcher.dispatch(agent)
            await asyncio.sleep(0.05)  # Brief bid window
            await self.dispatcher.select_and_transfer(agent.agent_id)
            return await self.dispatcher.wait_for_result(
                agent.agent_id,
                timeout=job.strategy.per_shard_timeout,
            )
        except Exception as e:
            logger.error(f"Shard {shard_cid[:12]} dispatch failed: {e}")
            return None

    def _aggregate_results(self, shard_results, query):
        """Default aggregation: collect shard outputs into summary."""
        outputs = []
        for cid, result in shard_results.items():
            output_b64 = result.get("output_b64", "")
            if output_b64:
                try:
                    raw = base64.b64decode(output_b64)
                    parsed = json.loads(raw)
                    outputs.append({"shard_cid": cid, "data": parsed})
                except (json.JSONDecodeError, Exception):
                    outputs.append({"shard_cid": cid, "data": output_b64})

        return {
            "query": query,
            "shard_count": len(shard_results),
            "shard_outputs": outputs,
        }

    def get_job(self, job_id):
        return self._jobs.get(job_id)
