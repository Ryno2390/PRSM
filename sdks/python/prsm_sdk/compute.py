"""
PRSM SDK Compute Client
Submit and manage compute jobs on the PRSM network
"""

import structlog
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

from .exceptions import PRSMError, NetworkError

logger = structlog.get_logger(__name__)


class JobStatus(str, Enum):
    """Status of a compute job"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class JobPriority(str, Enum):
    """Priority levels for compute jobs"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class JobRequest(BaseModel):
    """Request to submit a compute job"""
    prompt: str = Field(..., description="The prompt/query to process")
    model: str = Field("nwtn", description="Model to use for computation")
    max_tokens: int = Field(1000, ge=1, description="Maximum tokens in response")
    temperature: float = Field(0.7, ge=0, le=2, description="Response randomness")
    budget: Optional[float] = Field(None, description="Maximum FTNS to spend")
    priority: JobPriority = Field(JobPriority.NORMAL, description="Job priority")
    timeout: int = Field(300, description="Job timeout in seconds")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    tools: Optional[List[str]] = Field(None, description="Tools to enable")
    stream: bool = Field(False, description="Enable streaming response")


class JobResponse(BaseModel):
    """Response from job submission"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation time")
    estimated_cost: float = Field(..., description="Estimated FTNS cost")
    estimated_duration: float = Field(..., description="Estimated duration in seconds")
    queue_position: Optional[int] = Field(None, description="Position in queue")


class JobResult(BaseModel):
    """Result of a completed job"""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Final job status")
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    execution_time: float = Field(..., description="Execution time in seconds")
    token_usage: Dict[str, int] = Field(..., description="Token usage breakdown")
    ftns_cost: float = Field(..., description="Actual FTNS cost")
    reasoning_trace: Optional[List[str]] = Field(None, description="Reasoning steps")
    citations: Optional[List[Dict[str, Any]]] = Field(None, description="Citations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    completed_at: datetime = Field(..., description="Completion time")


class JobInfo(BaseModel):
    """Detailed information about a job"""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current status")
    request: JobRequest = Field(..., description="Original request")
    result: Optional[JobResult] = Field(None, description="Result if completed")
    progress: float = Field(0, ge=0, le=1, description="Progress (0-1)")
    created_at: datetime = Field(..., description="Creation time")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    error: Optional[str] = Field(None, description="Error message if failed")
    node_id: Optional[str] = Field(None, description="Processing node ID")


class JobListResponse(BaseModel):
    """Response from listing jobs"""
    jobs: List[JobInfo] = Field(default_factory=list, description="List of jobs")
    total: int = Field(..., description="Total number of jobs")
    offset: int = Field(..., description="Current offset")
    limit: int = Field(..., description="Current limit")


class ComputeClient:
    """
    Client for compute operations on PRSM network
    
    Provides methods for:
    - Submitting compute jobs
    - Monitoring job status
    - Retrieving results
    - Managing job lifecycle
    """
    
    def __init__(self, client):
        """
        Initialize compute client
        
        Args:
            client: PRSMClient instance for making API requests
        """
        self._client = client
    
    async def submit_job(
        self,
        prompt: str,
        model: str = "nwtn",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        budget: Optional[float] = None,
        priority: JobPriority = JobPriority.NORMAL,
        timeout: int = 300,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None
    ) -> JobResponse:
        """
        Submit a compute job to the network
        
        Args:
            prompt: The prompt/query to process
            model: Model to use (default: nwtn)
            max_tokens: Maximum tokens in response
            temperature: Response randomness (0-2)
            budget: Maximum FTNS to spend
            priority: Job priority level
            timeout: Job timeout in seconds
            context: Additional context
            tools: Tools to enable
            
        Returns:
            JobResponse with job ID and status
            
        Example:
            response = await client.compute.submit_job(
                prompt="Explain quantum computing",
                model="nwtn",
                max_tokens=2000,
                budget=5.0
            )
            print(f"Job submitted: {response.job_id}")
        """
        request = JobRequest(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            budget=budget,
            priority=priority,
            timeout=timeout,
            context=context,
            tools=tools
        )
        
        response = await self._client._request(
            "POST",
            "/compute/jobs",
            json_data=request.model_dump(exclude_none=True)
        )
        
        return JobResponse(**response)
    
    async def get_job(self, job_id: str) -> JobInfo:
        """
        Get detailed information about a job
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobInfo with job details
            
        Raises:
            PRSMError: If job not found
            
        Example:
            job = await client.compute.get_job("job_123")
            print(f"Status: {job.status}, Progress: {job.progress * 100}%")
        """
        response = await self._client._request(
            "GET",
            f"/compute/jobs/{job_id}"
        )
        
        return JobInfo(**response)
    
    async def get_result(self, job_id: str) -> JobResult:
        """
        Get the result of a completed job
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobResult with generated content
            
        Raises:
            PRSMError: If job not complete
            
        Example:
            result = await client.compute.get_result("job_123")
            print(f"Content: {result.content}")
            print(f"Cost: {result.ftns_cost} FTNS")
        """
        response = await self._client._request(
            "GET",
            f"/compute/jobs/{job_id}/result"
        )
        
        return JobResult(**response)
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancelled successfully
            
        Example:
            cancelled = await client.compute.cancel_job("job_123")
            if cancelled:
                print("Job cancelled")
        """
        response = await self._client._request(
            "POST",
            f"/compute/jobs/{job_id}/cancel"
        )
        
        return response.get("cancelled", False)
    
    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 10,
        offset: int = 0
    ) -> JobListResponse:
        """
        List recent jobs
        
        Args:
            status: Filter by status
            limit: Maximum results
            offset: Offset for pagination
            
        Returns:
            JobListResponse with list of jobs
            
        Example:
            jobs = await client.compute.list_jobs(status=JobStatus.RUNNING)
            for job in jobs.jobs:
                print(f"{job.job_id}: {job.status}")
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status.value
        
        response = await self._client._request(
            "GET",
            "/compute/jobs",
            params=params
        )
        
        return JobListResponse(**response)
    
    async def wait_for_completion(
        self,
        job_id: str,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
        on_progress: Optional[callable] = None
    ) -> JobResult:
        """
        Wait for a job to complete
        
        Args:
            job_id: Job identifier
            timeout: Maximum wait time in seconds
            poll_interval: Polling interval in seconds
            on_progress: Optional callback for progress updates
            
        Returns:
            JobResult when complete
            
        Raises:
            TimeoutError: If timeout exceeded
            PRSMError: If job fails
            
        Example:
            result = await client.compute.wait_for_completion(
                "job_123",
                timeout=300,
                on_progress=lambda j: print(f"Progress: {j.progress * 100}%")
            )
        """
        import asyncio
        
        start_time = asyncio.get_event_loop().time()
        
        while True:
            job = await self.get_job(job_id)
            
            if on_progress:
                on_progress(job)
            
            if job.status == JobStatus.COMPLETED:
                return await self.get_result(job_id)
            
            if job.status in (JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT):
                raise PRSMError(f"Job {job_id} ended with status: {job.status}")
            
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Timeout waiting for job {job_id}")
            
            await asyncio.sleep(poll_interval)
    
    async def estimate_cost(
        self,
        prompt: str,
        model: str = "nwtn",
        max_tokens: int = 1000
    ) -> float:
        """
        Estimate the FTNS cost for a job
        
        Args:
            prompt: The prompt to estimate
            model: Model to use
            max_tokens: Maximum tokens
            
        Returns:
            Estimated cost in FTNS
        """
        response = await self._client._request(
            "POST",
            "/compute/estimate",
            json_data={
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens
            }
        )
        
        return response.get("estimated_cost", 0.0)
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status
        
        Returns:
            Dictionary with queue statistics
        """
        response = await self._client._request(
            "GET",
            "/compute/queue/status"
        )
        
        return response
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available compute models
        
        Returns:
            List of model information dictionaries
        """
        response = await self._client._request(
            "GET",
            "/compute/models"
        )
        
        return response.get("models", [])