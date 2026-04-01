# PRSM Compute Skill

Compute job management on the PRSM decentralized network.

## Overview

This skill enables AI agents to submit, monitor, and manage compute jobs
across the PRSM distributed network. Jobs are executed on peer nodes that
contribute GPU/CPU resources, with costs denominated in FTNS tokens.

## Tools

| Tool | Description |
|------|-------------|
| `prsm_submit_job` | Submit training, inference, or evaluation jobs to the network |
| `prsm_job_status` | Check progress, metrics, and logs for a running job |
| `prsm_cancel_job` | Cancel a queued or running job |
| `prsm_list_queue` | View all jobs with filtering and sorting |

## Prompts

- **scheduler** — System prompt for an AI compute job management agent

## Example Usage

```
Submit a training job:
  prsm_submit_job(name="gpt2-finetune", task_type="training", dataset="ds-001", config={"epochs": 3, "lr": 1e-4})

Check job progress:
  prsm_job_status(job_id="job-abc123", verbose=true)

View running jobs:
  prsm_list_queue(status="running", sort_by="submitted")

Cancel a job:
  prsm_cancel_job(job_id="job-abc123", reason="Converged early")
```

## Resource Pricing

Compute resources are priced in FTNS tokens based on:
- GPU type and quantity (e.g., A100 > T4)
- Duration of execution
- Network demand (dynamic pricing)
- Priority level (high priority costs more)
