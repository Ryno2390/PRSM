# PRSM Datasets Skill

Dataset curation and management on the PRSM decentralized network.

## Overview

This skill enables AI agents to search, curate, validate, and publish datasets
across the PRSM network. It provides tools for discovering existing datasets,
combining and filtering data from multiple sources, running quality checks,
and publishing curated datasets for others to use.

## Tools

| Tool | Description |
|------|-------------|
| `prsm_search_datasets` | Search the network for datasets by query, domain, quality, and format |
| `prsm_curate_dataset` | Create new datasets by combining and filtering existing sources |
| `prsm_validate_dataset` | Run quality, schema, duplicate, and bias checks on datasets |
| `prsm_publish_dataset` | Publish datasets to the network with metadata and pricing |

## Prompts

- **curator** — System prompt for an AI dataset curation agent
- **analyst** — System prompt for a data quality analysis agent

## Example Usage

```
Search for high-quality NLP datasets in parquet format:
  prsm_search_datasets(query="transformer training data", domain="nlp", min_quality=0.8)

Curate a new code dataset:
  prsm_curate_dataset(name="code-python-v1", sources=["ds-001", "ds-042"], filters={"language": "python"})

Validate before publishing:
  prsm_validate_dataset(dataset_id="curated-001", checks=["schema", "duplicates", "quality"])

Publish to the network:
  prsm_publish_dataset(dataset_id="curated-001", metadata={"license": "MIT", "domain": "code"})
```

## FTNS Token Economy

Dataset operations interact with the PRSM token economy:
- **Searching** is free for all network participants
- **Curating** consumes compute resources priced in FTNS
- **Publishing** may set a per-access FTNS price (or free)
- **Accessing** published datasets costs the listed FTNS price
