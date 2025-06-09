# Diffing PRSM: Preventing Knowledge Base Divergence

## Overview

PRSM is a closed-loop, decentralized ecosystem that thrives on internally generated and user-contributed data. However, without external calibration, any closed system runs the risk of **epistemic divergence**—where the internal knowledge graph drifts from the broader world due to insular learning or synthetic feedback loops. This document outlines a proactive strategy for **"diffing" PRSM’s knowledge base** against the public internet to ensure alignment, transparency, and data quality over time.

---

## The Divergence Problem

As PRSM matures and becomes the dominant platform for scientific research and AI development, it faces two risks:

1. **Closed-Loop Drift**: Internal models trained on previously generated outputs without fresh input can result in degraded performance or hallucination inflation.
2. **Information Blind Spots**: Emerging data or insights from outside PRSM (e.g., proprietary or siloed knowledge) may be missed or incorporated too late.

To combat this, PRSM implements **automated external comparison cycles**.

---

## Diffing Protocol (PRSM ↔ External Web)

### 🔁 Periodic Syncing

At regular intervals, NWTN initiates a controlled “diffing” cycle to compare PRSM’s indexed corpus with:

- **Public Web Data**: Crawled from HTTPS endpoints (via open web scrapers)
- **ArXiv & Open Journals**: Using RSS and API feeds
- **Patent Databases**: To detect novel concepts not yet included in PRSM
- **Social Knowledge Signals**: Subreddits, GitHub activity, StackOverflow discussions

### 📊 Semantic Embedding Comparison

All data—both internal and external—is reduced to vector embeddings via a standardized transformer. These embeddings are:

- Clustered by topic/domain
- Compared for **novelty**, **coverage gaps**, and **semantic shifts**
- Scored using cosine similarity and concept drift detection algorithms

---

## Key Metrics Tracked

| Metric                     | Description                                          |
|----------------------------|------------------------------------------------------|
| **Coverage Delta**         | Missing topic clusters not yet represented in PRSM   |
| **Semantic Drift**         | Concepts that have changed meaning externally        |
| **Freshness Gap**          | Latency between new discoveries and PRSM ingestion   |
| **Discrepancy Hotspots**   | Domains where PRSM and the web have conflicting views|

NWTN uses these insights to flag and prioritize areas for data augmentation.

---

## FTNS-Incentivized Updating

Users can earn FTNS tokens for:

- Curating or importing “drifted” knowledge into PRSM
- Training distilled models to resolve detected divergence
- Hosting datasets flagged as coverage gaps

This turns drift correction into a community-driven process.

---

## Governance Integration

The PRSM governance layer can:

- Vote to prioritize high-drift domains
- Allocate additional FTNS incentives to critical knowledge gaps
- Commission open bounty challenges to resolve divergence

---

## Benefits

- **Quality Assurance**: Maintains PRSM as a gold-standard scientific corpus.
- **Anti-Stagnation**: Prevents recursive feedback loops from degrading performance.
- **Trust and Transparency**: Publicly logs knowledge divergence and PRSM’s response.

---

## Summary

By proactively diffing PRSM against the public web, the platform ensures:

- Continuous epistemic alignment
- Community-driven updates to remain future-relevant
- An always-fresh, always-grounded foundation for recursive scientific modeling

PRSM doesn't just grow—it grows wisely, with its eyes wide open.