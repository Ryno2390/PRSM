# PRSM Legal Structure

## Overview

PRSM is designed as a **two-entity structure** — a non-profit foundation governing the open protocol, paired with a for-profit commercial operator that funds the foundation through dividends. This architecture is modeled on precedents such as the Mozilla Foundation / Mozilla Corporation and Ethereum Foundation / ConsenSys, adapted for an AI coordination network.

This document is intended for investor due diligence. Fields marked **[PENDING]** reflect entities or filings not yet established as of the document date.

---

## Two-Entity Structure Summary

```
┌─────────────────────────────────────────────────────────┐
│               PRSM Foundation (Non-Profit)              │
│           501(c)(3) or equivalent · [PENDING]           │
│                                                         │
│  • Owns & governs the PRSM protocol                    │
│  • Holds protocol IP                                    │
│  • Ensures network remains open & decentralized         │
│  • Shareholder in Prismatica Inc.                       │
│  • Self-funds via Prismatica dividends                  │
└────────────────────────┬────────────────────────────────┘
                         │ Holds equity / receives dividends
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Prismatica Inc. (For-Profit)                │
│         Delaware C-Corp · [PENDING FORMATION]           │
│                                                         │
│  • Commercial arm and mega-node operator                │
│  • Provides base-load data, compute & storage           │
│  • Earns FTNS provenance royalties                      │
│  • Investor-facing entity for Series A                  │
└─────────────────────────────────────────────────────────┘
```

---

## Entity 1: PRSM Foundation

| Field | Value |
|-------|-------|
| **Legal Name** | PRSM Foundation |
| **Entity Type** | Non-profit corporation (501(c)(3) or equivalent) |
| **Jurisdiction** | United States — state TBD |
| **Formation Status** | **[PENDING]** — to be formed after Prismatica Inc. is established |
| **Formation Method** | Independent legal counsel; standard non-profit incorporation |
| **EIN/Tax ID** | **[PENDING]** |
| **501(c)(3) Application** | **[PENDING]** |
| **IRS Determination Letter** | **[PENDING]** |
| **Founder** | Ryne Schultz |

### Role and Mission

The PRSM Foundation is the neutral steward of the PRSM protocol. Its responsibilities include:

- **Protocol governance**: Setting standards, managing upgrades, and maintaining the openness of the PRSM network
- **IP ownership**: Holding trademarks, patents, and core protocol intellectual property
- **Decentralization assurance**: Ensuring no single commercial entity — including Prismatica — can capture or close the network
- **Financial sustainability**: Receiving dividend distributions from Prismatica Inc. to fund ongoing protocol operations, research, and community grants

### Relationship to Prismatica

The PRSM Foundation will hold an equity stake in Prismatica Inc. This arrangement means:

- The Foundation receives dividends as Prismatica generates royalty revenue
- The Foundation has a structural incentive to promote network growth (more nodes → more FTNS demand → more Prismatica revenue → larger dividends)
- Neither entity is financially dependent on token sales or grants alone

---

## Entity 2: Prismatica Inc.

| Field | Value |
|-------|-------|
| **Legal Name** | Prismatica Inc. |
| **Entity Type** | Delaware C-Corporation |
| **Jurisdiction** | Delaware, United States |
| **Formation Status** | **[PENDING]** — formation via Stripe Atlas in progress |
| **Registered Agent** | **[PENDING — via Stripe Atlas]** |
| **EIN/Tax ID** | **[PENDING]** |
| **Founder** | Ryne Schultz (Sole Founder) |

### Role and Business Model

Prismatica Inc. is the commercial arm of the PRSM ecosystem. It operates as a **mega-node** on the PRSM network — the primary provider of base-load compute, data, and storage capacity during the network's early growth phase.

**Revenue mechanism**: Prismatica earns **FTNS provenance royalties** for every unit of compute, data, and storage it contributes to the network. As network usage grows, demand for FTNS increases, and Prismatica's royalty position appreciates accordingly.

This is the **investor-facing entity** for the Series A funding round.

### Equity Structure

| Category | Detail |
|----------|--------|
| **Authorized Shares** | **[PENDING — post-formation]** |
| **Common Stock** | Founder shares — Ryne Schultz |
| **Preferred Stock (Series A)** | **[PENDING — post-formation, pre-investment]** |
| **PRSM Foundation stake** | To be issued upon Foundation formation |
| **Employee Option Pool** | **[PENDING — to be established at formation]** |

*Full cap table available to qualified investors under NDA upon formation.*

---

## Intellectual Property

### Ownership and Assignment

| IP Category | Current Owner | Planned Owner |
|-------------|--------------|---------------|
| Source Code | Ryne Schultz (individual) | PRSM Foundation (upon formation) |
| PRSM Trademark | **[PENDING registration]** | PRSM Foundation |
| Protocol Design / Architecture | Ryne Schultz (individual) | PRSM Foundation (upon formation) |
| Prismatica Trade Secrets | Prismatica Inc. (upon formation) | Prismatica Inc. |

**IP Transfer Plan**: Upon formation of both entities, all protocol-related IP will be assigned to the PRSM Foundation. Prismatica-specific commercial IP (infrastructure tooling, proprietary systems) will be assigned to or developed within Prismatica Inc.

### Open Source

The PRSM protocol implementation is open source. Key licenses:
- **Core Protocol**: Apache 2.0
- **SDKs**: MIT License
- **Tooling**: MIT / Apache 2.0 (component-specific)

All open-source usage is tracked and compliant with respective license terms.

---

## FTNS Token

The FTNS token is live on **Base mainnet** at [`0x5276a3756C85f2E9e46f6D34386167a209aa16e5`](https://basescan.org/address/0x5276a3756C85f2E9e46f6D34386167a209aa16e5).

| Field | Status |
|-------|--------|
| **Token Classification (US)** | Utility token (consumptive use) — formal legal opinion **[PENDING]** |
| **Howey Test Analysis** | Preliminary: utility token; no profit-sharing mechanism; consumptive use case |
| **MiCA Compliance (EU)** | **[PENDING review]** |
| **Token Legal Opinion** | **[PENDING — securities counsel to be engaged]** |

---

## Regulatory Compliance

| Framework | Status |
|-----------|--------|
| GDPR | Implemented |
| CCPA | Implemented |
| AML/KYC (token holders) | **[PENDING — planned pre-launch]** |
| SOC 2 Type II | In Progress |
| ISO 27001 | Planned (post-SOC 2) |

---

## Governance Documents

| Document | Status |
|----------|--------|
| Prismatica Certificate of Incorporation | **[PENDING — Stripe Atlas formation]** |
| Prismatica Bylaws | **[PENDING]** |
| PRSM Foundation Charter | **[PENDING]** |
| Stockholder Agreement | **[PENDING — post-formation]** |
| Investor Rights Agreement (Series A) | **[PENDING — to be drafted]** |
| IP Assignment Agreements | **[PENDING — upon entity formation]** |

*Governance documents will be made available to qualified investors under NDA upon formation.*

---

## Formation Timeline

| Milestone | Status | Target |
|-----------|--------|--------|
| Form Prismatica Inc. (Stripe Atlas) | **[PENDING]** | Near-term (Q2 2026) |
| Establish employee option pool | **[PENDING]** | At formation |
| Engage corporate / securities counsel | **[PENDING]** | At formation |
| Form PRSM Foundation | **[PENDING]** | Medium-term (post-Prismatica) |
| Transfer protocol IP to Foundation | **[PENDING]** | Post-Foundation formation |
| Issue Foundation equity stake in Prismatica | **[PENDING]** | Post-Foundation formation |
| Series A close | **[PENDING]** | 2026 |

---

## Legal Counsel

| Function | Firm | Status |
|----------|------|--------|
| Corporate Counsel | **[PENDING]** | To be engaged at formation |
| Securities Counsel | **[PENDING]** | To be engaged at formation |
| IP Counsel | **[PENDING]** | To be engaged at Foundation formation |
| Regulatory / Token Counsel | **[PENDING]** | To be engaged pre-launch |

---

## Due Diligence Materials

The following will be available to qualified investors under NDA upon formation:

- [ ] Prismatica Certificate of Incorporation
- [ ] Prismatica Bylaws and Board Resolutions
- [ ] Cap Table
- [ ] IP Assignment Agreements
- [ ] Material Contracts
- [ ] FTNS Token Legal Opinion (upon completion)
- [ ] PRSM Foundation Charter (upon formation)

---

## Contact

**Legal / Investment Inquiries**: funding@prsm.ai

---

*This document reflects the intended corporate structure as of March 2026. Prismatica Inc. and the PRSM Foundation are not yet formed. All legal matters should be verified with qualified legal counsel prior to investment. Fields marked [PENDING] will be updated as milestones are completed.*
