# Pull Request

## Summary

<!-- One or two sentences: what does this PR change, and why? -->

## Type

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (fix or feature that changes existing behavior)
- [ ] Performance improvement
- [ ] Refactor (no behavior change)
- [ ] Documentation update
- [ ] Build / CI / tooling
- [ ] Research / design doc

## Scope — which layer of PRSM is touched?

<!-- Check all that apply; helps reviewers route attention -->

- [ ] Data layer (`prsm/node/bittorrent_*`, `prsm/storage/`, content tiers)
- [ ] Compute layer (`prsm/compute/`, WASM / Wasmtime / SPRK runtime)
- [ ] Economic layer (`prsm/economy/`, Solidity contracts, settlement)
- [ ] Networking (`prsm/node/transport*`, bootstrap, R9 adapters)
- [ ] MCP server (`prsm/mcp_server.py`, `prsm/cli_modules/mcp_server.py`)
- [ ] CLI / TUI (`prsm/cli.py`, `prsm/cli_modules/`)
- [ ] Marketplace / orchestration (`prsm/marketplace/`, matching, routing)
- [ ] Docs
- [ ] Tests only
- [ ] Other: _______________________

## Testing

<!-- Describe the testing you performed. Delete rows that don't apply. -->

- [ ] `pytest --timeout=120` passes locally
- [ ] Added unit tests for new behavior
- [ ] Added integration tests
- [ ] Manually exercised the change via `prsm <command>` (note which)
- [ ] Solidity — `npx hardhat test` passes
- [ ] Not applicable (docs / research / non-code)

**Manual test notes (optional):**

<!-- If you ran a manual sanity check, briefly describe what you did + observed. -->

## Risk assessment

- [ ] Change is backwards-compatible (existing node configs, contracts, DBs continue to work)
- [ ] Change requires migration — migration script included and tested
- [ ] Change affects on-chain behavior — smart contracts reviewed + slashing invariants preserved
- [ ] Change touches Foundation governance surfaces — flagged in PR description for maintainer review
- [ ] Not applicable

## Linked issues / docs

<!-- Reference issues closed, or docs this PR implements against. -->

Closes #
Implements:

## Checklist for the author

- [ ] My commits are signed (`git config --get commit.gpgsign` returns `true`, or signed via SSH key)
- [ ] Code follows existing style — `ruff check` + `mypy prsm/` pass on modified files
- [ ] No secrets committed (`.env`, credential JSON, private keys)
- [ ] Documentation updated if behavior changed (README, `docs/INDEX.md` entry, or design doc)
- [ ] CHANGELOG / release notes updated if user-visible
- [ ] Commit messages follow the repository's style (see recent commits on `main`)

## For reviewers

- Load-bearing invariants to verify: see `docs/OPERATOR_GUIDE.md` §On-chain Keypairs + §Redundant-Execution Dispatch for Phase 7 / 7.1 invariants if this PR touches those surfaces.
- Audit-prep: if this PR touches any contract listed in
  `docs/2026-04-22-phase7.1x-audit-prep.md`, confirm the merge-ready
  freeze-tag status before merge.

---

*For the full contributor workflow, see [`CONTRIBUTING.md`](../CONTRIBUTING.md). For community guidelines, see [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md). Report security-relevant vulnerabilities privately via [GitHub Security Advisories](https://github.com/prsm-network/PRSM/security/advisories/new) rather than in a public PR.*
