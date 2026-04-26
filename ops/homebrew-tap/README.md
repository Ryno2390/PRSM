# PRSM Homebrew Tap (scaffold)

This directory contains the **source-of-truth** Homebrew formula for `prsm` plus the tooling to maintain it. The actual published tap lives at a separate repository per Homebrew convention.

## How users install PRSM via Homebrew

Once published, users run:

```bash
brew tap prsm-network/tap
brew install prsm
```

This installs PRSM with its full Python dependency tree in a vendored virtualenv, exposing the `prsm` CLI in `$(brew --prefix)/bin`.

## Why a scaffold instead of the actual tap repo?

Homebrew taps have a hard naming convention: `<org>/homebrew-<name>`, mapped at install time to `brew tap <org>/<name>`. So the canonical tap lives at `https://github.com/prsm-network/homebrew-tap` — a separate repo from the main `prsm-network/PRSM` codebase.

Keeping the **source of truth** here in the main monorepo gives us:
- Atomic version bumps with the corresponding code change
- Consistent formula edits across team members (one repo to PR against)
- Single CI surface for formula validation (Makefile targets, brew audit)
- Easy to audit formula history alongside code history

The `sync-to-tap.sh` script publishes from this scaffold to the public tap repo when ready.

## Layout

```
ops/homebrew-tap/
├── README.md                       (this file)
├── Formula/
│   └── prsm.rb                     (source-of-truth formula)
└── scripts/
    ├── regenerate-formula.sh       (refresh sha256 + resources after PyPI publish)
    └── sync-to-tap.sh              (push Formula/prsm.rb to prsm-network/homebrew-tap)
```

## Maintenance workflow

The full release cycle, end-to-end:

```bash
# 1. Bump version in pyproject.toml
# 2. Build + publish Python package
make publish-pypi

# 3. Regenerate Homebrew formula with real sha256 values + full transitive deps
bash ops/homebrew-tap/scripts/regenerate-formula.sh

# 4. Test the formula locally
brew install --build-from-source ops/homebrew-tap/Formula/prsm.rb
brew test prsm
brew uninstall prsm

# 5. Commit the regenerated formula
git add ops/homebrew-tap/Formula/prsm.rb
git commit -m "chore: regenerate Homebrew formula for prsm-network <VERSION>"

# 6. Sync to the published tap repo
bash ops/homebrew-tap/scripts/sync-to-tap.sh

# 7. Verify users can install
brew untap prsm-network/tap   # if previously tapped
brew tap prsm-network/tap
brew install prsm
```

`make publish-homebrew` and `make publish-homebrew-dry-run` wrap the relevant pieces of this flow.

## Status

- [x] Scaffold formula with placeholder sha256 (`Formula/prsm.rb`)
- [x] Regeneration script (`scripts/regenerate-formula.sh`)
- [x] Sync-to-tap script (`scripts/sync-to-tap.sh`)
- [x] Makefile target wiring (`make publish-homebrew{,-dry-run}`)
- [ ] Real sha256 values populated (requires post-mainnet PyPI publish at known version)
- [ ] Full transitive resource list generated via `homebrew-pypi-poet`
- [ ] `prsm-network/homebrew-tap` repository created on GitHub
- [ ] First successful `brew install prsm/tap/prsm` on Apple Silicon
- [ ] First successful `brew install prsm/tap/prsm` on Intel macOS
- [ ] Linux test via Linuxbrew

## Why placeholder sha256 right now

Homebrew formulas pin every download to a specific sha256 hash. This is the security property that prevents an adversary from swapping the source tarball server-side after the formula is published. Until prsm-network is actually published to PyPI at a stable version, we don't know what the canonical sha256 *should* be — so the scaffold uses zero-bytes placeholders that explicitly fail the integrity check.

This is intentional: a Homebrew formula with `0000…0000` sha256 cannot install. Anyone who tries gets a clear failure rather than a silent man-in-the-middle vulnerability. The placeholder forces the regeneration step to happen between PyPI publish and tap publish.

## Why we don't ship a binary

Idiomatic Homebrew for Python tools uses `Language::Python::Virtualenv` to create a vendored Python venv. This is the pattern PRSM follows — one Python dependency per `resource` block, all installed into a venv inside the Homebrew Cellar.

We don't ship a precompiled binary because:
- PRSM is fundamentally a Python package; bundling Python adds 50-100MB per platform
- The npm wrapper (`prsm-mcp`) covers the "I just want to launch the MCP server" use case with zero install overhead
- Homebrew's venv approach gives a stable, reproducible install that auto-updates with `brew upgrade`

## Why prsm-network is the source-of-truth

The Python package at https://pypi.org/project/prsm-network/ is the canonical install. The Homebrew formula and the npm wrapper are both convenience layers. There is one PRSM, in Python; the alternate channels are ergonomic delivery vehicles, not parallel implementations.

## Related documents

- [Phase 3.x.1 design plan](../../docs/2026-04-26-phase3.x.1-mcp-server-completion-design-plan.md) — full Task 10 scope
- [npm wrapper](../../npm/) — Task 9, parallel distribution channel
- [Distribution README](../../README.md#mcp-integration) — three install paths user-facing

## Versioning

- **0.1 (2026-04-26):** Initial scaffold with placeholder sha256. Real values + full resource list pending post-mainnet PyPI publish.
