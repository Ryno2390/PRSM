# Legacy pre-v1.6 test files

These test files were left untracked in the `tests/` root after the
v1.6.0 Scope Alignment Sprint (2026-04-09) deleted ~210K LoC of
pre-pivot AGI-framework code. They import modules that no longer
exist:

  * `prsm.compute.nwtn.*` (meta_reasoning_engine, voicebox,
    content_royalty_engine, content_ingestion_engine, orchestrator,
    context_manager)
  * `prsm.performance.*`
  * Deleted teacher / distillation / self-improvement modules
  * Non-existent top-level helpers (e.g., `bump_version`)

None of them was ever tracked in git (they predate the v1.6 merge
that removed their dependencies). They are preserved here — rather
than deleted — so future triage can:

  1. Confirm each test is superseded by a currently-green test in
     `tests/unit/` or `tests/integration/`.
  2. Salvage any still-relevant assertions into the new test suite.
  3. Delete this directory entirely once triage is complete.

## Relocation audit — 2026-04-22

  * 97 files had at least one import referencing a non-existent
    module path (verified via ast + filesystem resolution).
  * 10 files imported only surviving modules, but failed at
    pytest collection due to imported symbols that no longer
    exist (e.g., `TrainingJob` no longer exported from
    `prsm.node.node`).
  * 2 subdirectories (`prsm_validation/`, `regression/`) contain
    pre-pivot test harnesses (EG-CFG integration, regression-
    detection framework for deleted features).

Net: every file in this directory references code that was deleted
or materially restructured in the v1.6 pivot. None runs against the
current PRSM codebase.

## Do NOT add new tests here.

New tests go in `tests/unit/` or `tests/integration/` against the
current module surface.
