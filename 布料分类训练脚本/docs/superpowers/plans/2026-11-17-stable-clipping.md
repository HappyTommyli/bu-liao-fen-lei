# Stable gradient clipping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement task-by-task.

**Goal:** Safely clip finite large gradients, preserve true invalid-value protection and versioned validation continuation.
**Architecture:** numerical.py owns float64 norm clipping; engine aggregates diagnostics and uses explicit compatible recovery only in derived validation tasks. Existing UI continues the same button flow.
**Tech Stack:** Python, CPU PyTorch, NumPy, PyQt5, pytest.
**Spec:** docs/superpowers/specs/2026-11-17-stable-clipping.md

## Global Constraints
- Keep model, preprocessing, data admission, LR0.001 and clipping limit1 unchanged.
- No formal or long training; verify exact failing batches in memory and synthetic tests only.
- Original real failed tasks remain byte-for-byte unchanged.
- v1->v2 recovery compatibility must not bypass any other identity validation; never silently combine runtimes into a formal experiment.

## Review Focus
- Very large finite gradients must retain direction and nonzero magnitude after clipping.
- NaN/Inf or post-update corruption must still stop and preserve recovery.
- Compatibility must reject changed data/config/job/environment and preserve source experiment.
- A derived validation task must stop after its target fold and be barred from formal use.
- Restored history may lack new diagnostic fields; handle it without losing best state or scheduler/RNG.

### Task 1: Stable clipping primitive
**Files:** Modify cloth_trainer/numerical.py; create tests/test_stable_clipping.py.
**Interfaces:** clip_gradients(named_parameters,max_norm=1.) -> dict(norm,max_abs,coefficient,clipped_norm,clipped); NumericalFailure on invalid gradients/norm.
- [x] Write failing tests for finite large/ordinary/zero/mixed tensor gradients, direction/nonzero values, invalid gradients before mutation.
- [x] Implement float64 norm calculation and checked clipping. Run `.venv/bin/python -m pytest -q tests/test_stable_clipping.py`; expected all pass. Commit.

### Task 2: Diagnostics, compatible validation copies, integrated verification
**Files:** Modify engine.py, recovery.py, ui.py, README.md and tests; create runtime_validation.py if useful.
**Interfaces:** Consume clipping dict; recovery load accepts explicit runtime upgrade only with all remaining binding equal. Fork helper returns derived task and provenance; source never mutated.
- [x] Add failing integration tests for large finite gradients updating normally, epoch summaries, legacy-runtime fork/compatibility guards, stopping after one target fold and blocking formal loading.
- [x] Implement aggregation and compatibility plus simple UI notices, keep all existing true-nonfinite safety. Update the old overflow test to assert correct safe clipping, not a forced failure.
- [x] Run focused tests, then full `.venv/bin/python -m pytest -q --tb=short`; expected all pass.
- [x] Replay real epoch22 two batches with production clipping in memory, assert identities/losses, norm<=1 and finite optimizer; verify original task fingerprints unchanged; save a compact report.
- [x] Commit; one fresh read-only final review. Important fixes get RED->GREEN and full suite; record final evidence.
