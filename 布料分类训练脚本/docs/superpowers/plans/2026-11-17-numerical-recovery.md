# Numerical recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Prevent nonfinite updates and recover completed epochs with actionable diagnostics.
**Architecture:** A small recovery module owns atomic checksummed state; engine validates numerical boundaries and binds recovery to frozen jobs; existing UI consumes restore events.
**Tech Stack:** Python, PyTorch CPU, NumPy, PyQt5, pytest.
**Spec:** docs/superpowers/specs/2026-11-17-numerical-recovery.md

## Global Constraints
- No formal training; synthetic verification only. Do not modify actual failed experiment.
- Preserve architectures, inputs, admission, LR=0.001 and existing parameters.
- Stop on numerical failure, never skip a batch or silently change LR.
- Keep best export separate from current epoch recovery. Preserve old completed folds.

## Review Focus
- Interrupted checkpoint write must leave last committed state readable.
- Finite forward loss with nonfinite backward gradients must never update parameters.
- Failure after optimizer update or during validation must not replace last valid recovery.
- Resuming after final epoch/early stop must not add training epochs.
- Legacy history and failed diagnostics must remain inspectable; corrupt recovery must not restart silently.

### Task 1: Atomic bounded recovery state
**Files:** Create cloth_trainer/recovery.py, tests/test_recovery.py.
**Interfaces:** Produces save_recovery(folder,state,binding) and load_recovery(folder,binding); None means absent; invalid bytes/identity raise ValueError. Engine supplies tensors/primitive containers; torch weights_only deserialization only.
- [x] Write tests for roundtrip tensors, mismatch, corruption, interrupted replacement, rejection of nonfinite tensor state. Run `.venv/bin/python -m pytest -q tests/test_recovery.py` and expect failures from missing behavior.
- [x] Implement a single atomic checksummed envelope, fsync temporary before replace, reject nonfinite tensors. Run same command; expect all pass.
- [x] Commit and task-done using same test command.

### Task 2: Guarded fitting, full epoch continuation, UI and docs
**Files:** Modify cloth_trainer/engine.py, cloth_trainer/ui.py, README.md; create tests/test_numerical_training.py; update tests/test_ui.py.
**Interfaces:** Consumes Task1 recovery API. `_fit` optional job binding; restore events contain saved history and epoch. Existing model bundles remain unchanged.
- [x] Write tests for invalid input/logits/gradient, optimizer corruption, validation error reports, exact resume including scheduler and RNG, config/data binding, early stop/final budget, legacy no-checkpoint restart, UI history restore. Run focused tests; expect failures.
- [x] Implement guards/diagnostics and epoch state persistence plus verified continuation. Add UI restore/progress and concise notice; document usage. Run focused tests, expect pass.
- [x] Run full `.venv/bin/python -m pytest -q --tb=short`, expect all pass; hash original failed-task files before/after read-only compatibility verification, no changes.
- [x] Commit and task-done using full suite. One read-only fresh-context whole-branch review; important fixes get reproducing tests and full suite. Record evidence and final commit.
