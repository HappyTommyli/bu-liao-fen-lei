# GPU standard training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, one final independent review.

**Goal:** Deliver a GPU-first v4 classification workflow with grouped development CV, final refit and locked holdout.
**Architecture:** New focused modules coexist with legacy engine/config; dispatcher and existing Qt UI expose v4 without rewriting old experiments.
**Tech Stack:** PyTorch CUDA/CPU, NumPy/SciPy, PyQt5, pytest.
**Spec:** docs/superpowers/specs/2026-11-17-gpu-standard.md

## Global Constraints
- CUDA default, no silent CPU fallback; actual host nouveau blocks hardware acceptance.
- Preserve raw data/admission/protocol and old model formats. No formal real training or display driver changes.
- Test never influences scaler, CV, candidate choice or final epochs. Groups indivisible.
- Record CPU tests as CPU; conditional GPU tests must not fake availability.

## Review Focus
- CUDA/AMP optimizer state and RNG restore correctly, missing GPU fails clearly.
- Uneven microbatches have correct effective-batch loss weighting and padding masks.
- Candidate/final-epoch/test locking cannot be bypassed by UI resume or alternate keys.
- Legacy loaders/archives and audited snapshots remain usable without rewriting.
- Final-refit recovery uses full development scaler and never consults holdout.

### Task 1: Modern configuration and grouped partitions
**Files:** modern_config.py, modern_split.py, tests/test_modern_split.py.
**Interfaces:** ModernConfig extends legacy config with pipeline/device/precision/workers/microbatch; make_plan(records,seed) and validate_plan produce dev/test/cv index lists.
- [ ] RED config defaults/serialization and disjoint group/class coverage, insufficient groups, deterministic split.
- [ ] Implement, focused tests pass, commit.

### Task 2: Device-aware models and resumable training
**Files:** devices.py, modern_models.py, modern_training.py, tests/test_modern_training.py.
**Interfaces:** resolve_device(strict), make_loaders/cache, fit_fold/final and evaluate return standard prediction rows; atomic recovery bound to runtime.
- [ ] RED CPU explicit/CUDA unavailable, causal padding, conv lengths, microbatch weighting, numerical guards, resume equality and CUDA conditional smoke.
- [ ] Implement DataLoader, GPU transfer, AMP after unscale, stable clipping, metrics/early stop and full RNG recovery; focused tests pass, commit.

### Task 3: CV/final/test/inference orchestration and legacy routing
**Files:** modern_engine.py, engine.py dispatch, storage.py compatibility; tests/test_modern_engine.py.
**Interfaces:** train_request(v4), final_request, test_request, load_bundle/predict; preserve legacy helper pipeline for audit/freeze only.
- [ ] RED complete synthetic workflow, test isolation, final epoch binding, locked candidate, snapshot/weight tampering, portable package and recognition.
- [ ] Implement new task/version and snapshots, CV/final artifacts, locked holdout, archive/old compatibility; focused tests pass, commit.

### Task 4: Desktop controls, installation and environment diagnosis
**Files:** ui.py, settings_ui.py, train_cloth.py, installation/diagnosis scripts, README; tests/test_ui.py, test_startup.py.
**Interfaces:** Existing process worker adds final job; GPU default and explicit CPU; grouped CV library/final model only prediction.
- [ ] RED UI/device selection, CV->final->test background workflow, old library readonly, installer selection/diagnosis.
- [ ] Implement concise existing-page controls and plain-language defaults, installation CUDA wheels with explicit CPU option, diagnosis; focused tests pass, commit.

### Task 5: Full verification and delivery evidence
**Files:** tests/integration, docs verification records.
- [ ] Run `.venv/bin/python -m pytest -q --tb=short`, including updated v4 UI expectations and retained old-engine tests. No fake CUDA.
- [ ] Run offscreen desktop and GPU diagnostics; record actual skips/blocker, original experiment fingerprints unchanged. One final readonly review; Important fixes RED->GREEN and full suite.
- [ ] Commit durable evidence, keep local branch and original folder, no push/merge or formal training.
