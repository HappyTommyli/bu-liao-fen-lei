# Independent Recognition Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an independent TCN recognition page for newly collected data.

**Architecture:** Reuse final-model loading and prediction workers; keep recognition paths and label overrides separate from training state. Extend the focused UI through a recognition mixin.

**Tech Stack:** PyQt5, PyTorch, pytest.

**Spec:** ../specs/2026-11-20-independent-recognition-design.md (approved).

## Global Constraints

Preserve all existing uncommitted work. Do not train real data. Use saved model preprocessing and aggregation. Exclude source overlaps and unknown truth from independent metrics.

## Review Focus

- Empty or invalid directories show import errors without affecting training.
- Changing the selected model updates run eligibility.
- Unknown and out-of-class labels remain visible but unscored.
- Model selectors expose only completed TCN / |dBy| final weights.
- Busy workers prevent mutation of active recognition inputs; controls remain accessible at 1024×640.

### Task 1: Independent inputs and recognition page

Files: create cloth_trainer/tcn_recognition.py and tests/test_tcn_recognition.py; modify cloth_trainer/tcn_ui.py, train_cloth.py, tests/test_tcn_ui.py.

- [x] Write failing tests for four-page navigation, independent file expansion/labels/removal, final-model filtering and submitted prediction request.
- [x] Run targeted pytest and confirm intended failures.
- [x] Implement TCNRecognitionMixin with build_prediction, add_recognition, set_recognition_labels, remove_recognition, clear_recognition, start_prediction, refresh_recognition_models and result rendering. Keep saved config read-only and recognition requests separate.
- [x] Run targeted tests, including a genuine synthetic final-model inference lifecycle and overlap/unknown statistics.

### Task 2: Verification and documentation

- [x] Verify recognition layout at 1024×640 and 1366×768, plus export and busy-state behavior.
- [x] Update README with independent evaluation steps and limitations.
- [x] Run complete pytest suite; review only this task's changes without committing unrelated work.

Ruling: execute inline under the user's existing authorization; no extra implementation-method approval. Leave changes uncommitted because shared files contain pre-existing work.

Review: independent reviewer found no important/critical issues; import issue preservation was fixed after a failing regression test. Targeted tests: 7 passed. Final full-suite check: 245 passed in 111.37s. git diff --check passed. No real cloth training performed.
