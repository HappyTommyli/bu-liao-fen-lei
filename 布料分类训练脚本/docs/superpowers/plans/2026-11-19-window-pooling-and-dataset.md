# Window/pooling study and sixteen-class dataset implementation plan

Approved specification: user approved the preceding three-part proposal on 2026-11-19. Implement in the existing application directory, preserving the previous contact-window changes and historical experiments. No formal training.

## Task 1: Study presets and readable descriptions
- [x] Regression tests: old four-way candidates remain identical; new window_pool mode creates 900/mean,900/mean_std,450/mean,1800/mean candidates, all six blocks of128, kernel5, dilations1/2/4/8/16/32, dropout0.1. Locked fields cannot silently alter presets; training fields survive switching.
- [x] Implement common mode metadata in tcn_study, config validation, editor mode names and a central task/config explanation. Save explanatory notes alongside frozen execution and display historical notes with historical results. Contrast comparisons follow the actual study family.
- [x] Verify focused tests before proceeding.

## Task 2: Honest per-epoch logs
- [x] Regression tests exercise training event messages: validation macroF1, accuracy, loss, best F1/epoch; refit has no validation and says not applicable. Emit scheduler changes and completion reason. UI adds group/fold context and reflects frozen active tasks, not edited draft parameters.
- [x] Implement event formatting and completion metadata without additional evaluation or changing optimization. Preserve checkpoint/resume behavior.
- [x] Verify training and UI tests.

## Task 3: Derived dataset with preserved membership
- [x] Tests derive a class-filtered immutable snapshot: no removed arrays, unchanged retained arrays and source files; remap fivefold, dev/test indices without reshuffling; retain original admission batches and provenance; reject invalid retained coverage. Training must actually consume the saved split plan.
- [x] Implement dataset derivation and saved-plan validation. Hide intentionally removed-class review entries from the current dataset display. Preserve old snapshots/models.
- [x] Derive the actual latest 330-record/17-class dataset into310 records/16 classes, excluding印花丝绒. Create an active project pointing to the new version; do not train.

## Verification
- [x] Full test suite and real historical frozen-task load; new-mode tiny synthetic end-to-end CV/refit/test; inspect rendered task/config UI; verify actual310CSV/620segments/16classes and preserved memberships, original manifests/weights unchanged.

## Review focus
Old presets and study hashes remain loadable. New single-field mode switch never resets epochs/LR. No train F1 is invented from train accuracy. Refit says validation N/A. Derived-plan seed mismatch/partial selection fails instead of reshuffling. Removed labels do not remain as selectable training records. Active-task explanations cannot change with editor changes. No formal training or final-test evaluation on the real data.

## Completion evidence

- New preset, UI, frozen metadata, logs and immutable derived dataset implemented. Runtime training parameters remain unchanged (max200, early-stop30); no formal training started.
- Full suite:231 passed,7 CUDA tests automatically skipped,1 driver warning; later focused UI/study suite22 passed. Before that, seven actual CUDA tests failed during a system suspend/resume driver fault. This is not GPU acceptance: direct cuInit returned999; kernel Xid154 explicitly requests Node Reboot Required.
- Independent read-only review found four issues (override of parent partitions, ancestor exclusion loss, final-refit context, invalid draft overwriting busy preview); all fixed with failing-before/passing-after regressions. Synthetic end-to-end lifecycle including mean_std final weight load, holdout and portable export/import passed. Historical real20-model task still validates.
- Actual derived version a4679abce3aa4943a1e048f1e332f244:310CSV/620segments/16classes,247development/63test; original memberships preserved. Removed20CSV and metadata untouched; all old model weights and manifest hashes unchanged. Current project saved and actual Qt rendering/selection checked.
- No deferred implementation findings; real GPU validation remains pending system reboot. Work applied in the existing user-requested application checkout; earlier contact-window edits were retained.
