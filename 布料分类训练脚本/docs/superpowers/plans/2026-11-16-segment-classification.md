# Segment Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Deliver standalone full-segment classification and an orderly experiment/model library.
**Architecture:** New v2 classifiers and artifacts are isolated from read-only v1 inference. Training and held-out testing are separate worker actions. The library groups repetitions under each architecture/input/readout configuration.
**Tech Stack:** Python 3.10, PyTorch, NumPy/SciPy, PyQt5, Matplotlib.
**Spec:** ../../两阶段训练与模型库最终方案.md

## Global Constraints
- Work in the user-requested existing script folder; do not move the launcher or modify sources/saved models.
- 333Hz, causal holding, existing physical windows, grouped legs, train-only scaler.
- Genuine GRU/TCN/CERA, mean/last readout, one classification loss per segment, no new risk training.
- Defaults 200 epochs / patience30 / batch16 / lr0.001 / weight_decay0.0001 / dropout0.1 / hidden64.
- 70/10/20 grouped split; fixed split_seed42 across repetitions; manual test after validation selection.
- Stage two explicitly unavailable until its statistical specification is fixed.

## Review Focus
- Padding changes must not change valid segment classification (task1).
- Changed source/metadata or duplicate groups must not leak across fit/test (task2).
- Unfinished/quick artifacts must not masquerade as formal models; checkpoints must match manifest hashes (task2).
- Imported/moved packages and legacy models must retain correct loader behavior (task2).
- Repeated configs, unavailable stage2, and untested models must be visually unambiguous (task3).

## Task 1: Pure classifiers and grouped data
Files: config.py, classifiers.py, data.py; tests/test_classifiers.py, test_data.py.
Interface: SegmentClassifier(channels, classes, architecture, readout, hidden_size, dropout).forward(x,lengths)->logits[B,K]; split_records(records,seed)->train/val/test indices; TrainConfig.to_dict/from_dict.
- [x] Write failing tests for padding invariance, active modules, causal features, three-way exact quotas and reproducibility.
- [x] Run targeted tests and observe missing/new behavior failures.
- [x] Implement classifiers/config, add bx input, replace split allocator with whole-group three-way allocation.
- [x] Run classifier/data tests; update old split assertions to the approved protocol.

## Task 2: Training, separate testing, immutable storage
Files: engine.py, storage.py, train_cloth.py; tests/test_engine.py, test_storage_v2.py.
Interface: train_request(request,emit,cancelled)->task; test_request({task,keys},emit,cancelled)->task; predict_request; load_bundle supports cloth-classification-v2 and cera-net-v1.
- [x] Write failing tests for no test/calibration during fitting, isolated scaler, hash rejection, cancellation, repeated split identity, independent manual testing and portable loading.
- [x] Implement v2 engine with CE, val full F1 selection, scheduler, repeated seeds and guarded legacy inference.
- [x] Extend worker CLI to test; generate report/CSV and checksummed portable bundles.
- [x] Run integration tests and preserve all raw/previous result files.

## Task 3: Desktop workflow and model library
Files: ui.py, plots.py, settings_ui.py, catalog.py; tests/test_ui.py.
Interface: existing import/project/worker API retained; experiment list+grouped model table+detail tabs; model_selector holds exact leaf checkpoint; phase2 disabled with explanation.
- [x] Write failing UI tests for defaults, presets/task count, no automatic testing, grouped repetitions, legacy separation, phase2 unavailability and reloadable projects.
- [x] Build plain-language training/protocol settings and preset controls.
- [x] Refactor library and embedded results; adapt prediction/status/plots to v2 while preserving legacy result inspection.
- [x] Run UI tests including actual QProcess training, manual testing and recognition.

## Task 4: Delivery and verification
Files: README.md, docs/实现进度.md; screenshots in ignored outputs.
- [x] Update usage, stage2 boundary, parameters, legacy notes, new CLI.
- [x] Run complete tests, source checks, old model load, short real-data verification (not performance training), and headless screenshots.
- [x] Request one independent whole-change code review; fix important findings with regression tests.
- [x] Finish with launcher command, tested evidence, preserved artifacts and limitations.
