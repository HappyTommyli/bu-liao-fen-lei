# Batch capture and persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 批量重复采集，恢复工作台编辑状态，回收记录并复用编号。
**Architecture:** Catalog 负责编号/回收；preferences 加锁合并保存；Worker 批量调度等待持久保存；UI 只提交意图。
**Tech Stack:** Python / PySide6 / unittest / 现有模拟串口。
**Spec:** ../specs/2026-11-14-batch-persistence-design.md

## Global Constraints
23列CSV保持原格式；不访问实体设备、用户数据和固件。重开不自动运行/使能/有效基线；边界须确认启用。使用既有隔离工作树，交付仅覆盖散列核对通过的修改文件。

## Review Focus
累计位移越界启动前拒绝；最终写入失败不开始下一轮；间隔期间禁止手动操作；回收恢复拒绝链接/越界；自动保存与标签更新合并，损坏设置可启动并提示。

### Task 1: Catalog recycle and reusable names
**Files:** catalog.py; test_workbench_catalog.py
**Interfaces:** Catalog.trash(paths,blocked=()), trash_records(), restore(path,blocked=()).
- [x] RED: gap reuse, trash/restore bytes, active/path/link refusal, restore conflict.
- [x] Implement directory-locked whole-folder recycle and rollback.
- [x] GREEN: catalog tests; ledger evidence.

### Task 2: Persistent UI state
**Files:** new preferences.py; catalog.py; ui.py; test_workbench_preferences.py; test_workbench_ui.py.
**Interfaces:** load_settings(path), update_settings(path,changes); Window.save_session(), restore_session().
- [x] RED: settings merge, full draft roundtrip, keyboard off/bounds inactive, bad settings fallback.
- [x] Implement debounced save, close flush, directory preflight, preserve config mismatch.
- [x] GREEN: UI/preferences tests; ledger evidence.

### Task 3: Worker batch orchestration
**Files:** new batch.py; backend.py; test_workbench_batch.py.
**Interfaces:** validate_batch(recipe,current,config,count,interval,baseline); commands batch_record/batch_stop_after; snapshot batch.
- [x] RED: cumulative bounds, three captures, finalizer wait/failure, stop-after/immediate stop, exclusive interval.
- [x] Implement frozen settings, round metadata, durable-save gate and error cancellation.
- [x] GREEN: backend/batch tests; ledger evidence.

### Task 4: Batch and record UI
**Files:** ui.py; test_workbench_ui.py; LINUX_WORKBENCH.md.
- [x] RED: batch payload/progress/disabling, multiselect recycle and restore.
- [x] Implement compact batch controls, recovery UI; update guide.
- [x] Full suite, screenshot inspection, fresh final review.
- [x] Baseline hash comparison, deliver changed files, full delivered regression.

## Ledger
Initial: design approved by user “可以开始修改”; inline execution in existing isolated worktree. No new confirmation required.

- Baseline: delivered source snapshot isolated, original 137 unittest tests passed.
- Task 1: smallest free sequence and recycle/restore, catalog RED then 11 GREEN (including existing 7).
- Task 2: session merge and close/reopen draft tests RED then GREEN, UI 21 passed; no live settings used.
- Task 3: batch orchestration RED then 6 GREEN; distinct round capture metadata and cumulative bounds checked.
- Task 4: UI batch payload, lockout and multiselect confirmation RED then GREEN; first full suite 153 passed.
- Final independent review: batch_final_review identified four Important issues, no Critical/minor findings.
- Final: fixed receipt-temp symlink overwrite — test_receipt_temp_symlink_never_changes_external_file RED→GREEN using exclusive random temporary files.
- Final: fixed transient faults/reference loss during batch saving/interval — two packet injection tests RED→GREEN, cancel before future healthy packets can resume.
- Final: fixed active-batch leak on USB/STOP failure — test_motor_unplug_running_batch_does_not_leave_controls_reserved RED→GREEN, cancel batch independent of stop notification.
- Final: fixed structurally malformed nested settings — test_malformed_nested_settings_fall_back_without_crashing RED→GREEN, validate before widget population.
- Final full suite after one fix pass: 159/159 passed, includes actual C++ host firmware compilation tests. UI screenshots rendered and inspected; no physical device actions.
- User requested direct script update; integration is selective file copy into existing app after baseline hash checks, leaving unrelated dirty repository changes and data intact. No git merge/push/commit requested or performed.
- Delivered: 14 changed/new files synchronized after source-hash checks, prior versions retained in isolated delivery-backup. Full delivered unittest 159/159 passed in 22.382s; linux_workbench.py --help passed. No actual data or firmware files copied or changed.
