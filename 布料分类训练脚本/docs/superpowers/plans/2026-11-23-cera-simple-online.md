# CERA and simplified online implementation plan

> Execute inline using executing-plans, tests first, preserve dirty workspace.

Goal: train window CERA and use it online, simplify daily online controls.
Architecture: versioned CERA spec extends common TCN fields with GRU hidden/layers, dispatch via frozen candidate spec; existing parent-window trainer reused. UI model selector drives config, preview and history. Online worker unchanged apart from architecture acceptance; source selector and collapsed advanced/test controls replace parallel action buttons.
Spec: docs/superpowers/specs/2026-11-23-cera-simple-online.md
Stack: PyTorch/PyQt5/pytest.

- [x] TDD model spec/causality/padding/gradient tests; new CERA module, config/candidate/factory dispatch and frozen lifecycle, formal final/recognition and online loader compatibility.
- [x] TDD focused CERA selection/parameter/request/restore tests; selectable GRU parameters, correct previews, history and model lists.
- [x] TDD source-mode/start/control visibility tests; simple online entry points, advanced and formal test folding, clear current result, retain hash/policy guards.
- [x] Run full pytest, real tiny CERA training/load/replay and offscreen screenshots, final whole-change review, document actual evidence. No raw data edits, motor/serial commands or processes restarted.

Review focus: wrong architecture bound to candidate; CERA zero-state per window; legacy configs/bundles; selecting full-record model online; source changes while running; hidden formal test protections.

Implementation evidence: model/dispatch/final-load/replay3 tests green; editor and old-CERA window rejection reproduced then green. Focused suite42 passed before review; review fixes focused51 passed. Tiny CERA training used synthetic fixtures only, two paired candidates, validated final bundle and independent prediction. No serial/motor/raw data/process operations.
Final review (fresh cera_simple_review):3 P2 findings. Pause inaccessible after moving into disabled menu; legacy CERA project source missing on new-project action; legacy whole-record TCN without frozen window accepted online. All reproduced as failing tests before fixes. Added policy-save failure test (failed save must not start formal holdout) and fixed memory policy publication after successful write. Model selector now filters explicit versioned windows; both legacy architectures remain usable offline. UI checked1024×640, horizontal scrollbar0, advanced/formal default hidden, understandable model labels, three primary controls.

Final verification: full trainer suite338 passed in124.36s on2026-11-23 after review fixes; compile and git diff --check clean. All reviewed Important findings fixed with RED→GREEN regressions; no deferred findings. Physical CERA inference performance/accuracy remains unmeasured; user may start real training using model selector.
