# Cumulative-prefix CERA and responsive desktop implementation plan
> Execute inline with executing-plans and TDD; one fresh final reviewer.
Goal: approved spec docs/superpowers/specs/2026-11-23-prefix-stop-responsive.md.
Architecture: new prefix candidate/model revision reuses CERA and parent loss; prefix-aware evaluation selects final-prefix classification (never averages early evidence into final prediction). Explicit policy version and locking fields preserve old policy fingerprints. Focused desktop mixin caches metadata through isolated parser jobs and async I/O. Batched live events feed an incremental visualization; legacy UI backend remains callable.
- [x] TDD prefix spec/candidate/parent slicing, random subset training, complete evaluation/checkpoint/formal load tests; retain legacy revisions.
- [x] TDD cumulative online buffer, consecutive/time gate, lock/no further inference, quality/reset/end and policy compatibility, worker signal timeline/summary.
- [x] TDD process catalog/detail cache/latest-wins/cancel/heartbeat; fast analytic model preview, async dataset operations and focused UI routing.
- [x] TDD default prefix editor, four pages, clean visual status/timeline, buffered results with row review/class matrix, pause/export, responsive close.
- [x] Full suite, heartbeat under slow work, synthetic prefix end-to-end training/replay, offscreen narrow screenshots; fresh final review/fix and documentation.
Review focus: history memory after faults, lock persistence against later data faults, no future labels/probabilities in stopping, no metadata/GIL heavy work on GUI, stale async results, frozen test lock, old model/policy exact hashes, variable-prefix loss and normalization.

Final verification: 356 passed in134.81s (`QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q`, /tmp/cloth-prefix-final-all.log). Latest UI labels: 8 passed (/tmp/cloth-final-ui-literals.log). Compileall and git diff --check passed; CLI screenshot startup/shutdown exited0. No real-data training or acquisition action.
