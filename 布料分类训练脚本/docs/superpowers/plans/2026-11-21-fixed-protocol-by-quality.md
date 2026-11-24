# Fixed protocol and By quality implementation plan

> Execute inline with executing-plans and test-driven-development.

Goal: new sessions use fixed 51–140 mm outgoing / 30–120 mm return, automatically import 布料采集数据新, and evaluate only selected input axes.

Approved scope: user requested these three changes after the read-only By audit. Keep raw CSV/metadata, existing snapshots, saved projects and models unchanged. Ordinary startup imports the directory instead of restoring the previous frozen project; explicit --project / --import-directory override startup defaults. Screenshots and workers do not auto-import.

Files: config.py defines versioned defaults; sample_quality.py computes channel quality without changing sources; data.py applies it and records provenance; admission.py normalizes old serialized settings; settings_ui.py exposes the active rule and restore defaults; train_cloth.py selects startup source; tests cover behavior and compatibility; README documents defaults.

- [x] Write failing tests for default boundaries, legacy deserialization, startup precedence and By-only quality with true gaps / actual By spikes still rejected.
- [x] Implement selected-axis rule, preserving legacy recorded-validity policy for saved data settings missing a quality version. Preserve unknown invalid reasons.
- [x] Switch protocol defaults and startup import; update protocol UI wording and restore behavior.
- [x] Run focused and full suites; resolve regressions while keeping older relative-protocol tests explicit about their historical settings.
- [x] Run the production import path on the entire live source directory and inspect remaining rejected records. Review diff and document results.

Ruling: modify the existing dirty workspace because the requested change builds on its uncommitted TCN UI and recognition work; do not stash, reset, or relocate those changes. No automatic commits or hardware actions.

Verification: 253 tests passed in the complete offscreen suite. The production `train_cloth.py check` command scanned 651 candidates: 649 eligible, 1 excluded for directional inconsistency (顺毛双面呢_001), 1 incomplete-coverage record during ongoing collection. Raw source files were not edited. Actual protocol dialog screenshot reviewed with inherited application theme. Independent review identified and verified the fix for old review-batch diagnostics compatibility.

Ruling: compute quality on the feature channel (dBy when available); missing raw By provenance still produces undetermined admission rather than silently interpolating or accepting it. Preserve independent per-channel views for legacy multi-input comparisons. Old default-boundary tests now explicitly use the historical protocol; fixed-window endpoint and inference mismatch tests use the new default boundaries and an actual speed mismatch.
