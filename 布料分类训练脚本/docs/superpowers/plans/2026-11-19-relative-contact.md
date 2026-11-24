# Relative contact windows implementation plan

**Goal:** Apply the user's confirmed start-relative 44–144mm contact window to captures 035 onward, with the same spatial window on return (11–111mm from the 155mm turnaround).

**Architecture:** Version the contact rule in DataConfig. Determine actual origin from the first outgoing position, validate complete 155mm out-and-back coverage, preserve raw arrays and expose absolute bounds and relative distances in diagnostics. Earlier captures keep existing windows. Old serialized configurations lacking the new field use the fixed-X rule.

**Constraints:** Never modify raw CSV/metadata, weaken quality checks, start training, or overwrite old snapshots/models. Use metadata sequence with canonical filename fallback; conflicting identifiers fail review. Mixed old/new contact lengths must remain trainable with the existing variable-length batching.

### Task 1: Versioned cropping and compatibility
- [x] Add regression tests for negative/positive origins, sequence boundary 034/035, incomplete paths, conflicting sequence, mixed-length protocol, and old config dispatch; run failing tests.
- [x] Implement the rule, serialized compatibility, protocol matching and UI explanation.
- [x] Run focused and full test suite, then re-import real 035–045 CSVs and export fresh results without training.

**Review focus:** Raw sources unchanged; missing/truncated origin not silently inferred from min/max; unknown sequence retains fixed windows with explanation; old inference uses fixed coordinates; new contact gaps still fail.

Validation: 228 tests passed (94.25s); real 035–045 review 167 CSVs / 140 eligible / 27 contact gaps >20ms. Source hashes unchanged. Old 291-record snapshot loaded with fixed-x-v1. Mixed old/new 10-record, two-class snapshot roundtrip succeeded. No formal training or current-window restart performed.
