# Threshold Tuning TODO

## Current Status
- Initial refactor is complete.
- Round 1 rerun is complete under F1+Recall objective.
- Latest applied value: `matcher.projection_threshold = 0.67`.
- Round 2 is complete under F1+Recall objective.
- Latest applied value: `matcher.insight_threshold = 0.64`.
- Scope: tune 9 similarity thresholds one-by-one, with review gate after each round.

## Review Gate (per round)
- Deliverables for review before next round:
  - parameter-specific report (old vs new, metrics, tradeoff notes)
  - key positive/negative test results
  - synthetic evaluation summary
  - `mas/config.py` diff for that round

## Data and Test Requirements (per round)
- Synthetic dataset:
  - medium-large scale positive/negative cases (target 300-800 labeled pairs)
  - include boundary samples near threshold decision points
- Objective:
  - direct F1 + Recall tuning (F1 primary, Recall tie-break)
  - report macro Precision/Recall/F1 and positive-class Precision/Recall/F1

## Tuning Methods Standard (Apply To Every Next Round)
- Data construction:
  - Use fully synthetic AIGC-style data only.
  - Keep medium-large scale and include boundary samples near threshold.
  - Persist artifacts in `tests/optim/artifacts/`.
- Experiment protocol:
  - Use fixed multi-seed runs (default 5 seeds).
  - Perform two-stage scan: coarse scan + fine scan around local optimum.
  - Keep train/valid split and report both selection-side and validation-side metrics.
- Runtime consistency:
  - Reuse runtime embedding/cosine stack in tuning scripts.
  - Ensure threshold is wired to real runtime path before tuning.
- Test strategy:
  - Add explicit positive/negative unit tests for parameter behavior.
  - Add side-effect/contamination tests when parameter affects routing or side-specific outputs.
  - Keep key-case regression set for review readability.
- Deliverables:
  - Parameter report (`reports/round*_*.md`)
  - Scan table + valid comparison json
  - Key-case json
  - `mas/config.py` diff
  - `TODO.md` round status/metrics update

## Tuning Metrics Standard (Apply To Every Next Round)
- Core metrics:
  - Positive-class `Precision+ / Recall+ / F1+`
  - Macro `Precision / Recall / F1`
- Auxiliary metrics:
  - Confusion-matrix counts (`TP/FP/TN/FN`)
  - Task-specific risk metric (e.g., contamination rate for side-specific retrieval)
  - Style/data quality metrics when text format is constrained
- Selection policy:
  - Primary: maximize `F1+`
  - Tie-break: maximize `Recall+`
  - Additional tie-breakers: task-specific quality metrics, then threshold stability
- Validation policy:
  - Always compare baseline vs tuned with absolute deltas.
  - Record both gains and regressions explicitly.

## Completed Round Records (Methods + Metrics)
- Round 1 (`projection_threshold`):
  - Method: synthetic projection pairs + 5-seed threshold scan + key positive/negative cases.
  - Baseline -> tuned: `0.89 -> 0.67`.
  - Validation metrics:
    - `F1+`: `0.1220 -> 0.7390`
    - `Recall+`: `0.0692 -> 0.9896`
    - `Precision+`: `0.5128 -> 0.5897`
    - `Macro F1`: `0.3958 -> 0.6324`
- Round 2 (`insight_threshold`):
  - Method: synthetic retrieval+dedup dual-task tuning + side contamination audit + style-constrained insight generation.
  - Baseline -> tuned: `0.70 -> 0.64`.
  - Validation metrics:
    - Retrieval `F1+`: `0.6049 -> 0.7133`
    - Retrieval `Recall+`: `0.6307 -> 0.9119`
    - Retrieval contamination: `0.2042 -> 0.1752`
    - Dedup `F1+`: `0.7336 -> 0.7083` (regression recorded)
    - Style compliance: `100%`, actionable insight rate: `100%`

## Parameters and Round Plan

### Round 1 - `projection_threshold`
- File: `mas/config.py` (`matcher.projection_threshold`)
- Baseline in latest rerun: `0.89`
- Current applied value: `0.67`
- Focus:
  - projection match quality under noisy history cases
  - maximize positive-class F1 while preserving recall
- Status: `completed-awaiting-review`

### Round 2 - `insight_threshold`
- File: `mas/config.py` (`matcher.insight_threshold`)
- Current baseline: `0.70`
- Current applied value: `0.64`
- Focus:
  - insight retrieval precision vs coverage
  - side-specific contamination checks
- Status: `completed-awaiting-review`

### Round 3 - `fact_threshold`
- File: `mas/config.py` (`dedup.fact_threshold`)
- Current baseline: `0.95`
- Focus:
  - fact-node dedup false merge risk vs missed dedup
- Status: `pending`

### Round 4 - `other_threshold`
- File: `mas/config.py` (`dedup.other_threshold`)
- Current baseline: `0.90`
- Focus:
  - non-fact dedup (LAW/CLAIM) precision under semantic overlap noise
- Status: `pending`

### Round 5 - `fact_worker_min_similarity`
- File: `mas/config.py` (`worker_threshold.fact_worker_threshold`)
- Current baseline: `0.60`
- Focus:
  - ES fact retrieval filtering threshold
  - FP-heavy query suppression
- Status: `pending`

### Round 6 - `law_worker_min_similarity`
- File: `mas/config.py` (`worker_threshold.law_worker_threshold`)
- Current baseline: `0.60`
- Focus:
  - law retrieval filtering threshold
  - irrelevant statute injection reduction
- Status: `pending`

### Round 7 - `semantic_matcher_default_threshold`
- File: `mas/config.py` (`matcher.semantic_default_threshold`)
- Current baseline: `0.85`
- Focus:
  - default semantic matcher behavior consistency
  - downstream dedup sensitivity
- Status: `pending`

### Round 8 - `insight_fallback_min_similarity`
- File: `mas/config.py` (`matcher.insight_fallback_threshold`)
- Current baseline: `0.70`
- Focus:
  - fallback matching robustness when direct insight hit is missing
- Status: `pending`

### Round 9 - `law_jaccard_min_similarity`
- File: `mas/config.py` (`retrieval.law_jaccard_min_similarity`)
- Current baseline: `0.00` (currently used as `sim > min_sim`)
- Focus:
  - jurisprudence path noise filtering by law-overlap strength
- Status: `pending`

## Initial Refactor Changes Completed (before rounds)
- Added new config thresholds to `mas/config.py`:
  - `matcher.semantic_default_threshold`
  - `matcher.insight_fallback_threshold`
  - `retrieval.law_jaccard_min_similarity`
  - `worker_threshold.fact_worker_threshold`
  - `worker_threshold.law_worker_threshold`
- Wired usage in runtime:
  - `mas/core/system.py` for default semantic matcher threshold
  - `mas/agents/team.py` for Fact/Law worker thresholds
  - `mas/memory/insights.py` for fallback insight threshold
  - `mas/memory/legal_memory.py` for law Jaccard minimum similarity
