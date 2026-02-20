# Threshold Tuning TODO (Pre-Round Stage)

## Current Status
- Initial refactor is complete.
- Round-based tuning has **not started** yet.
- Scope: tune 9 similarity thresholds one-by-one, with review gate after each round.

## Review Gate (per round)
- Deliverables for review before next round:
  - parameter-specific report (old vs new, metrics, tradeoff notes)
  - key positive/negative test results
  - real-DB evaluation summary
  - `mas/config.py` diff for that round

## Data and Test Requirements (per round)
- Synthetic dataset:
  - medium-large scale positive/negative cases (target 300-800 labeled pairs)
  - include boundary samples near threshold decision points
- Real database dataset:
  - sample from existing ES indices (`rag_legal_cases`, `rag_legal_laws`)
  - light manual labeling for relevance
  - target 15-30 queries per domain, top-10 docs per query for audit
- Objective:
  - cost-sensitive scoring with FP:FN = 8:2
  - report macro Precision/Recall/F1 and cost score

## Parameters and Round Plan

### Round 1 - `projection_threshold`
- File: `mas/config.py` (`matcher.projection_threshold`)
- Current baseline: `0.58`
- Focus:
  - projection match quality under noisy history cases
  - false projection penalty (high priority)
- Status: `pending`

### Round 2 - `insight_threshold`
- File: `mas/config.py` (`matcher.insight_threshold`)
- Current baseline: `0.70`
- Focus:
  - insight retrieval precision vs coverage
  - side-specific contamination checks
- Status: `pending`

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

