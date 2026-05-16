# Phase 2 Evaluation Report — Preference Extraction Adapter

> **Date**: 2026-05-03 01:46  
> **Total Test Samples**: 50

---

## Summary

| Metric | Score |
|---|---|
| JSON Parseable | 50/50 (100%) |
| Schema Compliance | 50/50 (100%) |
| Preservation Rate | 49/50 (98%) |
| LLM Judge — Signal Detection | 4.68/5 |
| LLM Judge — Value Assignment | 4.60/5 |
| LLM Judge — Completeness | 4.68/5 |
| **LLM Judge — Overall** | **4.65/5** |

---

## Breakdown by Scenario

| Scenario | Count | Schema | Preservation | Judge Avg |
|---|---|---|---|---|
| empty_to_initial | 8 | 8/8 | 7/8 | 4.75/5 |
| incremental | 24 | 24/24 | 24/24 | 4.72/5 |
| no_change | 10 | 10/10 | 10/10 | 4.87/5 |
| override | 8 | 8/8 | 8/8 | 4.08/5 |

---

## Flagged Samples (Need Review)

### comm_override_002

- ⚠️ **Low Judge Score**: 1.0/5 — The evidence clearly indicates a preference for a more casual communication style, but the model failed to update the tone from 'academic' to something more casual.

### sched_incr_006

- ⚠️ **Low Judge Score**: 2.3/5 — The model detected a scheduling signal but incorrectly updated unavailable times instead of adding Wednesday as a preferred time.

### comm_empty_002

- ❌ **Preservation Errors**: 'scheduling' changed: {"preferred_times": [], "unavailable_times": []} → {"preferred_times": ["weekday afternoons 14:00-17:00"], "unavailable_times": ["Wednesday all day"]}
