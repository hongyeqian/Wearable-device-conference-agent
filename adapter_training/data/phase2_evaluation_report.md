# Phase 2 Evaluation Report — Preference Extraction Adapter

> **Date**: 2026-05-23 23:27  
> **Total Test Samples**: 50

---

## Summary

| Metric | Score |
|---|---|
| JSON Parseable | 50/50 (100%) |
| Schema Compliance | 50/50 (100%) |
| Preservation Rate | 49/50 (98%) |
| LLM Judge — Signal Detection | 4.72/5 |
| LLM Judge — Value Assignment | 4.68/5 |
| LLM Judge — Completeness | 4.68/5 |
| **LLM Judge — Overall** | **4.69/5** |

---

## Breakdown by Scenario

| Scenario | Count | Schema | Preservation | Judge Avg |
|---|---|---|---|---|
| empty_to_initial | 8 | 8/8 | 7/8 | 4.75/5 |
| incremental | 24 | 24/24 | 24/24 | 4.81/5 |
| no_change | 10 | 10/10 | 10/10 | 4.87/5 |
| override | 8 | 8/8 | 8/8 | 4.08/5 |

---

## Flagged Samples (Need Review)

### comm_override_002

- ⚠️ **Low Judge Score**: 1.0/5 — The evidence clearly indicates a preference for a more casual communication style, but the model did not update the preferences to reflect this change.

### comm_empty_002

- ❌ **Preservation Errors**: 'scheduling' changed: {"preferred_times": [], "unavailable_times": []} → {"preferred_times": ["weekday afternoons 14:00-17:00"], "unavailable_times": ["Wednesday all day"]}
