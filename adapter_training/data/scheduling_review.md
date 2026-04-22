# Scheduling — Training Data Review

**Total samples:** 9

---

## sched_empty_001 | empty_to_initial | Alice Chen

**Current Preferences:**
- (empty)

**New Evidence:**
> Alice Chen said: 'I am available on weekday mornings from 9:00 to 12:00, but I'm not available all day Friday or on Monday before 10:00.'

**Updated Preferences:**
- scheduling.preferred_times: weekday mornings 9:00-12:00
- scheduling.unavailable_times: Friday all day, Monday before 10:00

---

## sched_incr_001 | incremental | Bob Zhang

**Current Preferences:**
- communication_style.language: zh-en-mixed
- communication_style.length: detailed
- communication_style.format: paragraphs
- communication_style.tone: casual
- focus_areas: technical_details, all

**New Evidence:**
> User: Oh, by the way, I'm also not available on Fridays all day.
> Assistant: Got it, I've noted that down.

**Updated Preferences:**
- communication_style.language: zh-en-mixed
- communication_style.length: detailed
- communication_style.format: paragraphs
- communication_style.tone: casual
- scheduling.preferred_times: weekday afternoons 14:00-17:00
- scheduling.unavailable_times: Wednesday all day, weekends, Fridays all day
- focus_areas: technical_details, all

---

## sched_incr_002 | incremental | Charlie Wang

**Current Preferences:**
- scheduling.preferred_times: Tuesday and Thursday 10:00-12:00, Friday 14:00-16:00
- scheduling.unavailable_times: Monday all day, Wednesday afternoon
- communication_style.language: en
- communication_style.length: detailed
- communication_style.format: structured_sections
- communication_style.tone: academic
- focus_areas: research_findings

**New Evidence:**
> User: I also have availability on Tuesday evenings. 
> Assistant: Great to know, I'll update that.

**Updated Preferences:**
- scheduling.preferred_times: Tuesday and Thursday 10:00-12:00, Friday 14:00-16:00, Tuesday evening
- scheduling.unavailable_times: Monday all day, Wednesday afternoon
- communication_style.language: en
- communication_style.length: detailed
- communication_style.format: structured_sections
- communication_style.tone: academic
- focus_areas: research_findings

---

## sched_incr_003 | incremental | Diana Liu

**Current Preferences:**
- scheduling.preferred_times: weekday mornings 10:00-12:00
- scheduling.unavailable_times: Thursday all day, Friday after 15:00

**New Evidence:**
> User: I can also meet on Tuesday mornings from 9 to 12.
> Assistant: Great, I'll add that to your preferences.

**Updated Preferences:**
- communication_style.language: zh-CN
- communication_style.length: moderate
- communication_style.format: bullet_points
- communication_style.tone: friendly
- scheduling.preferred_times: weekday mornings 10:00-12:00, Tuesday morning 9-12
- scheduling.unavailable_times: Thursday all day, Friday after 15:00
- focus_areas: user_experience

---

## sched_incr_004 | incremental | Edward Park

**Current Preferences:**
- communication_style.language: en
- communication_style.length: concise
- communication_style.format: tables
- communication_style.tone: formal
- focus_areas: data_metrics

**New Evidence:**
> Edward Park said: 'I am available on Monday and Wednesday between 09:00 and 11:00.'

**Updated Preferences:**
- communication_style.language: en
- communication_style.length: concise
- communication_style.format: tables
- communication_style.tone: formal
- scheduling.preferred_times: Monday and Wednesday 09:00-11:00
- scheduling.unavailable_times: Tuesday all day, Friday after 14:00
- focus_areas: data_metrics

---

## sched_override_001 | override | Fiona Martinez

**Current Preferences:**
- communication_style.language: en
- communication_style.length: moderate
- communication_style.format: bullet_points
- communication_style.tone: direct
- scheduling.preferred_times: weekday afternoons 13:00-15:00
- scheduling.unavailable_times: Monday morning before 11:00, Friday all day
- focus_areas: decisions_and_blockers

**New Evidence:**
> Fiona Martinez said: 'I can actually do a meeting this Friday.'

**Updated Preferences:**
- communication_style.language: en
- communication_style.length: moderate
- communication_style.format: bullet_points
- communication_style.tone: direct
- scheduling.preferred_times: weekday afternoons 13:00-15:00
- scheduling.unavailable_times: Monday morning before 11:00
- focus_areas: decisions_and_blockers

---

## sched_nochange_001 | no_change | Grace Kim

**Current Preferences:**
- communication_style.language: en
- communication_style.length: detailed
- communication_style.format: structured_sections
- communication_style.tone: methodical
- scheduling.preferred_times: Tuesday and Thursday 14:00-16:00
- scheduling.unavailable_times: Wednesday all day, Monday morning
- focus_areas: quality_issues

**New Evidence:**
> User: Can you remind me what the release criteria are?
> Assistant: Yes, the release criteria include thorough testing and validation of quality metrics.

**Updated Preferences:**
- communication_style.language: en
- communication_style.length: detailed
- communication_style.format: structured_sections
- communication_style.tone: methodical
- scheduling.preferred_times: Tuesday and Thursday 14:00-16:00
- scheduling.unavailable_times: Wednesday all day, Monday morning
- focus_areas: quality_issues

---

## sched_nochange_002 | no_change | Hongye Qian

**Current Preferences:**
- communication_style.language: zh-en-mixed
- communication_style.length: moderate
- communication_style.format: structured_sections
- communication_style.tone: professional
- scheduling.preferred_times: Tuesday morning 9-12, weekday afternoons 14-17
- scheduling.unavailable_times: Friday all day, Monday before 10:00
- focus_areas: action_items, decisions, technical_details, all

**New Evidence:**
> User: Can we review the technical details from the last meeting?
> Assistant: Sure, let me pull that up.

**Updated Preferences:**
- communication_style.language: zh-en-mixed
- communication_style.length: moderate
- communication_style.format: structured_sections
- communication_style.tone: professional
- scheduling.preferred_times: Tuesday morning 9-12, weekday afternoons 14-17
- scheduling.unavailable_times: Friday all day, Monday before 10:00
- focus_areas: action_items, decisions, technical_details, all

---

## sched_nochange_003 | no_change | Ivan Petrov

**Current Preferences:**
- communication_style.language: en
- communication_style.length: concise
- communication_style.format: bullet_points
- communication_style.tone: technical
- scheduling.preferred_times: Monday and Friday 10:00-12:00
- scheduling.unavailable_times: Tuesday afternoon, Thursday all day
- focus_areas: infrastructure_updates

**New Evidence:**
> User: Can you remind me about the infrastructure updates?
> Assistant: Sure, I will provide a summary.

**Updated Preferences:**
- communication_style.language: en
- communication_style.length: concise
- communication_style.format: bullet_points
- communication_style.tone: technical
- scheduling.preferred_times: Monday and Friday 10:00-12:00
- scheduling.unavailable_times: Tuesday afternoon, Thursday all day
- focus_areas: infrastructure_updates

---
