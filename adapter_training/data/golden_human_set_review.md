# Golden Human Set - Manual Review Document

This document contains the 15 mock predictions alongside their expected human scores for manual verification.

## Case: `01_comm_empty_good`
**Scenario:** `empty_to_initial` | **Target Field:** `communication_style`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 2. New Evidence:
> Please write my summaries in English, use bullet points, and keep the tone professional.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "en",
    "length": "",
    "format": "bullet_points",
    "tone": "professional"
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 5
- **Value Assignment**: 5
- **Completeness**: 5
- **Reasoning**: Perfect extraction of all elements.

---

## Case: `02_comm_empty_medium`
**Scenario:** `empty_to_initial` | **Target Field:** `communication_style`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 2. New Evidence:
> Please write my summaries in English, use bullet points, and keep the tone professional.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "en",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 3
- **Value Assignment**: 5
- **Completeness**: 3
- **Reasoning**: Missed tone and format, only extracted language.

---

## Case: `03_comm_empty_bad`
**Scenario:** `empty_to_initial` | **Target Field:** `communication_style`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 2. New Evidence:
> Please write my summaries in English, use bullet points, and keep the tone professional.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "zh-CN",
    "length": "detailed",
    "format": "paragraphs",
    "tone": "casual"
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 1
- **Value Assignment**: 1
- **Completeness**: 1
- **Reasoning**: Total hallucination, completely contradicts the evidence.

---

## Case: `04_sched_incr_good`
**Scenario:** `incremental` | **Target Field:** `scheduling`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "en",
    "length": "concise",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [
      "Monday 9am"
    ],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 2. New Evidence:
> By the way, I am now completely unavailable on Friday afternoons.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "en",
    "length": "concise",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [
      "Monday 9am"
    ],
    "unavailable_times": [
      "Friday afternoon"
    ]
  },
  "focus_areas": []
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 5
- **Value Assignment**: 5
- **Completeness**: 5
- **Reasoning**: Correctly added Friday afternoon to unavailable_times.

---

## Case: `05_sched_incr_medium`
**Scenario:** `incremental` | **Target Field:** `scheduling`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "en",
    "length": "concise",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [
      "Monday 9am"
    ],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 2. New Evidence:
> By the way, I am now completely unavailable on Friday afternoons.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "en",
    "length": "concise",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [
      "Monday 9am",
      "Friday afternoon"
    ],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 4
- **Value Assignment**: 1
- **Completeness**: 5
- **Reasoning**: Found the time, but put it into preferred_times instead of unavailable_times.

---

## Case: `06_sched_incr_bad`
**Scenario:** `incremental` | **Target Field:** `scheduling`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "en",
    "length": "concise",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [
      "Monday 9am"
    ],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 2. New Evidence:
> By the way, I am now completely unavailable on Friday afternoons.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "en",
    "length": "concise",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [
      "Monday 9am"
    ],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 1
- **Value Assignment**: 1
- **Completeness**: 1
- **Reasoning**: Completely ignored the evidence, made no changes.

---

## Case: `07_focus_over_good`
**Scenario:** `override` | **Target Field:** `focus_areas`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "technical_details"
  ]
}
```

### 2. New Evidence:
> Actually, let's stop focusing on the technical details. I only care about action items from now on.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "action_items"
  ]
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 5
- **Value Assignment**: 5
- **Completeness**: 5
- **Reasoning**: Correctly overwrote technical_details with action_items.

---

## Case: `08_focus_over_medium`
**Scenario:** `override` | **Target Field:** `focus_areas`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "technical_details"
  ]
}
```

### 2. New Evidence:
> Actually, let's stop focusing on the technical details. I only care about action items from now on.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "technical_details",
    "action_items"
  ]
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 4
- **Value Assignment**: 3
- **Completeness**: 3
- **Reasoning**: Added action_items but failed to remove technical_details as requested.

---

## Case: `09_focus_over_bad`
**Scenario:** `override` | **Target Field:** `focus_areas`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "technical_details"
  ]
}
```

### 2. New Evidence:
> Actually, let's stop focusing on the technical details. I only care about action items from now on.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "all"
  ]
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 1
- **Value Assignment**: 1
- **Completeness**: 1
- **Reasoning**: Missed the specific signals and hallucinated 'all'.

---

## Case: `10_nochange_good`
**Scenario:** `no_change` | **Target Field:** `communication_style`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "en",
    "length": "moderate",
    "format": "paragraphs",
    "tone": "friendly"
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 2. New Evidence:
> Could you summarize what John said about the new marketing campaign?

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "en",
    "length": "moderate",
    "format": "paragraphs",
    "tone": "friendly"
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 5
- **Value Assignment**: 5
- **Completeness**: 5
- **Reasoning**: Correctly identified that there are no new preference signals.

---

## Case: `11_nochange_medium`
**Scenario:** `no_change` | **Target Field:** `communication_style`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "en",
    "length": "moderate",
    "format": "paragraphs",
    "tone": "friendly"
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 2. New Evidence:
> Could you summarize what John said about the new marketing campaign?

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "en",
    "length": "moderate",
    "format": "paragraphs",
    "tone": "friendly"
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "marketing"
  ]
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 2
- **Value Assignment**: 2
- **Completeness**: 3
- **Reasoning**: Over-interpreted a normal question as a persistent preference (added 'marketing').

---

## Case: `12_nochange_bad`
**Scenario:** `no_change` | **Target Field:** `communication_style`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "en",
    "length": "moderate",
    "format": "paragraphs",
    "tone": "friendly"
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 2. New Evidence:
> Could you summarize what John said about the new marketing campaign?

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "en",
    "length": "concise",
    "format": "bullet_points",
    "tone": "formal"
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": []
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 1
- **Value Assignment**: 1
- **Completeness**: 1
- **Reasoning**: Hallucinated massive changes to communication style with zero evidence.

---

## Case: `13_comm_incr_good`
**Scenario:** `incremental` | **Target Field:** `communication_style`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "zh-en-mixed",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "decisions"
  ]
}
```

### 2. New Evidence:
> I like how detailed the last report was, let's keep that length. But use a more academic tone.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "zh-en-mixed",
    "length": "detailed",
    "format": "",
    "tone": "academic"
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "decisions"
  ]
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 5
- **Value Assignment**: 5
- **Completeness**: 5
- **Reasoning**: Updated length to 'detailed' and tone to 'academic'.

---

## Case: `14_comm_incr_medium`
**Scenario:** `incremental` | **Target Field:** `communication_style`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "zh-en-mixed",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "decisions"
  ]
}
```

### 2. New Evidence:
> I like how detailed the last report was, let's keep that length. But use a more academic tone.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "zh-en-mixed",
    "length": "detailed",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "decisions"
  ]
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 4
- **Value Assignment**: 5
- **Completeness**: 3
- **Reasoning**: Updated length but completely missed the instruction about tone.

---

## Case: `15_comm_incr_bad`
**Scenario:** `incremental` | **Target Field:** `communication_style`

### 1. Current Preferences:
```json
{
  "communication_style": {
    "language": "zh-en-mixed",
    "length": "",
    "format": "",
    "tone": ""
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "decisions"
  ]
}
```

### 2. New Evidence:
> I like how detailed the last report was, let's keep that length. But use a more academic tone.

### 3. Predicted Output (Mock):
```json
{
  "communication_style": {
    "language": "zh-en-mixed",
    "length": "concise",
    "format": "",
    "tone": "casual"
  },
  "scheduling": {
    "preferred_times": [],
    "unavailable_times": []
  },
  "focus_areas": [
    "decisions"
  ]
}
```

### 4. EXPECTED HUMAN SCORE:
- **Signal Detection**: 1
- **Value Assignment**: 1
- **Completeness**: 1
- **Reasoning**: Assigned the exact opposite values ('concise' and 'casual').

---

