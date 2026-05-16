"""
Training Data Generator for Universal Preference Adapter (v2.1)

Generates training data in Unsloth ShareGPT format using GPT-4o-mini.
Each sample teaches the adapter:
    Current Preferences + New Evidence → Updated Preferences JSON

Key design:
- Fixed system prompt across all samples
- 4 scenarios: empty, incremental, override, nochange
- 3 field types: communication_style, scheduling, focus_areas
- GPT output is always the FULL updated profile (not delta)
- Separate output files per field type for human review before merging
- Each sample has a unique ID encoding (field_type, scenario, sequence)

Output per field type:
- {field}_train.jsonl   → training data (source of truth)
- {field}_review.md     → human-readable review (read-only view)

Usage:
    python generate_training_data.py                       # Generate 200 samples
    python generate_training_data.py --count 30            # Generate 30 samples
    python generate_training_data.py --merge               # Merge reviewed files
    python generate_training_data.py --validate file.jsonl # Validate format
"""

import json
import os
import sys
import argparse
import asyncio
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path for config imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from openai import AsyncOpenAI

# Fix SSL cert issue (mirrors run_server.py behavior)
if "SSL_CERT_FILE" in os.environ:
    cert_path = os.environ["SSL_CERT_FILE"]
    if cert_path and not os.path.exists(cert_path):
        del os.environ["SSL_CERT_FILE"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================
# Constants
# ============================

CURRENT_MODE = "train" # Can be updated to "test" via CLI

PERSONAS_FILE = Path(__file__).parent.parent / "personas" / "user_personas.json"
OUTPUT_DIR = Path(__file__).parent.parent / "data"

# Fixed system prompt — identical for every training sample.
# CRITICAL: always output the FULL schema skeleton. Empty fields use {} or [].
SYSTEM_PROMPT = (
    "You are a preference extraction model for a meeting assistant. "
    "Given new evidence about a user and their current preference profile, "
    "output the updated preference profile as JSON. "
    "Preserve all existing preferences unless directly contradicted by new evidence. "
    "Always output valid JSON with ALL fields present conforming to this schema: "
    '{"communication_style": {"language": str, "length": str, "format": str, "tone": str}, '
    '"scheduling": {"preferred_times": [str], "unavailable_times": [str]}, '
    '"focus_areas": [str]}. '
    "Use empty strings, empty objects {}, or empty arrays [] for fields with no evidence yet. "
    "Never omit any top-level or nested field from the schema."
)

# Scenario definitions: 4 categories with target ratios
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "empty": {
        "ratio": 0.15,
        "label": "empty_to_initial",
    },
    "incr": {
        "ratio": 0.50,
        "label": "incremental",
    },
    "override": {
        "ratio": 0.15,
        "label": "override",
    },
    "nochange": {
        "ratio": 0.20,
        "label": "no_change",
    },
}

# Field type definitions with ID prefixes
FIELD_TYPES: Dict[str, str] = {
    "communication_style": "comm",
    "scheduling": "sched",
    "focus_areas": "focus",
}


# ============================
# Persona loading
# ============================

def load_personas() -> List[Dict[str, Any]]:
    """Load user personas from the JSON file."""
    with open(PERSONAS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================
# Meta-prompt: scenario × field instructions
# ============================

# Each key is (field_type, scenario). Instructions tell GPT-4o-mini exactly
# what current_preferences + evidence + updated_preferences should look like.
# CRITICAL: Both current and updated must ALWAYS contain the full schema skeleton.
#   Empty fields are represented as {} or [] — never omitted.
_FULL_SKELETON_NOTE = (
    "\n\nIMPORTANT: Both current_preferences and updated_preferences must ALWAYS "
    "contain ALL 3 top-level keys (communication_style, scheduling, focus_areas) "
    "with their complete sub-field structure. "
    "Use empty strings, empty objects {{}}, or empty arrays [] for fields with no data. "
    "Example empty skeleton:\n"
    '{{"communication_style": {{"language": "", "length": "", "format": "", "tone": ""}}, '
    '"scheduling": {{"preferred_times": [], "unavailable_times": []}}, '
    '"focus_areas": []}}'
)

_SCENARIO_FIELD_INSTRUCTIONS: Dict[Tuple[str, str], str] = {
    # ---- communication_style × 4 scenarios ----
    ("communication_style", "empty"): (
        "The current_preferences uses the FULL skeleton with ALL fields empty.\n"
        "The new_evidence should be a user conversation or meeting statement that reveals "
        "the user's COMMUNICATION STYLE preferences (language, length, format, or tone).\n"
        "The updated_preferences must use the FULL skeleton — "
        "fill in the communication_style fields found in the evidence, "
        "keep scheduling and focus_areas empty."
        + _FULL_SKELETON_NOTE
    ),
    ("communication_style", "incr"): (
        "The current_preferences uses the FULL skeleton with some fields filled "
        "(e.g., scheduling or focus_areas have values, "
        "but communication_style sub-fields are mostly empty).\n"
        "The new_evidence should reveal NEW communication style information "
        "(preferred language, response length, format, or tone).\n"
        "The updated_preferences must use the FULL skeleton — "
        "preserve ALL existing values AND fill in the new communication_style fields."
        + _FULL_SKELETON_NOTE
    ),
    ("communication_style", "override"): (
        "The current_preferences uses the FULL skeleton with communication_style filled.\n"
        "The new_evidence should CONTRADICT one communication_style value. For example:\n"
        "  - Current format='bullet_points', user says 'give me detailed paragraphs'\n"
        "  - Current tone='formal', user says 'let's keep it casual'\n"
        "The updated_preferences must use the FULL skeleton — "
        "update the contradicted sub-field, preserve everything else as-is."
        + _FULL_SKELETON_NOTE
    ),
    ("communication_style", "nochange"): (
        "The current_preferences uses the FULL skeleton with some fields filled "
        "(including communication_style).\n"
        "The new_evidence is a NORMAL conversation with NO preference signals.\n"
        "Examples: 'What was discussed in the last meeting?', 'Summarize the Q3 report'\n"
        "The updated_preferences must be IDENTICAL to the current_preferences — "
        "copy every field unchanged."
        + _FULL_SKELETON_NOTE
    ),
    # ---- scheduling × 4 scenarios ----
    ("scheduling", "empty"): (
        "The current_preferences uses the FULL skeleton with ALL fields empty.\n"
        "The new_evidence should reveal the user's SCHEDULING preferences "
        "(preferred meeting times or unavailable times).\n"
        "The updated_preferences must use the FULL skeleton — "
        "fill in the scheduling fields, keep communication_style and focus_areas empty."
        + _FULL_SKELETON_NOTE
    ),
    ("scheduling", "incr"): (
        "The current_preferences uses the FULL skeleton with some fields filled "
        "(e.g., communication_style or focus_areas have values, "
        "but scheduling is partially filled or empty).\n"
        "The new_evidence should reveal NEW scheduling information.\n"
        "The updated_preferences must use the FULL skeleton — "
        "preserve ALL existing values AND add/enrich the scheduling fields."
        + _FULL_SKELETON_NOTE
    ),
    ("scheduling", "override"): (
        "The current_preferences uses the FULL skeleton with scheduling filled.\n"
        "The new_evidence should CONTRADICT one scheduling value. For example:\n"
        "  - unavailable_times includes 'Friday', user says 'this Friday works'\n"
        "  - preferred mornings, user says 'switch to afternoons'\n"
        "The updated_preferences must use the FULL skeleton — "
        "update the contradicted scheduling field, preserve everything else."
        + _FULL_SKELETON_NOTE
    ),
    ("scheduling", "nochange"): (
        "The current_preferences uses the FULL skeleton with some fields filled "
        "(including scheduling).\n"
        "The new_evidence is a NORMAL conversation with NO preference signals.\n"
        "Examples: 'Show me the action items', 'Who attended the sprint review?'\n"
        "The updated_preferences must be IDENTICAL to the current_preferences."
        + _FULL_SKELETON_NOTE
    ),
    # ---- focus_areas × 4 scenarios ----
    ("focus_areas", "empty"): (
        "The current_preferences uses the FULL skeleton with ALL fields empty.\n"
        "The new_evidence should reveal what content the user wants to FOCUS on "
        "(action_items, decisions, technical_details, or all).\n"
        "The updated_preferences must use the FULL skeleton — "
        "fill in focus_areas, keep communication_style and scheduling empty."
        + _FULL_SKELETON_NOTE
    ),
    ("focus_areas", "incr"): (
        "The current_preferences uses the FULL skeleton with some fields filled "
        "(e.g., communication_style or scheduling have values, "
        "but focus_areas is empty or has only 1 item).\n"
        "The new_evidence should reveal NEW focus area preferences.\n"
        "The updated_preferences must use the FULL skeleton — "
        "preserve ALL existing values AND add/enrich focus_areas."
        + _FULL_SKELETON_NOTE
    ),
    ("focus_areas", "override"): (
        "The current_preferences uses the FULL skeleton with focus_areas filled.\n"
        "The new_evidence should CHANGE the user's focus. For example:\n"
        "  - focus_areas=['action_items'], user says 'skip action items, "
        "I need technical details'\n"
        "The updated_preferences must use the FULL skeleton — "
        "replace focus_areas, preserve all other fields."
        + _FULL_SKELETON_NOTE
    ),
    ("focus_areas", "nochange"): (
        "The current_preferences uses the FULL skeleton with some fields filled "
        "(including focus_areas).\n"
        "The new_evidence is a NORMAL conversation with NO preference signals.\n"
        "Examples: 'What did Bob say about the API?', 'When is the next deadline?'\n"
        "The updated_preferences must be IDENTICAL to the current_preferences."
        + _FULL_SKELETON_NOTE
    ),
}


def build_meta_prompt(persona: Dict[str, Any], field_type: str, scenario: str) -> str:
    """
    Build the meta-prompt that instructs GPT-4o-mini to generate one training sample.
    """
    instruction = _SCENARIO_FIELD_INSTRUCTIONS[(field_type, scenario)]
    
    test_mode_str = ""
    if CURRENT_MODE == "test":
        test_mode_str = (
            "\n## TEST SET DIVERSITY REQUIREMENT\n"
            "CRITICAL: You are generating a TEST set to evaluate an AI's semantic parsing. "
            "You MUST use extremely diverse, idiomatic, obscure, or highly conversational "
            "natural language for the new_evidence. \n"
            "Do NOT use simple declarations like 'I prefer concise'. "
            "Instead use slang, passive aggression, implicit hints, corporate jargon, tangents, "
            "or very complex sentences to truly test the extraction capability. "
            "Invent highly out-of-distribution persona traits if needed!\n"
        )

    return f"""Generate exactly 1 training sample for a preference extraction model.
{test_mode_str}
## User Persona (use this for realistic evidence)
- Name: {persona['name']}
- Role: {persona['role']}
- Language: {persona['language']}
- Style: {persona['response_length']}, {persona['response_format']}
- Tone: {persona['tone']}
- Focus: {persona['focus_area']}
- Preferred times: {json.dumps(persona['preferred_meeting_times'])}
- Unavailable: {json.dumps(persona['unavailable_times'])}

## Preference Schema (ALL fields must ALWAYS be present)
```json
{{
  "communication_style": {{
    "language": "en | zh-CN | zh-en-mixed | (empty string if unknown)",
    "length": "concise | moderate | detailed | (empty string if unknown)",
    "format": "bullet_points | paragraphs | tables | structured_sections | (empty string if unknown)",
    "tone": "professional | casual | academic | formal | (empty string if unknown)"
  }},
  "scheduling": {{
    "preferred_times": ["Tuesday morning 9-12"],
    "unavailable_times": ["Friday all day"]
  }},
  "focus_areas": ["action_items", "decisions", "technical_details", "all"]
}}
```
STRICT ENUM WARNING: For `focus_areas`, you MUST ONLY use the EXACT strings provided above (`action_items`, `decisions`, `technical_details`, `all`). Do NOT invent new string values like `user_experience`.

## Empty Skeleton (use this when a field has no data)
```json
{{
  "communication_style": {{"language": "", "length": "", "format": "", "tone": ""}},
  "scheduling": {{"preferred_times": [], "unavailable_times": []}},
  "focus_areas": []
}}
```

## Target Field: {field_type}
## Scenario
{instruction}

## Evidence Types (mix both across calls)
- User-system conversation: "User: ... \\nAssistant: ..."
- Meeting statement: "{persona['name']} said: '...'"

## Output Format
Return a JSON object with exactly 3 fields:
```json
{{
  "current_preferences": {{ FULL skeleton, empty or filled }},
  "new_evidence": "<Text matching the required scenario>",
  "updated_preferences": {{ FULL skeleton after update }}
}}
```

IMPORTANT:
- Make the evidence natural and conversational, not robotic
- Only output valid JSON, no markdown
- BOTH current_preferences and updated_preferences must contain ALL 3 top-level keys
  with their complete sub-field structure. Never omit any field.
- Use empty strings (""), empty objects {{}}, or empty arrays [] for unknown fields
- For nochange scenario, updated_preferences must be identical to current_preferences
"""


# ============================
# Sample generation via GPT
# ============================

def assemble_sharegpt_sample(
    sample_id: str,
    field_type: str,
    scenario: str,
    persona_name: str,
    current_prefs: Dict,
    new_evidence: str,
    updated_prefs: Dict,
) -> Dict[str, Any]:
    """
    Package a single sample into the Unsloth ShareGPT format.

    Includes an 'id' and 'metadata' field for debugging and MD generation.
    Unsloth ignores extra fields outside 'conversations'.
    """
    human_value = (
        f"[Current Preferences]\n{json.dumps(current_prefs, ensure_ascii=False)}"
        f"\n\n[New Evidence]\n{new_evidence}"
    )
    gpt_value = json.dumps(updated_prefs, ensure_ascii=False)

    return {
        "id": sample_id,
        "metadata": {
            "field_type": field_type,
            "scenario": SCENARIOS[scenario]["label"],
            "persona": persona_name,
        },
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": gpt_value},
        ],
    }


async def generate_one_sample(
    client: AsyncOpenAI,
    persona: Dict[str, Any],
    field_type: str,
    scenario: str,
    sample_id: str,
    model: str = "gpt-4o-mini",
) -> Dict[str, Any] | None:
    """
    Generate a single training sample by calling GPT-4o-mini.

    Returns a ShareGPT-formatted dict with ID and metadata, or None on failure.
    """
    prompt = build_meta_prompt(persona, field_type, scenario)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise training data generator. "
                        "Output only a valid JSON object with keys: "
                        "current_preferences, new_evidence, updated_preferences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        current = parsed.get("current_preferences", {})
        evidence = parsed.get("new_evidence", "")
        updated = parsed.get("updated_preferences", {})

        if not evidence:
            logger.warning(f"  [{sample_id}] Empty evidence, skipping")
            return None

        # Validate schema completeness: both must have all 3 top-level keys
        required_keys = {"communication_style", "scheduling", "focus_areas"}
        for label, obj in [("current", current), ("updated", updated)]:
            missing = required_keys - set(obj.keys())
            if missing:
                logger.warning(
                    f"  [{sample_id}] {label} missing keys {missing}, skipping"
                )
                return None

        # Validate that updated_preferences is serializable
        json.dumps(updated, ensure_ascii=False)

        return assemble_sharegpt_sample(
            sample_id, field_type, scenario, persona["name"],
            current, evidence, updated,
        )

    except json.JSONDecodeError as e:
        logger.error(f"  [{sample_id}] JSON parse error: {e}")
        return None
    except Exception as e:
        logger.error(f"  [{sample_id}] API error: {e}")
        return None


# ============================
# Distribution calculation
# ============================

def compute_sample_counts(total: int) -> Dict[str, Dict[str, int]]:
    """
    Compute the number of samples per (field_type, scenario) pair.

    Distributes total evenly across field types, then applies scenario ratios
    within each field type. Returns {field_type: {scenario: count}}.
    """
    n_fields = len(FIELD_TYPES)
    base_per_field = total // n_fields
    remainder = total % n_fields

    result: Dict[str, Dict[str, int]] = {}
    for i, field_type in enumerate(FIELD_TYPES):
        field_total = base_per_field + (1 if i < remainder else 0)

        scenario_counts: Dict[str, int] = {}
        allocated = 0
        scenario_keys = list(SCENARIOS.keys())

        for j, scenario in enumerate(scenario_keys):
            if j == len(scenario_keys) - 1:
                # Last scenario absorbs rounding remainder
                scenario_counts[scenario] = field_total - allocated
            else:
                count = round(field_total * SCENARIOS[scenario]["ratio"])
                scenario_counts[scenario] = count
                allocated += count

        result[field_type] = scenario_counts

    return result


# ============================
# Markdown review file generation
# ============================

def _flatten_prefs(prefs: Dict[str, Any], prefix: str = "") -> List[str]:
    """
    Flatten a nested preference dict into human-readable bullet strings.

    Displays empty values explicitly so reviewers can see the full skeleton.
    Example:
        {"communication_style": {"language": "en", "length": ""}, "focus_areas": []}
        →
        ["communication_style.language: en", "communication_style.length: (empty)",
         "focus_areas: (empty)"]
    """
    lines = []
    for key, value in prefs.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"

        if isinstance(value, dict):
            if not value:
                # Empty dict — e.g. communication_style: {}
                lines.append(f"{full_key}: (empty)")
            else:
                lines.extend(_flatten_prefs(value, full_key))
        elif isinstance(value, list):
            if not value:
                lines.append(f"{full_key}: (empty)")
            else:
                items = ", ".join(str(v) for v in value)
                lines.append(f"{full_key}: {items}")
        elif value == "" or value is None:
            lines.append(f"{full_key}: (empty)")
        else:
            lines.append(f"{full_key}: {value}")

    if not lines:
        return ["(no fields)"]

    return lines


def _format_evidence(evidence: str) -> str:
    """Format evidence text as a markdown blockquote, splitting on newlines."""
    lines = evidence.replace("\\n", "\n").split("\n")
    return "\n".join(f"> {line}" for line in lines if line.strip())


def generate_review_markdown(samples: List[Dict[str, Any]], field_type: str) -> str:
    """
    Generate a human-readable markdown string for reviewing training samples.

    Each case is displayed with its ID, scenario, persona, current preferences,
    evidence text, and updated preferences in a clear, non-JSON format.
    """
    title = field_type.replace("_", " ").title()
    lines = [
        f"# {title} — Training Data Review",
        "",
        f"**Total samples:** {len(samples)}",
        "",
        "---",
        "",
    ]

    for sample in samples:
        sid = sample["id"]
        meta = sample["metadata"]

        # Parse the human turn to extract current_prefs and evidence
        human_value = sample["conversations"][1]["value"]
        gpt_value = sample["conversations"][2]["value"]

        # Split human turn into current prefs and evidence
        parts = human_value.split("\n\n[New Evidence]\n", 1)
        current_str = parts[0].replace("[Current Preferences]\n", "")
        evidence_str = parts[1] if len(parts) > 1 else ""

        # Parse JSON strings back to dicts for flattening
        try:
            current_prefs = json.loads(current_str)
        except json.JSONDecodeError:
            current_prefs = {}

        try:
            updated_prefs = json.loads(gpt_value)
        except json.JSONDecodeError:
            updated_prefs = {}

        # Build case header
        lines.append(
            f"## {sid} | {meta['scenario']} | {meta['persona']}"
        )
        lines.append("")

        # Current preferences
        lines.append("**Current Preferences:**")
        for item in _flatten_prefs(current_prefs):
            lines.append(f"- {item}")
        lines.append("")

        # New evidence
        lines.append("**New Evidence:**")
        lines.append(_format_evidence(evidence_str))
        lines.append("")

        # Updated preferences
        lines.append("**Updated Preferences:**")
        for item in _flatten_prefs(updated_prefs):
            lines.append(f"- {item}")
        lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ============================
# File I/O
# ============================

def save_field_data(field_type: str, samples: List[Dict[str, Any]]) -> None:
    """
    Save samples for one field type as both JSONL (training/test) and MD (review).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSONL for training (source of truth)
    jsonl_path = OUTPUT_DIR / f"{field_type}_{CURRENT_MODE}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Markdown for human review (read-only view)
    md_path = OUTPUT_DIR / f"{field_type}_{CURRENT_MODE}_review.md"
    md_content = generate_review_markdown(samples, field_type)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info(
        f"  Saved {len(samples)} samples → {jsonl_path.name}, {md_path.name}"
    )


# ============================
# Main generation pipeline
# ============================

async def generate_field_data(
    client: AsyncOpenAI,
    personas: List[Dict[str, Any]],
    field_type: str,
    scenario_counts: Dict[str, int],
    model: str = "gpt-4o-mini",
) -> List[Dict[str, Any]]:
    """
    Generate all samples for a single field type.

    Cycles through personas round-robin for diversity.
    Returns a list of ShareGPT-formatted samples.
    """
    prefix = FIELD_TYPES[field_type]
    samples: List[Dict[str, Any]] = []
    persona_idx = 0

    for scenario, count in scenario_counts.items():
        logger.info(
            f"  Generating {count} samples for "
            f"{field_type}/{SCENARIOS[scenario]['label']}..."
        )
        for i in range(count):
            persona = personas[persona_idx % len(personas)]
            persona_idx += 1

            sample_id = f"{prefix}_{scenario}_{i + 1:03d}"

            sample = await generate_one_sample(
                client, persona, field_type, scenario, sample_id, model
            )

            if sample is not None:
                samples.append(sample)
            else:
                logger.warning(f"  [{sample_id}] Failed, skipping")

            await asyncio.sleep(0.2)

    logger.info(
        f"  {field_type}: {len(samples)}/{sum(scenario_counts.values())} "
        f"samples generated"
    )
    return samples


# ============================
# Merge reviewed files
# ============================

def merge_all_files() -> None:
    """Merge all field-type JSONL files into combined.jsonl."""
    all_samples: List[Dict[str, Any]] = []

    for field_type in FIELD_TYPES:
        jsonl_path = OUTPUT_DIR / f"{field_type}_{CURRENT_MODE}.jsonl"
        if not jsonl_path.exists():
            logger.warning(f"  Missing {jsonl_path.name}, skipping")
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))

        logger.info(f"  Loaded {jsonl_path.name}")

    if not all_samples:
        logger.error("No samples found to merge")
        return

    random.shuffle(all_samples)

    combined_jsonl = OUTPUT_DIR / f"combined_{CURRENT_MODE}.jsonl"
    with open(combined_jsonl, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Merged {len(all_samples)} samples → {combined_jsonl.name}")


# ============================
# Validation
# ============================

def validate_file(filepath: Path) -> Dict[str, Any]:
    """
    Validate a JSONL file for correct ShareGPT format.

    Checks: valid JSONL, 3-turn conversations (system→human→gpt),
    GPT output is valid JSON, and each sample has an 'id' field.
    """
    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "has_id": 0,
        "errors": [],
    }

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            stats["total"] += 1
            try:
                sample = json.loads(line.strip())

                if "id" in sample:
                    stats["has_id"] += 1

                turns = sample.get("conversations", [])
                if len(turns) != 3:
                    stats["invalid"] += 1
                    stats["errors"].append(
                        f"Line {i}: expected 3 turns, got {len(turns)}"
                    )
                    continue

                if turns[0]["from"] != "system":
                    stats["invalid"] += 1
                    stats["errors"].append(f"Line {i}: first turn must be 'system'")
                    continue

                if turns[1]["from"] != "human" or turns[2]["from"] != "gpt":
                    stats["invalid"] += 1
                    stats["errors"].append(
                        f"Line {i}: expected human→gpt turn order"
                    )
                    continue

                # Validate GPT output is valid JSON
                json.loads(turns[2]["value"])

                stats["valid"] += 1

            except json.JSONDecodeError:
                stats["invalid"] += 1
                stats["errors"].append(f"Line {i}: invalid JSONL")

    return stats


# ============================
# CLI entry point
# ============================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Preference Adapter training data"
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "test"], default="train",
        help="Mode to run the generator in. 'test' generates overly complex evidence.",
    )
    parser.add_argument(
        "--count", type=int, default=200,
        help="Total samples to generate (default: 200)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="OpenAI model for generation (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge field-type files into combined.jsonl based on mode",
    )
    parser.add_argument(
        "--validate", type=str, metavar="FILE",
        help="Validate an existing JSONL file",
    )
    args = parser.parse_args()

    global CURRENT_MODE
    CURRENT_MODE = args.mode

    # --- Merge mode ---
    if args.merge:
        logger.info("Merging field-type files...")
        merge_all_files()
        return

    # --- Validate mode ---
    if args.validate:
        stats = validate_file(Path(args.validate))
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    # --- Generate mode ---
    personas = load_personas()
    logger.info(f"Loaded {len(personas)} user personas")

    from config.settings import OPENAI_API_KEY
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    counts = compute_sample_counts(args.count)
    logger.info(f"Target: {args.count} samples across {len(FIELD_TYPES)} field types")
    for ft, sc in counts.items():
        logger.info(f"  {ft}: {sc} (total={sum(sc.values())})")

    for field_type in FIELD_TYPES:
        logger.info(f"Generating {field_type}...")

        samples = await generate_field_data(
            client, personas, field_type, counts[field_type], args.model
        )

        if samples:
            save_field_data(field_type, samples)

            # Validate immediately after saving
            jsonl_path = OUTPUT_DIR / f"{field_type}_{CURRENT_MODE}.jsonl"
            stats = validate_file(jsonl_path)
            logger.info(
                f"  Validation: {stats['valid']}/{stats['total']} valid, "
                f"{stats['has_id']}/{stats['total']} have IDs"
            )
            if stats["errors"]:
                for err in stats["errors"][:3]:
                    logger.warning(f"    {err}")

    logger.info("Done! Review the _review.md files, then run --merge to combine.")


if __name__ == "__main__":
    asyncio.run(main())
