"""
Training Data Quality Judge for Universal Preference Adapter

Validates generated training data using both deterministic checks and
GPT-4o LLM-as-Judge for semantic correctness.

Two-layer validation:
1. Deterministic checks (no LLM cost):
   - Schema completeness: all 3 top-level keys with correct sub-fields
   - Preservation: non-target fields identical between current and updated
   - No-change identity: updated === current for nochange samples

2. LLM-as-Judge (GPT-4o):
   - Correctness: does the update correctly reflect the evidence?

Outputs a markdown report with summary stats and flagged samples.

Usage:
    python judge_training_data.py                              # Judge all field files
    python judge_training_data.py --file combined_train.jsonl  # Judge a specific file
    python judge_training_data.py --no-llm                     # Deterministic checks only
"""

import json
import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add project root to path for config imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from openai import AsyncOpenAI

# Fix SSL cert issue
if "SSL_CERT_FILE" in os.environ:
    cert_path = os.environ["SSL_CERT_FILE"]
    if cert_path and not os.path.exists(cert_path):
        del os.environ["SSL_CERT_FILE"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Field types and their expected target fields for preservation checks
FIELD_TYPES = ["communication_style", "scheduling", "focus_areas"]

# Expected sub-field keys for schema validation
EXPECTED_SCHEMA = {
    "communication_style": {"language", "length", "format", "tone"},
    "scheduling": {"preferred_times", "unavailable_times"},
    "focus_areas": None,  # Array, no sub-keys
}


# ============================
# Data loading
# ============================

def load_samples(filepath: Path) -> List[Dict[str, Any]]:
    """Load training samples from a JSONL file."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def parse_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract current_preferences, evidence, and updated_preferences
    from a ShareGPT-format sample.
    """
    human_value = sample["conversations"][1]["value"]
    gpt_value = sample["conversations"][2]["value"]

    parts = human_value.split("\n\n[New Evidence]\n", 1)
    current_str = parts[0].replace("[Current Preferences]\n", "")
    evidence_str = parts[1] if len(parts) > 1 else ""

    current = json.loads(current_str)
    updated = json.loads(gpt_value)

    return {
        "id": sample.get("id", "unknown"),
        "metadata": sample.get("metadata", {}),
        "current": current,
        "evidence": evidence_str,
        "updated": updated,
    }


# ============================
# Deterministic checks
# ============================

def check_schema(
    prefs: Dict[str, Any], label: str
) -> List[str]:
    """
    Validate that a preference dict contains the full schema skeleton.

    Returns a list of error messages (empty if valid).
    """
    errors = []

    # Check top-level keys
    for key in FIELD_TYPES:
        if key not in prefs:
            errors.append(f"{label} missing top-level key: {key}")
            continue

        value = prefs[key]

        if key == "focus_areas":
            if not isinstance(value, list):
                errors.append(f"{label}.focus_areas should be a list, got {type(value).__name__}")
        elif key in EXPECTED_SCHEMA and EXPECTED_SCHEMA[key] is not None:
            if not isinstance(value, dict):
                errors.append(f"{label}.{key} should be a dict, got {type(value).__name__}")
            else:
                missing = EXPECTED_SCHEMA[key] - set(value.keys())
                if missing:
                    errors.append(f"{label}.{key} missing sub-fields: {missing}")

    return errors


def check_preservation(
    current: Dict[str, Any],
    updated: Dict[str, Any],
    target_field: str,
) -> List[str]:
    """
    Verify that non-target fields are preserved between current and updated.

    For an 'incremental' communication_style sample, scheduling and focus_areas
    in updated must match current exactly.
    """
    errors = []

    for field in FIELD_TYPES:
        if field == target_field:
            continue

        current_val = current.get(field)
        updated_val = updated.get(field)

        if current_val != updated_val:
            errors.append(
                f"Preservation failed for '{field}': "
                f"current={json.dumps(current_val, ensure_ascii=False)}, "
                f"updated={json.dumps(updated_val, ensure_ascii=False)}"
            )

    return errors


def check_nochange(
    current: Dict[str, Any],
    updated: Dict[str, Any],
) -> List[str]:
    """For nochange samples, updated must be identical to current."""
    if current != updated:
        # Find which fields differ
        diffs = []
        for key in set(list(current.keys()) + list(updated.keys())):
            if current.get(key) != updated.get(key):
                diffs.append(key)
        return [f"nochange but fields differ: {diffs}"]
    return []


def run_deterministic_checks(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run all deterministic checks on a single parsed sample.

    Returns a result dict with pass/fail status and any errors.
    """
    sid = parsed["id"]
    meta = parsed["metadata"]
    scenario = meta.get("scenario", "")
    target_field = meta.get("field_type", "")

    result = {
        "id": sid,
        "scenario": scenario,
        "target_field": target_field,
        "schema_errors": [],
        "preservation_errors": [],
        "nochange_errors": [],
        "deterministic_pass": True,
    }

    # 1. Schema completeness for both current and updated
    result["schema_errors"].extend(
        check_schema(parsed["current"], "current")
    )
    result["schema_errors"].extend(
        check_schema(parsed["updated"], "updated")
    )

    # 2. Preservation check (non-target fields unchanged)
    if scenario != "empty_to_initial" and target_field:
        result["preservation_errors"] = check_preservation(
            parsed["current"], parsed["updated"], target_field
        )

    # 3. No-change identity check
    if scenario == "no_change":
        result["nochange_errors"] = check_nochange(
            parsed["current"], parsed["updated"]
        )

    # Overall deterministic pass/fail
    all_errors = (
        result["schema_errors"]
        + result["preservation_errors"]
        + result["nochange_errors"]
    )
    result["deterministic_pass"] = len(all_errors) == 0

    return result


# ============================
# LLM-as-Judge (GPT-4o)
# ============================

JUDGE_PROMPT = """You are evaluating a training sample for a preference extraction model.
The model should update a user's preference profile based on new evidence from conversations.

## Sample Details
- **Scenario**: {scenario}
- **Target field**: {target_field}

### Current Preferences
```json
{current}
```

### New Evidence
{evidence}

### Updated Preferences (model output)
```json
{updated}
```

## Evaluation Criteria

**Correctness** (1-5): Does the updated_preferences correctly reflect the evidence?
- 5: Perfect — the update exactly matches what the evidence indicates
- 4: Good — minor issues but essentially correct
- 3: Acceptable — partially correct but some fields wrong or hallucinated
- 2: Poor — significant errors in the update
- 1: Wrong — the update does not reflect the evidence at all

For "no_change" scenarios: the evidence MUST NOT contain any preference signals. \
If the evidence contains a preference signal (and updated == current), it means the generator failed the instruction. Score 1. \
If the evidence does not contain preference signals, score 5.
For "override" scenarios: check that ONLY the contradicted field changed.
For "empty_to_initial": check that only the target field is populated, others remain empty.
For "incremental": check that new fields are added AND existing fields are preserved.

Return ONLY a JSON object:
{{"score": <1-5>, "reason": "<brief explanation>"}}
"""


async def judge_one_sample(
    client: AsyncOpenAI,
    parsed: Dict[str, Any],
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Use GPT-4o to evaluate the semantic correctness of one sample.

    Returns {"score": int, "reason": str} or None on failure.
    """
    prompt = JUDGE_PROMPT.format(
        scenario=parsed["metadata"].get("scenario", "unknown"),
        target_field=parsed["metadata"].get("field_type", "unknown"),
        current=json.dumps(parsed["current"], indent=2, ensure_ascii=False),
        evidence=parsed["evidence"],
        updated=json.dumps(parsed["updated"], indent=2, ensure_ascii=False),
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict quality evaluator. Output only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        return {
            "score": int(result.get("score", 0)),
            "reason": result.get("reason", ""),
        }

    except Exception as e:
        logger.error(f"  [{parsed['id']}] Judge error: {e}")
        return {"score": 0, "reason": f"Judge failed: {e}"}


# ============================
# Report generation
# ============================

def generate_report(
    results: List[Dict[str, Any]],
    correctness_threshold: int = 3,
) -> str:
    """
    Generate a markdown quality report from judge results.

    Flags samples that:
    - Fail any deterministic check
    - Score below correctness_threshold on LLM judge
    """
    total = len(results)
    det_pass = sum(1 for r in results if r["deterministic_pass"])
    det_fail = total - det_pass

    # LLM scores (may be absent if --no-llm)
    has_llm = any("correctness" in r for r in results)
    if has_llm:
        scores = [r["correctness"]["score"] for r in results if "correctness" in r]
        avg_score = sum(scores) / len(scores) if scores else 0
        low_score = sum(1 for s in scores if s < correctness_threshold)
    else:
        avg_score = 0
        low_score = 0

    flagged = [
        r for r in results
        if not r["deterministic_pass"]
        or ("correctness" in r and r["correctness"]["score"] < correctness_threshold)
    ]

    # Build report
    lines = [
        "# Training Data Quality Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total samples:** {total}",
        "",
        "## Summary",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Schema valid | {det_pass}/{total} |",
        f"| Deterministic failures | {det_fail} |",
    ]

    if has_llm:
        lines.extend([
            f"| Correctness avg | {avg_score:.2f}/5 |",
            f"| Correctness < {correctness_threshold} | {low_score} |",
        ])

    lines.extend([
        f"| **Flagged samples** | **{len(flagged)}** |",
        "",
    ])

    # Scenario breakdown
    scenario_stats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        sc = r["scenario"]
        if sc not in scenario_stats:
            scenario_stats[sc] = {"total": 0, "det_pass": 0, "scores": []}
        scenario_stats[sc]["total"] += 1
        if r["deterministic_pass"]:
            scenario_stats[sc]["det_pass"] += 1
        if "correctness" in r:
            scenario_stats[sc]["scores"].append(r["correctness"]["score"])

    lines.extend([
        "## Breakdown by Scenario",
        "",
        "| Scenario | Count | Det. Pass | Correctness Avg |",
        "|---|---|---|---|",
    ])
    for sc, stats in scenario_stats.items():
        avg = (
            f"{sum(stats['scores']) / len(stats['scores']):.1f}"
            if stats["scores"] else "N/A"
        )
        lines.append(
            f"| {sc} | {stats['total']} | "
            f"{stats['det_pass']}/{stats['total']} | {avg} |"
        )

    lines.append("")

    # Flagged samples detail
    if flagged:
        lines.extend([
            "## Flagged Samples",
            "",
        ])

        for r in flagged:
            sid = r["id"]
            reasons = []

            if r["schema_errors"]:
                reasons.append("SCHEMA: " + "; ".join(r["schema_errors"]))
            if r["preservation_errors"]:
                reasons.append("PRESERVATION: " + "; ".join(r["preservation_errors"]))
            if r["nochange_errors"]:
                reasons.append("NOCHANGE: " + "; ".join(r["nochange_errors"]))
            if "correctness" in r and r["correctness"]["score"] < correctness_threshold:
                reasons.append(
                    f"CORRECTNESS ({r['correctness']['score']}/5): "
                    f"{r['correctness']['reason']}"
                )

            lines.append(f"### {sid} | {r['scenario']} | {r['target_field']}")
            lines.append("")
            for reason in reasons:
                lines.append(f"- ❌ {reason}")
            lines.append("")
            lines.append("---")
            lines.append("")
    else:
        lines.extend([
            "## ✅ No flagged samples — all data passed quality checks!",
            "",
        ])

    return "\n".join(lines)


# ============================
# CLI entry point
# ============================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Judge training data quality for Preference Adapter"
    )
    parser.add_argument(
        "--file", type=str, metavar="FILE",
        help="Specific JSONL file to judge (default: all field-type files)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM judge, run deterministic checks only",
    )
    parser.add_argument(
        "--threshold", type=int, default=3,
        help="Correctness score threshold for flagging (default: 3)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o",
        help="Model for LLM judge (default: gpt-4o)",
    )
    args = parser.parse_args()

    # Load samples
    if args.file:
        filepath = Path(args.file)
        if not filepath.is_absolute():
            filepath = DATA_DIR / filepath
        all_samples = load_samples(filepath)
        logger.info(f"Loaded {len(all_samples)} samples from {filepath.name}")
    else:
        all_samples = []
        for field_type in FIELD_TYPES:
            fp = DATA_DIR / f"{field_type}_train.jsonl"
            if fp.exists():
                samples = load_samples(fp)
                all_samples.extend(samples)
                logger.info(f"Loaded {len(samples)} from {fp.name}")
            else:
                logger.warning(f"Missing {fp.name}, skipping")

    if not all_samples:
        logger.error("No samples found to judge")
        return

    # Parse all samples
    parsed_samples = []
    for sample in all_samples:
        try:
            parsed_samples.append(parse_sample(sample))
        except Exception as e:
            logger.error(f"Failed to parse sample: {e}")

    logger.info(f"Parsed {len(parsed_samples)} samples")

    # 1. Run deterministic checks
    results = []
    for parsed in parsed_samples:
        result = run_deterministic_checks(parsed)
        results.append(result)

    det_pass = sum(1 for r in results if r["deterministic_pass"])
    logger.info(f"Deterministic checks: {det_pass}/{len(results)} passed")

    # 2. Run LLM judge (if enabled)
    if not args.no_llm:
        from config.settings import OPENAI_API_KEY
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)

        logger.info(f"Running LLM judge ({args.model}) on {len(parsed_samples)} samples...")

        for i, (parsed, result) in enumerate(zip(parsed_samples, results)):
            judgment = await judge_one_sample(client, parsed, args.model)
            result["correctness"] = judgment
            logger.info(
                f"  [{parsed['id']}] correctness={judgment['score']}/5"
            )
            # Rate limiting
            await asyncio.sleep(0.3)

        scores = [r["correctness"]["score"] for r in results]
        avg = sum(scores) / len(scores) if scores else 0
        logger.info(f"LLM judge avg correctness: {avg:.2f}/5")

    # 3. Generate report
    report = generate_report(results, args.threshold)
    report_path = DATA_DIR / "quality_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report saved to {report_path.name}")

    # Print summary to console
    flagged = sum(
        1 for r in results
        if not r["deterministic_pass"]
        or ("correctness" in r and r["correctness"]["score"] < args.threshold)
    )
    logger.info(
        f"Done! {flagged}/{len(results)} samples flagged. "
        f"See quality_report.md for details."
    )


if __name__ == "__main__":
    asyncio.run(main())
