"""
Skill: search_action_items
--------------------------
Answers queries like:
  - "What are my action items from the last 3 meetings with Ankit?"
  - "What action items did Hongye mention?"
  - "Summarize action items from recent meetings"

Strategy
--------
1. Lightweight query parse: extract person names + N-meeting count via regex / thefuzz.
2. Filter MeetingsDataFrame (reuses existing pandas_utils) → meeting_ids + raw action_tasks.
3. LiteLlm synthesis pass: format raw items into a readable answer.
4. Return canonical result dict (see skills/base.py).

No full ADK agent chain is needed here; the DataFrame metadata already contains
pre-extracted action_tasks from summary_metadata.json.  Heavy retrieval is
optional and enabled only when the vector store is already loaded.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# Make sure project root is importable regardless of cwd
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from skills.base import BaseSkill

# ---------------------------------------------------------------------------
# Lazy imports — only resolved at call time so the module itself is importable
# even when the venv / config is incomplete.
# ---------------------------------------------------------------------------

def _get_mdf():
    from sub_agents.pandas_utils import get_meetings_df
    return get_meetings_df()


def _get_llm():
    """Return a lightweight LiteLlm wrapper (same pattern as planner_agent.py)."""
    from litellm import completion as litellm_completion
    from config.settings import OPENAI_API_KEY, OPENAI_MODEL
    return litellm_completion, OPENAI_API_KEY, OPENAI_MODEL or "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Query parsing helpers
# ---------------------------------------------------------------------------

_LAST_N_RE = re.compile(
    r"\blast\s+(\d+)\s+meetings?\b", re.IGNORECASE
)
_RECENT_RE = re.compile(
    r"\b(recent|latest|newest)\s+meetings?\b", re.IGNORECASE
)


def _extract_n_meetings(query: str) -> int | None:
    """Return N from 'last N meetings', else None."""
    m = _LAST_N_RE.search(query)
    if m:
        return int(m.group(1))
    if _RECENT_RE.search(query):
        return 3  # default for "recent"
    return None


def _extract_person_mention(query: str, mdf) -> str | None:
    """
    Return the best-matched canonical participant name from the query,
    or None if no person is detected.

    Uses thefuzz (already a project dependency).
    """
    try:
        from thefuzz import fuzz, process as fuzz_process
    except ImportError:
        return None

    all_participants = mdf.get_all_participants()
    if not all_participants:
        return None

    # Split query into tokens (words + bigrams) to look for names
    tokens = query.split()
    candidates: list[tuple[str, int]] = []

    for length in (1, 2):
        for i in range(len(tokens) - length + 1):
            span = " ".join(tokens[i : i + length])
            # Skip stop words and short tokens
            if len(span) < 3 or span.lower() in {
                "what", "are", "my", "the", "from", "with", "did",
                "meeting", "meetings", "action", "items", "last", "recent",
                "summarize", "summary", "mentioned", "mention", "about",
                "give", "me", "all", "any",
            }:
                continue
            result = fuzz_process.extractOne(
                span, all_participants, scorer=fuzz.token_sort_ratio
            )
            if result and result[1] >= 60:
                candidates.append((result[0], result[1]))

    if not candidates:
        return None

    # Return the candidate with the highest fuzzy score
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Core action-item extraction from DataFrame
# ---------------------------------------------------------------------------

def _collect_action_items(
    mdf,
    person: str | None,
    n: int | None,
) -> tuple[list[dict], list[str]]:
    """
    Returns (meeting_records, meeting_ids).

    meeting_records: list of {meeting_id, datetime, participants, action_tasks}
    """
    if person:
        rows = mdf.filter_by_person(person)
    else:
        rows = mdf.df

    if rows is None or (hasattr(rows, "empty") and rows.empty):
        return [], []

    # Sort by date descending (already sorted in MeetingsDataFrame, but be safe)
    if hasattr(rows, "sort_values"):
        rows = rows.sort_values("date", ascending=False)

    if n is not None:
        rows = rows.head(n)

    records = []
    for _, row in rows.iterrows():
        records.append(
            {
                "meeting_id": row.get("meeting_id", ""),
                "datetime": row.get("datetime", ""),
                "participants": row.get("participants", []),
                "action_tasks": row.get("action_tasks", []),
            }
        )

    meeting_ids = [r["meeting_id"] for r in records if r["meeting_id"]]
    return records, meeting_ids


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a meeting assistant. You will be given raw action items extracted from \
meeting metadata. Produce a concise, clearly formatted summary of the action items.
Group by meeting when there are multiple meetings.
If a meeting has no action items, say "No action items found."
Cite the meeting_id in brackets after each group header, e.g. [con1].
Output plain text — no markdown headers, no JSON.
"""


def _synthesize(records: list[dict], query: str) -> str:
    """Use LiteLlm to format the raw action items into a readable answer."""
    try:
        litellm_completion, api_key, model = _get_llm()
        content = json.dumps(records, ensure_ascii=False, indent=2)
        resp = litellm_completion(
            model=model,
            api_key=api_key,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User question: {query}\n\n"
                        f"Meeting records with action items:\n{content}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        # Fallback: plain concatenation
        lines = []
        for r in records:
            mid = r.get("meeting_id", "?")
            tasks = r.get("action_tasks", [])
            if tasks:
                lines.append(f"[{mid}]")
                for t in tasks:
                    lines.append(f"  - {t}")
            else:
                lines.append(f"[{mid}] No action items.")
        return "\n".join(lines) if lines else "No action items found."


# ---------------------------------------------------------------------------
# Skill class
# ---------------------------------------------------------------------------

class SearchActionItemsSkill(BaseSkill):
    """
    Retrieval-type skill: searches meeting metadata for action items.

    run(query, context) parameters
    --------------------------------
    query   : natural-language question about action items
    context : optional dict with keys:
        - "person"   : override detected person name
        - "n"        : override N-meeting count
        - "meeting_ids" : explicit list of meeting IDs to filter
    """

    name = "search_action_items"
    description = "Search and retrieve action items or tasks from meeting records"
    trigger_keywords = (
        "action items", "action tasks", "tasks", "to-do", "todo",
        "what do i need to do", "assigned to", "my tasks",
        "what tasks", "follow-up tasks", "who is responsible",
    )
    examples = (
        "What are my action items from last meeting?",
        "Show tasks assigned to John",
        "What did we agree to do?",
        "List all action items from this week",
    )
    priority = 10
    enabled = True

    def run(self, query: str, context: dict | None = None) -> dict:
        ctx = context or {}

        try:
            mdf = _get_mdf()
        except Exception as exc:
            return self._error("METADATA_LOAD_FAILED", str(exc))

        # --- Parse query ---
        person: str | None = ctx.get("person") or _extract_person_mention(query, mdf)
        n: int | None = ctx.get("n") or _extract_n_meetings(query)

        # If explicit meeting_ids provided in context, use them directly
        explicit_ids: list[str] | None = ctx.get("meeting_ids")

        if explicit_ids:
            # Filter DataFrame to only those IDs
            df = mdf.df
            rows = df[df["meeting_id"].isin(explicit_ids)] if df is not None else None
            if rows is None or rows.empty:
                records, meeting_ids = [], []
            else:
                records = []
                for _, row in rows.iterrows():
                    records.append(
                        {
                            "meeting_id": row.get("meeting_id", ""),
                            "datetime": row.get("datetime", ""),
                            "participants": row.get("participants", []),
                            "action_tasks": row.get("action_tasks", []),
                        }
                    )
                meeting_ids = [r["meeting_id"] for r in records]
        else:
            try:
                records, meeting_ids = _collect_action_items(mdf, person, n)
            except Exception as exc:
                return self._error("FILTER_FAILED", str(exc))

        if not records:
            return self._success(
                structured_output={"action_items": []},
                text_output="No meetings found matching your query.",
                rewritten_query=query,
                meeting_ids=[],
            )

        # --- Flatten all action items for structured output ---
        all_items: list[dict[str, Any]] = []
        for r in records:
            for task in r.get("action_tasks", []):
                all_items.append(
                    {
                        "meeting_id": r["meeting_id"],
                        "datetime": r["datetime"],
                        "task": task,
                    }
                )

        # --- LLM synthesis ---
        try:
            text_output = _synthesize(records, query)
        except Exception as exc:
            text_output = f"(synthesis failed: {exc})"

        # Build a lightweight rewritten_query label
        parts = []
        if person:
            parts.append(f"person={person}")
        if n:
            parts.append(f"last {n} meetings")
        if explicit_ids:
            parts.append(f"ids={explicit_ids}")
        rewritten_query = f"action_items[{', '.join(parts)}]" if parts else query

        return self._success(
            structured_output={"action_items": all_items},
            text_output=text_output,
            rewritten_query=rewritten_query,
            meeting_ids=meeting_ids,
        )


# Singleton — imported by registry
skill = SearchActionItemsSkill()
