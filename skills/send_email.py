"""
Skill: send_email (mock)
------------------------
Action-type skill that simulates composing and "sending" an email.

No real email API, no OAuth, no SMTP.
Every "sent" email is appended to:
    <project_root>/logs/mock_email_history.jsonl

Content determination (three cases)
-------------------------------------
A) context["content"] is explicitly provided
   → use it directly, skip resolution

B) query references meeting content (summary, action items, follow-up, etc.)
   AND content can be resolved from MeetingsDataFrame
   → use resolved content

C) query references meeting content but resolution fails
   (e.g. "roadmap", no matching meeting data, or data is empty)
   → return CONTENT_RESOLUTION_FAILED error; never fabricate content

If the query contains no meeting content reference at all, the original
LLM / heuristic extraction path is used (suitable for ad-hoc emails).

Inputs (via query string OR context dict)
-----------------------------------------
query   : plain-language instruction, e.g.
          "Send meeting summary to Hongye"
context : optional overrides / pre-parsed fields:
    {
        "recipient":     "Hongye Qian",
        "subject":       "Meeting summary",
        "content":       "Here are the action items ...",  # triggers Case A
        "_context_text": "extra text for the LLM extractor",
        "use_llm":       True,   # False → force heuristic extract
        "person":        "Hongye Qian",  # hint for content resolution
    }

Output (success)
----------------
{
    "skill_name":  "send_email",
    "status":      "success",
    "rewritten_query": None,
    "meeting_ids":     None | [<ids used for content resolution>],
    "structured_output": {
        "recipient": "...",
        "subject":   "...",
        "content":   "...",
        "email_id":  "mock_<timestamp>_<uuid4[:8]>",
        "timestamp": "<ISO datetime>",
        "mock":      True,
    },
    "text_output": "Mock email prepared for ...",
    "error": None,
}

Output (error — Case C)
-----------------------
{
    "skill_name": "send_email",
    "status":     "error",
    ...
    "error": {
        "code":    "CONTENT_RESOLUTION_FAILED",
        "message": "Unable to resolve the requested email content ...",
    },
}
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from skills.base import BaseSkill

_LOG_DIR = _project_root / "logs"
_LOG_FILE = _LOG_DIR / "mock_email_history.jsonl"


# ---------------------------------------------------------------------------
# Content reference detection
# ---------------------------------------------------------------------------

_CONTENT_REF_RE = re.compile(
    r"\b("
    r"summary|minutes|meeting\s+notes|"
    r"action\s+items?|action\s+tasks?|"
    r"follow[\s\-]?ups?|next\s+steps?|"
    r"key\s+points?|key\s+takeaways?|takeaways?|"
    r"roadmap|agenda"
    r")\b",
    re.IGNORECASE,
)


def _has_content_reference(query: str) -> bool:
    """Return True if query implies the email body should come from meeting data."""
    return bool(_CONTENT_REF_RE.search(query))


def _classify_content_type(query_lower: str) -> str:
    """
    Classify what meeting data is being requested.

    Returns: 'action_items' | 'summary' | 'unknown'
    'unknown' → always CONTENT_RESOLUTION_FAILED (do not fabricate).
    """
    if any(kw in query_lower for kw in (
        "action item", "action items", "action task", "action tasks",
        "follow-up", "follow up", "followup", "next step", "next steps",
    )):
        return "action_items"
    if any(kw in query_lower for kw in (
        "summary", "minutes", "meeting notes",
        "key point", "key takeaway", "takeaway",
    )):
        return "summary"
    # roadmap, agenda, and anything else we cannot reliably map to structured data
    return "unknown"


# ---------------------------------------------------------------------------
# Content resolution from MeetingsDataFrame
# ---------------------------------------------------------------------------

def _resolve_content_from_meetings(
    query: str,
    ctx: dict,
    person_hint: str = "",
) -> tuple[str, str, list[str]]:
    """
    Attempt to resolve email body content from summary-level meeting data.

    Returns: (status, content, meeting_ids)
      status: "resolved" | "failed"

    Never fabricates content for unresolvable content types (roadmap, agenda, etc.).
    """
    content_type = _classify_content_type(query.lower())
    if content_type == "unknown":
        # roadmap / agenda / unrecognised → refuse to fabricate
        return "failed", "", []

    try:
        from sub_agents.pandas_utils import get_meetings_df
        mdf = get_meetings_df()
    except Exception:
        return "failed", "", []

    if mdf.df is None or mdf.df.empty:
        return "failed", "", []

    # Resolve person filter: context override > fuzzy-match on extracted recipient
    person: str | None = ctx.get("person") or None
    if not person and person_hint:
        try:
            matched = mdf.find_person(person_hint, threshold=0.5)
            person = matched[0] if matched else None
        except Exception:
            person = None

    # Determine N meetings (default: 1 = last meeting)
    n_match = re.search(r"\blast\s+(\d+)\s+meetings?\b", query, re.IGNORECASE)
    n = int(n_match.group(1)) if n_match else 1

    records = mdf.get_last_n_meetings(n, person_name=person)
    if not records:
        return "failed", "", []

    meeting_ids = [r.get("meeting_id", "") for r in records if r.get("meeting_id")]

    if content_type == "action_items":
        items: list[str] = []
        for r in records:
            tasks = r.get("action_tasks", []) or []
            if isinstance(tasks, list):
                items.extend(str(t) for t in tasks if t)
        if not items:
            return "failed", "", meeting_ids
        content = "Action items from recent meetings:\n" + "\n".join(f"- {t}" for t in items)
        return "resolved", content, meeting_ids

    # content_type == "summary"
    parts: list[str] = []
    for r in records:
        mid = r.get("meeting_id", "?")
        topics: list = r.get("topics", []) or []
        tasks: list = r.get("action_tasks", []) or []
        if topics or tasks:
            seg = f"Meeting {mid}:"
            if topics:
                seg += f"\n  Topics: {', '.join(str(t) for t in topics)}"
            if tasks:
                seg += f"\n  Action items: {', '.join(str(t) for t in tasks)}"
            parts.append(seg)
    if not parts:
        return "failed", "", meeting_ids
    content = "Meeting summary:\n\n" + "\n\n".join(parts)
    return "resolved", content, meeting_ids


# ---------------------------------------------------------------------------
# LLM extraction helpers
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = """\
You are an email-composition assistant.
Given a natural-language instruction, extract:
  - recipient  : the person / email address to send to (string)
  - subject    : a concise subject line (string)
  - content    : the email body text (string; use the surrounding context if provided)

Return ONLY a JSON object with exactly these three keys.
If a field cannot be determined, use an empty string "".
"""


def _extract_fields_via_llm(query: str, context_text: str = "") -> dict:
    """Use LiteLlm to extract recipient/subject/content from a free-form query."""
    try:
        from litellm import completion as litellm_completion
        from config.settings import OPENAI_API_KEY, OPENAI_MODEL

        model = OPENAI_MODEL or "gpt-4o-mini"
        user_msg = query
        if context_text:
            user_msg = f"{query}\n\nAdditional context:\n{context_text}"

        resp = litellm_completion(
            model=model,
            api_key=OPENAI_API_KEY,
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw[: raw.rfind("```")]
        return json.loads(raw)
    except Exception:
        return {"recipient": "", "subject": "", "content": query}


def _simple_extract(query: str) -> dict:
    """
    Regex/heuristic fallback when LLM is unavailable.
    Handles patterns like:
      "Send <subject> to <name>"
      "Email <name> about <subject>"
    """
    recipient = ""
    subject = ""
    content = query

    m = re.search(r"\bsend\b.+?\bto\b\s+(.+?)(?:\s+about|\s*$)", query, re.IGNORECASE)
    if m:
        recipient = m.group(1).strip().rstrip(".")
    m2 = re.search(r"\bemail\b\s+(.+?)\s+about\b", query, re.IGNORECASE)
    if m2 and not recipient:
        recipient = m2.group(1).strip()

    s_match = re.search(r"\babout\b\s+(.+?)(?:\s+to\b|\s*$)", query, re.IGNORECASE)
    if s_match:
        subject = s_match.group(1).strip().rstrip(".")
    elif not subject:
        words = query.split()
        subject = " ".join(words[:8]) + ("..." if len(words) > 8 else "")

    return {"recipient": recipient, "subject": subject, "content": content}


# ---------------------------------------------------------------------------
# Log helper
# ---------------------------------------------------------------------------

def _log_mock_email(record: dict) -> None:
    """Append one JSON line to logs/mock_email_history.jsonl."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def send_meeting_notification(pending: dict) -> str:
    """
    Build and log a mock meeting-invite notification from a pending_confirmation dict.

    Called by short_memory when the user confirms a schedule_meeting follow-up.
    Owns the output format; returns the user-visible text string.
    """
    participants     = pending.get("participants", [])
    title            = pending.get("title", "the meeting")
    when_str         = pending.get("suggested_datetime") or "(undetermined)"
    cal_event_id     = pending.get("calendar_event_id", "")
    participants_str = ", ".join(participants) if participants else "the participants"

    now      = datetime.now(timezone.utc)
    ts       = now.strftime("%Y%m%dT%H%M%S")
    email_id = f"mock_{ts}_{uuid.uuid4().hex[:8]}"

    try:
        _log_mock_email({
            "email_id":  email_id,
            "timestamp": now.isoformat(),
            "query":     pending.get("suggested_query", ""),
            "recipient": participants_str,
            "subject":   f"Meeting invite: {title}",
            "content":   f"Meeting scheduled for {when_str}. Calendar Event ID: {cal_event_id}",
        })
    except Exception:
        pass

    return (
        f"Mock email notification prepared.\n\n"
        f"Meeting invite sent to: {participants_str}\n\n"
        f"Title: {title}\n\n"
        f"When: {when_str}\n\n"
        f"Calendar Event ID: {cal_event_id}\n\n"
        f"Email ID: {email_id}\n\n"
        f"Note: No real email was sent. This is a mock notification."
    )


# ---------------------------------------------------------------------------
# Skill class
# ---------------------------------------------------------------------------

class SendEmailSkill(BaseSkill):
    """
    Action-type skill: composes and mock-sends an email.

    Content follows three cases (see module docstring).
    """

    name = "send_email"
    description = "Send an email to meeting participants with meeting content"
    trigger_keywords = (
        "send email", "email to", "email", "notify", "mail",
        "send summary", "send notes", "send action items",
        "send meeting notes", "send minutes",
    )
    examples = (
        "Send meeting summary to John",
        "Email action items to the team",
        "Notify Alice about the follow-up",
        "Send the meeting notes to participants",
    )
    priority = 10
    enabled = True

    def run(self, query: str, context: dict | None = None) -> dict:
        ctx = context or {}

        use_llm: bool = ctx.get("use_llm", True)
        context_text: str = ctx.get("_context_text", "")

        # Always extract recipient + subject (needed for all paths)
        if use_llm:
            extracted = _extract_fields_via_llm(query, context_text)
        else:
            extracted = _simple_extract(query)

        recipient: str = ctx.get("recipient") or extracted.get("recipient", "")
        subject: str = ctx.get("subject") or extracted.get("subject", "")

        # --- Content determination ---
        meeting_ids_ref: list[str] | None = None

        if ctx.get("content"):
            # Case A: content explicitly in context → send directly
            content: str = ctx["content"]

        elif _has_content_reference(query):
            # Cases B / C: query references meeting content — must resolve from data
            status, resolved_content, meeting_ids_ref = _resolve_content_from_meetings(
                query, ctx, person_hint=recipient
            )
            if status == "resolved":
                # Case B: resolved successfully
                content = resolved_content
            else:
                # Case C: cannot reliably resolve — do NOT fabricate content
                return self._error(
                    "CONTENT_RESOLUTION_FAILED",
                    "Unable to resolve the requested email content from existing meeting data. "
                    "Please provide the content explicitly or refine the meeting reference.",
                )

        else:
            # No meeting content reference → use extracted content (original path)
            content = extracted.get("content", query)

        # --- Build mock email ---
        now = datetime.now(timezone.utc)
        ts = now.strftime("%Y%m%dT%H%M%S")
        email_id = f"mock_{ts}_{uuid.uuid4().hex[:8]}"

        structured: dict = {
            "recipient": recipient,
            "subject":   subject,
            "content":   content,
            "email_id":  email_id,
            "timestamp": now.isoformat(),
            "mock":      True,
        }

        try:
            _log_mock_email({
                "email_id":  email_id,
                "timestamp": now.isoformat(),
                "query":     query,
                "recipient": recipient,
                "subject":   subject,
                "content":   content,
            })
        except Exception:
            pass

        to_label = recipient if recipient else "(unknown recipient)"
        lines = [
            "Mock email prepared.",
            "",
            f"To: {to_label}",
            f"Subject: {subject}",
            f"Email ID: {email_id}",
        ]
        if meeting_ids_ref:
            lines.append(f"Meeting IDs used: {', '.join(meeting_ids_ref)}")
        lines += [
            "",
            "Email content:",
            content,
            "",
            "Note: No real email was sent. Logged to logs/mock_email_history.jsonl.",
        ]
        text_output = "\n".join(lines)

        return self._success(
            structured_output=structured,
            text_output=text_output,
            meeting_ids=meeting_ids_ref,
        )


# Singleton — imported by registry
skill = SendEmailSkill()
