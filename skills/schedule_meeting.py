"""
Skill: schedule_meeting (mock)
------------------------------
Action-type skill that simulates scheduling a follow-up meeting.

No real calendar API, no OAuth.
Every "scheduled" event is appended to:
    <project_root>/logs/mock_calendar_history.jsonl

Workflow
--------
1. Load the most recent meeting from MeetingsDataFrame.
2. Determine participants: explicit names from query > last meeting's participants.
3. Mock-create a calendar event.
   Suggested date = last meeting date + 7 days.
   This is a mock scheduling heuristic, NOT a real scheduling decision.
4. Persist event to logs/mock_calendar_history.jsonl.
5. Return success with a follow_up_action field so callers can optionally
   chain to send_email for participant notifications.

Inputs (via query string OR context dict)
-----------------------------------------
query   : natural-language instruction, e.g.
          "Schedule the next meeting with Hongye and Ankit"
context : optional overrides:
    {
        "title":        "Sprint Planning",   # override event title
        "participants": ["Hongye Qian"],     # override participants list
        "date":         "2025-06-01",        # override suggested date (ISO)
    }

Output (success)
----------------
{
    "skill_name": "schedule_meeting",
    "status":     "success",
    "meeting_ids": ["<based_on_meeting_id>"],
    "structured_output": {
        "event_id":           "mock_cal_<timestamp>_<uuid[:8]>",
        "title":              "Follow-up: <topic> | Next Team Meeting",
        "participants":       [...],
        "based_on_meeting":   "<meeting_id>",
        "suggested_datetime": "<ISO datetime> | null",
        "datetime_source":    "explicit_user_query | context | mock_last_meeting_plus_7_days | unknown",
        "mock":               True,
        "follow_up_action": {
            "type":              "send_email",
            "suggested_query":   "Send meeting invite for '...' to ...",
            "participants":      [...],
            "title":             "...",
            "suggested_datetime": "...",
            "datetime_source":   "...",
        },
    },
    "text_output": "Mock calendar event created ...",
}

Output (error)
--------------
{
    "skill_name": "schedule_meeting",
    "status":     "error",
    "error": {
        "code":    "MEETING_REFERENCE_NOT_FOUND" | "CALENDAR_MOCK_FAILED",
        "message": "...",
    },
}
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from skills.base import BaseSkill

_LOG_DIR = _project_root / "logs"
_LOG_FILE = _LOG_DIR / "mock_calendar_history.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_mdf():
    from sub_agents.pandas_utils import get_meetings_df
    return get_meetings_df()


def _extract_explicit_participants(query: str, mdf) -> list[str]:
    """
    Extract participants explicitly named in query, e.g. "with Hongye and Ankit".
    Stops at boundary words (time/date markers) so they are not treated as names.
    Uses fuzzy matching against known participants when possible.
    """
    m = re.search(
        r"\bwith\b\s+(.+?)(?=\s+(?:about|for|on|at|tomorrow|today|next|this|by|to\b)|$)",
        query,
        re.IGNORECASE,
    )
    if not m:
        return []

    raw = m.group(1).strip()
    tokens = [t.strip() for t in re.split(r"\band\b|,", raw, flags=re.IGNORECASE) if t.strip()]

    all_participants = mdf.get_all_participants()
    if not all_participants:
        return [t for t in tokens if len(t) >= 2]

    try:
        from thefuzz import fuzz, process as fuzz_process
    except ImportError:
        return [t for t in tokens if len(t) >= 2]

    found: list[str] = []
    for token in tokens:
        if len(token) < 2:
            continue
        result = fuzz_process.extractOne(token, all_participants, scorer=fuzz.token_sort_ratio)
        if result and result[1] >= 60:
            found.append(result[0])
    return found


def _compute_next_date(last_date_str: str) -> str | None:
    """
    Mock heuristic: suggest next meeting = last meeting date + 7 days.
    This is not a real scheduling algorithm.
    """
    if not last_date_str:
        return None
    try:
        last = date.fromisoformat(last_date_str)
        return (last + timedelta(days=7)).isoformat()
    except Exception:
        return None


def _log_mock_event(record: dict) -> None:
    """Append one JSON line to logs/mock_calendar_history.jsonl."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Datetime extraction helpers
# ---------------------------------------------------------------------------

_WEEKDAYS: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

# Only attempt datetime extraction when the query contains recognisable hints.
_DT_HINT = re.compile(
    r"\b(today|tomorrow|next|this|at\s+\d|\d{1,2}\s*[ap]m|\d{2}:\d{2}|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


def _extract_time_from_query(q_lower: str) -> str | None:
    """Return a raw time string ('3 PM', '15:00') from an already-lowercased query."""
    m = re.search(r"\bat\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", q_lower, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(\d{1,2}:\d{2})\b", q_lower)
    if m:
        return m.group(1)
    return None


def _parse_time_to_hhmm(time_str: str) -> str:
    """Normalise '3 PM', '15:00', '3:30pm' → 'HH:MM'."""
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_str.strip(), re.IGNORECASE)
    if not m:
        return "00:00"
    hour = int(m.group(1))
    minutes = int(m.group(2) or 0)
    meridiem = (m.group(3) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minutes:02d}"


def _extract_explicit_datetime(
    query: str,
    base_date: "date | None" = None,
) -> "tuple[str | None, str]":
    """
    Extract an explicit date/time from a natural-language query.

    Priority:
      1. dateparser (lazy import) — handles most natural-language expressions
      2. Manual regex fallback: ISO date, 'next <weekday>', 'tomorrow', 'this <weekday>'

    Returns (datetime_str | None, source_label).
    datetime_str is 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM'.
    source_label is 'explicit_user_query' when something was found, else 'unknown'.
    """
    if not _DT_HINT.search(query):
        return None, "unknown"

    q_lower = query.lower()
    today = date.today()

    # 1. Try dateparser (optional dependency)
    try:
        import dateparser as _dp
        parsed = _dp.parse(
            query,
            settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False},
        )
        if parsed and parsed.date() >= today:
            return parsed.strftime("%Y-%m-%d %H:%M"), "explicit_user_query"
    except Exception:
        pass

    # 2. ISO date: YYYY-MM-DD [HH:MM]
    iso_m = re.search(r"(\d{4}-\d{2}-\d{2})(?:\s+(\d{2}:\d{2}))?", query)
    if iso_m:
        return f"{iso_m.group(1)} {iso_m.group(2) or '00:00'}", "explicit_user_query"

    # 3. Extract optional time component
    time_str = _extract_time_from_query(q_lower)

    # 4. Extract day reference
    day: "date | None" = None

    if "tomorrow" in q_lower:
        day = today + timedelta(days=1)

    if day is None:
        pat = r"\bnext\s+(" + "|".join(_WEEKDAYS) + r")\b"
        m = re.search(pat, q_lower)
        if m:
            target_wd = _WEEKDAYS[m.group(1)]
            delta = (target_wd - today.weekday()) % 7 or 7  # always in the future
            day = today + timedelta(days=delta)

    if day is None:
        pat = r"\bthis\s+(" + "|".join(_WEEKDAYS) + r")\b"
        m = re.search(pat, q_lower)
        if m:
            target_wd = _WEEKDAYS[m.group(1)]
            delta = (target_wd - today.weekday()) % 7
            day = today + timedelta(days=delta)

    if day and time_str:
        return f"{day.isoformat()} {_parse_time_to_hhmm(time_str)}", "explicit_user_query"
    if day:
        return day.isoformat(), "explicit_user_query"
    if time_str:
        ref = base_date if isinstance(base_date, date) else today
        return f"{ref.isoformat()} {_parse_time_to_hhmm(time_str)}", "explicit_user_query"

    return None, "unknown"


# ---------------------------------------------------------------------------
# Skill class
# ---------------------------------------------------------------------------

class ScheduleMeetingSkill(BaseSkill):
    """
    Action-type skill: mock-schedules a follow-up meeting.

    Infers scheduling context from the most recent meeting in MeetingsDataFrame.
    Returns a follow_up_action field so callers can chain to send_email.
    """

    name = "schedule_meeting"
    description = "Schedule or book a follow-up meeting with participants"
    trigger_keywords = (
        "schedule meeting", "book meeting", "arrange meeting",
        "set up meeting", "follow-up meeting", "follow up meeting",
        "calendar", "schedule a call", "book a call", "schedule",
    )
    examples = (
        "Schedule a follow-up meeting with Alice and Bob",
        "Book a meeting next week",
        "Arrange a team sync",
        "Set up a call with the project team",
    )
    priority = 10
    enabled = True

    def run(self, query: str, context: dict | None = None) -> dict:  # noqa: C901
        ctx = context or {}

        # ------------------------------------------------------------------
        # Try to import real MCP calendar integration (optional dependency).
        # Falls back gracefully to mock if packages are not installed.
        # ------------------------------------------------------------------
        try:
            import skills.schedule_meeting_mcp  # noqa: F401  — probe only
            _MCP_AVAILABLE = False
        except ImportError:
            _MCP_AVAILABLE = False

        # ------------------------------------------------------------------
        # Step 1: Load meeting data (needed for participant/title/date context
        #         in both the real and mock paths).
        # ------------------------------------------------------------------
        try:
            mdf = _get_mdf()
        except Exception as exc:
            return self._error(
                "MEETING_REFERENCE_NOT_FOUND",
                f"Failed to load meeting data: {exc}",
            )

        if mdf.df is None or mdf.df.empty:
            return self._error(
                "MEETING_REFERENCE_NOT_FOUND",
                "No meeting data available to infer scheduling context.",
            )

        # Step 2: Get most recent meeting as the reference basis
        last_meetings = mdf.get_last_n_meetings(1)
        if not last_meetings:
            return self._error(
                "MEETING_REFERENCE_NOT_FOUND",
                "No meetings found in data.",
            )
        last = last_meetings[0]

        # Step 3: Determine participants
        #   Priority: context override > explicit names in query > last meeting's list
        explicit = _extract_explicit_participants(query, mdf)
        participants: list[str] = (
            ctx.get("participants")
            or explicit
            or (last.get("participants") or [])
        )

        # Step 4: Determine event title
        topics: list = last.get("topics") or []
        title: str = ctx.get("title") or (
            f"Follow-up: {topics[0]}" if topics else "Next Team Meeting"
        )

        # Step 5: Determine suggested datetime
        # Priority: context["datetime"] > context["date"] > explicit in query > mock fallback
        suggested_datetime: str | None = None
        datetime_source: str = "unknown"

        if ctx.get("datetime"):
            suggested_datetime = ctx["datetime"]
            datetime_source = "context"
        elif ctx.get("date"):
            suggested_datetime = ctx["date"]
            datetime_source = "context"
        else:
            explicit_dt, dt_src = _extract_explicit_datetime(query)
            if explicit_dt and dt_src == "explicit_user_query":
                suggested_datetime = explicit_dt
                datetime_source = "explicit_user_query"
            else:
                last_date: str = last.get("date", "")
                mock_dt = _compute_next_date(last_date)
                if mock_dt:
                    suggested_datetime = mock_dt
                    datetime_source = "mock_last_meeting_plus_7_days"

        based_on: str = last.get("meeting_id", "")

        # ------------------------------------------------------------------
        # Real calendar path (when MCP packages are available)
        # ------------------------------------------------------------------
        if _MCP_AVAILABLE:
            from skills.schedule_meeting_mcp.auth.token_store import init_db
            from skills.schedule_meeting_mcp.mcp_client import create_event
            from skills.schedule_meeting_mcp.schemas import CalendarEventRequest
            from skills.schedule_meeting_mcp.providers.google_calendar import GoogleCalendarProvider

            init_db()
            user_id = ctx.get("user_id") or os.getenv("DEFAULT_USER_ID", "local_user")

            # Check Google auth status
            auth_status = GoogleCalendarProvider().check_auth(user_id)

            if not auth_status["connected"]:
                # Not connected — return auth_required so the user can authorize.
                # follow_up_action is None because no event was created yet.
                oauth_base = os.getenv("CALENDAR_OAUTH_BASE_URL", "http://127.0.0.1:8001")
                auth_link = f"{oauth_base}/auth/google/start?user_id={user_id}"
                text_output = (
                    "Google Calendar is not connected.\n\n"
                    f"Please authorize Google Calendar access:\n{auth_link}\n\n"
                    "After authorizing, schedule the meeting again.\n\n"
                    "Outlook Calendar is also available (currently scaffold/mock)."
                )
                structured: dict[str, Any] = {
                    "event_id":           None,
                    "title":              title,
                    "participants":       participants,
                    "based_on_meeting":   based_on,
                    "suggested_datetime": suggested_datetime,
                    "datetime_source":    datetime_source,
                    "status":             "auth_required",
                    "auth_url":           auth_link,
                    "mock":               False,
                    "follow_up_action":   None,
                }
                return self._success(
                    structured_output=structured,
                    text_output=text_output,
                    meeting_ids=[based_on] if based_on else None,
                )

            # Google is connected — create the real event.
            tz = os.getenv("DEFAULT_TIMEZONE", "Asia/Singapore")

            # Build CalendarEventRequest — filter attendees (emails) vs display names
            import re as _re
            _email_re = _re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
            attendees = [p for p in participants if _email_re.match(p.strip())]
            display_names = [p for p in participants if not _email_re.match(p.strip())]

            cal_req = CalendarEventRequest(
                title=title,
                start_time=suggested_datetime or datetime.now(timezone.utc).isoformat(),
                end_time=None,  # provider will add DEFAULT_EVENT_DURATION_MINUTES
                timezone=tz,
                participants=display_names,
                attendees=attendees,
                description=ctx.get("description", ""),
                user_id=user_id,
                based_on_meeting=based_on or None,
            )

            result = create_event(cal_req)

            if result.status == "success":
                names_str = ", ".join(participants) if participants else "the participants"
                follow_up: dict[str, Any] = {
                    "type":               "send_email",
                    "suggested_query":    f"Send meeting invite for '{title}' to {names_str}",
                    "participants":       participants,
                    "title":              title,
                    "suggested_datetime": suggested_datetime,
                    "datetime_source":    datetime_source,
                }
                structured = {
                    "event_id":           result.event_id,
                    "title":              result.title or title,
                    "participants":       participants,
                    "based_on_meeting":   based_on,
                    "suggested_datetime": result.start_time or suggested_datetime,
                    "datetime_source":    datetime_source,
                    "mock":               False,
                    "html_link":          result.html_link,
                    "follow_up_action":   follow_up,
                }
                who_display = ", ".join(participants) if participants else "(unknown)"
                link_line = f"\nEvent link: {result.html_link}\n" if result.html_link else ""
                warnings_text = ""
                if result.warnings:
                    warnings_text = "\n\nWarnings:\n" + "\n".join(f"- {w}" for w in result.warnings)
                text_output = (
                    f"Google Calendar event created.\n\n"
                    f"Title: {result.title or title}\n\n"
                    f"When: {result.start_time or suggested_datetime or '(undetermined)'}\n\n"
                    f"Who: {who_display}\n\n"
                    f"ID: {result.event_id}"
                    f"{link_line}"
                    f"{warnings_text}\n\n"
                    "Would you like to send email notifications to the participants?"
                )
                return self._success(
                    structured_output=structured,
                    text_output=text_output,
                    meeting_ids=[based_on] if based_on else None,
                )

            elif result.status == "auth_required":
                # Token expired / revoked mid-session
                oauth_base = os.getenv("CALENDAR_OAUTH_BASE_URL", "http://127.0.0.1:8001")
                auth_link = result.auth_url or f"{oauth_base}/auth/google/start?user_id={user_id}"
                structured = {
                    "event_id":           None,
                    "title":              title,
                    "participants":       participants,
                    "based_on_meeting":   based_on,
                    "suggested_datetime": suggested_datetime,
                    "datetime_source":    datetime_source,
                    "status":             "auth_required",
                    "auth_url":           auth_link,
                    "mock":               False,
                    "follow_up_action":   None,
                }
                return self._success(
                    structured_output=structured,
                    text_output=(
                        "Google Calendar authorization is required.\n\n"
                        f"Please re-authorize:\n{auth_link}\n\n"
                        "After authorizing, schedule the meeting again."
                    ),
                    meeting_ids=[based_on] if based_on else None,
                )

            else:
                # API error — fall through to mock below, but note the error
                print(
                    f"[schedule_meeting] Real calendar API error "
                    f"({result.error_code}): {result.error_message}. Falling back to mock."
                )
                # Fall through to mock path

        # ------------------------------------------------------------------
        # Mock / fallback path
        # ------------------------------------------------------------------

        # Step 6: Build mock event record
        now = datetime.now(timezone.utc)
        event_id = f"mock_cal_{now.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"

        event: dict[str, Any] = {
            "event_id":           event_id,
            "title":              title,
            "participants":       participants,
            "based_on_meeting":   based_on,
            "suggested_datetime": suggested_datetime,
            "datetime_source":    datetime_source,
            "mock":               True,
        }

        # Step 7: Persist to log
        try:
            _log_mock_event({
                "event_id":           event_id,
                "timestamp":          now.isoformat(),
                "query":              query,
                "title":              title,
                "participants":       participants,
                "based_on_meeting":   based_on,
                "suggested_datetime": suggested_datetime,
                "datetime_source":    datetime_source,
                "mock":               True,
            })
        except Exception:
            pass

        # Step 8: Build follow_up_action for optional email chaining
        names_str = ", ".join(participants) if participants else "the participants"
        follow_up = {
            "type":               "send_email",
            "suggested_query":    f"Send meeting invite for '{title}' to {names_str}",
            "participants":       participants,
            "title":              title,
            "suggested_datetime": suggested_datetime,
            "datetime_source":    datetime_source,
        }

        structured = {**event, "follow_up_action": follow_up}

        # Step 9: Build formatted text output
        date_display = suggested_datetime or "(undetermined)"
        who_display  = ", ".join(participants) if participants else "(unknown)"

        _source_labels = {
            "context":                       "provided in context",
            "explicit_user_query":           "explicit user query",
            "mock_last_meeting_plus_7_days": "mock estimate (last meeting date + 7 days)",
            "unknown":                       "unknown",
        }
        source_label = _source_labels.get(datetime_source, datetime_source)

        notes: list[str] = [
            "Note: This is a mock calendar event. No real Google Calendar event was created.",
        ]
        if datetime_source == "mock_last_meeting_plus_7_days":
            notes.append(
                "Note: Suggested date is a mock estimate based on last meeting date + 7 days."
            )

        text_output = (
            f"Mock calendar event created.\n\n"
            f"Title: {title}\n\n"
            f"When: {date_display}\n\n"
            f"Time source: {source_label}\n\n"
            f"Who: {who_display}\n\n"
            f"ID: {event_id}\n\n"
            + "\n\n".join(notes)
            + "\n\nWould you like to send email notifications to the participants?"
        )

        return self._success(
            structured_output=structured,
            text_output=text_output,
            meeting_ids=[based_on] if based_on else None,
        )


# Singleton — imported by registry
skill = ScheduleMeetingSkill()
