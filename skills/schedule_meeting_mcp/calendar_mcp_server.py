"""
calendar_mcp_server.py — Real MCP server exposing calendar tools via FastMCP.

Run this as a standalone process when you want to expose calendar tools
over the MCP protocol (e.g. for Claude Desktop or another MCP client):

    python -m skills.schedule_meeting_mcp.calendar_mcp_server

Port: the FastMCP default (stdio transport by default; use mcp.run(transport="sse")
for HTTP-SSE if needed).
"""
from __future__ import annotations

import os
import secrets
import time

from mcp.server.fastmcp import FastMCP

from .calendar_service import create_calendar_event, get_provider_status
from .schemas import CalendarEventRequest, CalendarProviderName
from .auth.token_store import (
    get_preferred_provider,
    set_preferred_provider,
    get_token,
    delete_token,
    init_db,
    save_oauth_state,
)
from .auth.google_oauth import get_auth_url, exchange_code_for_token, get_user_email


mcp = FastMCP("calendar-mcp")


# ---------------------------------------------------------------------------
# Provider management tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_calendar_providers(user_id: str = "local_user") -> dict:
    """List all supported calendar providers and their connection status."""
    init_db()
    statuses = get_provider_status(user_id)
    return {
        "providers": [
            {
                "name": s.name,
                "connected": s.connected,
                "user_email": s.user_email,
                "auth_url": s.auth_url,
            }
            for s in statuses
        ]
    }


@mcp.tool()
def get_connected_calendar_provider(user_id: str = "local_user") -> dict:
    """Get the currently preferred / connected calendar provider for a user."""
    init_db()
    preferred = get_preferred_provider(user_id)
    if not preferred:
        return {"preferred_provider": None, "message": "No provider configured yet."}
    return {"preferred_provider": preferred}


@mcp.tool()
def set_calendar_provider(user_id: str, provider: str) -> dict:
    """Set the preferred calendar provider for a user."""
    init_db()
    supported = [CalendarProviderName.GOOGLE, CalendarProviderName.OUTLOOK]
    if provider not in supported:
        return {"status": "error", "message": f"Unsupported provider '{provider}'. Choose from: {supported}"}
    set_preferred_provider(user_id, provider)
    return {"status": "success", "preferred_provider": provider}


@mcp.tool()
def get_calendar_auth_url(user_id: str = "local_user", provider: str = "google") -> dict:
    """Generate an OAuth authorization URL for the given provider."""
    init_db()
    if provider == CalendarProviderName.GOOGLE:
        state = secrets.token_urlsafe(16)
        save_oauth_state(state, user_id, provider)
        auth_url = get_auth_url(user_id, state)
        return {"auth_url": auth_url, "provider": provider}
    else:
        return {
            "status": "error",
            "message": f"OAuth for provider '{provider}' is not yet configured.",
        }


# ---------------------------------------------------------------------------
# Event management tools
# ---------------------------------------------------------------------------

@mcp.tool()
def create_calendar_event_tool(
    title: str,
    start_time: str,
    end_time: str = None,
    timezone: str = "Asia/Singapore",
    participants: list = None,
    description: str = "",
    user_id: str = "local_user",
    provider: str = None,
    create_online_meeting: bool = False,
) -> dict:
    """
    Create a calendar event.

    Args:
        title: Event title / summary.
        start_time: ISO datetime string (e.g. '2025-06-15T10:00:00').
        end_time: ISO datetime string. If omitted, defaults to start_time + 30 min.
        timezone: IANA timezone name (e.g. 'Asia/Singapore').
        participants: List of participant names or email addresses.
        description: Optional event description.
        user_id: User identifier (used to look up stored tokens).
        provider: 'google' or 'outlook'. Defaults to preferred provider.
        create_online_meeting: If True, request a Google Meet link.
    """
    init_db()

    # Separate emails from display names
    import re
    _email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    parts = participants or []
    attendees = [p for p in parts if _email_re.match(p.strip())]
    display_names = [p for p in parts if not _email_re.match(p.strip())]

    req = CalendarEventRequest(
        title=title,
        start_time=start_time,
        end_time=end_time,
        timezone=timezone,
        participants=display_names,
        attendees=attendees,
        description=description,
        user_id=user_id,
        provider=provider,
        create_online_meeting=create_online_meeting,
    )
    result = create_calendar_event(req)
    return {
        "provider":      result.provider,
        "status":        result.status,
        "event_id":      result.event_id,
        "html_link":     result.html_link,
        "title":         result.title,
        "start_time":    result.start_time,
        "end_time":      result.end_time,
        "timezone":      result.timezone,
        "attendees":     result.attendees,
        "warnings":      result.warnings,
        "error_code":    result.error_code,
        "error_message": result.error_message,
        "auth_url":      result.auth_url,
    }


@mcp.tool()
def find_free_slots(user_id: str, date: str, duration_minutes: int = 30) -> dict:
    """
    Find free time slots on a given date (stub — not yet implemented).

    Args:
        user_id: User identifier.
        date: Date string in YYYY-MM-DD format.
        duration_minutes: Required slot duration in minutes.
    """
    return {
        "status": "not_implemented",
        "message": "Free slot detection is not yet implemented. Requires Google Calendar freebusy API.",
    }


@mcp.tool()
def update_calendar_event(event_id: str, user_id: str, **kwargs) -> dict:
    """
    Update an existing calendar event (stub — not yet implemented).

    Args:
        event_id: The Google Calendar event ID.
        user_id: User identifier.
        **kwargs: Fields to update (title, start_time, end_time, description, etc.)
    """
    return {
        "status": "not_implemented",
        "message": "Event update is not yet implemented.",
    }


@mcp.tool()
def delete_calendar_event(
    event_id: str,
    user_id: str = "local_user",
    provider: str = "google",
) -> dict:
    """
    Delete a calendar event (stub — not yet implemented).

    Args:
        event_id: The calendar event ID to delete.
        user_id: User identifier.
        provider: Calendar provider name ('google' or 'outlook').
    """
    return {
        "status": "not_implemented",
        "message": "Event deletion is not yet implemented.",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    mcp.run()
