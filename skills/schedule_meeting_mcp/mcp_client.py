"""
mcp_client.py — Thin synchronous wrapper over calendar_service.

No MCP transport is needed for the ADK path — this module calls
calendar_service functions directly so skill.run() stays synchronous.
"""
from __future__ import annotations

from .calendar_service import create_calendar_event, get_provider_status
from .schemas import CalendarEventRequest, CalendarEventResult, CalendarProviderStatus


def create_event(request: CalendarEventRequest) -> CalendarEventResult:
    """Create a calendar event via the appropriate provider."""
    return create_calendar_event(request)


def list_providers(user_id: str) -> list[CalendarProviderStatus]:
    """Return auth status for all supported calendar providers."""
    return get_provider_status(user_id)
