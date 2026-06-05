"""
outlook_calendar.py — Outlook / Microsoft 365 Calendar provider scaffold.

This provider is currently a mock stub. Full implementation requires:
  - Microsoft Graph API credentials (AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID)
  - A completed microsoft_oauth.py
  - microsoft-graph or requests-based Graph API calls

Returns status="mock" so callers know the event was not actually created.
"""
from __future__ import annotations

from ..schemas import CalendarEventRequest, CalendarEventResult, CalendarProviderName
from .base import CalendarProvider


class OutlookCalendarProvider(CalendarProvider):
    """Mock scaffold for Outlook Calendar. Not yet implemented."""

    def get_provider_name(self) -> str:
        return CalendarProviderName.OUTLOOK

    def check_auth(self, user_id: str) -> dict:
        return {
            "connected": False,
            "user_email": None,
            "auth_url": "http://127.0.0.1:8001/auth/microsoft/start",
        }

    def create_event(self, request: CalendarEventRequest) -> CalendarEventResult:
        """Return a mock result — Microsoft Graph integration not yet implemented."""
        return CalendarEventResult(
            provider=CalendarProviderName.OUTLOOK,
            status="mock",
            title=request.title,
            start_time=request.start_time,
            end_time=request.end_time,
            timezone=request.timezone,
            attendees=request.attendees,
            warnings=[
                "Outlook Calendar is not yet implemented. "
                "This is a mock/scaffold result. "
                "Configure AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_TENANT_ID to enable it."
            ],
            error_code="NOT_IMPLEMENTED",
            error_message=(
                "Outlook Calendar integration is a scaffold. "
                "See skills/schedule_meeting_mcp/providers/outlook_calendar.py and "
                "skills/schedule_meeting_mcp/auth/microsoft_oauth.py."
            ),
        )
